import gc

import logging
import optuna
import torch
import typer
import yaml
import math

from functools import partial

from .trainer import TrainerFactory
from .device import resolve_device
from .datamodules import DataModuleFactory
from .configurationmanager import ConfigurationManager
from .experimentmanager import save_study, log_final_epsilon, log_noise_multiplier, save_hpo_metrics

log = logging.getLogger(__name__)


class HyperparameterOptimizer:

    # helper function for reading Optuna hyperparameter configuration that
    # includes types, options, lower and upper bounds, etc for hyperparams
    @staticmethod
    def read_optuna_config(config_fpath):
        with open(config_fpath, 'rb') as fh:
            config = yaml.safe_load(fh)

        # coerce numeric types
        for name, cfg in config.items():
            t = cfg.get('type')
            if t == 'float':
                cfg['min']       = float(cfg['min'])
                cfg['max']       = float(cfg['max'])
                cfg['log_space'] = bool(cfg.get('log_space', False))
            elif t == 'int':
                cfg['min'] = int(cfg['min'])
                cfg['max'] = int(cfg['max'])
            elif t == 'ordered' and 'options' in cfg:
                cfg['options'] = [float(v) for v in cfg['options']]

        return config

    # helper function for reading possible manual trials configued
    @staticmethod
    def read_manual_trials(manual_trials_fpath):
        with open(manual_trials_fpath, 'rb') as fh:
            manual_trials = yaml.safe_load(fh)

        return manual_trials.get('trials', [])

    @staticmethod
    def validate_manual_trials(manual_trials, target_hypers):
        for trial in manual_trials:
            invalid_keys = [key for key in trial if key not in target_hypers]
            if invalid_keys:
                raise ValueError(f'Invalid hyperparameters in manual trial: {invalid_keys}. These should be defined in `--target-hypers`')

    @staticmethod
    def optimize_hypers(config_manager: ConfigurationManager) -> None:

        # we do not want to record gradient norms during the HPO trials. it makes much
        # more sense to record the stuff only for the final eveluation. so save the state,
        # disable gradient recording, and later restore the state before final eval round.
        config_manager.disable_recording()

        configuration = config_manager.configuration

        # the targeted hypers from command line
        target_hypers = configuration.target_hypers

        if len(target_hypers) == 0:
            raise typer.BadParameter('Bayesian optimization enabled and no target hyperparameters for optimization.')

        optuna_config = HyperparameterOptimizer.read_optuna_config(configuration.optuna_config)

        # check that the target hypers are known and appear in Optuna configuration
        for target_hyper in target_hypers:
            if not hasattr(config_manager.hyperparams, target_hyper):
                raise typer.BadParameter(f'Cannot optimize unknown hyperparameter "{target_hyper}".')
            if not target_hyper in optuna_config:
                config_fpath = configuration.optuna_config
                raise typer.BadParameter(f'Hyperparameter "{target_hyper}" not found in Optuna configuration file "{config_fpath}".')

        # optuna needs a global process group with 'gloo' backend for communication.
        # "Create a global gloo backend when group is None and WORLD is nccl."
        # # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
        optuna_process_group = torch.distributed.new_group(
            backend='gloo',
        )

        if torch.distributed.get_rank() == 0:
            log.info('Determining maximum batch size for optimization.')

        max_batch_size = HyperparameterOptimizer.get_max_batch_size(config_manager)

        if torch.distributed.get_rank() == 0:
            log.info(f'- Maximum batch size for optimization: {max_batch_size}.')

        # the optimization objective
        objective = partial(
            HyperparameterOptimizer.objective,
            config_manager,
            optuna_config,
            target_hypers,
            max_batch_size,
            optuna_process_group,
        )

        # start Optuna study only on rank zero
        if torch.distributed.get_rank() == 0:
            journal_fpath = configuration.optuna_journal

            # we will store the information about the trials on disk in a journal file
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(journal_fpath),
            )

            # we manually define the sampler to be able to set the seed
            if configuration.optuna_sampler == 'BoTorchSampler':
                sampler_cls = getattr(optuna.integration, configuration.optuna_sampler)
                sampler = sampler_cls(seed=configuration.seed)
            else:
                sampler_cls = getattr(optuna.samplers, configuration.optuna_sampler)
                sampler = sampler_cls(seed=configuration.seed)

            # should we try to resume an existing study?
            load_if_exists = configuration.optuna_resume
            study = optuna.create_study(
                storage=storage,
                study_name=configuration.experiment_name,
                sampler=sampler,
                load_if_exists=load_if_exists,
                direction=configuration.optuna_direction,
            )

            # add any possible manual trials that we want to run
            if configuration.optuna_manual_trials:
                manual_trials = HyperparameterOptimizer.read_manual_trials(configuration.optuna_manual_trials)
                HyperparameterOptimizer.validate_manual_trials(manual_trials, target_hypers)

                # add the trials to the queue if they are not there yet
                enqueued_trial_count = 0
                for trial in manual_trials:
                    # if batch size is set to full batch (-1), then we need to map
                    # it to the actual full batch to keep optuna happy
                    if trial.get('batch_size') == -1:
                        trial['batch_size'] = max_batch_size

                    # Optuna calls this method inside `enqueue_trial`, but we
                    # need to also keep track of how many manual trials we are
                    # enqueuing that have not been completed yet
                    if not study._should_skip_enqueue(trial):
                        log.info(f'Enqueuing trial: {trial}')
                        study.enqueue_trial(trial, skip_if_exists=True)
                        enqueued_trial_count += 1

                log.info(f'Enqueued {enqueued_trial_count} manual trials.')

                if enqueued_trial_count >= configuration.n_trials:
                    raise ValueError('The number of enqueued trials exceeds or matches the total number of trials. Please reduce the number of manual trials or increase `n_trials` to allow Optuna to perform additional trials.')

                # Adjust number of random trials to account for enqueued trials
                remaining_random_trials = max(configuration.optuna_random_trials - enqueued_trial_count, 0)
                log.info(f'Setting n_startup_trials to {remaining_random_trials} to account for enqueued trials.')
                sampler = sampler_cls(
                    n_startup_trials=remaining_random_trials,
                    seed=configuration.seed,
                )

            study.optimize(
                objective,
                n_trials=configuration.n_trials,
                gc_after_trial=True, # garbage collect after each trial
            )
        else:
            # "Please set trial object in rank-0 node and set `None` in the other rank node.
            # - https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.TorchDistributedTrial.html
            for _ in range(configuration.n_trials):
                objective(None)

        # log the results of the best trial
        if torch.distributed.get_rank() == 0:
            trial = study.best_trial
            log.info(f'Best objective ralue: {trial.value}')
            log.info('Params: ')
            for key, value in trial.params.items():
                log.info(f' - {key}: {value}')

        # first we need to broadcast the best parameters to all ranks,
        # so we pack them into a list for sending
        if torch.distributed.get_rank() == 0:
            raw = study.best_trial.params
            actual = {}

            # remap ordered hypers to their actual values
            for key, idx in raw.items():
                if key.endswith('_idx'):
                    hyper = key[:-4]
                    if hyper == 'batch_size':
                        min_exp = int(math.log2(optuna_config[hyper]['min']))
                        max_exp = int(math.log2(max_batch_size))
                        vals = [2**e for e in range(min_exp, max_exp+1)] + [-1]
                        raw_val = vals[idx]
                        actual[hyper] = max_batch_size if raw_val == -1 else raw_val
                    elif hyper == 'max_grad_norm':
                        actual[hyper] = optuna_config[hyper]['options'][idx]
                else:
                    actual[key] = idx

            # rank 0 broadcasts the known best hypers
            broadcast_objects = [actual]
        else:
            # other receive
            broadcast_objects = [None]

        # now, broadcast the list from rank 0 to all the other ranks
        torch.distributed.broadcast_object_list(broadcast_objects, src=0)

        # and finally, let's unpack the best params from the list
        best_params = broadcast_objects[0]

        # restore the state of gradient/loss recording. if it was enabled, we want
        # to keep it enabled for the final training round
        config_manager.restore_recording()

        # now we can train/evaluate for the final time with best params
        metrics = HyperparameterOptimizer._final_evaluation_round(best_params, config_manager)

        if torch.distributed.get_rank() == 0:
            # save this study to experiment directory
            save_study(config_manager, study, metrics)

    @staticmethod
    def _final_evaluation_round(best_params, config_manager):
        # lastly, we'll train with the training data and the validation data
        # using the best params to get a model that we can evaluate on the test
        # for the final accuracy set
        if torch.distributed.get_rank() == 0:
            log.info('Training final model with best hyperparameters for evaluation.')

        # update the training hypers to the best values from the optimization
        for hyper, best_value in best_params.items():
            setattr(config_manager.hyperparams, hyper, best_value)

        # enable evaluation mode (train on train+valid and validate on test)
        config_manager.configuration.evaluation_mode = True

        # we want to validate during the final evaluation round
        config_manager.configuration.validation_frequency = 1

        # construct the final model
        trainer = TrainerFactory.get_trainer(config_manager)

        if torch.distributed.get_rank() == 0:
            log.info('!! Final training round on the full training dataset (train + valid) and evaluating on test.')
            log.info('--------------------------------------------------------------------------------------------')

        # fit model using training AND validation data. we also use the test
        # set for validation here.
        trainer.fit()

        # now we can evaluate the final performance of the best model
        if torch.distributed.get_rank() == 0:
            log.info('Evaluating final model on the test set.')

        # All ranks participate in test() because _evaluate uses collective
        # ops (all_reduce for loss, torchmetrics sync for metrics).
        loss, metrics = trainer.test()

        if torch.distributed.get_rank() == 0:
            log.info(f'Final loss: {loss:.4f}')

        if torch.distributed.get_rank() == 0:
            if metrics:
                log.info('Final metrics:')
                for key, value in metrics.items():
                    log.info(f' - {key}: {value}.')

            log_final_epsilon(config_manager, trainer)
            log_noise_multiplier(config_manager, trainer)

            # save model if requested
            if save_model := config_manager.configuration.save_model:
                if model_weights_path in config_manager.configuration:
                    save_path = config_manager.configuration.model_weights_path
                else:
                    save_path = Path(config_manager.configuration.log_dir, config_manager.configuration.experiment_name, 'final_model.pt')
                log.info(f'Saving final model after HPO to "{save_path}"...')
                trainer.save_model(save_path)

        return metrics

    @staticmethod
    def objective(
        config_manager: ConfigurationManager,
        optuna_config: dict,
        target_hypers: list,
        max_batch_size: int,
        process_group: torch.distributed.ProcessGroupGloo,
        trial: optuna.trial.Trial,
    ):
        # make the trial support distributed
        # https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
        trial = optuna.integration.TorchDistributedTrial(trial, group=process_group)

        for target_hyper in target_hypers:
            cfg = optuna_config[target_hyper]

            if cfg['type'] == 'ordered':
                if target_hyper == 'batch_size':
                    min_exp = int(math.log2(cfg['min']))
                    max_exp = int(math.log2(max_batch_size))

                    # E.g. 256, 512, 1024, ..., Full batch
                    vals = [2**e for e in range(min_exp, max_exp+1)] + [-1]
                else:
                    vals = cfg['options']

                idx = trial.suggest_int(f'{target_hyper}_idx', 0, len(vals)-1)
                raw = vals[idx]
                hyper_value = max_batch_size if (target_hyper=='batch_size' and raw==-1) else raw

            elif cfg['type'] == 'float':
                hyper_value = trial.suggest_float(
                    target_hyper,
                    cfg['min'],
                    cfg['max'],
                    log=cfg.get('log_space', False),
                )

            elif cfg['type'] == 'int':
                max_value = cfg['max']

                # Using -1 as max in batch size configuration signals full batch
                if target_hyper == 'batch_size' and cfg['max'] == -1:
                    max_value = max_batch_size

                hyper_value = trial.suggest_int(
                    target_hyper,
                    cfg['min'],
                    max_value,
                )

            elif cfg['type'] == 'categorical':
                hyper_value = trial.suggest_categorical(
                    target_hyper,
                    cfg['options'],
                )
            else:
                raise ValueError(f'Unknown type in Optuna config: {cfg["type"]}')

            # update the hyperparameter value in configuration
            setattr(config_manager.hyperparams, target_hyper, hyper_value)

        # while tuning hypers, do not validate
        config_manager.configuration.validation_frequency = 0

        # get the trainer with updated config
        trainer = TrainerFactory.get_trainer(config_manager)

        if torch.distributed.get_rank() == 0:
            log.info(f'Starting trial {trial.number}.')
            log.info(config_manager.hyperparams)

        trainer.fit()

        # optimization objective is the validation loss
        # Again, we need all ranks to participate in loss and metrics computation
        loss, metrics = trainer.validate()

        if torch.distributed.get_rank() == 0:
            log.info('Writing the loss and metrics of current trial into file.')
            save_hpo_metrics(
                config_manager,
                loss,
                metrics,
                trial_index=trial.number,
            )

        # find the correct metric value to use as optimization objective
        target_metric = config_manager.configuration.optuna_target_metric
        if target_metric == 'loss':
            objective = loss
        elif target_metric in metrics:
            objective = metrics[target_metric]
        else:
            raise RuntimeError(f'Metric "{target_metric}" not found in metrics (' + ', '.join(metrics.keys()) + ')')

        # without these, this randomly leads to a nasty segmentation fault
        # on AMD, and a memory related CUDA error on Nvidia.
        del trainer, process_group, trial
        gc.collect()

        return objective

    @staticmethod
    def get_max_batch_size(config_manager: ConfigurationManager):
        device = resolve_device(config_manager.configuration.device)
        datamodule = DataModuleFactory.get_datamodule(
            config_manager.configuration,
            config_manager.hyperparams,
            device,
        )
        datamodule.initialize_datasets_only()
        max_batch_size = datamodule.get_dataset_size()

        return max_batch_size
