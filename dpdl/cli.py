import logging
import sys
import time
from typing import List, Optional, Literal

import torch
import typer
from pydantic import ValidationError
from typing_extensions import Annotated
from pathlib import Path

from .configurationmanager import ConfigurationManager
from .experimentmanager import (
    log_final_epsilon,
    log_runtime,
    log_test_metrics,
    log_train_metrics,
    start_experiment_logging,
)
from .hyperparameteroptimizer import HyperparameterOptimizer
from .models.model_factory import ModelFactory
from .predictor import PredictorFactory
from .trainer import TrainerFactory
from .utils import seed_everything

log = logging.getLogger(__name__)


def _format_validation_error(exc: ValidationError) -> str:
    """
    Format a clean message of possibly multiple validation errors.
    """

    errors = exc.errors() or []
    if not errors:
        return str(exc)

    messages = []
    for err in errors:
        msg = err.get('msg', str(exc))
        loc = err.get('loc') or ()
        loc_parts = []
        for item in loc:
            if item == '__root__':
                continue
            loc_parts.append(str(item))
        if loc_parts:
            messages.append(f"{'.'.join(loc_parts)}: {msg}")
        else:
            messages.append(msg)

    return '; '.join(messages)

def cli(
        ctx: typer.Context,
        command: Annotated[
            str,
            typer.Argument(
                help='Command to run ("train", "optimize", "predict", "train-predict", or "show-layers")',
            )
        ],
        use_steps: Annotated[
            bool,
            typer.Option(
                help='Use steps instead of epochs',
                rich_help_panel='Training options',
            )
        ] = False,
        epochs: Annotated[
            Optional[int],
            typer.Option(
                help='Number of epochs to train',
                rich_help_panel='Training options',
            )
        ] = None,
        total_steps: Annotated[
            Optional[int],
            typer.Option(
                help='Total number of gradient updates',
                rich_help_panel='Training options',
            )
        ] = None,
        learning_rate: Annotated[
            float,
            typer.Option(
                help='Learning rate',
                rich_help_panel='Training options',
            )
        ] = 1e-3,
        lr_scheduler: Annotated[
            Literal['none', 'bnb_linear_decay'],
            typer.Option(
                help='Learning-rate scheduler',
                rich_help_panel='Training options',
            )
        ] = 'none',
        batch_size: Annotated[
            Optional[int],
            typer.Option(
                help='Batch size',
                rich_help_panel='Training options',
            )
        ] = 1024,
        sample_rate: Annotated[
            Optional[float],
            typer.Option(
                help='Sample rate',
                rich_help_panel='Training options',
            )
        ] = None,
        max_length: Annotated[
            Optional[int],
            typer.Option(
                help='Max tokenizer length',
                rich_help_panel='Training options'
            )
        ] = None,
        optimizer: Annotated[
            str,
            typer.Option(
                help='Optimizer',
                rich_help_panel='Training options',
            )
        ] = 'Adam',
        optimizer_momentum: Annotated[
            Optional[float],
            typer.Option(
                help='Optimizer momentum (BSR paper `beta`)',
                rich_help_panel='Training options',
            )
        ] = None,
        optimizer_weight_decay: Annotated[
            float,
            typer.Option(
                help='Optimizer weight decay (BSR paper `alpha`)',
                rich_help_panel='Training options',
            )
        ] = 0.0,
        physical_batch_size: Annotated[
            Optional[int],
            typer.Option(
                help='Largest size batch that fits in GPU memory',
                rich_help_panel='Training options',
            )
        ] = 40,
        num_workers: Annotated[
            int,
            typer.Option(
                help='Number of workers for data loading (per GPU)',
                rich_help_panel='Training options',
            )
        ] = 7,
        validation_frequency: Annotated[
            float,
            typer.Option(
                help='Validation frequency',
                rich_help_panel='Training options',
            )
        ] = 1.0,
        seed: Annotated[
            int,
            typer.Option(
                help='Random seed',
                rich_help_panel='Training options',
            )
        ] = 0,
        privacy: Annotated[
            bool,
            typer.Option(
                help='Enable privacy (Opacus)',
                rich_help_panel='Training options',
            )
        ] = True,
        cache_features: Annotated[
            Optional[bool],
            typer.Option(
                help='Cache features from feature extractor (requires head-only training)',
                rich_help_panel='Training options',
            )
        ] = False,
        evaluation_mode: Annotated[
            bool,
            typer.Option(
                help='Enable evaluation mode (train on train+valid and validate on test)',
                rich_help_panel='Training options',
            )
        ] = False,
        checkpoint_step_interval: Annotated[
            Optional[int],
            typer.Option(
                help='Save model checkpoint on every nth step',
                rich_help_panel='Training options',
            )
        ] = None,
        prediction_save_gradient_data: Annotated[
            Optional[bool],
            typer.Option(
                help='Save also gradient information when predicting',
                rich_help_panel='Prediction options',
            )
        ] = False,
        model_name: Annotated[
            str,
            typer.Option(
                help='PyTorch Image Models (timm) model name',
                rich_help_panel='Model options',
            )
        ] = 'resnetv2_50x1_bit.goog_in21k',
        llm: Annotated[
            bool,
            typer.Option(
                help='Enable LLM model mode (use HuggingFace models and tokenization)',
                rich_help_panel='Training options',
            )
        ] = False,
        task: Annotated[
            str,
            typer.Option(
                help='Training task [ImageClassification, SequenceClassification, CausalLM, InstructLM]',
                rich_help_panel='Training options',
            )
        ] = 'ImageClassification',
        load_in_4bit: Annotated[
            bool,
            typer.Option(
                help='Quantization in 4 bit',
                rich_help_panel='Model options'
            )
        ] = False,
        loss_function: Annotated[
            str,
            typer.Option(
                help='PyTorch Module Loss Function Name',
                rich_help_panel='Loss Function options',
            )
        ] = 'CrossEntropyLoss',
        pretrained: Annotated[
            bool,
            typer.Option(
                help='Use pretrained model',
                rich_help_panel='Model options',
            )
        ] = True,
        zero_head: Annotated[
            bool,
            typer.Option(
                help='Set model head weights to zero',
                rich_help_panel='Model options',
            )
        ] = False,
        peft: Annotated[
            str,
            typer.Option(
                help='Use Parameter Efficient Fine-tuning ("lora", "film", "head-only")',
                rich_help_panel='Model options',
            )
        ] = None,
        lora_rank: Annotated[
            Optional[int],
            typer.Option(
                help='Rank for LoRA fine-tuning',
                rich_help_panel='Model options',
            )
        ] = None,
        weight_perturbation_level: Annotated[
            float,
            typer.Option(
                help='Pretrained weight perturbation noise level',
                rich_help_panel='Model options',
            )
        ] = 0,
        model_weights_path: Annotated[
            Optional[str],
            typer.Option(
                help='File path to loading or saving model weights',
                rich_help_panel='Model options',
            )
        ] = None,
        save_model: Annotated[
            Optional[bool],
            typer.Option(
                help='Save the resulting model under log directory',
                rich_help_panel='Model options',
            )
        ] = False,
        dataset_name: Annotated[
            str,
            typer.Option(
                help='Dataset name',
                rich_help_panel='Dataset options',
            )
        ] = 'uoft-cs/cifar10',
        dataset_path: Annotated[
            Optional[str],
            typer.Option(
                help='Load local dataset on disk from given path',
                rich_help_panel='Dataset options',
            )
        ] = None,
        subset_size: Annotated[
            float,
            typer.Option(
                help='Only load subset of the dataset (0.1 indicate 10%)',
                rich_help_panel='Dataset options',
            )
        ] = 1.0,
        validation_size: Annotated[
            Optional[float],
            typer.Option(
                help='Validation set size, if we need to split it from train (0.1 indicates 10%)',
                rich_help_panel='Dataset options',
            )
        ] = 0.1,
        test_size: Annotated[
            Optional[float],
            typer.Option(
                help='Test set size, if we need to split it from train (0.1 indicates 10%)',
                rich_help_panel='Dataset options',
            )
        ] = 0.1,
        shots: Annotated[
            Optional[int],
            typer.Option(
                help='Number of shots (training example per class) to use',
                rich_help_panel='Dataset options',
            )
        ] = None,
        stratify_shots: Annotated[
            Optional[bool],
            typer.Option(
                help='Use stratified sampling when constructing few-shot dataset',
                rich_help_panel='Dataset options',
            )
        ] = True,
        dataset_label_field: Annotated[
            Optional[str],
            typer.Option(
                help='Name of the field that determines label for the dataset',
                rich_help_panel='Dataset options',
            )
        ] = None,
        dataset_text_fields: Annotated[
            Optional[List[str]],
            typer.Option(
                help='Name of the field(s) where the text is located for language tasks',
                rich_help_panel='Dataset options',
            )
        ] = None,
        imbalance_factor: Annotated[
            Optional[float],
            typer.Option(
                help='Parameter of the exponential distribution for imbalancing an dataset',
                rich_help_panel='Dataset options',
            )
        ] = None,
        imbalance_reverse: Annotated[
            Optional[bool],
            typer.Option(
                help='Flip the order of classes when imbalancing',
                rich_help_panel='Dataset options',
            )
        ] = None,
        fairness_imbalance_class: Annotated[
            Optional[int],
            typer.Option(
                help="Class to imbalance for fairness-style experiments, i.e., only one class is subsampled",
                rich_help_panel="Dataset options",
            )
        ] = None,
        max_test_examples: Annotated[
            Optional[int],
            typer.Option(
                help='Limit for the maximum number of examples in test/validation set (truncate)',
                rich_help_panel='Dataset options',
            )
        ] = None,
        cache_dataset_transforms: Annotated[
            Optional[bool],
            typer.Option(
                help='Cache the image transformations on disk (faster to disable for small images)',
                rich_help_panel='Dataset options',
            )
        ] = False,
        split_seed: Annotated[
            Optional[int],
            typer.Option(
                help='Random seed for creating dataset subsets',
                rich_help_panel='Dataset options',
            )
        ] = 42,
        log_dir: Annotated[
            str,
            typer.Option(
                help='Log directory',
                rich_help_panel='Logging options',
            )
        ] = 'logs',
        experiment_name: Annotated[
            Optional[str],
            typer.Option(
                help='Experiment name for logging',
                rich_help_panel='Logging options',
            )
        ] = 'default',
        overwrite_experiment: Annotated[
            bool,
            typer.Option(
                help='Overwrite existing experiment logs',
                rich_help_panel='Logging options',
            )
        ] = False,
        device: Annotated[
            str,
            typer.Option(
                help="Device to run on ('cuda', 'cpu', or 'auto')",
                rich_help_panel='Runtime options',
                envvar='DPDL_DEVICE',
            )
        ] = 'auto',
        record_clipping: Annotated[
            Optional[bool],
            typer.Option(
                help='Record clipping stats (MSE)',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_memory_usage: Annotated[
            bool,
            typer.Option(
                help='Record GPU memory usage peaks.',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_training_time: Annotated[
            bool,
            typer.Option(
                help='Record epoch duration and mean time per logical optimizer step.',
                rich_help_panel='Logging options',
            )
        ] = False,
        noise_history_mode: Annotated[
		Optional[str],
		typer.Option(
			help="Inverse-noise history: regenerate (default) or stored."
		)
	] = None,
        record_snr: Annotated[
            Optional[bool],
            typer.Option(
                help='Record signal-to-noise-ratio',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_llm_samples: Annotated[
            Optional[bool],
            typer.Option(
                help='Generate and log LLM samples at epoch end',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_gradient_norms: Annotated[
            Optional[bool],
            typer.Option(
                help='Record layer-wise gradients before clipping',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_gradient_norms_quantiles: Annotated[
            Optional[List[int]],
            typer.Option(
                help='Quantiles for gradient norms',
                rich_help_panel='Logging options',
            )
        ] = [25, 50, 75],
        record_loss_by_step: Annotated[
            Optional[bool],
            typer.Option(
                help='Record train loss by step',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_loss_by_epoch: Annotated[
            Optional[bool],
            typer.Option(
                help='Record train/validation loss by epoch',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_per_class_accuracy: Annotated[
            Optional[bool],
            typer.Option(
                help='Record per-class accuracy',
                rich_help_panel='Logging options',
            )
        ] = False,
        record_final_train_accuracy: Annotated[
            Optional[bool],
            typer.Option(
                help='Evaluate the final accuracy also on the training set',
                rich_help_panel='Logging options',
            )
        ] = False,
        disable_epsilon_logging: Annotated[
            Optional[bool],
            typer.Option(
                help='Disable logging of final epsilon (only needed if this causes problems with e.g. low noise multiplier)',
                rich_help_panel='Logging options',
            )
        ] = False,
        noise_multiplier: Annotated[
            Optional[float],
            typer.Option(
                help='Noise multiplier',
                rich_help_panel='Opacus options',
            )
        ] = None,
        max_grad_norm: Annotated[
            Optional[float],
            typer.Option(
                help='Maximum gradient norm (for clipping)',
                rich_help_panel='Opacus options',
            )
        ] = 1.0,
        clipping_mode: Annotated[
            Optional[str],
            typer.Option(
                help='Opacus clipping mode ("flat" or "per_layer" or "adaptive")',
                rich_help_panel='Opacus options',
            )
        ] = 'flat',
        secure_mode: Annotated[
            Optional[bool],
            typer.Option(
                help='Enable secure mode for production use',
                rich_help_panel='Opacus options',
            )
        ] = False,
        poisson_sampling: Annotated[
            Optional[bool],
            typer.Option(
                help='Enable Opacus Poisson sampling',
                rich_help_panel='Opacus options',
            )
        ] = True,
        normalize_clipping: Annotated[
            Optional[bool],
            typer.Option(
                help='Normalize clipping (to decouple the learning rate and max_grad_norm)',
                rich_help_panel='Opacus options',
            )
        ] = False,
        accountant: Annotated[
            Optional[str],
            typer.Option(
                help='Privacy accountant',
                rich_help_panel='Opacus options',
            )
        ] = 'prv',
        noise_mechanism: Annotated[
            Optional[str],
            typer.Option(
                help='Noise mechanism ("gaussian", "bandmf", "bsr", "bisr", "bandinvmf", "bifr", or "blt"); BLT is workload-driven and does not expose lambda',
                rich_help_panel='Opacus options',
            )
        ] = 'gaussian',
        sampling_mode: Annotated[
            Optional[str],
            typer.Option(
                help='Sampling semantics ("torch_sampler", "cyclic_poisson", "b_min_sep", "balls_in_bins")',
                rich_help_panel='Opacus options',
            )
        ] = None,
        bsr_coeffs: Annotated[
            Optional[List[float]],
            typer.Option(
                help='BSR Toeplitz coefficients (repeat flag for multiple values)',
                rich_help_panel='BSR options',
            )
        ] = None,
        bsr_bands: Annotated[
            Optional[int],
            typer.Option(
                help='BSR bands (required for cyclic_poisson semantics)',
                rich_help_panel='BSR options',
            )
        ] = None,
        bifr_frac: Annotated[
            Optional[float],
            typer.Option(
                help='BIFR interpolation parameter frac in [0,1] for fixed-batch BIFR',
                rich_help_panel='BSR options',
            )
        ] = None,
        blt_buffers: Annotated[
            Optional[int],
            typer.Option(
                help='BLT workload complexity control; Opacus resolves theta/theta_hat implicitly from workload inputs and this buffer-count knob',
                rich_help_panel='BSR options',
            )
        ] = None,
        bsr_mf_sensitivity: Annotated[
            Optional[float],
            typer.Option(
                help='BSR MF sensitivity override for accounting/calibration',
                rich_help_panel='BSR options',
            )
        ] = None,
        bnb_b: Annotated[
            Optional[int],
            typer.Option(
                help='Optional BNB structured-sampling integer: bins for balls_in_bins, min-separation b for b_min_sep; if omitted, DPDL derives balls_in_bins bins from steps per epoch and b_min_sep b from bands when available',
                rich_help_panel='BNB options',
            )
        ] = None,
        bnb_p: Annotated[
            Optional[float],
            typer.Option(
                help='Optional BNB b-min-separation eligibility probability p in (0,1]; if omitted, DPDL derives it from the target average batch size and b',
                rich_help_panel='BNB options',
            )
        ] = None,
        bnb_bands: Annotated[
            Optional[int],
            typer.Option(
                help='BNB Toeplitz bands (required for BNB accounting)',
                rich_help_panel='BNB options',
            )
        ] = None,
        bnb_num_samples: Annotated[
            Optional[int],
            typer.Option(
                help='BNB Monte Carlo sample budget',
                rich_help_panel='BNB options',
            )
        ] = None,
        bnb_chunk_size: Annotated[
            Optional[int],
            typer.Option(
                help='BNB Monte Carlo chunk size',
                rich_help_panel='BNB options',
            )
        ] = None,
        bnb_seed: Annotated[
            Optional[int],
            typer.Option(
                help='BNB Monte Carlo RNG seed',
                rich_help_panel='BNB options',
            )
        ] = None,
        bnb_calibration_mode: Annotated[
            Optional[Literal['evr', 'optimistic']],
            typer.Option(
                help='BNB calibration mode ("evr" keeps the EVR guard, "optimistic" disables it)',
                rich_help_panel='BNB options',
            )
        ] = None,
        target_epsilon: Annotated[
            Optional[float],
            typer.Option(
                help='Target epsilon for the privacy accountant (delta defaults to min(1e-5, 1/10^ceil(log10(N))))',
                rich_help_panel='Opacus options',
            )
        ] = None,
        noise_batch_ratio: Annotated[
            Optional[float],
            typer.Option(
                help='Noise-batch ratio (https://arxiv.org/abs/2501.18914)',
                rich_help_panel='Opacus options',
            )
        ] = None,
        target_hypers: Annotated[
            Optional[List[str]],
            typer.Option(
                help='Hyperparameters to optimize (use multiple times if necessary)',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = [],
        n_trials: Annotated[
            Optional[int],
            typer.Option(
                help='Number of optimization rounds',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 20,
        optuna_random_trials: Annotated[
            Optional[int],
            typer.Option(
                help='Number of random trials to start the optimization with',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 10,
        optuna_target_metric: Annotated[
            Optional[str],
            typer.Option(
                help='Target metric for Bayesian optimization',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'loss',
        optuna_direction: Annotated[
            Optional[str],
            typer.Option(
                help='Direction for Bayesian optimization ("minimize" or "maximize")',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'minimize',
        optuna_config: Annotated[
            Optional[str],
            typer.Option(
                help='Configuration file containing ranges/options for hypers',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'conf/optuna_hypers.conf',
        optuna_manual_trials: Annotated[
            Optional[str],
            typer.Option(
                help='Configuration file defining manual trials to start the optimziation with',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = None,
        optuna_journal: Annotated[
            Optional[str],
            typer.Option(
                help='Optuna journal (logging) file path',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'optuna.journal',
        optuna_resume: Annotated[
            Optional[bool],
            typer.Option(
                help='Resume previous Optuna study',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = False,
        optuna_sampler: Annotated[
            Optional[str],
            typer.Option(
                help='Optuna sampler (a class from optuna.samplers)',
                rich_help_panel='Bayesian optimization (Optuna) options',
            )
        ] = 'BoTorchSampler',
        verbose_callback: Annotated[
            Optional[bool],
            typer.Option(
                help='Enable debug callback for detailed output',
                rich_help_panel='',
            )
        ] = False,
        predict_dataset_split: Annotated[
            Optional[str],
            typer.Option(
                '--predict-dataset-split',
                help='Dataset split to use for prediction',
                rich_help_panel='Prediction options',
            )
        ] = 'test',
    ):

    # Map from commands to functions
    HANDLERS = {
        'train': run_train,
        'optimize': run_optimize,
        'predict': run_predict,
        'train-predict': run_train_and_predict,
    }

    # Throw a clean error to typer instead of outputting a stacktrace
    try:
        config_manager = ConfigurationManager(ctx.params)
    except ValidationError as exc:
        raise typer.BadParameter(_format_validation_error(exc))
    except ValueError as exc:
        raise typer.BadParameter(str(exc))

    command = config_manager.get_command()

    if command == 'show-layers':
        run_show_layers(config_manager)
        return

    # ConfigurationManager knows our experiment
    # directory, so let's start logging also there
    if torch.distributed.get_rank() == 0:
        start_experiment_logging(log.parent, config_manager)

    torch.distributed.barrier()

    handler = HANDLERS.get(command)
    if handler is None:
        raise typer.BadParameter(f'Unknown command "{command}".')

    handler(config_manager)


def run_show_layers(config_manager: ConfigurationManager) -> None:
    log.info(config_manager.configuration)
    log.info('Showing model layers.')
    model, _ = ModelFactory.get_model(
        config_manager.configuration,
        config_manager.hyperparams,
    )
    model.show_layers()


def run_train(config_manager: ConfigurationManager) -> Optional[Path]:
    rank_zero = torch.distributed.get_rank() == 0

    if rank_zero:
        log.info('Starting training.')
        log.info(config_manager.hyperparams)
        log.info(config_manager.configuration)

    seed_everything(config_manager.configuration.seed)

    trainer = TrainerFactory.get_trainer(config_manager)

    start_time = time.time()
    trainer.fit()
    end_time = time.time()

    # log final train accuracy if needed
    if config_manager.configuration.record_final_train_accuracy:
        if rank_zero:
            log.info('Evaluating on train set..')

        train_loss, train_metrics = trainer._evaluate('train', enable_callbacks=False)

        if rank_zero:
            log_train_metrics(config_manager, train_metrics, train_loss)

    # log test accuracy and run time, and save model if asked
    if rank_zero:
        log.info('Evaluating on test set..')
        test_loss, test_metrics = trainer.test()

        log_test_metrics(config_manager, test_metrics, test_loss)
        log_runtime(config_manager, start_time, end_time)

    if not config_manager.configuration.disable_epsilon_logging:
        # Correlated noise accounting can use Monte-Carlo methods, which are embarassingly
        # parallel. To distribute this slow calculation is the only reason to resolve the
        # final epsilon on all ranks.
        log_final_epsilon(config_manager, trainer)

    saved_model_path = None

    # Should we save the model?
    if config_manager.configuration.save_model:
        if config_manager.configuration.model_weights_path:
            save_path = Path(config_manager.configuration.model_weights_path)
        else:
            save_path = Path(
                config_manager.configuration.log_dir,
                config_manager.configuration.experiment_name,
                'final_model.pt',
            )
            config_manager.configuration.model_weights_path = str(save_path)

        if rank_zero:
            log.info(f'Saving model to "{save_path}"...')
            trainer.save_model(save_path)
            log.info('Saving model done.')
            saved_model_path = save_path

        torch.distributed.barrier()

    return saved_model_path


def run_optimize(config_manager: ConfigurationManager) -> None:
    if torch.distributed.get_rank() == 0:
        log.info('Starting hyperparameter optimization.')
        log.info(config_manager.configuration)

    seed_everything(config_manager.configuration.seed)

    start_time = time.time()
    HyperparameterOptimizer.optimize_hypers(config_manager)
    end_time = time.time()

    # log the runtime
    if torch.distributed.get_rank() == 0:
        log_runtime(config_manager, start_time, end_time)


def run_predict(config_manager: ConfigurationManager) -> None:
    if torch.distributed.get_rank() == 0:
        log.info('Starting prediction.')
        log.info(config_manager.configuration)

    seed_everything(config_manager.configuration.seed)

    start_time = time.time()

    predictor = PredictorFactory.get_predictor(config_manager)
    predictor.predict(config_manager.configuration)

    end_time = time.time()
    log_runtime(config_manager, start_time, end_time)


def run_train_and_predict(config_manager: ConfigurationManager) -> None:
    train_config_manager = config_manager.clone_with_overrides(
        command='train',
        save_model=True,
    )
    saved_model_path = run_train(train_config_manager)

    # Only rank 0 saved the model, so we need to synchronize the
    # saved model path to other ranks. Otherwise they don't know
    # where to load th emodel from.
    saved_model_path = synchronize_saved_model_path(saved_model_path)

    predict_config_manager = config_manager.clone_with_overrides(
        command='predict',
        model_weights_path=str(saved_model_path),
        privacy=False,  # We of course predict without Opacus.
        record_memory_usage=False,
        record_training_time=False,
        num_workers=0,  # Avoid reusing workers, we don't need many for this.
    )
    run_predict(predict_config_manager)


def synchronize_saved_model_path(saved_model_path: Optional[Path]) -> Path:
    path_list = [str(saved_model_path) if saved_model_path else None]
    torch.distributed.broadcast_object_list(path_list, src=0)
    path_str = path_list[0]

    if path_str is None:
        raise RuntimeError('Prediction failed: Could not find saved model.')

    return Path(path_str)
