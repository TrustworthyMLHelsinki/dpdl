import logging
import pathlib
import torch
import typer

from pydantic import BaseModel, root_validator
from typing import Optional, List, Literal

log = logging.getLogger(__name__)

class Hyperparameters(BaseModel):
    learning_rate: float = 1e-3
    epochs: Optional[int] = None
    total_steps: Optional[int] = None
    batch_size: Optional[int] = None
    sample_rate: Optional[float] = None
    noise_multiplier: Optional[float]
    max_grad_norm: Optional[float]
    target_epsilon: Optional[float]
    privacy: bool = True # Only used in __str__
    max_length: Optional[int] = None

    @root_validator(pre=True)
    def check_batch_size_or_sample_rate(cls, values):
        batch_size, sample_rate = values.get('batch_size'), values.get('sample_rate')

        if all([batch_size, sample_rate]):
            raise ValueError('Either batch_size or sample_rate must be set, but not both.')

        return values

    @root_validator(pre=True)
    def check_target_epsilon_or_noise_multiplier(cls, values):
        target_epsilon, noise_multiplier = values.get('target_epsilon'), values.get('noise_multiplier')

        if all([target_epsilon, noise_multiplier]):
            raise ValueError('Both, target_epsilon and noise_multiplier given.')

        return values

    @root_validator(pre=True)
    def check_epochs(cls, values):
        epochs = values.get('epochs')
        total_steps = values.get('total_steps')

        if epochs is None and total_steps is None:
            raise ValueError(
                'Missing training length: set `--epochs` or `--total-steps`.'
            )

        return values

    def __str__(self):
        hypers = [
            ('Epochs', self.epochs),
            ('Total steps', self.total_steps),
            ('Learning rate', self.learning_rate),
            ('Batch size', self.batch_size),
            ('Max length', self.max_length)
        ]

        if self.privacy:
            privacy_hypers = [
                ('Sample rate', self.sample_rate),
                ('Noise multiplier', self.noise_multiplier),
                ('Max grad norn', self.max_grad_norm),
                ('Target epsilon', self.target_epsilon),
            ]
            hypers.extend(privacy_hypers)

        max_key_length = max(len(hyper[0]) for hyper in hypers)
        hyper_str = [f'{hyper[0]:<{max_key_length}}: {hyper[1]}' for hyper in hypers]

        return 'Hyperparameters:\n  ' + '\n  '.join(hyper_str) + '\n'

class Configuration(BaseModel):
    command: Literal['train', 'optimize', 'predict', 'show-layers', 'train-predict']
    privacy: bool = True
    model_name: str = 'resnetv2_50x1_bit.goog_in21k'
    loss_function: str = 'CrossEntropyLoss'
    optimizer: str = 'Adam'
    dataset_name: str = 'uoft-cs/cifar10'
    dataset_path: Optional[str] = None
    llm: bool = False
    task: Literal['ImageClassification', 'SequenceClassification', 'CausalLM', 'InstructLM' ] = 'ImageClassification'
    physical_batch_size: int = 40
    num_workers: int = 7
    validation_frequency: float = 1.0
    seed: int = 0
    log_dir: str = 'logs'
    checkpoints_dir: str = None
    experiment_name: str = 'default'
    overwrite_experiment: bool = False
    device: Literal['cuda', 'cpu', 'auto'] = 'auto'
    clipping_mode: str = 'flat'
    secure_mode: bool = False
    accountant: str = 'prv'
    poisson_sampling: bool = True
    normalize_clipping: bool = False
    n_trials: int = 20
    optuna_random_trials: int = 10
    target_hypers: List[str] = []
    optuna_target_metric: str = 'loss'
    optuna_direction: Literal['minimize', 'maximize'] = 'minimize'
    optuna_config: str = 'conf/optuna_hypers.conf'
    metric_config: Optional[str] = None
    optuna_manual_trials: Optional[str] = None
    optuna_journal: str = 'optuna.journal'
    optuna_resume: bool = False
    optuna_sampler: str = 'BoTorchSampler'
    subset_size: Optional[float] = 1.0
    shots: Optional[int] = None
    stratify_shots: Optional[bool] = True
    zero_head: bool = False
    peft: Optional[Literal['lora', 'film', 'head-only']] = None
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    pretrained: bool = True
    cache_features: Optional[bool] = False
    use_steps: Optional[bool] = False
    evaluation_mode: Optional[bool] = False
    dataset_label_field: Optional[str] = None
    dataset_text_fields: Optional[List[str]] = None
    max_test_examples: Optional[int] = None
    imbalance_factor: Optional[float] = None
    imbalance_reverse: Optional[bool] = False
    fairness_imbalance_class: Optional[int] = None
    validation_size: Optional[float] = 0.1
    test_size: Optional[float] = 0.1
    save_model: Optional[bool] = False
    model_weights_path: Optional[str] = None
    record_clipping: Optional[bool] = False
    record_snr: Optional[bool] = False
    record_optimizer_stats: Optional[bool] = False
    record_llm_samples: Optional[bool] = False
    record_gradient_norms: Optional[bool] = False
    record_gradient_norms_quantiles: Optional[List[int]] = [25, 50, 75]
    verbose_callback: Optional[bool] = False
    cache_dataset_transforms: Optional[bool] = False
    weight_perturbation_level: float = 0
    record_loss_by_step: Optional[bool] = False
    record_loss_by_epoch: Optional[bool] = False
    record_metrics_by_epoch: Optional[bool] = False
    record_per_class_accuracy: Optional[bool] = False
    record_final_train_accuracy: Optional[bool] = False
    checkpoint_step_interval: Optional[int] = None
    checkpoint_steps: Optional[List[int]] = None
    skip_test: Optional[bool] = False
    disable_epsilon_logging: Optional[bool] = False
    split_seed: Optional[int] = 42
    predict_dataset_split: Optional[str] = 'test'
    prediction_save_gradient_data: Optional[bool] = False
    load_in_4bit: bool = False

    class Config:
        # Fix Pydantic warning:
        # UserWarning: Field "model_name" has conflict with protected namespace "model_".
        protected_namespaces = ()

    @root_validator(pre=True)
    def check_fairness_imbalance_factor(cls, values):
        imbalance_factor = values.get('imbalance_factor')
        fairness_imbalance_class = values.get('fairness_imbalance_class')

        if fairness_imbalance_class and not imbalance_factor:
            raise ValueError(
                'Parameter "imbalance_factor" is required when using "fairness_imbalance_class".'
            )

        return values

    @root_validator(pre=True)
    def check_command(cls, values):
        command = values.get('command')

        if command not in ['train', 'optimize', 'predict', 'show-layers', 'train-predict']:
            raise ValueError('Command must be "train", "optimize", "predict", "show-layers", or "train-predict".')

        return values

    @root_validator(pre=True)
    def check_total_steps(cls, values):
        total_steps = values.get('total_steps')
        epochs = values.get('epochs')

        if total_steps and epochs:
            raise ValueError('Parameters "epochs" and "total_steps" are exclusive.')

        return values

    @root_validator(pre=True)
    def check_shots_and_subset_size(cls, values):
        shots = values.get('shots')
        subset_size = values.get('subset_size')

        if shots and subset_size:
            raise ValueError('Parameters "shots" and "subset_size" are exclusive.')

        return values

    @root_validator(pre=True)
    def check_feature_cache(cls, values):
        cache_features = values.get('cache_features')
        peft_method = values.get('peft')

        if cache_features and peft_method != 'head-only':
            raise ValueError(f'Head only training required if feature cache is enabled.')

        return values

    @root_validator(pre=True)
    def check_lora_rank(cls, values):
        if values.get('peft') == 'lora':
            lora_rank = values.get('lora_rank')

            if lora_rank is None:
                raise ValueError('Parameter "lora_rank" is required when using LoRA.')

            if lora_rank <= 0:
                raise ValueError('Parameter "lora_rank" must be positive.')

            lora_alpha = values.get('lora_alpha')
            if lora_alpha is None:
                values['lora_alpha'] = lora_rank
            elif lora_alpha <= 0:
                raise ValueError('Parameter "lora_alpha" must be positive.')

        return values

    def __str__(self):
        attributes = [
            ('Command', self.command),
            ('Privacy', self.privacy),
            ('Model name', self.model_name),
            ('Optimizer', self.optimizer),
            ('Dataset name', self.dataset_name),
            ('Dataset path', self.dataset_path),
            ('Dataset label field', self.dataset_label_field),
            ('Dataset text field(s) for LLM tasks', self.dataset_text_fields),
            ('Dataset imbalance factor', self.imbalance_factor),
            ('Dataset imbalance reverse', self.imbalance_reverse),
            ('Cache dataset transforms', self.cache_dataset_transforms),
            ('Validation size', self.validation_size),
            ('Test size', self.test_size),
            ('Physical batch size', self.physical_batch_size),
            ('Num workers', self.num_workers),
            ('Validation frequency', self.validation_frequency),
            ('Seed', self.seed),
            ('Log dir', self.log_dir),
            ('Experiment name', self.experiment_name),
            ('Overwrite experiment', self.overwrite_experiment),
            ('Device', self.device),
            ('Shots', self.shots),
            ('Use stratified sampling for few-shot dataset', self.stratify_shots),
            ('Subset size', self.subset_size),
            ('Zero head weights', self.zero_head),
            ('PEFT method', self.peft),
            ('LoRA rank', self.lora_rank),
            ('LoRA alpha', self.lora_alpha),
            ('Use pretrained model', self.pretrained),
            ('Pretrained model weight perturbation noise level', self.weight_perturbation_level),
            ('Use precomputed features', self.cache_features),
            ('Use steps instead of epochs', self.use_steps),
            ('Evaluation mode', self.evaluation_mode),
            ('Save final model', self.save_model),
            ('Path for saving/loding model weights', self.model_weights_path),
            ('Record clipping stats (MSE)', self.record_clipping),
            ('Record signal-to-noise ratio', self.record_snr),
            ('Record optimizer (gradient & Adam) stats', self.record_optimizer_stats),
            ('Record LLM samples', self.record_llm_samples),
            ('Record gradient norms', self.record_gradient_norms),
            ('Record gradient norms quantiles', self.record_gradient_norms_quantiles),
            ('Record train loss by step', self.record_loss_by_step),
            ('Record train/valid loss and accuracy by epoch', self.record_loss_by_epoch),
            ('Record train/valid metrics by epoch', self.record_metrics_by_epoch),
            ('Record per-class accuracy', self.record_per_class_accuracy),
            ('Record final training accuracy', self.record_final_train_accuracy),
            ('Checkpoint every nth step', self.checkpoint_step_interval),
            ('Checkpoint at exact steps', self.checkpoint_steps),
            ('Skip test evaluation', self.skip_test),
            ('Enable callback debug logging', self.verbose_callback),
            ('Fairness-style subsampling class', self.fairness_imbalance_class),
            ('Random seed for creating dataset subsets', self.split_seed),
            ('Metric configuration', self.metric_config),
            ('LLM use', self.llm),
            ('Task', self.task),
        ]

        if self.privacy:
            privacy_attributes = [
                ('Clipping mode', self.clipping_mode),
                ('Secure mode', self.secure_mode),
                ('Accountant', self.accountant),
                ('Poisson sampling', self.poisson_sampling),
                ('Normalize clipping', self.normalize_clipping),
            ]
            attributes.extend(privacy_attributes)

        if self.command == 'optimize':
            optuna_attributes = [
                ('N trials', self.n_trials),
                ('Target hypers', ', '.join(self.target_hypers)),
                ('Optuna target metric', self.optuna_target_metric),
                ('Optuna direction', self.optuna_direction),
                ('Optuna config', self.optuna_config),
                ('Optuna manual trials configuration', self.optuna_manual_trials),
                ('Optuna journal', self.optuna_journal),
                ('Optuna resume', self.optuna_resume),
                ('Optuna number of random trials', self.optuna_random_trials),
            ]
            attributes.extend(optuna_attributes)
        elif self.command == 'predict':
            predict_attributes = [
                ('Prediction dataset split', self.predict_dataset_split),
                ('Save gradient information when predicting', self.prediction_save_gradient_data),
            ]
            attributes.extend(predict_attributes)

        max_key_length = max(len(attr[0]) for attr in attributes)
        attribute_str = [f'{attr[0]:<{max_key_length}}: {attr[1]}' for attr in attributes]

        return 'Configuration:\n  ' + '\n  '.join(attribute_str) + '\n'

class ConfigurationManager:
    def __init__(self, cli_params: dict):
        self._cli_params = dict(cli_params)
        self.command = cli_params['command']

        self.configuration = Configuration(**cli_params)
        self.hyperparams = Hyperparameters(**cli_params)

        # remove the target hypers from hyperparams as they will be set in trials
        for target_hyper in self.configuration.target_hypers:
            setattr(self.hyperparams, target_hyper, None)

    def get_command(self):
        return self.command

    def disable_recording(self):
        """
        Disable all the recording flags.

        This is especially for HPO where we don't want to do the recordings for all
        the trials, but only for the final evaluation round.
        """

        cfg = self.configuration

        # get all record_ flags from the Configuration object
        self._record_backup = {
            k: getattr(cfg, k)
            for k in vars(cfg)
            if k.startswith('record_') and isinstance(getattr(cfg, k), bool)
        }
        for k in self._record_backup:
            setattr(cfg, k, False)

    def restore_recording(self):
        cfg = self.configuration
        for k, v in self._record_backup.items():
            setattr(cfg, k, v)

        self._record_backup.clear()

    def save_configuration(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'configuration.txt', 'w') as fh:
                fh.write(str(self.configuration))

            with open(directory / 'configuration.json', 'w') as fh:
                fh.write(self.configuration.json())

            log.info(f'Configuration saved to {directory}.')

    def save_hyperparameters(self, directory: pathlib.Path):
        if torch.distributed.get_rank() == 0:
            with open(directory / 'hyperparameters.txt', 'w') as fh:
                fh.write(str(self.hyperparams))

            with open(directory / 'hyperparameters.json', 'w') as fh:
                fh.write(self.hyperparams.json())

            log.info(f'Hyperparameters saved to {directory}/.')

    def clone_with_overrides(self, **overrides) -> 'ConfigurationManager':
        params = dict(self._cli_params)
        params.update(overrides)
        return ConfigurationManager(params)
