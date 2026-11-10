import logging
import pathlib
import json
import torch
import typer

from pydantic import BaseModel, ConfigDict, model_validator
from typing import Any, Optional, List, Literal

log = logging.getLogger(__name__)

# DPDL MF paper-comment convention for this file:
# - Short tags: `BSR (Kalinin and Lampert, 2024)`, `Balls-n-Bins (Chua et al., 2024)`.
# - Math-note format: one intuition line + one compact equation + variable mapping.
# - Inline variable map at first semantic use in validators.
# - Scope note: implemented DPDL->Opacus plumbing only (no AOF commentary).

"""
Configuration parsing and validation for DPDL privacy/accounting surfaces.

This file owns the CLI/configuration contract for MF-related training runs. The
validators do not prove paper claims; they enforce that user inputs can be
interpreted as the paper quantities expected by Opacus runtime/accountant code.
"""


_FAMILY_CONTRACTS = {
    'blt': {
        'accountants': {'blt', 'bnb'},
        'sampling_modes': {None, 'torch_sampler', 'balls_in_bins', 'b_min_sep'},
        'requires_non_poisson': True,
    },
    'bandmf': {
        'accountants': {'bandmf', 'bnb'},
        'sampling_modes': {None, 'torch_sampler', 'cyclic_poisson', 'balls_in_bins', 'b_min_sep'},
        'requires_non_poisson': True,
    },
    'bsr': {
        'accountants': {'bsr', 'bnb'},
        'sampling_modes': {None, 'torch_sampler', 'cyclic_poisson', 'balls_in_bins', 'b_min_sep'},
        'requires_non_poisson': True,
    },
    'bisr': {
        'accountants': {'bsr', 'bnb'},
        'sampling_modes': {None, 'torch_sampler', 'cyclic_poisson', 'balls_in_bins', 'b_min_sep'},
        'requires_non_poisson': True,
    },
    'bandinvmf': {
        'accountants': {'bsr', 'bnb'},
        'sampling_modes': {None, 'torch_sampler', 'cyclic_poisson', 'balls_in_bins', 'b_min_sep'},
        'requires_non_poisson': True,
    },
    'bifr': {
        'accountants': {'bsr', 'bnb'},
        'sampling_modes': {None, 'torch_sampler', 'balls_in_bins', 'b_min_sep'},
        'requires_non_poisson': True,
    },
}


def _is_explicitly_set(v: object) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return True


def _validate_bandmf_bsr_contracts(
    *,
    mechanism: str,
    sampling_mode: str | None,
    accountant: str,
    poisson_sampling: bool,
    target_hypers: set[str],
    bsr_bands: int | None,
    bsr_z_std: float | None,
    bsr_iterations_number: int | None,
    bsr_mf_sensitivity: float | None,
    bsr_max_participations: int | None,
    bsr_min_separation: int | None,
    bifr_frac: float | None,
) -> None:
    contract = _FAMILY_CONTRACTS[mechanism]
    if accountant not in contract['accountants']:
        allowed = ', '.join(sorted(contract['accountants']))
        raise ValueError(f'{mechanism.upper()} mechanism requires --accountant in {{{allowed}}}.')

    if contract['requires_non_poisson'] and poisson_sampling:
        raise ValueError(
            f'{mechanism.upper()} mechanism requires non-Poisson semantics: set --poisson-sampling False.'
        )

    if sampling_mode not in contract['sampling_modes']:
        allowed_modes = ', '.join(str(m) for m in sorted(contract['sampling_modes'], key=lambda v: str(v)))
        raise ValueError(
            f'{mechanism.upper()} mechanism does not support --sampling-mode {sampling_mode!r}; '
            f'supported: {allowed_modes}.'
        )

    bands_missing = bsr_bands is None and 'bsr_bands' not in target_hypers
    if bands_missing:
        raise ValueError(f'{mechanism.upper()} mechanism requires --bsr-bands.')

    if mechanism == 'bandmf' and sampling_mode == 'cyclic_poisson':
        if bsr_mf_sensitivity is not None:
            raise ValueError(
                '--bsr-mf-sensitivity is fixed-batch BSR only and cannot be used with --sampling-mode cyclic_poisson.'
            )
        if bsr_max_participations is not None:
            raise ValueError(
                '--bsr-max-participations is fixed-batch BSR only and cannot be used with --sampling-mode cyclic_poisson.'
            )
        if bsr_min_separation is not None:
            raise ValueError(
                '--bsr-min-separation is fixed-batch BSR only and cannot be used with --sampling-mode cyclic_poisson.'
            )

    if bsr_bands is not None and int(bsr_bands) < 1:
        raise ValueError('--bsr-bands must be >= 1.')

    if bsr_z_std is not None and bsr_z_std < 0:
        raise ValueError('--bsr-z-std must be >= 0.')

    if bsr_iterations_number is not None and bsr_iterations_number < 1:
        raise ValueError('--bsr-iterations-number must be >= 1.')

    if mechanism == 'bifr' and bifr_frac is not None:
        if float(bifr_frac) < 0.0 or float(bifr_frac) > 1.0:
            raise ValueError('--bifr-frac must satisfy 0 <= frac <= 1.')


def _validate_blt_contracts(
    *,
    sampling_mode: str | None,
    accountant: str,
    poisson_sampling: bool,
    target_hypers: set[str],
    blt_buffers: int | None,
    bnb_b: int | None,
    bnb_p: float | None,
) -> None:
    contract = _FAMILY_CONTRACTS['blt']
    if accountant not in contract['accountants']:
        allowed = ', '.join(sorted(contract['accountants']))
        raise ValueError(f'BLT mechanism requires --accountant in {{{allowed}}}.')

    if contract['requires_non_poisson'] and poisson_sampling:
        raise ValueError('BLT mechanism requires non-Poisson semantics: set --poisson-sampling False.')

    if sampling_mode not in contract['sampling_modes']:
        allowed_modes = ', '.join(str(m) for m in sorted(contract['sampling_modes'], key=lambda v: str(v)))
        raise ValueError(
            f'BLT mechanism does not support --sampling-mode {sampling_mode!r}; '
            f'supported: {allowed_modes}.'
        )

    if blt_buffers is not None and int(blt_buffers) < 1:
        raise ValueError('--blt-buffers must be >= 1.')

    if sampling_mode in (None, 'torch_sampler'):
        if accountant != 'blt':
            raise ValueError('Fixed-batch BLT requires --accountant blt.')
        if bnb_b is not None or bnb_p is not None:
            raise ValueError('Fixed-batch BLT does not use --bnb-b or --bnb-p.')
        return

    if accountant != 'bnb':
        raise ValueError('Amplified BLT sampling requires --accountant bnb.')

    if bnb_b is not None and int(bnb_b) < 1:
        raise ValueError('--bnb-b must be >= 1.')

    if sampling_mode == 'balls_in_bins':
        if bnb_p is not None:
            raise ValueError('BLT balls_in_bins path does not use --bnb-p.')
        return

    if bnb_p is not None and not (0.0 < float(bnb_p) <= 1.0):
        raise ValueError('--bnb-p must satisfy 0 < p <= 1.')

    if bnb_b is None and blt_buffers is None and 'blt_buffers' not in target_hypers:
        raise ValueError(
            'BLT b_min_sep path requires --bnb-b or a workload-facing BLT control such as --blt-buffers.'
        )


def _validate_bnb_contracts(
    *,
    mechanism: str,
    sampling_mode: str | None,
    accountant: str,
    poisson_sampling: bool,
    target_hypers: set[str],
    bnb_b: int | None,
    bnb_p: float | None,
    bnb_bands: int | None,
    bsr_bands: int | None,
    bnb_num_samples: int | None,
    bnb_chunk_size: int | None,
    bnb_seed: int | None,
    bnb_calibration_mode: str | None,
) -> None:
    if sampling_mode == 'b_min_sep':
        if mechanism not in ('gaussian', 'bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr', 'blt'):
            raise ValueError(
                'b_min_sep sampling requires --noise-mechanism gaussian, bandmf, bsr, bisr, bandinvmf, bifr, or blt.'
            )

        if accountant != 'bnb':
            raise ValueError('b_min_sep sampling requires --accountant bnb.')

        if poisson_sampling:
            raise ValueError(
                'b_min_sep sampling requires non-Poisson semantics: set --poisson-sampling False.'
            )

        if bnb_b is not None and int(bnb_b) < 1:
            raise ValueError('--bnb-b must be >= 1.')

        if bnb_p is not None:
            if float(bnb_p) <= 0.0 or float(bnb_p) > 1.0:
                raise ValueError('--bnb-p must satisfy 0 < p <= 1.')
        if (
            bnb_b is None
            and bsr_bands is None
            and bnb_bands is None
            and 'bsr_bands' not in target_hypers
        ):
            raise ValueError(
                'b_min_sep sampling requires --bnb-b or a mechanism/accountant band '
                'parameter (--bsr-bands or --bnb-bands).'
            )

        if (
            mechanism != 'gaussian'
            and mechanism != 'blt'
            and bsr_bands is None
            and bnb_bands is None
            and 'bsr_bands' not in target_hypers
        ):
            raise ValueError(
                f'{mechanism.upper()} b_min_sep path requires --bsr-bands.'
            )

        if bsr_bands is not None and int(bsr_bands) < 1:
            raise ValueError('--bsr-bands must be >= 1.')

        if bnb_bands is not None and int(bnb_bands) < 1:
            raise ValueError('--bnb-bands must be >= 1.')

        if mechanism == 'gaussian' and bnb_bands is not None and int(bnb_bands) != 1:
            raise ValueError('Gaussian b_min_sep path only supports --bnb-bands 1.')

    if bnb_num_samples is not None and int(bnb_num_samples) < 1:
        raise ValueError('--bnb-num-samples must be >= 1.')

    if bnb_chunk_size is not None and int(bnb_chunk_size) < 1:
        raise ValueError('--bnb-chunk-size must be >= 1.')

    if bnb_seed is not None and int(bnb_seed) < 0:
        raise ValueError('--bnb-seed must be >= 0.')

    if bnb_calibration_mode is not None and str(bnb_calibration_mode) not in {'evr', 'optimistic'}:
        raise ValueError("--bnb-calibration-mode must be one of {'evr', 'optimistic'}.")


def _validate_balls_in_bins_mf_contracts(
    *,
    mechanism: str,
    sampling_mode: str | None,
    accountant: str,
    target_hypers: set[str],
    bnb_b: int | None,
    bnb_bands: int | None,
    bsr_bands: int | None,
) -> None:
    if sampling_mode != 'balls_in_bins' or mechanism not in ('bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr', 'blt'):
        return

    if accountant != 'bnb':
        raise ValueError(
            f'{mechanism.upper()} balls_in_bins path requires --accountant bnb.'
        )

    if bnb_b is not None and int(bnb_b) < 1:
        raise ValueError('--bnb-b must be >= 1.')

    if mechanism != 'blt' and bsr_bands is None and bnb_bands is None and 'bsr_bands' not in target_hypers:
        raise ValueError(
            f'{mechanism.upper()} balls_in_bins path requires --bsr-bands.'
        )

    if bsr_bands is not None and int(bsr_bands) < 1:
        raise ValueError('--bsr-bands must be >= 1.')

    if bnb_bands is not None and int(bnb_bands) < 1:
        raise ValueError('--bnb-bands must be >= 1.')


def _validate_balls_in_bins_gaussian_contract(
    *,
    mechanism: str,
    sampling_mode: str | None,
    accountant: str,
    poisson_sampling: bool,
    bnb_b: int | None,
    bnb_bands: int | None,
) -> None:
    if mechanism != 'gaussian' or sampling_mode != 'balls_in_bins':
        return

    if accountant != 'bnb':
        raise ValueError('Gaussian balls_in_bins path requires --accountant bnb.')

    if poisson_sampling:
        raise ValueError(
            'Gaussian balls_in_bins path requires non-Poisson semantics: set --poisson-sampling False.'
        )

    if bnb_b is not None and int(bnb_b) < 1:
        raise ValueError('--bnb-b must be >= 1.')

    if bnb_bands is not None and int(bnb_bands) != 1:
        raise ValueError('Gaussian balls_in_bins path only supports --bnb-bands 1.')


def _validate_privacy_contracts(
    *,
    mechanism: str,
    sampling_mode: str | None,
    accountant: str,
    poisson_sampling: bool,
    target_hypers: set[str],
    bsr_coeffs: list[float] | None,
    bsr_z_std: float | None,
    bsr_bands: int | None,
    bsr_max_participations: int | None,
    bsr_min_separation: int | None,
    bsr_mf_sensitivity: float | None,
    bsr_iterations_number: int | None,
    bifr_frac: float | None,
    blt_buffers: int | None,
    bnb_b: int | None,
    bnb_p: float | None,
    bnb_bands: int | None,
    bnb_num_samples: int | None,
    bnb_chunk_size: int | None,
    bnb_seed: int | None,
    bnb_calibration_mode: str | None,
) -> None:
    """
    Validate mechanism/accountant/sampler compatibility at config-parse time.
    Math: enforce accountant/runtime contracts, e.g. cyclic q = bands·sample_rate ∈ (0,1],
    fixed-batch BSR uses S_{k,b}(C;T), and BNB uses structured sampling metadata.
    Mapping: mechanism=noise_mechanism, accountant=accountant, sampler=sampling_mode.
    """
    has_any_bsr_field = any(
        _is_explicitly_set(v)
        for v in [
            bsr_coeffs,
            bsr_z_std,
            bsr_bands,
            bsr_max_participations,
            bsr_min_separation,
            bsr_mf_sensitivity,
            bsr_iterations_number,
        ]
    )
    if mechanism not in ('bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr') and has_any_bsr_field:
        raise ValueError(
            'BSR/BandMF/BISR/BandInvMF/BIFR-specific parameters require --noise-mechanism bandmf, bsr, bisr, bandinvmf, or bifr.'
        )

    if mechanism != 'bifr' and _is_explicitly_set(bifr_frac):
        raise ValueError('BIFR-specific parameters require --noise-mechanism bifr.')

    if mechanism != 'blt' and _is_explicitly_set(blt_buffers):
        raise ValueError('BLT-specific parameters require --noise-mechanism blt.')

    # `sampling_mode` carries runtime sampler semantics; `cyclic_poisson` is only valid for BandMF/BSR.
    if sampling_mode == 'cyclic_poisson' and mechanism not in ['bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr']:
        raise ValueError('Cyclic Poisson sampling requires --noise-mechanism bandmf, bsr, bisr, bandinvmf, or bifr.')

    if mechanism in ('bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr'):
        _validate_bandmf_bsr_contracts(
            mechanism=mechanism,
            sampling_mode=sampling_mode,
            accountant=accountant,
            poisson_sampling=poisson_sampling,
            target_hypers=target_hypers,
            bsr_bands=bsr_bands,
            bsr_z_std=bsr_z_std,
            bsr_iterations_number=bsr_iterations_number,
            bsr_mf_sensitivity=bsr_mf_sensitivity,
            bsr_max_participations=bsr_max_participations,
            bsr_min_separation=bsr_min_separation,
            bifr_frac=bifr_frac,
        )

    if mechanism == 'blt':
        _validate_blt_contracts(
            sampling_mode=sampling_mode,
            accountant=accountant,
            poisson_sampling=poisson_sampling,
            target_hypers=target_hypers,
            blt_buffers=blt_buffers,
            bnb_b=bnb_b,
            bnb_p=bnb_p,
        )

    has_any_bnb_field = any(
        _is_explicitly_set(v)
        for v in [
            bnb_b,
            bnb_p,
            bnb_bands,
        ]
    )
    if mechanism not in ('gaussian', 'bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr', 'blt') and has_any_bnb_field:
        raise ValueError(
            'BNB-specific parameters require --noise-mechanism gaussian, bandmf, bsr, bisr, bandinvmf, bifr, or blt.'
        )

    _validate_bnb_contracts(
        mechanism=mechanism,
        sampling_mode=sampling_mode,
        accountant=accountant,
        poisson_sampling=poisson_sampling,
        target_hypers=target_hypers,
        bnb_b=bnb_b,
        bnb_p=bnb_p,
        bnb_bands=bnb_bands,
        bsr_bands=bsr_bands,
        bnb_num_samples=bnb_num_samples,
        bnb_chunk_size=bnb_chunk_size,
        bnb_seed=bnb_seed,
        bnb_calibration_mode=bnb_calibration_mode,
    )

    _validate_balls_in_bins_mf_contracts(
        mechanism=mechanism,
        sampling_mode=sampling_mode,
        accountant=accountant,
        target_hypers=target_hypers,
        bnb_b=bnb_b,
        bnb_bands=bnb_bands,
        bsr_bands=bsr_bands,
    )
    _validate_balls_in_bins_gaussian_contract(
        mechanism=mechanism,
        sampling_mode=sampling_mode,
        accountant=accountant,
        poisson_sampling=poisson_sampling,
        bnb_b=bnb_b,
        bnb_bands=bnb_bands,
    )

class Hyperparameters(BaseModel):
    learning_rate: float = 1e-3
    epochs: Optional[int] = None
    total_steps: Optional[int] = None
    batch_size: Optional[int] = None
    sample_rate: Optional[float] = None
    noise_multiplier: Optional[float]
    max_grad_norm: Optional[float]
    target_epsilon: Optional[float]
    noise_batch_ratio: Optional[float]
    bsr_bands: Optional[int] = None
    bnb_bands: Optional[int] = None
    blt_buffers: Optional[int] = None
    privacy: bool = True # Only used in __str__
    max_length: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def check_batch_size_or_sample_rate(cls, values):
        batch_size, sample_rate = values.get('batch_size'), values.get('sample_rate')

        if all([batch_size, sample_rate]):
            raise ValueError('Either batch_size or sample_rate must be set, but not both.')

        return values

    @model_validator(mode="before")
    @classmethod
    def check_target_epsilon_or_noise_multiplier(cls, values):
        target_epsilon, noise_multiplier = values.get('target_epsilon'), values.get('noise_multiplier')

        if all([target_epsilon, noise_multiplier]):
            raise ValueError('Both, target_epsilon and noise_multiplier given.')

        return values

    @model_validator(mode="before")
    @classmethod
    def check_target_epsilon_or_noise_batch_ratio(cls, values):
        target_epsilon, noise_batch_ratio = values.get('target_epsilon'), values.get('noise_batch_ratio')

        if all([target_epsilon, noise_batch_ratio]):
            raise ValueError('Both, target_epsilon and noise_batch_ratio given.')

        return values

    @model_validator(mode="before")
    @classmethod
    def check_noise_batch_ratio_or_noise_multiplier(cls, values):
        noise_multiplier, noise_batch_ratio = values.get('noise_multiplier'), values.get('noise_batch_ratio')

        if all([noise_multiplier, noise_batch_ratio]):
            raise ValueError('Both, noise_multiplier and noise_batch_ratio given.')

        return values

    @model_validator(mode="before")
    @classmethod
    def check_epochs(cls, values):
        epochs = values.get('epochs')
        total_steps = values.get('total_steps')

        if epochs is None and total_steps is None:
            raise ValueError(
                'Missing training length: set `--epochs` or use `--use-steps` with `--total-steps`.'
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
                ('Noise-batch ratio', self.noise_batch_ratio),
                ('BSR bands', self.bsr_bands),
                ('BNB bands', self.bnb_bands),
                ('BLT buffers', self.blt_buffers),
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
    optimizer_momentum: Optional[float] = None
    optimizer_weight_decay: float = 0.0
    lr_scheduler: Literal['none', 'bnb_linear_decay'] = 'none'
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
    noise_mechanism: Literal['gaussian', 'bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr', 'blt'] = 'gaussian'
    sampling_mode: Optional[Literal['torch_sampler', 'cyclic_poisson', 'b_min_sep', 'balls_in_bins']] = None
    bsr_coeffs: Optional[List[float]] = None
    bsr_bands: Optional[int] = None
    bsr_z_std: Optional[float] = None
    bsr_max_participations: Optional[int] = None
    bsr_min_separation: Optional[int] = None
    bsr_mf_sensitivity: Optional[float] = None
    bsr_iterations_number: Optional[int] = None
    bifr_frac: Optional[float] = None
    bnb_b: Optional[int] = None
    bnb_p: Optional[float] = None
    bnb_bands: Optional[int] = None
    blt_buffers: Optional[int] = None
    bnb_num_samples: Optional[int] = None
    bnb_chunk_size: Optional[int] = None
    bnb_seed: Optional[int] = None
    bnb_calibration_mode: Optional[Literal['evr', 'optimistic']] = None
    n_trials: int = 20
    optuna_random_trials: int = 10
    target_hypers: List[str] = []
    optuna_target_metric: str = 'loss'
    optuna_direction: Literal['minimize', 'maximize'] = 'minimize'
    optuna_config: str = 'conf/optuna_hypers.conf'
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
    record_memory_usage: bool = False
    record_training_time: bool = False
    noise_history_mode: Optional[Literal['regenerate', 'stored']] = None
    record_clipping: Optional[bool] = False
    record_snr: Optional[bool] = False
    record_llm_samples: Optional[bool] = False
    record_gradient_norms: Optional[bool] = False
    record_gradient_norms_quantiles: Optional[List[int]] = [25, 50, 75]
    verbose_callback: Optional[bool] = False
    cache_dataset_transforms: Optional[bool] = False
    weight_perturbation_level: float = 0
    record_loss_by_step: Optional[bool] = False
    record_loss_by_epoch: Optional[bool] = False
    record_per_class_accuracy: Optional[bool] = False
    record_final_train_accuracy: Optional[bool] = False
    checkpoint_step_interval: Optional[int] = None
    disable_epsilon_logging: Optional[bool] = False
    split_seed: Optional[int] = 42
    predict_dataset_split: Optional[str] = 'test'
    prediction_save_gradient_data: Optional[bool] = False
    load_in_4bit: bool = False

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="after")
    def check_resource_options(self):
        if self.noise_history_mode is not None and self.noise_mechanism not in ('bisr', 'bandinvmf', 'bifr'):
            raise ValueError('noise history selection requires an inverse family')
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_empty_bsr_coeffs(cls, values):
        coeffs = values.get('bsr_coeffs')
        if isinstance(coeffs, (list, tuple)) and len(coeffs) == 0:
            values['bsr_coeffs'] = None
        return values

    @model_validator(mode="before")
    @classmethod
    def check_fairness_imbalance_factor(cls, values):
        imbalance_factor = values.get('imbalance_factor')
        fairness_imbalance_class = values.get('fairness_imbalance_class')

        if fairness_imbalance_class and not imbalance_factor:
            raise ValueError(
                'Parameter "imbalance_factor" is required when using "fairness_imbalance_class".'
            )

        return values

    @model_validator(mode="before")
    @classmethod
    def check_command(cls, values):
        command = values.get('command')

        if command not in ['train', 'optimize', 'predict', 'show-layers', 'train-predict']:
            raise ValueError('Command must be "train", "optimize", "predict", "show-layers", or "train-predict".')

        return values

    @model_validator(mode="before")
    @classmethod
    def check_total_steps(cls, values):
        total_steps = values.get('total_steps')
        use_steps = values.get('use_steps')
        epochs = values.get('epochs')

        if total_steps and epochs:
            raise ValueError('Parameters "epochs" and "total_steps" are exclusive.')

        if total_steps and not use_steps:
            raise ValueError('Parameter "total_steps" requires also "use_steps".')

        return values

    @model_validator(mode="before")
    @classmethod
    def normalize_sampling_mode_aliases(cls, values):
        """Normalize legacy sampling aliases to canonical runtime names."""
        sampling_mode = values.get('sampling_mode')
        if sampling_mode == 'balls_n_bins':
            values = dict(values)
            values['sampling_mode'] = 'balls_in_bins'
        return values

    @model_validator(mode="before")
    @classmethod
    def check_total_steps_nonpoisson_sampling_mode(cls, values):
        """
        Guard non-poisson step-based mode with explicit sampler semantics.
        Math: with horizon T=total_steps and non-Poisson sampling, q depends on sampler law
        (cyclic/b-min-sep/balls-in-bins), so sampling_mode must be explicit.
        Mapping: T=total_steps, q=sample_rate seen by Opacus accountants.
        """
        total_steps = values.get('total_steps')
        use_steps = values.get('use_steps')
        poisson_sampling = values.get('poisson_sampling', True)
        sampling_mode = values.get('sampling_mode')
        privacy = values.get('privacy', True)
        mechanism = values.get('noise_mechanism', 'gaussian')

        # `total_steps` is optimization horizon `T`; non-poisson requires explicit sampling contract.
        if (
            privacy
            and total_steps
            and use_steps
            and not poisson_sampling
            and sampling_mode is None
        ):
            raise ValueError(
                'Setting total_steps with non-Poisson sampling requires '
                '--sampling-mode cyclic_poisson, b_min_sep, or balls_in_bins.'
            )

        if (
            privacy
            and total_steps
            and use_steps
            and not poisson_sampling
            and sampling_mode == 'torch_sampler'
            and mechanism not in ('bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr', 'blt')
        ):
            raise ValueError(
                'Setting total_steps with non-Poisson sampling requires '
                '--sampling-mode cyclic_poisson, b_min_sep, or balls_in_bins.'
            )

        return values

    @model_validator(mode="before")
    @classmethod
    def check_shots_and_subset_size(cls, values):
        shots = values.get('shots')
        subset_size = values.get('subset_size')

        if shots and subset_size:
            raise ValueError('Parameters "shots" and "subset_size" are exclusive.')

        return values

    @model_validator(mode="before")
    @classmethod
    def check_feature_cache(cls, values):
        cache_features = values.get('cache_features')
        peft_method = values.get('peft')

        if cache_features and peft_method != 'head-only':
            raise ValueError(f'Head only training required if feature cache is enabled.')

        return values

    @model_validator(mode="after")
    def check_accountant_mechanism_compatibility(self):
        """
        Validate high-level accountant/mechanism compatibility.
        Math: this gate enforces valid pairs (mechanism, accountant) so ε(δ) is computed
        under the intended contract for correlated vs Gaussian mechanisms.
        Mapping: mechanism=noise_mechanism, accountant=accountant.
        """
        mechanism = self.noise_mechanism
        accountant = self.accountant

        if mechanism == 'gaussian' and accountant == 'bsr':
            raise ValueError(
                'Gaussian mechanism does not support mechanism-specific accountants; '
                'use --accountant prv/rdp/gdp, or use a supported runtime mechanism with --accountant bnb and --sampling-mode balls_in_bins.'
            )

        return self

    @model_validator(mode="after")
    def check_optimizer_parameters(self):
        if self.optimizer_momentum is not None and not (0.0 <= float(self.optimizer_momentum) < 1.0):
            raise ValueError('--optimizer-momentum must be in [0, 1).')

        if self.optimizer_weight_decay is not None and float(self.optimizer_weight_decay) < 0.0:
            raise ValueError('--optimizer-weight-decay must be >= 0.')

        if self.optimizer == 'paper-sgd':
            if self.optimizer_momentum is None:
                raise ValueError('paper-sgd requires --optimizer-momentum.')
            if float(self.optimizer_weight_decay or 0.0) == 0.0:
                raise ValueError(
                    'paper-sgd requires --optimizer-weight-decay as the paper shrinkage factor alpha.'
                )

        return self

    @model_validator(mode="after")
    def check_privacy_parameter_contracts(self):
        """
        Run full privacy-parameter contract checks.
        Math: validates that provided privacy parameters define one coherent accounting path
        (target ε/δ, explicit σ, or mechanism-specific state constraints).
        Mapping: ε=target_epsilon, δ=target_delta, σ=noise_multiplier or bsr_z_std.
        """
        targeted_hypers = set(self.target_hypers) if self.command == 'optimize' else set()
        _validate_privacy_contracts(
            mechanism=self.noise_mechanism,
            sampling_mode=self.sampling_mode,
            accountant=self.accountant,
            poisson_sampling=self.poisson_sampling,
            target_hypers=targeted_hypers,
            bsr_coeffs=self.bsr_coeffs,
            bsr_z_std=self.bsr_z_std,
            bsr_bands=self.bsr_bands,
            bsr_max_participations=self.bsr_max_participations,
            bsr_min_separation=self.bsr_min_separation,
            bsr_mf_sensitivity=self.bsr_mf_sensitivity,
            bsr_iterations_number=self.bsr_iterations_number,
            bifr_frac=self.bifr_frac,
            blt_buffers=self.blt_buffers,
            bnb_b=self.bnb_b,
            bnb_p=self.bnb_p,
            bnb_bands=self.bnb_bands,
            bnb_num_samples=self.bnb_num_samples,
            bnb_chunk_size=self.bnb_chunk_size,
            bnb_seed=self.bnb_seed,
            bnb_calibration_mode=self.bnb_calibration_mode,
        )
        return self

    def __str__(self):
        attributes = [
            ('Command', self.command),
            ('Privacy', self.privacy),
            ('Model name', self.model_name),
            ('Optimizer', self.optimizer),
            ('Optimizer momentum', self.optimizer_momentum),
            ('Optimizer weight decay', self.optimizer_weight_decay),
            ('LR scheduler', self.lr_scheduler),
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
            ('Use pretrained model', self.pretrained),
            ('Pretrained model weight perturbation noise level', self.weight_perturbation_level),
            ('Use precomputed features', self.cache_features),
            ('Use steps instead of epochs', self.use_steps),
            ('Evaluation mode', self.evaluation_mode),
            ('Save final model', self.save_model),
            ('Path for saving/loding model weights', self.model_weights_path),
            ('Record clipping stats (MSE)', self.record_clipping),
            ('Record signal-to-noise ratio', self.record_snr),
            ('Record LLM samples', self.record_llm_samples),
            ('Record gradient norms', self.record_gradient_norms),
            ('Record gradient norms quantiles', self.record_gradient_norms_quantiles),
            ('Record train loss by step', self.record_loss_by_step),
            ('Record train/valid loss and accuracy by epoch', self.record_loss_by_epoch),
            ('Record per-class accuracy', self.record_per_class_accuracy),
            ('Record final training accuracy', self.record_final_train_accuracy),
            ('Checkpoint every nth step', self.checkpoint_step_interval),
            ('Enable callback debug logging', self.verbose_callback),
            ('Fairness-style subsampling class', self.fairness_imbalance_class),
            ('Random seed for creating dataset subsets', self.split_seed),
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
                ('Noise mechanism', self.noise_mechanism),
                ('Sampling mode', self.sampling_mode),
                ('BSR coeffs', self.bsr_coeffs),
                ('BSR z std', self.bsr_z_std),
                ('BIFR frac', self.bifr_frac),
                ('BLT buffers', self.blt_buffers),
                ('BSR max participations', self.bsr_max_participations),
                ('BSR min separation', self.bsr_min_separation),
                ('BSR MF sensitivity', self.bsr_mf_sensitivity),
                ('BSR iterations number', self.bsr_iterations_number),
                ('BNB b', self.bnb_b),
                ('BNB p', self.bnb_p),
                ('BNB MC samples', self.bnb_num_samples),
                ('BNB chunk size', self.bnb_chunk_size),
                ('BNB MC seed', self.bnb_seed),
                ('BNB calibration mode', self.bnb_calibration_mode),
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

        if 'bsr_sensitivity_scale' in cli_params:
            raise ValueError(
                'Legacy key `bsr_sensitivity_scale` is no longer supported. '
                'Use canonical `bsr_sensitivity_scale` in mechanism state (resolved by trainer/runtime).'
            )
        if 'blt_lambda' in cli_params or 'lambda' in cli_params:
            raise ValueError(
                'Diagnostic BLT lambda inputs are not supported in DPDL. '
                'Use workload-driven BLT with --blt-buffers instead.'
            )

        privacy = cli_params.get('privacy', True)
        command = cli_params.get('command')
        target_epsilon = cli_params.get('target_epsilon')
        noise_multiplier = cli_params.get('noise_multiplier')
        noise_batch_ratio = cli_params.get('noise_batch_ratio')
        noise_mechanism = cli_params.get('noise_mechanism', 'gaussian')
        bsr_z_std = cli_params.get('bsr_z_std')
        # Treat explicit bsr_z_std as an explicit privacy path at this gate so
        # mechanism-specific validation can produce the authoritative error
        # when noise_mechanism is incompatible.
        explicit_bsr_z_std_path = bsr_z_std is not None

        if (
            privacy
            and command in ('train', 'optimize', 'train-predict')
            and not explicit_bsr_z_std_path
            and target_epsilon is None
            and noise_multiplier is None
            and noise_batch_ratio is None
        ):
            raise ValueError(
                'Privacy mode requires one explicit target path. '
                'Set one of --target-epsilon (or -1 for clip-only), '
                '--noise-multiplier, --noise-batch-ratio, or --bsr-z-std (BandMF/BSR/BISR/BandInvMF/BIFR only).'
            )

        self.configuration = Configuration(**cli_params)
        self.hyperparams = Hyperparameters(**cli_params)

        if (
            self.configuration.noise_mechanism in ('bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr')
            and self.configuration.bsr_z_std is not None
            and (
                (
                    self.hyperparams.target_epsilon is not None
                    and float(self.hyperparams.target_epsilon) != -1.0
                )
                or self.hyperparams.noise_multiplier is not None
                or self.hyperparams.noise_batch_ratio is not None
            )
        ):
            raise ValueError(
                '--bsr-z-std cannot be combined with --target-epsilon (except clip-only -1), '
                '--noise-multiplier, or --noise-batch-ratio for BandMF/BSR/BISR/BandInvMF/BIFR. '
                'Use either explicit --bsr-z-std alone, or accounting-driven noise controls.'
            )

        self._validate_privacy_parameter_contracts()

        # Remove target hypers from hyperparams; they will be set per HPO trial.
        for target_hyper in self.configuration.target_hypers:
            if not hasattr(self.hyperparams, target_hyper):
                raise ValueError(
                    f'Unsupported target hyperparameter "{target_hyper}". '
                    'Target hypers must be fields of Hyperparameters.'
                )
            setattr(self.hyperparams, target_hyper, None)

        self._log_bsr_trace_from_config_parse()

    def _validate_privacy_parameter_contracts(self) -> None:
        """
        Apply centralized privacy contract validation for resolved config/hypers.
        Math: validates parsed runtime tuple (mechanism, sampler, accountant, metadata)
        before forwarding to Opacus where ε(δ) accounting is executed.
        Mapping: metadata includes bands/bsr_max_participations/bsr_min_separation/bsr_mf_sensitivity.
        """
        cfg = self.configuration
        targeted_hypers = set(cfg.target_hypers) if cfg.command == 'optimize' else set()
        _validate_privacy_contracts(
            mechanism=cfg.noise_mechanism,
            sampling_mode=cfg.sampling_mode,
            accountant=cfg.accountant,
            poisson_sampling=cfg.poisson_sampling,
            target_hypers=targeted_hypers,
            bsr_coeffs=cfg.bsr_coeffs,
            bsr_z_std=cfg.bsr_z_std,
            bsr_bands=self.hyperparams.bsr_bands,
            bsr_max_participations=cfg.bsr_max_participations,
            bsr_min_separation=cfg.bsr_min_separation,
            bsr_mf_sensitivity=cfg.bsr_mf_sensitivity,
            bsr_iterations_number=cfg.bsr_iterations_number,
            bifr_frac=cfg.bifr_frac,
            blt_buffers=self.hyperparams.blt_buffers,
            bnb_b=cfg.bnb_b,
            bnb_p=cfg.bnb_p,
            bnb_bands=self.hyperparams.bnb_bands,
            bnb_num_samples=cfg.bnb_num_samples,
            bnb_chunk_size=cfg.bnb_chunk_size,
            bnb_seed=cfg.bnb_seed,
            bnb_calibration_mode=cfg.bnb_calibration_mode,
        )

    def _log_bsr_trace_from_config_parse(self) -> None:
        """
        Emit config-parse trace payload for BSR/BandMF accounting inputs.
        Math: logs inputs that determine correlated accounting scale terms
        (fixed-batch S_{k,b}(C;T) or cyclic κ(T)) and z_std path selection.
        Mapping: coeffs/bands/bsr_iterations_number/bsr_max_participations/bsr_min_separation/bsr_mf_sensitivity.
        """
        cfg = self.configuration
        if cfg.noise_mechanism not in ('bandmf', 'bsr', 'bisr', 'bandinvmf', 'bifr'):
            return

        coeffs = cfg.bsr_coeffs if cfg.bsr_coeffs is not None else []
        # `target_epsilon`/`target_delta` are DP targets; `bands/k/min_separation` fields parameterize sensitivity contracts.
        payload = {
            'stage': 'dpdl_config_parse',
            'command': cfg.command,
            'noise_mechanism': cfg.noise_mechanism,
            'accountant': cfg.accountant,
            'sampling_mode': cfg.sampling_mode,
            'poisson_sampling': cfg.poisson_sampling,
            'use_steps': cfg.use_steps,
            'epochs': self.hyperparams.epochs,
            'total_steps': self.hyperparams.total_steps,
            'batch_size': self.hyperparams.batch_size,
            'target_epsilon': self.hyperparams.target_epsilon,
            'noise_multiplier': self.hyperparams.noise_multiplier,
            'bsr': {
                'coeff_count': len(coeffs),
                'coeff_head': list(coeffs[:5]),
                'bsr_bands': self.hyperparams.bsr_bands,
                'bsr_iterations_number': cfg.bsr_iterations_number,
                'bifr_frac': cfg.bifr_frac,
                'bsr_mf_sensitivity': cfg.bsr_mf_sensitivity,
                'bsr_min_separation': cfg.bsr_min_separation,
                'bsr_max_participations': cfg.bsr_max_participations,
                'z_std': cfg.bsr_z_std,
            },
        }
        log.info('BSR_TRACE %s', json.dumps(payload, sort_keys=True))

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
