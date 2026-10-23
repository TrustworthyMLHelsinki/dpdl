import logging
import pathlib
from typing import List

from ..configurationmanager import Configuration, Hyperparameters
from ..utils import tensor_to_python_type
from .base_callback import Callback
from .body_head_gradient import RecordBodyAndHeadGradientNormsPerClassCallback
from .checkpoint import CheckpointCallback
from .clipping_bias import ClipMSEDecompositionCallback
from .cosine_similarity import (
    RecordCosineSimilarityCallback,
    RecordPerClassCosineSimilarityCallback,
)
from .debug import DebugProbeCallback
from .epoch_stats import RecordEpochStatsCallback
from .gradient_proportion import RecordClippedProportionsPerClassCallback
from .gradient_stats import RecordGradientStatisticsCallback
from .optimizer_stats import RecordOptimizerStatsCallback
from .per_class_accuracy import RecordPerClassAccuracyCallback
from .record_epoch_metrics import RecordMetricsByEpochCallback
from .record_losses import RecordLossesByEpochCallback, RecordTrainLossByStepCallback
from .record_accuracy import RecordAccuracyByEpochCallback
from .record_snr import RecordSNRCallback
from .llm_sampling import LLMSamplingCallback

log = logging.getLogger(__name__)


class CallbackHandler:
    def __init__(self, callbacks: list = []):
        self.callbacks = callbacks

    def call(self, event, *args, **kwargs):
        for callback in self.callbacks:
            event_handler = getattr(callback, event)
            event_handler(*args, **kwargs)


class CallbackFactory:
    @staticmethod
    def get_callbacks(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        device=None,
    ) -> List[Callback]:

        log_dir = configuration.log_dir
        experiment_name = configuration.experiment_name
        full_log_dir = pathlib.Path(f'{log_dir}/{experiment_name}')

        callbacks = [
            RecordEpochStatsCallback(use_steps=configuration.use_steps, device=device),
        ]

        if configuration.record_gradient_norms:
            max_grad_norm = hyperparams.max_grad_norm

            callbacks.append(
                RecordBodyAndHeadGradientNormsPerClassCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm, quantiles=configuration.record_gradient_norms_quantiles
                )
            )

            callbacks.append(
                RecordCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordPerClassCosineSimilarityCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordClippedProportionsPerClassCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            callbacks.append(
                RecordGradientStatisticsCallback(
                    log_dir=full_log_dir, max_grad_norm=max_grad_norm
                )
            )

            configuration.record_per_class_accuracy = True

        if configuration.verbose_callback:
            callbacks.append(DebugProbeCallback())

        if configuration.record_loss_by_step:
            callbacks.append(RecordTrainLossByStepCallback(log_dir=full_log_dir))

        if configuration.record_loss_by_epoch:
            callbacks.append(RecordLossesByEpochCallback(log_dir=full_log_dir, device=device))
            callbacks.append(RecordAccuracyByEpochCallback(log_dir=full_log_dir))

        if configuration.record_metrics_by_epoch:
            callbacks.append(RecordMetricsByEpochCallback(log_dir=full_log_dir))

        if configuration.record_per_class_accuracy:
            callbacks.append(
                RecordPerClassAccuracyCallback(
                    log_dir=full_log_dir,
                )
            )

        if configuration.record_snr:
            callbacks.append(
                RecordSNRCallback(
                    log_dir=full_log_dir,
                    max_grad_norm=hyperparams.max_grad_norm,
                )
            )

        if configuration.record_optimizer_stats:
            callbacks.append(
                RecordOptimizerStatsCallback(
                    log_dir=full_log_dir,
                    max_grad_norm=hyperparams.max_grad_norm,
                )
            )

        if configuration.record_clipping:
            callbacks.append(
                ClipMSEDecompositionCallback(
                    log_dir=full_log_dir,
                    max_grad_norm=hyperparams.max_grad_norm,
                    normalize_clipping=configuration.normalize_clipping,
                )
            )

        if configuration.checkpoint_step_interval or configuration.checkpoint_steps or configuration.checkpoint_epoch_interval:
            callbacks.append(
                CheckpointCallback(
                    log_dir=full_log_dir,
                    checkpoint_step_interval=configuration.checkpoint_step_interval,
                    checkpoint_epoch_interval=configuration.checkpoint_epoch_interval,
                    checkpoint_steps=configuration.checkpoint_steps,
                    remove_old_checkpoints = configuration.remove_old_checkpoints,
                    device=device,
                )
            )

        if configuration.record_llm_samples:
            callbacks.append(LLMSamplingCallback())

        return callbacks
