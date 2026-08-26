import logging
import math

import torch
import torchmetrics

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordEpochStatsCallback(Callback):
    def __init__(self, use_steps=False, device=None):
        super().__init__()

        self.use_steps = use_steps

        device = device or torch.device('cuda')
        self.train_loss = torchmetrics.aggregation.MeanMetric().to(device)

        # Do not log these metrics
        self._metrics_to_ignore = [
            'ConfusionMatrix',
            'MulticlassAccuracyPerClass',
        ]

    def on_train_start(self, trainer):
        if self._is_global_zero():
            if self.use_steps:
                batch_size = trainer.datamodule.batch_size
                data_size = len(trainer.get_dataloader("train").dataset)
                steps_per_epoch = data_size / batch_size
                epochs = math.ceil(trainer.total_steps / steps_per_epoch)

                log.info(
                    f"!!! Starting training for approximately {epochs} epochs ({trainer.total_steps} steps)."
                )
            else:
                log.info(f"!!! Starting training for {trainer.epochs} epochs.")

    def on_train_end(self, trainer):
        if self._is_global_zero():
            log.info("!!! Training finished.")

    def on_train_epoch_start(self, trainer, epoch):
        self.train_loss.reset()

        if self._is_global_zero():
            log.info(f"--------------------------------------------------")
            if not self.use_steps:
                log.info(f"Starting training epoch {epoch+1}.")
            else:
                log.info(f"Starting training approximate epoch {epoch+1}.")

    def on_train_epoch_end(self, trainer, epoch, metrics):
        loss = self.train_loss.compute()

        if self._is_global_zero():
            if not self.use_steps:
                log.info(f"Epoch {epoch+1} finished. Loss: {loss:.4f}.")
            else:
                log.info(f"Approximate epoch {epoch+1} finished. Loss: {loss:.4f}.")

            self._log_metrics(metrics, "Train metrics")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.train_loss.update(loss)

    def on_validation_epoch_end(self, trainer, epoch, metrics, loss):
        if self._is_global_zero():
            log.info(f"Validation finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Validation metrics")

    def on_test_epoch_end(self, trainer, epoch, metrics, loss):
        if self._is_global_zero():
            log.info(f"Test finished. Loss: {loss:.4f}.")
            self._log_metrics(metrics, "Test metrics")

    def _log_metrics(self, metrics, annotation="Metrics"):
        if not metrics:
            return

        metrics = tensor_to_python_type(metrics)

        log.info(annotation + ":")
        for key, value in metrics.items():
            if not key in self._metrics_to_ignore:
                log.info(f" - {key}: {value}.")
