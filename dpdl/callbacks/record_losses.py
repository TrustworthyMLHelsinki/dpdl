import csv
import logging
import os

import torch
import torchmetrics

from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordTrainLossByStepCallback(Callback):
    def __init__(self, log_dir: str):
        super().__init__() # We'll get `global_step` from super

        self.log_dir = log_dir
        self.train_losses = []

        os.makedirs(self.log_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        super().on_train_batch_end(trainer, batch_idx, batch, loss, **kwargs)

        # reduce the rank-local loss to a mean loss across all ranks before storing
        mean_loss = self._mean_across_ranks(loss, trainer.device)

        self.train_losses.append({'step': self.global_step, 'train_loss': mean_loss})

    def on_train_end(self, trainer, *args, **kwargs):
        if self._is_global_zero():
            train_loss_path = os.path.join(self.log_dir, 'train_loss_by_step.csv')

            with open(train_loss_path, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=['step', 'train_loss'])
                writer.writeheader()
                writer.writerows(self.train_losses)

            log.info(f'Training losses (by step) saved to {train_loss_path}')


class RecordLossesByEpochCallback(Callback):
    def __init__(self, log_dir, device=None):
        super().__init__()

        self.log_dir = log_dir
        device = device or torch.device('cuda')
        self.train_loss = torchmetrics.aggregation.MeanMetric().to(device)
        self.train_losses = []
        self.val_losses = []

    def on_train_start(self, trainer):
        if self._is_global_zero():
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.csv_path = os.path.join(self.log_dir, 'epoch_losses.csv')

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss'])

    def on_train_epoch_start(self, trainer, epoch):
        self.train_loss.reset()

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.train_loss.update(loss)

    def on_train_epoch_end(self, trainer, epoch, metrics):
        train_loss = self.train_loss.compute().item()
        self.train_losses.append(train_loss)

    def on_validation_epoch_end(self, trainer, epoch, metrics, loss):
        self.val_losses.append(loss)

    def on_train_end(self, trainer):
        if self._is_global_zero():
            epochs = len(self.train_losses)

            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)

                for i in range(epochs):
                    train_loss_val = self.train_losses[i]
                    val_loss_val = self.val_losses[i] if i < len(self.val_losses) else ''
                    writer.writerow([i+1, train_loss_val, val_loss_val])

            log.info('Training finished and all epoch losses have been logged to CSV.')
