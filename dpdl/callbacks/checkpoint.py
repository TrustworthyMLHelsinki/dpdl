import json
import logging
import os
import re

import torch
import torchmetrics

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by modification time"""

    if not os.path.exists(checkpoint_dir):
        return 0

    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint_step_')]

    if not checkpoints:
        return 0

    # Sort by modification time
    latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))

    # Extract step number
    match = re.search(r'checkpoint_step_(\d+)', latest)

    if match:
        return int(match.group(1))

    return 0


class CheckpointCallback(Callback):
    def __init__(self, log_dir: str, checkpoint_step_interval=None, checkpoint_steps=None, device=None):
        super().__init__()

        self.log_dir = log_dir
        self.checkpoint_step_interval = checkpoint_step_interval
        self.checkpoint_steps = set(checkpoint_steps or [])
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints')
        self.global_step = get_latest_checkpoint(self.checkpoints_dir)

        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # Initialize mean metric for accumulating train loss over interval
        # sync_on_compute=True (default) so compute() reduces into the global mean across the distributed group.
        device = device or torch.device('cuda')
        self.interval_loss = torchmetrics.aggregation.MeanMetric().to(device)

    def on_train_start(self, trainer):
        super().on_train_start(trainer)

        if self._is_global_zero() and 0 in self.checkpoint_steps:
            checkpoint_path = os.path.join(self.checkpoints_dir, 'checkpoint_step_0.pt')
            self.save_checkpoint(trainer, checkpoint_path)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        # Accumulate on every rank, because loss is rank-local
        # update() reduces this into the global mean.
        self.interval_loss.update(loss)
        self.global_step += 1

        interval_checkpoint = self.checkpoint_step_interval and self.global_step % self.checkpoint_step_interval == 0
        if interval_checkpoint or self.global_step in self.checkpoint_steps:
            if self._is_global_zero():
                checkpoint_path = os.path.join(
                    self.checkpoints_dir, f'checkpoint_step_{self.global_step}.pt'
                )
                self.save_checkpoint(trainer, checkpoint_path)

            # Exact checkpoints are state snapshots. Validate only at the last one.
            # Decided identically on every rank, so all ranks skip or validate together.
            final_exact_checkpoint = self.checkpoint_steps and self.global_step == max(self.checkpoint_steps)
            if not interval_checkpoint and not final_exact_checkpoint:
                return

            # All ranks must participate because _evaluate uses collective ops to return global metrics
            val_loss, metrics = trainer.validate(enable_callbacks=False)

            # Compute the average train loss since the last checkpoint
            # compute() syncs across ranks, so every rank must call it
            # Only rank 0 writes the metrics file.
            avg_train_loss = self.interval_loss.compute().item()
            self.interval_loss.reset()

            if self._is_global_zero():
                # Add the average train loss to the metrics dictionary
                metrics = {
                    'loss': val_loss,
                    'avg_train_loss_since_last_checkpoint': avg_train_loss,
                    **metrics,
                }

                metrics_path = os.path.join(
                    self.checkpoints_dir, f'checkpoint_step_{self.global_step}_metrics.json'
                )
                self.save_metrics(metrics, metrics_path)

    def on_train_end(self, trainer, *args, **kwargs):
        # The last exact checkpoint and its metrics were already saved at batch end.
        # Checked on every rank (global_step is in sync) so all ranks agree before
        # the collective validate() below.
        if self.checkpoint_steps and self.global_step == max(self.checkpoint_steps):
            return

        if self._is_global_zero():
            final_checkpoint_path = os.path.join(
                self.checkpoints_dir, f'final_checkpoint_step_{self.global_step}.pt'
            )
            self.save_checkpoint(trainer, final_checkpoint_path)

        val_loss, metrics = trainer.validate(enable_callbacks=False)

        # Compute avg loss since last checkpoint across ranks, every rank must call it
        # Only rank 0 writes the metrics file
        avg_train_loss = self.interval_loss.compute().item()
        self.interval_loss.reset()

        if self._is_global_zero():
            metrics = {
                'loss': val_loss,
                'avg_train_loss_since_last_checkpoint': avg_train_loss,
                **metrics,
            }

            metrics_path = os.path.join(
                self.checkpoints_dir, f'final_checkpoint_step_{self.global_step}_metrics.json'
            )
            self.save_metrics(metrics, metrics_path)

    def save_checkpoint(self, trainer, checkpoint_path: str):
        if self.checkpoint_steps:
            # Exact-step probes need both the parameters and Adam's first and second moments.
            torch.save(
                {
                    'step': self.global_step,
                    'model_state_dict': trainer._unwrap_model().state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                },
                checkpoint_path,
            )
            log.info(f'Model and optimizer checkpoint saved at {checkpoint_path}')
        else:
            trainer.save_model(checkpoint_path)
            log.info(f'Model checkpoint saved at {checkpoint_path}')

    def save_metrics(self, metrics, metrics_path: str):
        metrics = tensor_to_python_type(metrics)

        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh)

        log.info(f'Model checkpoint metrics saved at {metrics_path}')
