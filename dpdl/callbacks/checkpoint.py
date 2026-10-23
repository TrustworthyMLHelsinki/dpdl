import json
import logging
import numpy as np
import os
import random
import re
import shutil

import torch
import torchmetrics
from opacus import distributed
import wandb

from ..utils import tensor_to_python_type
from .base_callback import Callback

log = logging.getLogger(__name__)

class CheckpointCallback(Callback):
    def __init__(self, log_dir: str, checkpoint_step_interval=None, checkpoint_epoch_interval=None, checkpoint_steps=None, remove_old_checkpoints:bool=False, device=None):
        super().__init__()

        self.log_dir = log_dir
        self.checkpoint_step_interval = checkpoint_step_interval
        self.checkpoint_epoch_interval = checkpoint_epoch_interval
        self.checkpoint_steps = set(checkpoint_steps or [])
        self.checkpoints_dir = os.path.join(self.log_dir, 'checkpoints') # NOTE: should use separate checkpoint_dir?
        self.global_step = 0
        self.global_epoch = 0
        self.remove_old_checkpoints = remove_old_checkpoints

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        # NOTE: need to check if distributed is handled correctly somewhere!

        # Initialize mean metric for accumulating train loss over interval
        # sync_on_compute=True (default) so compute() reduces into the global mean across the distributed group.
        self.device = device or torch.device('cuda')
        self.interval_loss = torchmetrics.aggregation.MeanMetric().to(self.device)

    def on_train_start(self, trainer):
        super().on_train_start(trainer)

        # should check, on train_start is not necessarily start of entire training, might be restart
        # currently set again after loading the state
        if self._is_global_zero() and 0 in self.checkpoint_steps:
            checkpoint_path = os.path.join(self.checkpoints_dir, 'checkpoint_step_0.pt')
            self.save_checkpoint(trainer, checkpoint_path)

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, **kwargs):
        # Accumulate on every rank, because loss is rank-local
        # update() reduces this into the global mean.
        self.interval_loss.update(loss)
        self.global_step += 1

        # note: this checkpointing code has not been updated
        if self.checkpoint_step_interval or self.checkpoint_steps:

            interval_checkpoint = self.checkpoint_step_interval and self.global_step % self.checkpoint_step_interval == 0
            if interval_checkpoint or self.global_step in self.checkpoint_steps:
                if self._is_global_zero():
                    checkpoint_path = os.path.join(
                        self.checkpoints_dir, f'checkpoint_step_{self.global_step}'
                    )
                    os.makedirs(checkpoint_path, exist_ok=True)
                    self.save_checkpoint(trainer, checkpoint_path)

                # Exact checkpoints are state snapshots. Validate only at the last one.
                # Decided identically on every rank, so all ranks skip or validate together.
                #final_exact_checkpoint = self.checkpoint_steps and self.global_step == max(self.checkpoint_steps)
                #if not interval_checkpoint and not final_exact_checkpoint:
                #    return

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
                        checkpoint_path, f'checkpoint_step_{self.global_step}_metrics.json'
                    )
                    self.save_metrics(metrics, metrics_path)
                    if self.remove_old_checkpoints:
                        self._prune_old_checkpoints(checkpoints_dir=self.checkpoints_dir, keep=checkpoint_path)
                    trainer.start_step = self.global_step

    def on_train_epoch_end(self, trainer, epoch, metrics, **kwargs):

        # Accumulate on every rank, because loss is rank-local
        # update() reduces this into the global mean.
        #self.interval_loss.update(loss)

        # NOTE: epoch checkpoints don't currently write metrics out
        #log.info(f"on train epoch end, global_step: {self.global_step}, global_epoch: {self.global_epoch}")
        self.global_epoch += 1

        if self.checkpoint_epoch_interval:
            interval_checkpoint = self.checkpoint_epoch_interval and self.global_epoch % self.checkpoint_epoch_interval == 0
            if interval_checkpoint:

                checkpoint_path = os.path.join(
                    self.checkpoints_dir, f'checkpoint_epoch_{self.global_epoch}'
                )
                if self._is_global_zero():
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.distributed.barrier()
                    self.save_checkpoint(trainer, checkpoint_path)
                    if os.path.exists(checkpoint_path):
                        if self.remove_old_checkpoints:
                            self._prune_old_checkpoints(checkpoints_dir=self.checkpoints_dir, keep=checkpoint_path)
                        trainer.start_epoch = self.global_epoch
                else:
                    # need to save cuda & datamodule rng state from all workers, possibly also other rngs
                    # writing to disk with all ranks is faster than distributed communication and saving only on rank0
                    # wait until folder is created by rank0
                    torch.distributed.barrier()
                    self.save_checkpoint(trainer, checkpoint_path)

    def on_train_end(self, trainer, *args, **kwargs):
        # The last exact checkpoint and its metrics were already saved at batch end.
        # Checked on every rank (global_step is in sync) so all ranks agree before
        # the collective validate() below.
        if self.checkpoint_steps and self.global_step == max(self.checkpoint_steps):
            return
        log.warning('skipping final checkpoint!')
        return
        # rest of this fn is out of date, check on_train_epoch_end above for current saving version

        if self._is_global_zero():
            folder_name = f'final_checkpoint_'
            folder_name += f'epoch_{trainer.epochs}' if self.checkpoint_epoch_interval else f'step_{self.global_step}'
            final_checkpoint_path = os.path.join(
                self.checkpoints_dir, folder_name
            )
            os.makedirs(final_checkpoint_path, exist_ok=True)
            self.save_checkpoint(trainer, final_checkpoint_path)
        else:
            # rank0 needs datamodule rng state from all workers
            torch.distributed.gather(tensor=trainer.datamodule.generator.get_state().to(self.device), gather_list=None)
            torch.distributed.gather(tensor=torch.stack(torch.cuda.get_rng_state_all()).to(self.device), gather_list=None)

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
                final_checkpoint_path, f'final_checkpoint_step_{self.global_step}_metrics.json'
            )
            self.save_metrics(metrics, metrics_path)
            if self.remove_old_checkpoints and os.path.exists(final_checkpoint_path):
                self._prune_old_checkpoints(checkpoints_dir=self.checkpoints_dir, keep=final_checkpoint_path)

    def save_checkpoint(self, trainer, checkpoint_path: str):
        # NOTE: any cuda vs cpu issues here?
        # need to check same for loading
        # each worker saves their own rngs, rank0 also other necessary params and model
        to_save = {
                'rng': {
                    'cuda': torch.cuda.get_rng_state_all(),
                    'torch': torch.get_rng_state(),
                    'numpy': np.random.get_state(),
                    'random': random.getstate(),
                    'datamodule': trainer.datamodule.generator.get_state(),
                },
                'global_step': self.global_step,
                'global_epoch': self.global_epoch,
                'start_step': trainer.start_step,
                'start_epoch': trainer.start_epoch,
                # 'model_state_dict': trainer._unwrap_model().state_dict(),
                # should be enough to only save most of these on rank0, saving on all ranks only for debugging
                'optimizer_state': trainer.optimizer.state_dict(),
                'noise_multiplier': float(getattr(trainer.optimizer, 'noise_multiplier', 0.0)),
                'scheduler_state': trainer.scheduler.state_dict() if trainer.scheduler is not None else None,
                # reproducibility currently doesn't seem to require saving these, need to check if something can break without saving
                #'callback_handler': trainer.callback_handler,
                #'train_metrics': getattr(trainer._unwrap_model(), "train_metrics", None),
                #'valid_metrics': getattr(trainer._unwrap_model(), "valid_metrics", None),
                #'test_metrics': getattr(trainer._unwrap_model(), "test_metrics", None),
                'accountant_state': trainer.privacy_engine.accountant.state_dict() if getattr(trainer, "privacy_engine",
                                                                                              None) else None,
                'target_epsilon': trainer.target_epsilon if getattr(trainer, "target_epsilon", None) else None,
                'target_delta': trainer.target_delta if getattr(trainer, "target_delta", None) else None,
            }
        """ if want to save stuff only on rank0, like optimizer state
        if torch.distributed.get_rank() == 0:
            to_save.update(
                {
                'accountant_state': trainer.privacy_engine.accountant.state_dict() if getattr(trainer, "privacy_engine", None) else None,
                'target_epsilon': trainer.target_epsilon if getattr(trainer, "target_epsilon", None) else None,
                'target_delta': trainer.target_delta if getattr(trainer, "target_delta", None) else None,
                }
            )
            to_save['rng'].update({'datamodule': trainer.datamodule.generator.get_state()},)"""
        torch.save(to_save, checkpoint_path + f'/rank{torch.distributed.get_rank()}_training_state.pt',)

        if torch.distributed.get_rank() == 0:
            trainer.save_model(fpath=checkpoint_path, adapters_only=True)
            log.info(f'Checkpoint saved at {checkpoint_path}')

    def _prune_old_checkpoints(self, checkpoints_dir: str, keep: str) -> None:
        """Delete every checkpoint_step_* & checkpoint_epoch_* dir except `keep`. Best-effort."""
        log.info(f"Trying to remove from {checkpoints_dir}")
        keep_base = os.path.basename(os.path.normpath(keep))
        for d in os.listdir(checkpoints_dir):
            log.info(f"found {d}")
            if not (d.startswith('checkpoint_step_') or d.startswith('checkpoint_epoch_')) or d == keep_base:
                continue
            path = os.path.join(checkpoints_dir, d)
            if not os.path.isdir(path):
                continue
            try:
                shutil.rmtree(path)
                log.info(f'Pruned old checkpoint {path}')
            except OSError as e:
                log.warning(f'Could not prune old checkpoint {path}: {e}')

    def save_metrics(self, metrics, metrics_path: str):
        metrics = tensor_to_python_type(metrics)

        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh)

        log.info(f'Model checkpoint metrics saved at {metrics_path}')


