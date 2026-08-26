import torch
import logging
from dpdl.callbacks.base_callback import Callback

log = logging.getLogger(__name__)


class DebugProbeCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_start")

    def on_train_end(self, trainer):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_end")

    def on_train_epoch_start(self, trainer, epoch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_epoch_start")

    def on_train_epoch_end(self, trainer, epoch, epoch_loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_epoch_end")

    def on_train_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_batch_start")

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_physical_batch_start")

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_batch_end")

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_train_physical_batch_end")

    def on_validation_epoch_start(self, trainer, epoch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_epoch_start")

    def on_validation_epoch_end(self, trainer, epoch, metrics, loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_epoch_end")

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_batch_start")

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_validation_batch_end")

    def on_test_epoch_start(self, trainer, epoch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_epoch_start")

    def on_test_epoch_end(self, trainer, epoch, metrics, loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_epoch_end")

    def on_test_batch_start(self, trainer, batch_idx, batch):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_batch_start")

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        log.info(f"[DEBUG - RANK {torch.distributed.get_rank()}] on_test_batch_end")
