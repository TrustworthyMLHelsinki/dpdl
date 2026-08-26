import logging
import torch

from ..utils import distributed_world_size, is_global_zero

log = logging.getLogger(__name__)


class Callback:
    def __init__(self):
        self.global_step = 0

    def _is_global_zero(self):
        return is_global_zero()

    def _distributed_world_size(self):
        return distributed_world_size()

    @staticmethod
    def _mean_across_ranks(loss, device):
        world_size = distributed_world_size()
        if world_size == 1:
            return loss

        total = torch.tensor(float(loss), device=device)
        torch.distributed.all_reduce(total)
        return (total / world_size).item()

    def on_train_start(self, trainer):
        self.global_step = 0

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        self.global_step += 1

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer, epoch):
        pass

    def on_train_epoch_end(self, trainer, epoch, metrics):
        pass

    def on_train_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_train_physical_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_validation_epoch_start(self, trainer, epoch):
        pass

    def on_validation_epoch_end(self, trainer, epoch, metrics, loss):
        pass

    def on_validation_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_validation_batch_end(self, trainer, batch_idx, batch, loss):
        pass

    def on_test_epoch_start(self, trainer, epoch):
        pass

    def on_test_epoch_end(self, trainer, epoch, metrics, loss):
        pass

    def on_test_batch_start(self, trainer, batch_idx, batch):
        pass

    def on_test_batch_end(self, trainer, batch_idx, batch, loss):
        pass
