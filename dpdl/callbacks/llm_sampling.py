import torch

from .base_callback import Callback


class LLMSamplingCallback(Callback):
    """
    Generates and logs a text sample on rank 0 at the end of each training epoch.
    Other ranks wait on a barrier so they stay in sync while sampling runs.
    """

    def on_train_epoch_end(self, trainer, epoch, metrics):
        if self._is_global_zero():
            trainer._sample_impl()

        torch.distributed.barrier()
