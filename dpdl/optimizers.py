import math
import torch

from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from .configurationmanager import Configuration, Hyperparameters
from .datamodules import DataModule

class OptimizerFactory:
    @staticmethod
    def get_optimizer(configuration: Configuration, hyperparams: Hyperparameters, model: torch.nn.Module):
        optimizer_cls = getattr(torch.optim, configuration.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=hyperparams.learning_rate)
        return optimizer

    @staticmethod
    def get_scheduler(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,
        total_steps: int = None,
    ):
        """Get a learning rate scheduler with warmup and decay.

        Supported `lr_scheduler_type` values: 'cosine', 'linear', 'constant'.
        When `total_steps` is not provided it is derived from the number of
        epochs and the training dataloader size.
        """
        lr_scheduler_type = configuration.lr_scheduler_type

        if total_steps is None and hyperparams.epochs is not None:
            dataloader = datamodule.get_dataloader('train')
            N = len(dataloader.dataset)
            B = datamodule.batch_size
            steps_per_epoch = math.ceil(N / B)
            total_steps = steps_per_epoch * hyperparams.epochs
        elif total_steps is None:
            raise ValueError('Total steps must be provided if epochs is not specified.')

        warmup_steps = int(total_steps * 0.15)

        if lr_scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif lr_scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        elif lr_scheduler_type == 'constant':
            # Linear warmup -> flat LR forever. Recommended for DP fine-tuning:
            # avoids the post-peak decay window where DP noise dominates the
            # gradient signal and the model drifts away from the warmup minimum.
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            raise ValueError(f'Unknown scheduler type: {lr_scheduler_type}')

        return scheduler

    @staticmethod
    def get_scheduler_factory(configuration, hyperparams, datamodule, total_steps=None):
        """Return a closure that builds the scheduler for a (possibly wrapped) optimizer.

        Under DP, Opacus wraps the optimizer only after the privacy engine is
        attached, so the scheduler must be created against that wrapped optimizer.
        """
        def factory(optimizer):
            return OptimizerFactory.get_scheduler(
                configuration, hyperparams, optimizer, datamodule, total_steps,
            )
        return factory

