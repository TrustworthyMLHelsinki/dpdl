from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping

import opacus
import torch
from opacus import GradSampleModule
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel
from opacus.utils.batch_memory_manager import BatchMemoryManager

from peft import PeftModel

from .callbacks.callback_factory import CallbackFactory, CallbackHandler
from .configurationmanager import Configuration, ConfigurationManager, Hyperparameters
from .datamodules import DataModule, DataModuleFactory
from .device import resolve_device
from .loss_factory import LossFactory
from .metrics_factory import MetricsFactory
from .models.model_base import ModelBase
from .models.model_factory import ModelFactory
from .optimizers import OptimizerFactory
from .utils import is_global_zero, seed_everything, shift_and_flatten

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,

        # essentials
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        datamodule: DataModule,
        adapter: TaskAdapter,

        # generic params
        epochs: int = 10,
        total_steps: int = 0,
        validation_frequency: int = 1,
        seed: int = 0,
        physical_batch_size: int = 40,
        callback_handler: CallbackHandler | None = None,
        peft: str | None = None,
        task: str | None = None,
        device: torch.device | None = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.datamodule = datamodule
        self.epochs = epochs
        self.total_steps = total_steps
        self.validation_frequency = validation_frequency
        self.seed = seed
        self.physical_batch_size = physical_batch_size
        self.peft = peft
        self.task = task
        self.device = device or torch.device('cuda')
        self.adapter = adapter
        self.adapter.device = self.device

        if not callback_handler:
            self.callback_handler = CallbackHandler()
        else:
            self.callback_handler = callback_handler

        if self.epochs and self.total_steps:
            raise ValueError('You should provide either "epochs" or "total_steps", not both.')

        self.setup()

    def setup(self):
        self.model = self.model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model)

    def fit(self):
        self.callback_handler.call('on_train_start', self)

        if self.total_steps:
            self._fit_total_steps()
        else:
            self._fit_epochs()

        self.callback_handler.call('on_train_end', self)

    def _fit_epochs(self):
        for epoch in range(self.epochs):
            self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                self.validate(epoch)

    def _fit_total_steps(self):
        step = 0
        virtual_epoch = 0
        steps_per_epoch = self._calculate_steps_per_epoch()

        # start the first virtual epoch
        self._handle_virtual_epoch_start(virtual_epoch)

        while step < self.total_steps:
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
                if step >= self.total_steps:
                    break

                self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)
                logical_batch_loss = self.fit_one_batch(batch_idx, batch)
                self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)

                step += 1

                if step % steps_per_epoch == 0:
                    self._handle_virtual_epoch_end(virtual_epoch)
                    virtual_epoch += 1

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        self.validate(virtual_epoch)

                    # are we finished?
                    if step >= self.total_steps:
                        break

                    # start the next virtual epoch
                    self._handle_virtual_epoch_start(virtual_epoch)

        last_step_in_epoch = step % steps_per_epoch
        if last_step_in_epoch != 0:
            self._handle_virtual_epoch_end(virtual_epoch)

        assert step == self.total_steps, f'Mismatch in total steps count: Expected {self.total_steps} total steps, but stepped {step} times!'

    def _handle_virtual_epoch_start(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

    def _handle_virtual_epoch_end(self, epoch):
        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        for batch_idx, batch in enumerate(self.datamodule.get_dataloader('train')):
            self.callback_handler.call('on_train_batch_start', self, batch_idx, batch)

            logical_batch_loss = self.fit_one_batch(batch_idx, batch)

            self.callback_handler.call('on_train_batch_end', self, batch_idx, batch, logical_batch_loss)

        # compute the epoch metrics
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()

        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)

    def fit_one_batch(self, batch_idx, batch):
        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
        # then process the sub batches one at a time and call backward.
        # when all the sub batches have been processed we can finally step the optimizer.

        # the adapter handles the physical batches, as it's a different operation depending on the task.
        physical_batches = list(self.adapter.iterate_physical_batches((X, y), self.physical_batch_size))
        N = len(physical_batches)

        logical_batch_loss = 0.0

        # zero the grads as usually before doing anything
        self.optimizer.zero_grad()

        logical_batch_loss = 0
        for i, physical_batch in enumerate(physical_batches):
            self.callback_handler.call('on_train_physical_batch_start', self, i, physical_batch)

            forward_output = self.adapter.forward(self._unwrap_model(), physical_batch)
            loss = self.adapter.compute_loss(self._unwrap_model(), physical_batch, forward_output, normalize_by=N)
            self.adapter.update_metrics(self._unwrap_model(), physical_batch, forward_output)
            loss.backward()

            logical_batch_loss += loss.item()

            # notify the callbacks of a physical batch end
            self.callback_handler.call('on_train_physical_batch_end', self, i, physical_batch, loss.item())

        # after accumulating the gradients for all the sub batches we can finally update weights.
        self.optimizer.step()

        return logical_batch_loss

    def validate(self, epoch=None, enable_callbacks=True):
        return self._evaluate('validation', epoch, enable_callbacks)

    def test(self):
        return self._evaluate('test')

    def get_dataloader(self, name):
        return self.datamodule.get_dataloader(name)

    def get_datamodule(self):
        return self.datamodule

    def get_parameter_count(self):
        from .peft import count_parameters

        return count_parameters(self._unwrap_model())

    def _evaluate(self, mode, epoch=None, enable_callbacks=True):
        if enable_callbacks and is_global_zero():
            self.callback_handler.call(f'on_{mode}_epoch_start', self, epoch)

        self.model.eval()
        torch.set_grad_enabled(False)

        # record the loss separately, as we need to return it when
        # performing hyperparameter optimization
        evaluation_loss = 0

        if mode == 'validation':
            dataloader_name = 'valid'
            metrics_evaluator = self._unwrap_model().valid_metrics
        elif mode == 'test':
            dataloader_name = 'test'
            metrics_evaluator = self._unwrap_model().test_metrics
        elif mode == 'train':
            dataloader_name = 'train_eval'
            metrics_evaluator = self._unwrap_model().train_metrics
        else:
            raise ValueError(f'Unknown evaluation mode: "{mode}"')

        dataloader = self.datamodule.get_dataloader(dataloader_name)

        metrics_evaluator.reset()

        # Weight each batch by the number of terms its mean loss averages over
        # (examples for classification, non-ignored tokens for LM),
        # so the reduction is a per-item mean rather than a mean of per-batch means
        loss_sum = 0.0
        sample_count = 0

        for batch_idx, batch in enumerate(dataloader):
            loss, weight = self._evaluate_one_batch(mode, batch_idx, batch, enable_callbacks, metrics_evaluator)
            loss_sum += loss * weight
            sample_count += weight

        # all_reduce the weighted sum and the count separately, then divide once
        # so the global loss is sum(loss*n) / sum(n) across every rank
        totals = torch.tensor([loss_sum, float(sample_count)], device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        evaluation_loss = (totals[0] / totals[1]).item() if totals[1] > 0 else 0.0

        # metrics_evaluator.compute() performs an all_reduce internally when
        # sync_on_compute=True, so every rank gets the globally aggregated values
        metrics = metrics_evaluator.compute()

        torch.set_grad_enabled(True)
        self.model.train()

        # Callbacks run on rank 0 only, so they cannot reduce a loss themselves
        # Hand them the already-reduced one
        if enable_callbacks and is_global_zero():
            self.callback_handler.call(f'on_{mode}_epoch_end', self, epoch, metrics, evaluation_loss)

        return evaluation_loss, metrics

    def _evaluate_one_batch(self, mode, batch_idx, batch, enable_callbacks, metrics_evaluator):
        # Eval batch callbacks run on rank 0 only
        if enable_callbacks and is_global_zero():
            self.callback_handler.call(f'on_{mode}_batch_start', self, batch_idx, batch)

        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        forward_output = self.adapter.forward(self._unwrap_model(), (X, y))
        loss = self.adapter.compute_loss(self._unwrap_model(), (X, y), forward_output)

        # Exact denominator the mean loss is divided by,
        # so _evaluate can reduce to a true per-item mean across batches and ranks.
        weight = self.adapter.num_loss_items(self._unwrap_model(), (X, y), forward_output)
        self.adapter.update_metrics(
            self._unwrap_model(),
            (X, y),
            forward_output,
            metrics=metrics_evaluator,  # record into the provided evaluator
        )

        if enable_callbacks and is_global_zero():
            self.callback_handler.call(f'on_{mode}_batch_end', self, batch_idx, batch, loss.item())

        return loss.item(), weight

    def _unwrap_model(self):
        m = self.model

        # model can be wrapped inside many module, such as
        # DDP, Opacus' DPDDP or GradSampleModule, and HuggingFace's
        # PeftModule. Let's just unwrap the all the get to ModelBase
        while hasattr(m, 'module'):
            m = m.module

        return m  # ModelBase

    def _calculate_steps_per_epoch(self):
        N = len(self.datamodule.get_dataloader('train').dataset)
        B = self.datamodule.batch_size
        return math.ceil(N / B)

    def save_model(self, fpath, adapters_only=False):
        def unwrap_model_for_saving(m):
            # strip opacus and distributed models until we hit
            # either a ModelBase or HuggingFace's PeftModel
            while True:
                # Strip Opacus' GradSampleModule
                if isinstance(m, opacus.GradSampleModule):
                    m = m._module
                    continue

                # Strip Opacus' DP DPDDP
                if isinstance(m, opacus.distributed.DifferentiallyPrivateDistributedDataParallel):
                    m = m.module
                    continue

                # Strip standard DDP
                if isinstance(m, torch.nn.parallel.DistributedDataParallel):
                    m = m.module
                    continue

                # Stop when we if we found what we want
                if isinstance(m, (PeftModel, ModelBase)):
                    return m

            return m

        model = unwrap_model_for_saving(self.model)

        if isinstance(model, PeftModel):
            if adapters_only:
                # PeftModel knows to save the adapters only
                model.save_pretrained(fpath)

                log.info(f'Saved merged HF PEFT adapters to {fpath}')
            else:
                # Merge PEFT into model and save the whole model
                merged = model.merge_and_unload()

                log.info(f'GOT A NEW MODEL FROM MERGE_AND_UNLOAD: {merged}')
                # The `merge_and_unload` will incorporate the LoRA layers in
                # the model. Then it will return as ModelBase.
                merged.save_model(fpath)

                if is_global_zero():
                    log.info(f'Saved merged HF PEFT model to {fpath}')

            return

        model.save_model(fpath)

    def _sample_impl(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('sample')):
                X = batch
                X = self.adapter.move_to_device(X)

                is_mapping = isinstance(X, Mapping)  # covers dict and HF BatchEncoding
                # gradient accumulation. split the batch to sub batches that fit in the GPU memory.
                # then process the sub batches one at a time and call backward.
                # when all the sub batches have been processed we can finally step the optimizer.
                if is_mapping:
                    # split each tensor in the dict
                    X_split = {k: v.split(self.physical_batch_size, dim=0) for k, v in X.items()}
                else:
                    X_split = X.split(self.physical_batch_size, dim=0)

                N = len(X_split['input_ids'])

                for i in range(N):
                    if is_mapping:
                        X_splitted = {k: X_split[k][i] for k in X_split}
                    else:
                        X_splitted = X_split[i]

                    generated_ids = self._unwrap_model().generate(
                        X_splitted,
                        max_new_tokens=250,
                        temperature=0.5,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.datamodule.tokenizer.pad_token_id,
                        eos_token_id=self.datamodule.tokenizer.eos_token_id,
                    )

                    log.info('Sampled text decoded', self.datamodule.decode(generated_ids))

        self.model.train()


class DifferentiallyPrivateTrainer(Trainer):
    def __init__(
        self,
        *,
        # privacy params
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        clipping_mode: str = 'flat',
        accountant: str = 'prv',
        poisson_sampling: bool = True,
        normalize_clipping: bool = False,
        secure_mode: bool = False,
        target_epsilon: float | None = None,
        target_delta: float | None = None,
        seed: int = 0,
        **kwargs,
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.clipping_mode = clipping_mode
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.seed = seed
        self.poisson_sampling = poisson_sampling
        self.normalize_clipping = normalize_clipping

        # setup opacus privacy engine
        privacy_engine_args = {
            'accountant': accountant,
            'secure_mode': secure_mode,
        }

        self.privacy_engine = opacus.PrivacyEngine(**privacy_engine_args)

        super().__init__(seed=seed, **kwargs)

    def _has_target_privacy_params(self):
        if self.target_epsilon == -1:
            return False

        if not self.target_epsilon:
            return False

        if self.target_epsilon and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" and "target_delta" not given.')

        if all([self.target_epsilon, self.noise_multiplier]):
            raise ValueError('Parameters "target_epsilon" and "noise_multiplier" are exlusive.')

        if self.target_epsilon and not self.target_delta:
            raise ValueError('Parameter "target_epsilon" present, but "target_delta" is missing.')

        return True

    def setup(self):
        noise_generator = torch.Generator(device=self.device)
        if self.seed:
            noise_generator.manual_seed(self.seed)

        self.model = self.model.to(self.device)

        # let's be distributed by default and wrap the model for Opacus DDP.
        # DifferentiallyPrivateDistributedDataParallel is actually a no-op in Opacus, but
        # let's wrap anyway in case of future api changes. https://opacus.ai/tutorials/ddp_tutorial
        model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(self.model)

        optimizer = self.optimizer
        train_dataloader = self.datamodule.get_dataloader('train')

        # setup differential privacy for the model, optimize, and dataloader
        if self._has_target_privacy_params():
            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
            )
        else:
            if self.target_epsilon == -1:
                self.noise_multiplier = 0

            dp_model, dp_optimizer, dp_dataloader = self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                clipping=self.clipping_mode,
                noise_generator=noise_generator,
                poisson_sampling=self.poisson_sampling,
                normalize_clipping=self.normalize_clipping,
                total_steps=self.total_steps,
            )

        # now we can start using the DP'ifyed stuff
        self.model = dp_model
        self.datamodule.set_dataloader('train', dp_dataloader)
        self.optimizer = dp_optimizer

        # When target_epsilon is set, Opacus computes the noise multiplier for us; capture the
        # effective value (also correct for the directly-configured and no-noise paths).
        self.noise_multiplier = float(dp_optimizer.noise_multiplier)

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(self.target_delta)

    def get_noise_multiplier(self):
        return self.noise_multiplier

    def _unwrap_model(self):
        # the model is wrapped inside Opacus, and Opacus distributed.
        # let's unwrap the vanilla model and return it
        return self.model._module.module

    def _fit_total_steps(self):
        # here we'll keep track of our approximate epochs
        virtual_epoch = 0

        # number of total steps taken
        step = 0

        # number of logical batches in an approximate epoch
        n_logical_batches = 0

        # track the logical batch loss here
        logical_batch_loss = 0

        # track the number of physical batches in a logical batch
        n_physical_batch_in_logical = 0

        # flag to indicate the beginning of a new logical batch
        logical_batch_begin = True

        # flag to indicate that a logical batch has been completed (set via the optimizer check)
        logical_batch_completed = False

        # to calculate the start/end of an epoch, we need the number
        # of steps in an epoch.
        steps_per_epoch = self._calculate_steps_per_epoch()

        # At the very start, call on_train_batch_start for the first logical batch.
        if logical_batch_begin:
            self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
            logical_batch_begin = False

        # if 'total_steps' is set then Opacus will do the stepping for us, or
        # more precisely: the dataloader will have exactly 'total_steps' batches.
        # Here, we will spend approximately an epoch worth of those.
        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:
            for batch_idx, batch in enumerate(virtual_dataloader):
                # first batch, we can start first epoch
                if batch_idx == 0:
                    self._handle_virtual_epoch_start(virtual_epoch)

                # now, let's check if we are going to reach the end of logical batch.
                # the optimizer will not skip next gradient update if we are not at
                # the end of the logical batch. there's currently pretty much no other
                # way to do it than this, because we don't know the size of the logical
                # batch that was sampled.
                if not self.optimizer._check_skip_next_step(False):
                    step += 1
                    logical_batch_completed = True
                else:
                    logical_batch_completed = False

                # notify the callbacks of a physical batch start
                self.callback_handler.call('on_train_physical_batch_start', self, batch_idx, batch)

                # let's fit this physical batch
                batch_loss = self.fit_one_batch(batch_idx, batch)

                # notify the callbacks of a physical batch end
                self.callback_handler.call('on_train_physical_batch_end', self, batch_idx, batch, batch_loss)

                # accumulate loss and count the number of physical batches in a logical batch
                logical_batch_loss += batch_loss
                n_physical_batch_in_logical += 1

                # if the logical batch is complete, notify batch end and reset counters
                if logical_batch_completed:
                    self.callback_handler.call(
                        'on_train_batch_end',
                        self,
                        n_logical_batches,
                        None,
                        logical_batch_loss / n_physical_batch_in_logical,  # mean of physical batch losses
                    )
                    n_logical_batches += 1
                    logical_batch_loss = 0
                    n_physical_batch_in_logical = 0

                    # the next iteration starts a new logical batch
                    logical_batch_begin = True

                # At the beginning of a new logical batch, call on_train_batch_start.
                if logical_batch_begin:
                    self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                    logical_batch_begin = False

                # and next we check for epoch end
                if (logical_batch_completed and step % steps_per_epoch == 0) or step == self.total_steps:
                    self._handle_virtual_epoch_end(virtual_epoch)

                    if self.validation_frequency and virtual_epoch % self.validation_frequency == 0:
                        self.validate(virtual_epoch)

                    if step < self.total_steps:
                        virtual_epoch += 1
                        self._handle_virtual_epoch_start(virtual_epoch)
                        # Start a new logical batch for the new epoch.
                        self.callback_handler.call('on_train_batch_start', self, n_logical_batches, None)
                        logical_batch_begin = False

                # Reset the logical batch completion flag for the next iteration.
                logical_batch_completed = False

        if step != self.total_steps:
            log.warn(f'Was going to step for {self.total_steps}, but stepped only {step} steps.')

    def fit_one_batch(self, batch_idx, batch):
        self.optimizer.zero_grad()

        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        forward_output = self.adapter.forward(self._unwrap_model(), (X, y))
        loss = self.adapter.compute_loss(self._unwrap_model(), (X, y), forward_output, normalize_by=None)
        self.adapter.update_metrics(self._unwrap_model(), (X, y), forward_output)
        loss.backward()

        self.optimizer.step()

        loss = loss.item()

        return loss

    def fit_one_epoch(self, epoch):
        self.model.train()
        self.callback_handler.call('on_train_epoch_start', self, epoch)

        logical_idx = 0
        logical_loss = 0.0
        phys_in_logical = 0
        in_new_logical = True

        with BatchMemoryManager(
            data_loader=self.datamodule.get_dataloader('train'),
            max_physical_batch_size=self.physical_batch_size,
            optimizer=self.optimizer,
        ) as virtual_dataloader:

            for phys_idx, batch in enumerate(virtual_dataloader):

                # if we're starting a new logical batch, signal start
                if in_new_logical:
                    self.callback_handler.call(
                        'on_train_batch_start', self, logical_idx, None
                    )
                    in_new_logical = False

                # physical‐batch callbacks
                self.callback_handler.call(
                    'on_train_physical_batch_start', self, phys_idx, batch
                )

                loss = self.fit_one_batch(phys_idx, batch)

                self.callback_handler.call(
                    'on_train_physical_batch_end', self, phys_idx, batch, loss
                )

                # accumulate
                logical_loss += loss
                phys_in_logical += 1

                # check for logical‐batch boundary
                if not self.optimizer._check_skip_next_step(False):
                    avg = logical_loss / phys_in_logical
                    self.callback_handler.call(
                        'on_train_batch_end',
                        self,
                        logical_idx,
                        None,
                        avg,
                    )
                    logical_idx += 1
                    logical_loss = 0.0
                    phys_in_logical = 0
                    in_new_logical = True

        # wrap up epoch
        metrics = self._unwrap_model().train_metrics.compute()
        self._unwrap_model().train_metrics.reset()
        self.callback_handler.call('on_train_epoch_end', self, epoch, metrics)


class TaskAdapter:
    """
        Adapter class for different Tasks.

        One adapter per task family: classification, Causal-LM, ..

        These are to follow the open/close principle: instead of changing
        the Trainer(s), we can just create a new adapter for a new task.

        Handles per-task splitting, moving to device, forward/loss/metrics calls.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def move_to_device(self, X, y=None):
        device = self.device

        def move(obj):
            if isinstance(obj, Mapping):
                return {k: move(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor):
                return obj.to(device=device, non_blocking=True)
            else:
                return obj

        X = move(X)
        y = move(y) if y is not None else None

        return (X, y) if y is not None else X

    def iterate_physical_batches(self, batch, physical_batch_size):
        """
        Return an iterator over physical batches.
        """
        ...

    def forward(self, model, batch):
        raise NotImplementedError

    def compute_loss(self, model, batch, forward_output, normalize_by: int | None = None):
        raise NotImplementedError

    def num_loss_items(self, model, batch, forward_output) -> int:
        raise NotImplementedError

    def update_metrics(self, model, batch, forward_output, metrics = None):
        raise NotImplementedError


class ClassificationAdapter(TaskAdapter):
    def iterate_physical_batches(self, batch, physical_batch_size):
        X, y = batch
        for Xs, ys in zip(X.split(physical_batch_size, 0), y.split(physical_batch_size, 0)):
            yield (Xs, ys)

    def forward(self, model, batch):
        X, _ = batch
        logits = model(X)
        return logits

    def compute_loss(self, model, batch, forward_output, normalize_by: int | None = None):
        _, y = batch

        loss = model.criterion(forward_output, y)

        if normalize_by:
            loss = loss / normalize_by

        return loss

    def num_loss_items(self, model, batch, forward_output) -> int:
        # No ignored targets in classification, so the mean divides by batch size.
        _, y = batch
        return y.shape[0]

    def update_metrics(self, model, batch, forward_output, metrics = None):
        _, y = batch

        if metrics is not None:
            metrics_to_update = metrics
        else:
            metrics_to_update = model.train_metrics if model.training else model.valid_metrics

        metrics_to_update.update(forward_output, y)


class LanguageModelAdapter(TaskAdapter):
    def iterate_physical_batches(self, batch, physical_batch_size):
        X, y = batch
        splits = {k: v.split(physical_batch_size, dim=0) for k, v in X.items()}
        y_splits = y.split(physical_batch_size, dim=0)

        for i in range(len(y_splits)):
            yield ({k: splits[k][i] for k in splits}, y_splits[i])

    def forward(self, model, batch):
        X, _ = batch
        logits = model(X)
        return logits

    def compute_loss(self, model, batch, forward_output, normalize_by: int | None = None):
        _, y = batch
        preds, y_flat = shift_and_flatten(forward_output, y)

        loss = model.criterion(preds, y_flat)

        if normalize_by:
            loss = loss / normalize_by

        return loss

    def num_loss_items(self, model, batch, forward_output) -> int:
        # criterion averages over non-ignored, shifted targets; mirror the shift
        # in shift_and_flatten (drop the first column) and count what remains.
        _, y = batch
        ignore_index = getattr(model.criterion, 'ignore_index', -100)
        return int((y[:, 1:] != ignore_index).sum())

    def update_metrics(self, model, batch, forward_output, metrics = None):
        _, y = batch
        if metrics is not None:
            metrics_to_update = metrics
        else:
            metrics_to_update = model.train_metrics if model.training else model.valid_metrics

        with torch.no_grad():
            metrics_to_update.update(forward_output, y)

# Define task specific adapters
_ADAPTERS = {
    'ImageClassification': ClassificationAdapter,
    'SequenceClassification': ClassificationAdapter,
    'CausalLM': LanguageModelAdapter,
    'InstructLM': LanguageModelAdapter,
}

class TrainerFactory:

    @staticmethod
    def _make_adapter(configuration, device):
        task = configuration.task or 'classification'

        if task not in _ADAPTERS:
            raise ValueError(f'No adapter for task "{task}"')

        return _ADAPTERS[task](device)

    @staticmethod
    def get_trainer(config_manager: ConfigurationManager) -> Trainer:
        device = resolve_device(config_manager.configuration.device)

        # are we differentially private?
        if config_manager.configuration.privacy:
            return TrainerFactory._get_differentially_private_trainer(
                config_manager.configuration,
                config_manager.hyperparams,
                device,
            )

        # XXX: checkpoint dir??? We should have this from the new save model implementation?
        if config_manager.configuration.checkpoint_step_interval is not None:
            config_manager.configuration.checkpoints_dir = os.path.join(
                config_manager.configuration.log_dir,
                config_manager.configuration.experiment_name,
                'checkpoints',
        )

        return TrainerFactory._get_basic_trainer(
            config_manager.configuration,
            config_manager.hyperparams,
            device,
        )

    @staticmethod
    def _get_basic_trainer(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        device: torch.device,
    ) -> Trainer:

        # First create DataModule, it can figure out the number of classes
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams, device)
        output_dim = datamodule.get_output_dim()

        # Now, setup data, model, and optimizer
        loss_fn = LossFactory.get_loss(configuration)

        # This also return effective number of classes, as for LM tasks
        # it is vocabulary size and for classification tasksk it's number
        # of classes as usually.
        model, transforms, output_dim_eff = ModelFactory.get_model(
            configuration,
            hyperparams,
            output_dim,
            loss_fn,
        )

        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)
        metrics = MetricsFactory.get_metrics(configuration, output_dim_eff)
        model.set_metrics(metrics)

        # Initialize the datamodule with the transformations
        datamodule.initialize(transforms)

        # should we cache outputs from the feature extractor?
        if configuration.cache_features:
            # compute cache on rank 0 only
            if is_global_zero():
                datamodule.cache_features(model)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                datamodule.cache_features(model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams, device=device)
        )

        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        adapter = TrainerFactory._make_adapter(configuration, device)

        # instantiate a trainer without dp
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            adapter=adapter,
            callback_handler=callback_handler,
            physical_batch_size=configuration.physical_batch_size,
            epochs=epochs,
            total_steps=total_steps,
            seed=configuration.seed,
            validation_frequency=configuration.validation_frequency,
            peft=configuration.peft,
            task=configuration.task,
            device=device,
        )

        return trainer

    @staticmethod
    def _get_differentially_private_trainer(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        device: torch.device,
    ) -> Trainer:
        # Target delta calculation: A common heuristic is to use 1/N', with N'
        # being the size of the dataset rounded up to the nearest power of 10.
        # To avoid too large values of delta, let's pick a somewhat sensible
        # minimum of 1e-5.
        def _round_up_to_nearest_power_of_10(n):
            return 10 ** math.ceil(math.log10(n))

        def _calculate_target_delta(N):
            N_prime = _round_up_to_nearest_power_of_10(N)
            return min(1e-5, 1 / N_prime)

        def _get_target_privacy_params(hyperparams):
            N = len(datamodule.get_dataloader('train').dataset)
            target_delta = _calculate_target_delta(N)

            if is_global_zero():
                log.info(f'Dataset size is {N}, setting target delta to: {target_delta}.')

            # are we given a target epsilon?
            if hyperparams.target_epsilon is not None:
                target_epsilon = hyperparams.target_epsilon
            else:
                target_epsilon = None

            return target_delta, target_epsilon

        # First initialize the DataModule, it will know about the number of classes
        datamodule = DataModuleFactory.get_datamodule(configuration, hyperparams, device)
        output_dim = datamodule.get_output_dim()

        # Now, setup data, model, and optimizer
        loss_fn = LossFactory.get_loss(configuration)

        model, transforms, output_dim_eff = ModelFactory.get_model(
            configuration,
            hyperparams,
            output_dim,
            loss_fn,
        )

        metrics = MetricsFactory.get_metrics(configuration, output_dim_eff)
        model.set_metrics(metrics)

        optimizer = OptimizerFactory.get_optimizer(configuration, hyperparams, model)

        # The datamodule needs to be aware of the transformations, now we can initialize it
        datamodule.initialize(transforms)
        dataloader = datamodule.get_dataloader('train')

        # Are we caching the outputs of the feature extractor
        if configuration.cache_features:
            # compute cache on rank 0 only
            if is_global_zero():
                datamodule.cache_features(model)
                torch.distributed.barrier()
            else:
                torch.distributed.barrier()
                datamodule.cache_features(model)

        callback_handler = CallbackHandler(
            CallbackFactory.get_callbacks(configuration, hyperparams, device=device)
        )

        target_delta, target_epsilon = _get_target_privacy_params(hyperparams)
        epochs, total_steps = TrainerFactory._get_epochs_and_steps(configuration, hyperparams, datamodule)

        adapter = TrainerFactory._make_adapter(configuration, device)

        # instantiate a differentialy private trained
        trainer = DifferentiallyPrivateTrainer(
            model=model,
            optimizer=optimizer,
            datamodule=datamodule,
            adapter=adapter,
            # hypers
            epochs=epochs,
            total_steps=total_steps,
            noise_multiplier=hyperparams.noise_multiplier,
            max_grad_norm=hyperparams.max_grad_norm,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            poisson_sampling=configuration.poisson_sampling,
            normalize_clipping=configuration.normalize_clipping,
            # config
            accountant=configuration.accountant,
            secure_mode=configuration.secure_mode,
            clipping_mode=configuration.clipping_mode,
            physical_batch_size=configuration.physical_batch_size,
            seed=configuration.seed,
            callback_handler=callback_handler,
            validation_frequency=configuration.validation_frequency,
            peft=configuration.peft,
            task=configuration.task,
            device=device,
        )

        return trainer

    @staticmethod
    def _get_epochs_and_steps(
        configuration: Configuration,
        hyperparams: Hyperparameters,
        datamodule: DataModule,
    ):
        """
        Compute the number of training epochs and total optimizer steps.

        If `--total-steps` is given, we step for exactly that many optimizer
        updates (`use_steps` is not required for this).

        If `use_steps=True` together with `--epochs`, we convert epochs to
        total_steps using ceil(N / B), which matches the default logic in Opacus:
            - sample_rate = 1 / ceil(N / B)
            - steps = int(1 / sample_rate) = ceil(N / B)

        However, default Opacus might still make more steps than us, because we
        cap the total number of steps exactly at `total_steps` and Opacus default
        (`use_steps=False`) always makes a full pass on the dataloader when feeding
        batches through the BatchMemoryManager.

        Returns:
            (epochs, total_steps): One of the values will be None depending on mode.
        """

        # If we're using step-based training and the number of epochs is specified,
        # convert epochs to total steps using the default Opacus logic.
        if configuration.use_steps and hyperparams.epochs:
            dataloader = datamodule.get_dataloader('train')

            # Match Opacus: steps_per_epoch = ceil(N / B)
            N = len(dataloader.dataset)
            B = datamodule.batch_size
            steps_per_epoch = math.ceil(N / B)
            total_steps = steps_per_epoch * hyperparams.epochs
            epochs = None

        # If total steps are manually specified, step for exactly that amount
        # regardless of `use_steps` (which only governs the epochs->steps conversion above).
        elif hyperparams.total_steps:
            total_steps = hyperparams.total_steps
            epochs = None

        # Standard epoch-based training
        else:
            total_steps = None
            epochs = hyperparams.epochs

        return epochs, total_steps
