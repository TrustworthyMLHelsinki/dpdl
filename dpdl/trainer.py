from __future__ import annotations

import logging
import math
import os
import re
import shutil
from collections.abc import Mapping

import opacus
import torch
from opacus import GradSampleModule
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel
from opacus.utils.batch_memory_manager import BatchMemoryManager

from peft import PeftModel

from .callbacks.callback_factory import CallbackFactory, CallbackHandler
from .configurationmanager import Configuration, ConfigurationManager, Hyperparameters
from .datamodules import (
    DataModule,
    DataModuleFactory,
    DISEASE_EVAL_FIELDS,
    normalize_disease_text,
    strip_emojis,
)
#from .datamodules import DataModule, DataModuleFactory
from .device import resolve_device
from .loss_factory import LossFactory
from .metrics_factory import MetricsFactory
from .models.model_base import ModelBase
from .models.model_factory import ModelFactory
from .optimizers import OptimizerFactory
from .utils import seed_everything, shift_and_flatten

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

        # Resume support: epoch to start the training loop from (0 = fresh run).
        # Set by load_training_state() when resuming from a checkpoint.
        # note: not implemented (yet)
        self.start_epoch = 0

        # Some tasks (e.g. DiseaseTask) run a generation-based test/val eval that
        # issues collectives (all_reduce); those must run on ALL ranks. Others
        # keep the rank-0-only eval path. Driven by the adapter.
        self.eval_all_ranks = getattr(adapter, 'needs_generation_eval', False)

        # Populated by DiseaseTaskAdapter.eval_acc; read by the CLI to persist
        # the per-disease/confusion/per-sample memorization-study artifacts.
        self.last_per_disease_accuracy = None
        self.last_disease_confusion = None
        self.last_per_sample_eval = None

        if not callback_handler:
            self.callback_handler = CallbackHandler()
        else:
            self.callback_handler = callback_handler

        if self.epochs and self.total_steps:
            raise ValueError('You should provide either "epochs" or "total_steps", not both.')

        self.setup()

        # note: do we need to shard this, could only run on rank 0? seems like motivation is to avoid inflating counts,
        # but this shouldn't happen if check rank before evals
        # Shard the generation eval loader across ranks so its all_reduce counts
        # are not inflated by the world size (no-op for single-GPU / non-disease).
        if self.eval_all_ranks:
            self._distribute_eval_dataloaders()

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

    def _distribute_eval_dataloaders(self):
        """Shard the generation-eval ('sample') loader across ranks.

        The 'sample' loader feeds the DiseaseTask generation eval, which
        all_reduces its per-rank counts. Without sharding, every rank would
        evaluate the full set and the reduction would over-count by world_size.
        Standard valid/test loaders are intentionally left unsharded (their
        metrics are computed rank-0-only).
        """
        if torch.distributed.get_world_size() <= 1:
            return
        if self.datamodule.get_dataloader('sample') is not None:
            self.datamodule.set_dataloader(
                'sample', self.datamodule._get_distributed_dataloader('sample'),
            )

    def _run_validation(self, epoch):
        """Run validation honoring the per-task rank participation rule.

        For generation-eval tasks all ranks participate (collectives inside
        eval_acc); otherwise only rank 0 evaluates while the rest wait.
        """
        if self.eval_all_ranks or torch.distributed.get_rank() == 0:
            self.validate(epoch)
        torch.distributed.barrier()

    def _fit_epochs(self):
        for epoch in range(self.epochs):
            self.fit_one_epoch(epoch)

            if self.validation_frequency and epoch % self.validation_frequency == 0:
                self._run_validation(epoch)
                #if torch.distributed.get_rank() == 0:
                # self.validate(epoch)

                # other ranks will wait for validation
                torch.distributed.barrier()

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
                        self._run_validation(virtual_epoch)
                        #if torch.distributed.get_rank() == 0:
                        #    self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

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
        if enable_callbacks:
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

        for batch_idx, batch in enumerate(dataloader):
            loss = self._evaluate_one_batch(mode, batch_idx, batch, enable_callbacks, metrics_evaluator)
            evaluation_loss += loss

        evaluation_loss /= len(dataloader)

        # Generation-based eval (e.g. DiseaseTask disease accuracy). No-op for
        # tasks without it. Runs before compute() so its metrics are included.
        self.adapter.eval_acc(self, metrics_evaluator)

        metrics = metrics_evaluator.compute()

        torch.set_grad_enabled(True)
        self.model.train()

        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_epoch_end', self, epoch, metrics)

        return evaluation_loss, metrics

    def _evaluate_one_batch(self, mode, batch_idx, batch, enable_callbacks, metrics_evaluator):
        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_batch_start', self, batch_idx, batch)

        X, y = batch
        X, y = self.adapter.move_to_device(X, y)

        forward_output = self.adapter.forward(self._unwrap_model(), (X, y))
        loss = self.adapter.compute_loss(self._unwrap_model(), (X, y), forward_output)
        self.adapter.update_metrics(
            self._unwrap_model(),
            (X, y),
            forward_output,
            metrics=metrics_evaluator,  # record into the provided evaluator
        )

        if enable_callbacks:
            self.callback_handler.call(f'on_{mode}_batch_end', self, batch_idx, batch, loss.item())

        return loss.item()

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

                if torch.distributed.get_rank() == 0:
                    log.info(f'Saved merged HF PEFT model to {fpath}')

            return

        model.save_model(fpath)

    # Note: should this be moved to DiseaseTaskAdapter?
    def _sample_impl(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.datamodule.get_dataloader('sample')):
                # DiseaseTask's sample loader yields (tokenized, raw_fields);
                # other tasks yield just the tokenized inputs.
                X = batch[0] if isinstance(batch, tuple) else batch
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

                    # NOTE: should move all defaults to cli.py
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
                        # Rank-0-only for standard tasks; all ranks for
                        # generation-eval tasks whose validate() issues collectives.
                        self._run_validation(virtual_epoch)
                        # validate only on rank 0. no need to do distributed here,
                        # the computation is not heavy because we don't need gradients.
                        #if torch.distributed.get_rank() == 0:
                        #    self.validate(virtual_epoch)

                        # other ranks will wait for validation
                        torch.distributed.barrier()

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

    def update_metrics(self, model, batch, forward_output, metrics = None):
        _, y = batch
        if metrics is not None:
            metrics_to_update = metrics
        else:
            metrics_to_update = model.train_metrics if model.training else model.valid_metrics

        with torch.no_grad():
            metrics_to_update.update(forward_output, y)


class DiseaseTaskAdapter(LanguageModelAdapter):
    """Instruction-LM task with a generation-based diagnosis evaluation.

    Training/validation reuse the LanguageModelAdapter forward/loss/metrics
    (the disease collate yields (tokenized, labels) just like InstructLM).
    On top of that, eval_acc generates a completion per test sample and scores
    disease accuracy + PII leakage + confidence, with distributed all_reduce.
    """
    needs_generation_eval = True

    def __init__(self, device):
        super().__init__(device)
        self.tokens_labels = None

    def sample(self, trainer):
        trainer._sample_impl()

    def evaluate_diseases_accuracy_exact_matching(self, trainer):
        log.info('Evaluating diseases accuracy with exact matching...')
        is_dist = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        rank = torch.distributed.get_rank() if is_dist else 0
        trainer.model.eval()

        # Per-field counters: {field: {value_text: count}}
        all_fields = ['disease'] + DISEASE_EVAL_FIELDS
        count_correct = {f: {} for f in all_fields}
        count_true = {f: {} for f in all_fields}

        corr_total = 0
        total = 0

        # Confidence tracking: collect (log_prob, is_correct) per sample.
        # log_prob is None when the disease tokens were not found in the generation.
        confidence_records = []  # list of (mean_log_prob: float | None, correct: bool)

        # Per-sample evaluation records — one dict per example. Drives the
        # memorization-study CSVs (canary scoring s(c), reconstruction lift,
        # PII reconstruction). Gathered across ranks below and exposed on
        # trainer.last_per_sample_eval for the caller to persist.
        per_sample_records = []

        # Confusion-matrix bookkeeping. We pick a SINGLE predicted disease per
        # sample (earliest mention in the generation, ties broken by longest
        # disease name -- handles substring overlap like Pneumonia vs Bronchopneumonia).
        disease_ids_sorted   = sorted(self.tokens_labels.keys())
        disease_texts_sorted = [self.tokens_labels[i]['text'] for i in disease_ids_sorted]
        n_d = len(disease_ids_sorted)
        id_to_row           = {did: r for r, did in enumerate(disease_ids_sorted)}
        disease_text_to_col = {t: c for c, t in enumerate(disease_texts_sorted)}
        # rows = truth disease, cols = predicted disease, last col = "no prediction".
        confusion_local = torch.zeros(n_d, n_d + 1, dtype=torch.float64)

        with torch.no_grad():
            for batch_idx, batch in enumerate(trainer.datamodule.get_dataloader('sample')):
                X, raw_fields = batch
                X = trainer.adapter.move_to_device(X)

                is_mapping = isinstance(X, Mapping)
                if is_mapping:
                    X_split = {k: v.split(trainer.physical_batch_size, dim=0) for k, v in X.items()}
                    N = len(X_split['input_ids'])
                    chunk_sizes = [len(X_split['input_ids'][i]) for i in range(N)]
                else:
                    X_split = X.split(trainer.physical_batch_size, dim=0)
                    N = len(X_split)
                    chunk_sizes = [len(X_split[i]) for i in range(N)]

                offset = 0
                for i in range(N):
                    chunk_size = chunk_sizes[i]

                    if is_mapping:
                        X_splitted = {k: X_split[k][i] for k in X_split}
                    else:
                        X_splitted = X_split[i]

                    # Slice raw_fields for this physical batch
                    raw_chunk = {}
                    for k, v in raw_fields.items():
                        raw_chunk[k] = v[offset:offset + chunk_size]

                    # output_scores=True gives us per-step logits so we can compute
                    # the model's confidence in the disease tokens it generated.
                    generate_output = trainer._unwrap_model().generate(
                        X_splitted,
                        max_new_tokens=60,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        pad_token_id=trainer.datamodule.tokenizer.pad_token_id,
                        eos_token_id=trainer.datamodule.tokenizer.eos_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=4,
                    )
                    generated_ids = generate_output.sequences  # (B, full_seq_len)
                    scores = generate_output.scores            # tuple of (B, vocab) tensors
                    input_len = X_splitted['input_ids'].shape[1]
                    decoded_text = trainer.datamodule.decode(generated_ids[:, input_len:])

                    # Decode the prompt portion too — needed for reconstruction
                    # / leakage analysis. Same skip_special_tokens convention.
                    decoded_prompt = trainer.datamodule.decode(generated_ids[:, :input_len])

                    # Evaluate disease label (via integer → text lookup)
                    if '_disease_id' in raw_chunk:
                        disease_texts = [self.tokens_labels[idx.item()]['text'] for idx in raw_chunk['_disease_id']]
                        count_true['disease'] = count_true_labels(disease_texts, count_true['disease'])

                        # exact_matching gives per-batch aggregate; we need per-sample for confidence.
                        for b in range(chunk_size):
                            disease_text = disease_texts[b]
                            # Strip emojis from the decoded generation before any
                            # text-space disease matching (DP models tend to prepend
                            # decorative emojis that carry no diagnostic signal).
                            cleaned_text = strip_emojis(decoded_text[b])

                            is_correct = bool(
                                re.search(re.escape(disease_text), cleaned_text, flags=re.IGNORECASE)
                            )
                            if is_correct:
                                count_correct['disease'][disease_text] = count_correct['disease'].get(disease_text, 0) + 1
                                corr_total += 1

                            log_prob = _disease_log_prob(
                                scores,
                                generated_ids,
                                disease_text,
                                trainer.datamodule.tokenizer,
                                input_len,
                                b,
                            )
                            confidence_records.append((log_prob, is_correct))

                            # Confusion matrix: pick a single predicted disease for this sample.
                            truth_id  = raw_chunk['_disease_id'][b].item()
                            truth_row = id_to_row[truth_id]
                            predicted = extract_predicted_disease(cleaned_text, disease_texts_sorted)
                            pred_col  = disease_text_to_col[predicted] if predicted is not None else n_d
                            confusion_local[truth_row, pred_col] += 1

                            # Per-sample record for the memorization-study CSV.
                            record = {
                                'disease': disease_text,
                                'disease_id': truth_id,
                                'predicted_disease': predicted,
                                'is_correct': is_correct,
                                'log_prob_answer': log_prob,
                                'generated_text': decoded_text[b],
                                'prompt_text': decoded_prompt[b],
                            }

                            # Per-field truth values + per-sample substring match.
                            for field in DISEASE_EVAL_FIELDS:
                                if field in raw_chunk:
                                    truth_val = raw_chunk[field][b]
                                    if isinstance(truth_val, torch.Tensor):
                                        truth_val = truth_val.item()
                                    truth_val = str(truth_val) if truth_val is not None else ''
                                    record[f'field_truth_{field}'] = truth_val
                                    record[f'field_match_{field}'] = bool(
                                        truth_val and re.search(
                                            re.escape(truth_val), decoded_text[b], flags=re.IGNORECASE,
                                        )
                                    )
                            per_sample_records.append(record)

                    # Evaluate all other fields (PII leakage + utility)
                    for field in DISEASE_EVAL_FIELDS:
                        if field in raw_chunk:
                            field_values = raw_chunk[field]
                            count_true[field] = count_true_labels(field_values, count_true[field])
                            _, count_correct[field] = exact_matching(decoded_text, field_values, count_correct[field])

                    total += chunk_size
                    offset += chunk_size

        trainer.model.train()

        # Local confidence intermediates
        with_conf_local    = [(lp, c) for lp, c in confidence_records if lp is not None]
        conf_correct_local   = [lp for lp, c in with_conf_local if c]
        conf_incorrect_local = [lp for lp, c in with_conf_local if not c]

        if is_dist:
            # Pack all local scalars into one float64 tensor and all-reduce SUM across ranks.
            n_f = len(all_fields)
            base_n = 2 + 2 * n_f + 7
            device = next(trainer.model.parameters()).device
            buf = torch.zeros(base_n + 2 * n_d, dtype=torch.float64, device=device)
            buf[0] = total
            buf[1] = corr_total
            for i, field in enumerate(all_fields):
                buf[2 + i]       = sum(count_true[field].values())
                buf[2 + n_f + i] = sum(count_correct[field].values())
            buf[2 + 2*n_f + 0] = len(with_conf_local)
            buf[2 + 2*n_f + 1] = sum(math.exp(lp) for lp, _ in with_conf_local) if with_conf_local else 0.0
            buf[2 + 2*n_f + 2] = sum(math.exp(lp) * float(c) for lp, c in with_conf_local) if with_conf_local else 0.0
            buf[2 + 2*n_f + 3] = sum(conf_correct_local) if conf_correct_local else 0.0
            buf[2 + 2*n_f + 4] = len(conf_correct_local)
            buf[2 + 2*n_f + 5] = sum(conf_incorrect_local) if conf_incorrect_local else 0.0
            buf[2 + 2*n_f + 6] = len(conf_incorrect_local)
            for d_idx, disease_text in enumerate(disease_texts_sorted):
                buf[base_n + d_idx]       = count_true['disease'].get(disease_text, 0)
                buf[base_n + n_d + d_idx] = count_correct['disease'].get(disease_text, 0)

            torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)
            buf = buf.cpu()

            total      = int(buf[0].item())
            corr_total = int(buf[1].item())
            field_accuracy = {
                field: (
                    buf[2 + n_f + i].item() / buf[2 + i].item()
                    if buf[2 + i].item() > 0 else 0.0
                )
                for i, field in enumerate(all_fields)
            }

            # Replace local per-disease counts with the all-reduced totals.
            count_true['disease'] = {
                disease_text: int(buf[base_n + d_idx].item())
                for d_idx, disease_text in enumerate(disease_texts_sorted)
                if buf[base_n + d_idx].item() > 0
            }
            count_correct['disease'] = {
                disease_text: int(buf[base_n + n_d + d_idx].item())
                for d_idx, disease_text in enumerate(disease_texts_sorted)
                if buf[base_n + n_d + d_idx].item() > 0
            }

            g_n_with_conf   = buf[2 + 2*n_f + 0].item()
            g_sum_probs     = buf[2 + 2*n_f + 1].item()
            g_sum_weights   = buf[2 + 2*n_f + 2].item()
            g_sum_lp_corr   = buf[2 + 2*n_f + 3].item()
            g_n_lp_corr     = int(buf[2 + 2*n_f + 4].item())
            g_sum_lp_incorr = buf[2 + 2*n_f + 5].item()
            g_n_lp_incorr   = int(buf[2 + 2*n_f + 6].item())

            confidence_weighted_acc = g_sum_weights / g_sum_probs if g_sum_probs > 0 else 0.0
            mean_conf_correct   = g_sum_lp_corr   / g_n_lp_corr   if g_n_lp_corr   > 0 else 0.0
            mean_conf_incorrect = g_sum_lp_incorr / g_n_lp_incorr if g_n_lp_incorr > 0 else 0.0

            confidence_stats = {
                'confidence_weighted_accuracy': confidence_weighted_acc,
                'mean_log_prob_correct':        mean_conf_correct,
                'mean_log_prob_incorrect':      mean_conf_incorrect,
                'n_with_confidence':            int(g_n_with_conf),
            }
        else:
            field_accuracy = {
                field: (
                    sum(count_correct[field].values()) / sum(count_true[field].values())
                    if count_true[field] else 0.0
                )
                for field in all_fields
            }
            if with_conf_local:
                probs   = [math.exp(lp) for lp, _ in with_conf_local]
                weights = [math.exp(lp) * float(c) for lp, c in with_conf_local]
                confidence_weighted_acc = sum(weights) / sum(probs)
            else:
                confidence_weighted_acc = 0.0
            mean_conf_correct   = sum(conf_correct_local)   / len(conf_correct_local)   if conf_correct_local   else 0.0
            mean_conf_incorrect = sum(conf_incorrect_local) / len(conf_incorrect_local) if conf_incorrect_local else 0.0
            confidence_stats = {
                'confidence_weighted_accuracy': confidence_weighted_acc,
                'mean_log_prob_correct':        mean_conf_correct,
                'mean_log_prob_incorrect':      mean_conf_incorrect,
                'n_with_confidence':            len(with_conf_local),
            }

        # All-reduce the confusion matrix so rank 0 sees the full eval set.
        if is_dist:
            device = next(trainer.model.parameters()).device
            confusion_buf = confusion_local.to(device)
            torch.distributed.all_reduce(confusion_buf, op=torch.distributed.ReduceOp.SUM)
            confusion_total = confusion_buf.cpu()
        else:
            confusion_total = confusion_local

        # Gather per-sample records across ranks. Every rank must participate.
        if is_dist:
            world = torch.distributed.get_world_size()
            gathered_records = [None] * world
            torch.distributed.all_gather_object(gathered_records, list(per_sample_records))
            per_sample_records = [r for part in gathered_records for r in part]

        # Logging and per-value breakdowns: rank 0 only to avoid duplicate output
        if not is_dist or rank == 0:
            n_with_conf = confidence_stats['n_with_confidence']
            log.info(f'Total correct: {corr_total} / {total}')
            for field, acc in field_accuracy.items():
                log.info(f'  {field} accuracy: {acc:.4f}')
            log.info(
                f'  confidence-weighted accuracy: {confidence_stats["confidence_weighted_accuracy"]:.4f} '
                f'(mean log-prob correct={confidence_stats["mean_log_prob_correct"]:.3f}, '
                f'incorrect={confidence_stats["mean_log_prob_incorrect"]:.3f}, '
                f'n_with_scores={n_with_conf}/{total})'
            )
            accuracy_per_disease = compute_accuracy_per_disease(
                self.tokens_labels, count_correct['disease'], count_true['disease']
            )
            log.info(f'Per-disease breakdown: {accuracy_per_disease}')
            for field in ['name', 'country']:
                breakdown = compute_accuracy_per_value(count_correct[field], count_true[field])
                log.info(f'Per-{field} breakdown ({len(breakdown)} unique values):')
                for value, stats in breakdown.items():
                    log.info(
                        f'  {value!r:40s}  true={stats["true_count"]:4d}  '
                        f'correct={stats["correct"]:4d}  acc={stats["accuracy"]:.3f}'
                    )

            # Build a sparse dict-of-dicts confusion matrix.
            disease_confusion = {}
            for r, truth_text in enumerate(disease_texts_sorted):
                row = {}
                for c, pred_text in enumerate(disease_texts_sorted):
                    cnt = int(confusion_total[r, c].item())
                    if cnt > 0:
                        row[pred_text] = cnt
                na = int(confusion_total[r, n_d].item())
                if na > 0:
                    row[NO_PREDICTION_KEY] = na
                if row:
                    disease_confusion[truth_text] = row

            # Quick sanity log: top off-diagonal confusions.
            off_diag = []
            for r, truth_text in enumerate(disease_texts_sorted):
                for c, pred_text in enumerate(disease_texts_sorted):
                    if r == c:
                        continue
                    cnt = int(confusion_total[r, c].item())
                    if cnt > 0:
                        off_diag.append((cnt, truth_text, pred_text))
            off_diag.sort(reverse=True)
            if off_diag:
                log.info('Top disease confusions (truth -> predicted, count):')
                for cnt, t, p in off_diag[:10]:
                    log.info(f'  {t!r} -> {p!r}: {cnt}')
        else:
            accuracy_per_disease = {}
            disease_confusion = {}

        return (
            field_accuracy['disease'],
            accuracy_per_disease,
            field_accuracy,
            confidence_stats,
            disease_confusion,
            per_sample_records,
        )

    def eval_acc(self, trainer, metrics_evaluator):
        is_dist = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        is_rank0 = (not is_dist) or (torch.distributed.get_rank() == 0)
        if is_rank0:
            log.info('Evaluating diseases accuracy with exact matching after the epoch...')

        acc, accuracy_per_disease, field_accuracy, confidence_stats, disease_confusion, per_sample_records = \
            self.evaluate_diseases_accuracy_exact_matching(trainer)

        # Only rank 0 updates the metrics evaluator to avoid double counting
        if is_rank0:
            def _safe_update(key, value):
                if key in metrics_evaluator:
                    metrics_evaluator[key].update(value)
                else:
                    log.warning(
                        f'Metric "{key}" not found in metrics collection '
                        f'(collection has: {list(metrics_evaluator.keys())}).'
                    )

            _safe_update('MulticlassAccuracyDisease', acc)

            field_to_metric = {
                'name':       'AccuracyName',
                'country':    'AccuracyCountry',
                'occupation': 'AccuracyOccupation',
                'hobby':      'AccuracyHobby',
                'symptoms':   'AccuracySymptoms',
                'treatment':  'AccuracyTreatment',
            }
            for field, metric_key in field_to_metric.items():
                if field in field_accuracy:
                    _safe_update(metric_key, field_accuracy[field])

            _safe_update('ConfidenceWeightedAccuracyDisease', confidence_stats['confidence_weighted_accuracy'])
            _safe_update('MeanLogProbCorrect',                confidence_stats['mean_log_prob_correct'])
            _safe_update('MeanLogProbIncorrect',              confidence_stats['mean_log_prob_incorrect'])

            log.info(f'Accuracy after the epoch for diseases: {acc}')
            log.info(f'Accuracy per disease: {accuracy_per_disease}')
            log.info(f'Per-field accuracy: {field_accuracy}')

            trainer.last_per_disease_accuracy = accuracy_per_disease
            trainer.last_disease_confusion    = disease_confusion
            trainer.last_per_sample_eval      = per_sample_records

        return acc

    def set_label_tokens(self, datamodule):
        label_field = datamodule._label_field
        splits = datamodule._dataset_splits
        train_class_label = splits['train'].features[label_field]

        # Guard against silent label drift: every split must share the same
        # ClassLabel.names as train, otherwise _disease_id from valid/test will
        # decode to a different disease text via tokens_labels (built from train).
        train_names = list(train_class_label.names)
        for split_name, split in splits.items():
            if split_name == 'train':
                continue
            split_feature = split.features[label_field]
            split_names = list(getattr(split_feature, 'names', []))
            if split_names != train_names:
                raise ValueError(
                    f'ClassLabel mismatch between train and "{split_name}" for '
                    f'label field "{label_field}". '
                    f'This means _disease_id will map to the wrong disease text. '
                    f'train.names={train_names!r}  {split_name}.names={split_names!r}'
                )

        class_number = splits['train'][label_field]
        diseases = {}
        for i in class_number:
            # Normalize so the disease string used by substring eval, the
            # confusion matrix, and the per-disease CSV matches the form that
            # appears in dataset narratives (and in the training prepend).
            disease_text = normalize_disease_text(train_class_label.int2str(i))
            if i in diseases:
                diseases[i]['count'] += 1
            else:
                tokens = datamodule.tokenizer.encode(disease_text, add_special_tokens=False)
                diseases[i] = {'count': 1, 'tokens': tokens, 'text': disease_text}

        self.tokens_labels = diseases


def find_disease_token_span(token_ids, disease_text, tokenizer):
    """Find the [start, end) token span covering `disease_text` in `token_ids`.

    The search happens in *text* space, then we walk the tokens once to project
    the character range back onto token indices. Assumes a byte-level /
    SentencePiece tokenizer where decode is additive over tokens (OLMo, GPT-*,
    Llama, Mistral, ...). Returns (-1, -1) when not present.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if not token_ids or not disease_text:
        return -1, -1

    full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    char_pos = full_text.lower().find(disease_text.lower())
    if char_pos < 0:
        return -1, -1
    char_end = char_pos + len(disease_text)

    cursor = 0
    start_token = None
    end_token = None
    for i, tid in enumerate(token_ids):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        next_cursor = cursor + len(tok_str)
        if start_token is None and cursor <= char_pos < next_cursor:
            start_token = i
        if start_token is not None and char_end <= next_cursor:
            end_token = i + 1
            break
        cursor = next_cursor

    if start_token is None or end_token is None:
        return -1, -1
    return start_token, end_token


def _disease_log_prob(scores, generated_ids, disease_text, tokenizer, input_len, sample_idx):
    """Mean log-prob assigned by the model to the disease span it produced.

    Returns the mean log-prob (float) over the in-context tokens spanning the
    disease, or None if the disease string is not present in the generation or
    no scores exist.
    """
    if len(scores) == 0:
        return None

    new_ids = generated_ids[sample_idx, input_len:]  # generated tokens only
    start, end = find_disease_token_span(new_ids, disease_text, tokenizer)
    if start < 0:
        return None

    total_lp = 0.0
    n_tokens = 0
    new_ids_list = new_ids.tolist() if isinstance(new_ids, torch.Tensor) else list(new_ids)
    for j in range(start, end):
        if j >= len(scores):
            return None
        lp = torch.log_softmax(scores[j][sample_idx], dim=-1)
        total_lp += lp[new_ids_list[j]].item()
        n_tokens += 1

    if n_tokens == 0:
        return None
    return total_lp / n_tokens

NO_PREDICTION_KEY = '__no_prediction__'

def extract_predicted_disease(text, disease_texts):
    """Return the single disease name the model most plausibly predicted in `text`.

    Heuristic: earliest case-insensitive occurrence of any known disease name;
    ties broken by length (longer wins) so substring overlaps like "Pneumonia"
    inside "Bronchopneumonia" don't masquerade as the prediction.
    """
    if not text:
        return None
    text_lower = text.lower()
    best_pos = None
    best_len = -1
    best_name = None
    for name in disease_texts:
        if not name:
            continue
        idx = text_lower.find(name.lower())
        if idx < 0:
            continue
        if (
            best_pos is None
            or idx < best_pos
            or (idx == best_pos and len(name) > best_len)
        ):
            best_pos = idx
            best_len = len(name)
            best_name = name
    return best_name


def count_true_labels(label_text, count_true):
    for label in label_text:
        count_true[label] = count_true.get(label, 0) + 1
    return count_true

def exact_matching(texts, labels, count_correct=None):
    corr = 0
    for i in range(len(texts)):
        label = labels[i]
        if not label:
            continue
        if re.search(re.escape(label), texts[i], flags=re.IGNORECASE):
            count_correct[label] = count_correct.get(label, 0) + 1
            corr += 1
    return corr, count_correct


def compute_accuracy_per_disease(token_labels, count_correct, count_true):
    accuracy_per_disease = {}
    for _, info in token_labels.items():
        disease_name = info['text']

        correct_count = count_correct.get(disease_name, 0)
        true_count = count_true.get(disease_name, 0)

        accuracy = correct_count / true_count if true_count > 0 else 0.0

        accuracy_per_disease[disease_name] = {
            'correct': correct_count,
            'true_count': true_count,
            'accuracy': accuracy,
        }
    return accuracy_per_disease


def compute_accuracy_per_value(count_correct, count_true):
    """Per-value accuracy breakdown for arbitrary string fields (name, country, …)."""
    result = {}
    for value, true_count in count_true.items():
        correct_count = count_correct.get(value, 0)
        result[value] = {
            'true_count': true_count,
            'correct': correct_count,
            'accuracy': correct_count / true_count if true_count > 0 else 0.0,
        }
    return dict(sorted(result.items(), key=lambda kv: kv[1]['true_count'], reverse=True))

# Define task specific adapters
_ADAPTERS = {
    'ImageClassification': ClassificationAdapter,
    'SequenceClassification': ClassificationAdapter,
    'CausalLM': LanguageModelAdapter,
    'InstructLM': LanguageModelAdapter,
    'DiseaseTask': DiseaseTaskAdapter,
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
            if torch.distributed.get_rank() == 0:
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
        # Note: can move this?
        # Build the disease label->token/text mapping (no-op for other tasks).
        adapter.set_label_tokens(datamodule)

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

            if torch.distributed.get_rank() == 0:
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
            if torch.distributed.get_rank() == 0:
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
        # note: possible to move this?
        # Build the disease label->token/text mapping (no-op for other tasks).
        adapter.set_label_tokens(datamodule)

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

        If `use_steps=True`, we convert epochs to total_steps using ceil(N / B),
        which matches the default logic in Opacus:
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

        # If total steps are manually specified in config
        elif configuration.use_steps and hyperparams.total_steps:
            total_steps = hyperparams.total_steps
            epochs = None

        # Standard epoch-based training
        else:
            total_steps = None
            epochs = hyperparams.epochs

        return epochs, total_steps
