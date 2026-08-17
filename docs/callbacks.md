# Callback system

DPDL uses a lightweight callback interface to plug in logging, metrics, and checkpointing without changing the trainer.
The callback system is akin to the one first introduced in the [fastai library](https://github.com/fastai/fastai).
Callbacks are created via [CallbackFactory](../dpdl/callbacks/callback_factory.py) that constructs the callbacks according to the requested settings in [ConfigurationManager](../dpdl/configurationmanager.py).
To trigger the callbacks, [Trainer](../dpdl/trainer.py) emits callback events during training, validation, and testing.

## Events and batch types

DPDL distinguishes between **logical batches** (one optimizer update) and **physical batches** (or *micro-batches*).
Callbacks receive events for both:

- `on_train_batch_start/end`: logical batch boundaries (one optimizer step).
- `on_train_physical_batch_start/end`: micro-batch boundaries.
- `on_train_epoch_start/end`: epoch boundaries (or ["approximate" epochs](./epochs-vs-steps.md) when using `--use-steps`).
- Validation/testing routines have analogous `on_validation_*` and `on_test_*` events.

## Building a new callback

1) Subclass `Callback` and override only the hooks you need.
2) If you need step count, override `on_train_batch_end`, and call `super()` to keep `global_step` in sync.
3) Use `self._is_global_zero()` to avoid duplicated logging in DDP.
4) Register your callback in `CallbackFactory.get_callbacks()` and add a CLI flag for it in [dpdl/cli.py](../dpdl/cli.py) (see `record_snr`, `record_clipping`, etc. for the pattern).

Minimal example:

```python
import logging

from dpdl.callbacks.base_callback import Callback

log = logging.getLogger(__name__)


class MyCallback(Callback):
    def on_train_start(self, trainer):
        super().on_train_start(trainer)

        if self._is_global_zero():
            log.info('Starting training')

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        # This keeps `self.global_step` in sync.
        super().on_train_batch_end(trainer, batch_idx, batch, loss)

        if self._is_global_zero() and self.global_step % 100 == 0:
            print(f'Logical batch ended, step={self.global_step} loss={loss:.4f}')
```

## Practical guidance

- If you need statistics on a logical batch, you need to accumulate the data via physical batches and then do your magic in `on_train_batch_end`.
- Similarly, to get epoch statistics, accumulate logical batches (note that this can be memory heavy).
- **For step-based training (`--use-steps`), epoch callbacks are approximate.** ([See here for explanation](./epochs-vs-steps.md)).

## Some example callbacks 

- [`RecordEpochStatsCallback`](../dpdl/callbacks/epoch_stats.py): logs start/end of training and epoch progress.
- [`RecordTrainLossByStepCallback`](../dpdl/callbacks/record_losses.py): writes **per-step** train loss to CSV.
- [`RecordLossesByEpochCallback`](../dpdl/callbacks/record_losses.py): writes **per-epoch** train/validation loss to CSV.
- [`RecordAccuracyByEpochCallback`](../dpdl/callbacks/record_accuracy.py): logs train/validation accuracy per epoch.
- [`RecordPerClassAccuracyCallback`](../dpdl/callbacks/per_class_accuracy.py): records per-class accuracy over time.
- [`RecordOptimizerStatsCallback`](../dpdl/callbacks/optimizer_stats.py): writes per-step gradient-norm quantiles, clipping/retained-weight stats, and Adam update statistics to CSV.
- [`CheckpointCallback`](../dpdl/callbacks/checkpoint.py): saves checkpoints and validation metrics every N steps.
- [`DebugProbeCallback`](../dpdl/callbacks/debug.py): prints every hook call to help verify event ordering.
