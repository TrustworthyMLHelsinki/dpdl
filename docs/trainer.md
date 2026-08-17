# Trainer and DifferentiallyPrivateTrainer

This document summarizes how DPDL's trainers manage training, including handling of data, models, and optimization.
We keep the documentation intentionally brief and point to the relevant source files for deeper understanding.

Relevant sources:
- Trainers and adapters: [`dpdl/trainer.py`](../dpdl/trainer.py)
- Metrics setup: [`dpdl/metrics_factory.py`](../dpdl/metrics_factory.py)
- Model base + metric storage: [`dpdl/models/model_base.py`](../dpdl/models/model_base.py)
- Callback system: [Callback system](callbacks.md)
- Step/epoch rounding notes: [Epochs vs steps](epochs-vs-steps.md)

## Overview

DPDL uses two trainer paths:
- `Trainer`: standard (non-DP) training loop.
- `DifferentiallyPrivateTrainer`: extends `Trainer` and employs [Opacus](https://github.com/meta-pytorch/opacus/) for DP.

`TrainerFactory` initializes a [datamodule](../dpdl/datamodules.py), a [model](../dpdl/models/model_base.py), an [optimizer](../dpdl/optimizers.py), and [callbacks](./callbacks.md) and [metrics](../dpdl/metrics_factory.py) based on the configuration passed in [ConfigurationManager](../dpdl/configurationmanager.py).

## Metrics

We support [torchmetrics](https://github.com/Lightning-AI/torchmetrics) through the [metrics module](../dpdl/metrics_factory.py).

Metrics live in the [model](../dpdl/models/model_base.py) (`train_metrics`, `valid_metrics`, `test_metrics`) and the Trainer(s) update them during different training phases.

### Built-in metrics

`MetricsFactory` always initialises a task-appropriate baseline set:

- **Classification** (`ImageClassification`, `SequenceClassification`): macro-, micro-, and per-class-averaged `MulticlassAccuracy`, and a `ConfusionMatrix` on the test split.
- **Language modelling** (`CausalLM`, `InstructLM`): micro `MulticlassAccuracy` and `Perplexity`.

### Custom metrics

Additional metrics can be declared in a YAML config file and passed via `--metric-config`. The file has three optional sections (`train_metrics`, `valid_metrics`, `test_metrics`), each a list of entries. An entry is either a plain metric class name or a mapping:

```yaml
train_metrics:
  - Accuracy                   # shorthand: just the class name

valid_metrics:
  - name: AUROC                # torchmetrics class name
    params:
      thresholds: 10
      average: macro
```

See [`conf/metrics/example.conf`](../conf/metrics/example.conf) for a working example.

`CustomMetricsFactory` resolves each name against the torchmetrics submodule hierarchy (`torchmetrics`, `torchmetrics.classification`, `torchmetrics.text`, `torchmetrics.regression`, `torchmetrics.image`, `torchmetrics.audio`). Task-level defaults such as `num_classes` and `task` are injected automatically where the metric accepts them; `sync_on_compute` is also set to match the train/eval setting. Params declared in the config always take precedence over these defaults.

## Task adapters

The trainers use **task adapters** to keep task-specific logic out of the training loop.
Adapters are defined in [`dpdl/trainer.py`](../dpdl/trainer.py) each provides functions:
- `move_to_device`: to transfer data to device.
- `iterate_physical_batches`: to split a logical batch into micro-batches.
- `forward`, `compute_loss`, `update_metrics`: task-specific training behavior.

Current adapters include:
- `ClassificationAdapter` for image and sequence classification.
- `LanguageModelAdapter` for causal/LLM tasks.

Should you need to add a new task, implement a new adapter and register it in `_ADAPTERS`.

## Training loop (epoch vs step mode)

The core loop lives in `Trainer.fit()` and chooses between:
- **Epoch mode**: iterate dataloader for each epoch.
- **Step mode**: iterate until a target number of optimizer steps is reached.

_Note: Please install [our Opacus fork](https://github.com/DPBayes/opacus) to use the step mode._

When `--use-steps` is enabled, epoch counts are **approximate** and derived from batch size and dataset size.
This behavior is documented in [Epochs vs steps](epochs-vs-steps.md).

## Logical vs physical batches

DPDL supports gradient accumulation for both DP and non-DP paths.
This means that logical batches are divided into **physical batches** (or *micro-batches*) that fit in memory according to physical batch size (`--physical-batch-size`).
If your GPU is spinning below 100%, a first tuning step to improve performance would be to increase the physical batch size.

For e.g. calculating batch statistics, use the [Callback system](callbacks.md) that receives events for both logical and physical batches.

## DifferentiallyPrivateTrainer

In addition to basic `Trainer` functionality, `DifferentiallyPrivateTrainer` creates an Opacus [Privacy Engine](https://opacus.ai/api/privacy_engine.html) to manage differential privacy.
For this, we:
- Wrap the model in DP-aware modules.
- Replace the training dataloader with a DP-compatible sampler.
- Configure sampling and accounting via `make_private` (if `--noise-multiplier` is given) or `make_private_with_epsilon` (if `--target-epsilon` is given).

In essence, DP training mode just uses Opacus to wrap the model, optimizer, and dataloader while keeping the rest of the training loop structure the same.

## Callbacks and logging

Callbacks are created by `CallbackFactory` and invoked through `CallbackHandler` (see [callback factory](../dpdl/callbacks/callback_factory.py)) inside the trainer loops.
For further information on hooks and the callback system in general, see [Callback system](callbacks.md).
