# Epochs vs steps, rounding errors in logging, and sample-rate rounding

DPDL supports both epoch-based and step-based training.
Under the hood, Opacus uses Poisson sampling, so the number of optimizer updates per epoch depends on the sampling scheme.
This section documents how `--use-steps`, `--epochs`, and `--total-steps` interact, why rounding appears in logs, and why this mirrors Opacus behavior.

We will use `S` for steps, `N` for the size of the dataset, `B` for batch sizes, `E` for epochs.

## Modes and conversions

Note: Please install [our Opacus fork](https://github.com/DPBayes/opacus) to use the step mode.

- Epoch mode: provide `--epochs E`.
  - DPDL trains for E epochs (full passes through the dataloader).
  - By default Opacus only supports [discrete sample-rates](https://github.com/meta-pytorch/opacus/blob/f17f254ab8f1f1095e8257bf278769d549748bbc/opacus/privacy_engine.py#L408), e.g. `sample_rate = 1 / len(data_loader)`.
    For standard dataloaders, `len(data_loader) = ceil(N / B)`. This setting limits the choice of batch size, which will be discussed in detail [later](#sample-rate-discrete-vs-smooth).

- Step mode: provide `--use-steps`, and then specify the value for either `--total-steps` or `--epochs`.
  - `--use-steps --total-steps S`: DPDL trains for exactly `S` optimizer updates.
    We set `sample_rate = batch_size / dataset_size` and the sampler runs for exactly `S` steps.
  - `--use-steps --epochs E`: DPDL converts epochs to steps and then trains in step mode.
    The conversion is:

```
steps_per_epoch = ceil(N / B)
total_steps = steps_per_epoch * E
```

## Rounding of epochs up in logs when using `--use-steps`

When step-based training is enabled, DPDL reconstructs an "approximate epoch" for logging and [callbacks](./callbacks.md), which enables the analysis of the dynamics during the training.
This is intentionally approximate because Poisson sampling does not yield a fixed number of samples per step.
The conversion that logging uses:

```
steps_per_epoch ~= N / B
approx_epochs = ceil(total_steps / (N / B))
```

Because `total_steps` was computed with `ceil(N / B)`, the forward and backward conversions do not exactly match.
For small epoch counts this mismatch can be significant.

Example:
- `N = 1000`, `B = 256`
- `steps_per_epoch = ceil(1000/256) = 4`
- `--use-steps --epochs 1` => `total_steps = 4`
- `approx_epochs = ceil(4 / (1000/256)) = ceil(1.024) = 2`

So logs may show "approximate epoch 2" even though you asked for 1 epoch.
**This only affects logging**; the number of optimizer updates is still exactly `total_steps`.

## Comparison to Opacus default behavior

When `total_steps` is not provided, Opacus computes:
- `sample_rate = 1 / len(data_loader)`
- [UniformSampler](https://github.com/meta-pytorch/opacus/blob/f17f254ab8f1f1095e8257bf278769d549748bbc/opacus/privacy_engine.py#L408) uses `steps = int(1 / sample_rate)`.

With standard dataloaders, `len(data_loader) = ceil(N / B)`, so Opacus implicitly rounds to `ceil(N / B)` steps per epoch.
DPDL uses the same conversion, so step counts are consistent with Opacus.

## Sample-rate: discrete vs smooth

The choice of `total_steps` also controls which sample rates are possible.

- Without `total_steps`:
  - `sample_rate` in Opacus is [forced to discrete values](https://github.com/meta-pytorch/opacus/blob/f17f254ab8f1f1095e8257bf278769d549748bbc/opacus/privacy_engine.py#L408) `1 / len(data_loader)` (examples: 1/2, 1/3, 1/4, ...).
  - This forces discrete sample rates tied to the dataloader length.

- With `total_steps`:
  - Opacus sets `sample_rate = batch_size / dataset_size`.
  - This supports "smooth" rates such as 999/1000 or 0.2, as long as your batch size reflects that rate.
  - DPDL exposes this mode via `--use-steps --total-steps S`.

## Practical guidance

- For exact update counts or smooth sample rates, prefer `--use-steps --total-steps`.
- To stay aligned with Opacus defaults, use `--epochs` (without `--use-steps`) or `--use-steps --epochs S`, and expect rounding in the logs.
- If you want epoch counts that map cleanly to steps, choose batch sizes that divide the dataset size evenly.
