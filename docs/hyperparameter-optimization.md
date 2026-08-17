# Hyperparameter optimization (or HPO using Optuna)

DPDL relies on [Optuna](https://github.com/optuna/optuna) to optimize selected hyperparameters via the `optimize` command.
In this document, we summarize how the optimizer is configured and how CLI flags work with the HPO.
We aim to keep the document brief and point to code for deeper understanding.

Relevant sources:
- Optimizer logic: [`dpdl/hyperparameteroptimizer.py`](../dpdl/hyperparameteroptimizer.py)
- Search space example: [`conf/optuna_hypers.conf`](../conf/optuna/optuna_hypers.conf)
- Ordered search space example: [`conf/optuna_hypers_ordered.conf`](../conf/optuna/optuna_hypers_ordered.conf)
- Manual trials example: [`conf/optuna_trials.conf`](../conf/optuna/optuna_trials.conf)

For a hands-on, end-to-end walkthrough (search space, manual trials, running
`dpdl optimize`, and the reporting/aggregation scripts under `bin/`), see
[tutorial_hpo.ipynb](../tutorials/tutorial_hpo.ipynb).

## Overview

When running `dpdl optimize`, the DPDL system starts an Optuna study on rank 0 and broadcasts the best hyperparameters (of each trial) to all ranks.
Each trial trains a model (without validation during training) and evaluates the optimization objective (e.g. loss or accuracy) on validation data at the end.
After the best trial is found, we combine the training and validation sets into a new training set (to maximize the amount of training data) and train on it using the hyperparameters from the best trial.
The final metrics are evaluated on the test set.

## Search space config (`conf/optuna_hypers.conf`)

For HPO, we need to specify the ranges of hyperparameters to search for.
For this, there exists a config file in YAML format.
For each hyperparameter to optimize, there *must* be an entry that defines a **type** and its bounds/options.

Supported types:
- `float`: `min`, `max`, and optional boolean `log_space`
- `int`: `min`, `max`
- `categorical`: `options`
- `ordered`: ordered discrete values (see below)

In the config, we can use `-1` to indicate the [maximum batch size](../dpdl/hyperparameteroptimizer.py#L146) (i.e. full dataset).

See [`conf/optuna_hypers.conf`](../conf/optuna/optuna_hypers.conf) for an example.

## Manual trials (`conf/optuna_trials.conf`)

It is also possible to configure manual trials.
These define hypers that Optuna will try before starting its own algorithm for suggesting the next hypers.

In [`conf/optuna_trials.conf`](../conf/optuna/optuna_trials.conf), each entry is a dictionary of hyperparameters and the respective values to try.

## Ordered discrete type (`ordered`)

Optuna does not support *ordered* categorical values natively.
DPDL works around this by implementing an ordered discrete search by mapping values to integer indices during the trial and then remapping the index back to the true value after optimization.
This enables, for example, using discrete learning rate values while still being aware of their relative order in HPO.

Gotchas:
- For convenience, for `batch_size` with `ordered` type, the values are auto‑generated as powers of two from `min` up to the dataset size, plus `-1` for full batch.
- For other ordered parameters (e.g., `learning_rate`), you must provide `options` explicitly (see [`conf/optuna_hypers_ordered.conf`](../conf/optuna/optuna_hypers_ordered.conf) for an example).

## CLI flags and behavior

Core flags for HPO:
- `--target-hypers`: list of hyperparameters to optimize (required). (Repeat the switch to provide multiple values.)
- `--optuna-config`: search space file (default `conf/optuna/optuna_hypers.conf`).
- `--optuna-manual-trials`: optional manual trials file.
- `--optuna-target-metric`: objective metric (`loss` or a metric key, such as `MulticlassAccuracy`).
- `--optuna-direction`: `minimize` or `maximize`, indicating whether to minimize (e.g. loss) or maximize (e.g. accuracy) the optimization objective.
- `--n-trials`: total number of trials.
- `--optuna-random-trials`: warmup random trials (Optuna defaults to 10 completely random trial before starting its algorithm).
- `--optuna-resume`: to resume an existing study with the same experiment name.

## Sampler default

The default sampler is `BoTorchSampler` (Bayesian optimization).
You can switch samplers by providing another Optuna sampler class name via `--optuna-sampler`.
See [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) for more information about samplers.
