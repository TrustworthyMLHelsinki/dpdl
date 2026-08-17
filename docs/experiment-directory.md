# Experiment directory contents

In this document, we summarize what DPDL writes into the experiment directory `<log_dir>/<experiment_name>` and when those files appear.

## Location and lifecycle

The experiment directory is `<log_dir>/<experiment_name>`. Its parts are defined by `--log-dir` and `--experiment-name` and it is created at the start of any DPDL run.
If `--overwrite-experiment` is set, any existing directory with the same name is removed; otherwise, DPDL tries to resume the experiment.

## Always‑written artifacts (all commands)

Written at startup by `start_experiment_logging`:
- `configuration.txt` and `configuration.json`
- `hyperparameters.txt` and `hyperparameters.json`
- `stdout.txt` and `stderr.txt`
- `gpu_type` and `gpu_count`
- `git-hash`

The `txt` files are for humans and `json` for computers.

When any command finishes, a `runtime` file is written to the experiment directory.
As the name suggests, this file contains the time of the run.
The existence of this file is a good way to find finished experiments.
And conversely, unfinished experiments can be found by looking for experiment directories *without* this file.

## Train command (`dpdl train`)

Written before training starts:
- `parameter_counts.txt` and `parameter_counts.json`: model parameter count breakdown
- `dataset_sizes.txt` and `dataset_sizes.json`: number of examples per split

The parameter counts are:
- `P_backbone`: parameters in the backbone (base model without head or adapters)
- `P_frozen`: parameters that remain frozen during training
- `P_model`: all instantiated parameters (backbone + PEFT adapters + head)
- `d_peft`: trainable PEFT parameters (`d_train - d_head`)
- `d_head`: trainable parameters in the classification head
- `d_train`: all trainable parameters (`d_peft + d_head`)

Written at the end of training:
- `test_metrics`: test loss + metrics
- `final_epsilon`: the resulting epsilon (only in DP mode)

If `--save-model` is specified also the resulting model is saved as `final_model.pt`.

Furthermore, many callbacks can save their own data at the end of training (see [callback documentation](./callbacks.md) for `on_train_end` event.)
For example, if `--record-final-train-accuracy` is used, then training metrics after last epoch are saved in `train_metrics` under experiment directory.

## Predict command (`dpdl predict`)

Written after prediction:
- `predictions_<split>.json`: labels/preds/probabilities for the selected split
- `predict_metrics.json`: metrics on the predicted split
- `gradient_diagnostics_<split>.csv`: only if `--prediction-save-gradient-data`

## Train‑predict command (`dpdl train-predict`)

First runs train and then predicts using the resulting model.

## Hyperparameter optimization (`dpdl optimize`)

Files written:
- `hpo_metrics.json`: per‑trial validation loss/metrics
- `trials.json`, `trials.csv`: full Optuna trials table
- `best-params.json`: resolved best hyperparameters
- `best-params-raw-idx.json`: raw Optuna params (ordered indices)
- `best-value`: best objective value (evaluated on the **validation** set)
- `final-metrics`: metrics from the final evaluation round (evaluated on the **test** set)
- `final_epsilon`, resulting epsilon on the final evaluation round
- `noise_multiplier`: DP-SGD noise multiplier in the final evaluation round
- `results-and-configuration.json`: combined summary bundle
- `optuna.conf`: copy of the Optuna search space config
- `optuna.journal`: Optuna journal for the study

Gotchas:
- HPO disables callback recording during trials and re‑enables it for the final evaluation round.
