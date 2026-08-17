<p align="center">
  <img src="images/dpdl-logo.png" alt="DPDL logo" width="300">
</p>

<h1 align="center" alt="Easy experimentation for Differentially Private Deep Learning">
  <b>Experiment framework for Differentially Private Deep Learning</b>
</h1>

## Tutorials

New to DPDL? Start with [tutorial.ipynb](tutorials/tutorial.ipynb) (training a DP image classifier on a medical dataset) and then [tutorial_hpo.ipynb](tutorials/tutorial_hpo.ipynb) (hyperparameter optimization for that same setup).

## Installation and usage

### Prerequisites

- Python >= 3.10
- PyTorch (CPU or GPU build appropriate for your system)

### Install from source

Clone the repository:

```bash
git clone https://github.com/DPBayes/dpdl.git
cd ./dpdl
```

Create and activate a virtual environment, then install DPDL.

Note that you might want to use `--system-site-packages`, if you are installing DPDL on your cluster.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# You might want to install PyTorch for your platform/CUDA/ROCm first.
# See https://pytorch.org/get-started/locally/

pip install -e .
```

Some features (`--use-steps` and `--normalize-clipping`) require our fork of Opacus:

```bash
pip install "git+https://github.com/DPBayes/opacus.git"
```

Otherwise, the official Opacus can be installed by

```bash
pip install opacus
```

### Test your installation

Run the CPU-only test suite (uses the fake dataset; no downloads):

```bash
pip install -e ".[test]"
pytest -m "not gpu"
```

To run GPU smoke tests (requires CUDA and a visible GPU):

```bash
pytest -m gpu
```

### Command line usage

The entry point is [run.py](run.py) (also installed as the `dpdl` CLI).

### Example usage

At minimum, specify `--epochs` (or `--use-steps` with `--total-steps`).

A real-world example (CIFAR-10 + ViT tiny). This will download data and weights:

```bash
dpdl train --epochs 10 --dataset-name uoft-cs/cifar10 --model-name vit_tiny_patch16_224.augreg_in21k --device auto
```

## Usage

### Command line help

Run `dpdl --help` (or `python run.py --help`).

![](images/usage.png)

### Creating a Slurm script

There is a tool for creating Slurm run scripts for LUMI

```
$ bin/create-run-script.sh
Usage: bin/create-run-script.sh script_name [options...]

script_name               Name of the script to be created.

Options:
  --help                  Show this help message.
  project                 Slurm project (default: project_462000213).
  partition               Slurm partition (default: standard-g).
  gpus                    Number of GPUs (default: 8).
  time                    Time allocation (default: 1:00:00, 00:15:00 for dev-g).
  mem_per_gpu             Memory per GPU (default: 60G).
  cpus_per_task           Number of CPUs per task (default: 7).

Example:
  bin/create-run-script.sh run.sh project_462000213 small-g 1
```

### Training under DP / without DP

See [tutorial.ipynb](tutorials/tutorial.ipynb) for a runnable, end-to-end example (with and without DP).

## Architecture

![DPDL Architecture](images/dpdl-architecture.svg)

### Entry point

The entrypoint [run.py](run.py) provides a CLI using Python's Typer module.

### Command-line interface

The CLI implementation is in [dpdl/cli.py](dpdl/cli.py)

### Training

The CLI calls the `fit` method of [trainer](dpdl/trainer.py)

### Hyperparameter optimization

The CLI calls the `optimize_hypers` method of [hyperparameteroptimizer](dpdl/hyperparameteroptimizer.py).

The ranges/options for the different hyperparameters is in `conf/optuna/optuna_hypers.conf`.

See the detailed guide: [docs/hyperparameter-optimization.md](docs/hyperparameter-optimization.md).

Example (optimize learning rate and batch size):

```
dpdl optimize --target-hypers learning_rate --target-hypers batch_size --n-trials 20 --optuna-config conf/optuna/optuna_hypers.conf
```

### Callbacks

The system provides a flexible [callback system](dpdl/callbacks/callback_factory.py) (see [docs/callbacks.md](docs/callbacks.md)).

### Add a new dataset?

Create a new [datamodule](dpdl/datamodules.py).

NB: The code currently should support all Huggingface image datasets by using, for example, a `--dataset-name cifar100` command-line parameter.

### Add a new model?

Create a new model in `dpdl/models/` and register it in [ModelFactory](dpdl/models/model_factory.py) (see [docs/models.md](docs/models.md) for a detailed description).

### Add a new optimizer?

Add a new optimizer in [optimizers](dpdl/optimizers.py).

## Acknowledgements

We borrow the callback idea from [fastai](https://github.com/fastai/fastai) and the datamodule idea from [PyTorch Lightning](https://github.com/Lightning-AI/lightning).
