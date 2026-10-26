#!/bin/bash
#SBATCH --account=project_462001244
#SBATCH --partition=dev-g
##SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --error=log/slurm-%x.%j.out
#SBATCH --output=log/slurm-%x.%j.stdout
##SBATCH --signal=USR1@300

## note: LUMI has limit of max 56 cores/node

# Load CSC PyTorch
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export SIF=/appl/local/laifs/containers/lumi-multitorch-latest.sif

# Project specific settings
export PROJECT="project_462001244"
export DATA_DIR="/scratch/$PROJECT/data"
export HF_DATASETS_CACHE="$DATA_DIR/huggingface"
export HF_HUB_CACHE="$DATA_DIR/huggingface_hub"
#export TORCH_HOME="$DATA_DIR/torch"
export TYPER_STANDARD_TRACEBACK=1

echo SLURM_NPROCS: $SLURM_NPROCS
echo SLURM_LOCALID: $SLURM_LOCALID
echo $SLURM_PROCID: $SLURM_PROCID
echo ROCR_VISIBLE_DEVICES: $ROCR_VISIBLE_DEVICES

# Start ONE container instance for the whole job, shared by both ranks, instead
# of each srun-launched task starting its own separate `singularity run`. Each
# run_wrapper.sh invocation below execs into this same running instance.
export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container
export SINGULARITY_INSTANCE_NAME="dpdl_mixheikk_${SLURM_JOB_ID}"
singularity instance start -B /scratch/$PROJECT/venvs/dpdl-mixheikk.sqsh:/user-software:image-src=/ "$SIF" "$SINGULARITY_INSTANCE_NAME"
trap 'singularity instance stop "$SINGULARITY_INSTANCE_NAME"' EXIT


cpu_bind_mask="0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

# Run the wrapper script with srun
set -xv

# should only use cpu-bind-mask when reserving full core
#srun --cpu-bind=mask_cpu:$cpu_bind_mask --exclusive ./run_wrapper.sh $@
srun ./run_wrapper.sh run.py train --loss-function CrossEntropyLoss --dataset-name sarus-tech/medical_v3 \
--dataset-label-field disease --llm --model-name HuggingFaceTB/SmolLM2-135M-Instruct --task DiseaseTask --num-workers 0 \
--epochs 2 --batch-size 8 --validation-frequency 0 --device auto --lora-rank 4 \
--lora-alpha 4 --seed 2807 --split-seed 2303 --wandb-logging --wandb-username mixheikk --wandb-project DPDL-LLMs \
--max-length 200 --llm-max-new-tokens 128 --llm-temperature 0.1 --llm-top-p 0.9 --llm-repetition-penalty 1.2 --llm-no-repeat-ngram-size 0 \
--physical-batch-size 8 --optimizer SGD --learning-rate 1e-2 --experiment-name default --validation-size 0.1 --test-size 0.1 \
--max-grad-norm 1 --target-epsilon 8 --no-privacy --peft head-only --skip-test \
--checkpoint-epoch-interval 1 --resume-training-from-checkpoint
#--skip-test
#--overwrite-experiment