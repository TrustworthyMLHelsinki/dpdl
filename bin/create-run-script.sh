#!/bin/bash

show_help() {
    echo "Usage: $0 script_name env_path [options...]"
    echo ""
    echo "script_name               Name of the script to be created."
    echo "env_path                  Path to the environment (see env_type below)."
    echo ""
    echo "Options:"
    echo "  --help                  Show this help message."
    echo "  env_type                Environment type: 'singularity' or 'venv' (default: venv)."
    echo "                          env_path is the .sqsh image for singularity, or the venv"
    echo "                          directory (containing bin/activate) for venv."
    echo "  project                 Slurm project (default: project_xxxxxxxxx)."
    echo "  partition               Slurm partition (default: standard-g)."
    echo "  gpus                    Number of GPUs (default: 8)."
    echo "  time                    Time allocation (default: 1:00:00, 00:15:00 for dev-g)."
    echo "  mem_per_gpu             Memory per GPU (default: 60G)."
    echo "  cpus_per_task           Number of CPUs per task (default: 7)."
    echo ""
    echo "Example:"
    echo "  $0 run.sh /scratch/project_xxxxxxxxx/venvs/dpdl venv project_xxxxxxxxx small-g 1"
}

# Check for --help option
if [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# First argument is the script name
script_name=$1
if [[ "$script_name" == "" ]]; then
    show_help
    exit 0
fi

# Second argument is the environment path, required just like script_name
env_path=$2
if [[ "$env_path" == "" ]]; then
    echo "Error: env_path is required." >&2
    show_help
    exit 1
fi

env_type=${3:-"venv"}
if [[ "$env_type" != "singularity" && "$env_type" != "venv" ]]; then
    echo "Error: env_type must be either 'singularity' or 'venv'." >&2
    show_help
    exit 1
fi

# Wrapper script sets the environment variables after "srun" has been called
wrapper_script="run_wrapper.sh"

project=${4:-"project_xxxxxxxxx"}
partition=${5:-"standard-g"}
gpus=${6:-8}
ntasks_per_node=$gpus
time=${7:-"1:00:00"}
mem_per_gpu=${8:-"60G"}
cpus_per_task=${9:-7}
cpu_bind_mask="0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"
nodes=1

srun_args=""

# if we are using all the GPUs, then set GPU binding and reserve the whole node
if [ "$gpus" == "8" ]; then
    srun_args="$srun_args --cpu-bind=mask_cpu:$cpu_bind_mask"
    srun_args="$srun_args --exclusive"
fi

if [ "$partition" == "dev-g" ]; then
    time="00:15:00"
fi

# Only the final line differs between the two environment types
if [[ "$env_type" == "singularity" ]]; then
    run_cmd='singularity exec "instance://$SINGULARITY_INSTANCE_NAME" python3 -u "$@"'
else
    run_cmd='python3 -u "$@"'
fi

# Create the wrapper script dynamically
cat <<EOF > $wrapper_script
#!/bin/bash

MIOPEN_DIR=\$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_DIR/cache
export MIOPEN_USER_DB=\$MIOPEN_DIR/config

# Distributed settings
export MASTER_PORT=\$(expr 30000 + \$(echo -n \$SLURM_JOBID | tail -c 4))
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=\$SLURM_NPROCS
export LOCAL_RANK=\$SLURM_LOCALID
export RANK=\$SLURM_PROCID
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
export HSA_VISIBLE_DEVICES=\$SLURM_LOCALID

# Same-node job: keep bootstrap on loopback
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

getent hosts "\$MASTER_ADDR" || echo "getent could not resolve \$MASTER_ADDR"

# Finally, run the program
$run_cmd
EOF

chmod +x $wrapper_script

# Create the specified main script dynamically
cat <<EOF > $script_name
#!/bin/bash
#SBATCH --account=$project
#SBATCH --partition=$partition
#SBATCH --nodes=$nodes
#SBATCH --ntasks-per-node=$ntasks_per_node
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --gpus=$gpus
#SBATCH --time=$time
#SBATCH --mem-per-gpu=$mem_per_gpu
#SBATCH --threads-per-core=1
#SBATCH --error=slurm-%x.%j.out
#SBATCH --output=slurm-%x.%j.stdout

# Fix for illegal memory access with convolutional networks
export MIOPEN_DEBUG_CONV_CK_IGEMM_FWD_V6R1_DLOPS_NCHW=0

# Project specific settings
export PROJECT="$project"
export DATA_DIR="/scratch/\$PROJECT/data"
export HF_DATASETS_CACHE="\$DATA_DIR/huggingface"
export HUGGINGFACE_HUB_CACHE="\$DATA_DIR/huggingface_hub"
export TORCH_HOME="\$DATA_DIR/torch"
export _TYPER_STANDARD_TRACEBACK=1

EOF

if [[ "$env_type" == "venv" ]]; then
cat <<EOF >> $script_name
# Activate virtual environment
source "$env_path/bin/activate"

EOF
else
cat <<EOF >> $script_name
# Start one container instance for the whole job, shared by all ranks
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
export SINGULARITY_INSTANCE_NAME="dpdl_\${SLURM_JOB_ID}"
singularity instance start -B "$env_path":/user-software:image-src=/ "\$SIF" "\$SINGULARITY_INSTANCE_NAME"
trap 'singularity instance stop "\$SINGULARITY_INSTANCE_NAME"' EXIT

EOF
fi

cat <<EOF >> $script_name
# Run the wrapper script with srun
set -xv
srun $srun_args ./$wrapper_script \$@
EOF

# Make the main script executable
chmod +x $script_name

echo "Created scripts: $script_name and $wrapper_script."
