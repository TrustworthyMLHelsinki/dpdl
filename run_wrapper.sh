#!/bin/bash

# Distributed settings
export MASTER_PORT=$(expr 30000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export HSA_VISIBLE_DEVICES=$SLURM_LOCALID

# Same-node job: keep bootstrap/rendezvous on loopback rather than letting
# Gloo/NCCL try to negotiate over the cluster's HSN/Slingshot interface.
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,ENV,ALLOC
export TORCH_CPP_LOG_LEVEL=INFO
export PYTHONUNBUFFERED=1

echo "rank=$RANK local_rank=$LOCAL_RANK world_size=$WORLD_SIZE rocr_visible_devices=$ROCR_VISIBLE_DEVICES"
echo "master_addr=$MASTER_ADDR master_port=$MASTER_PORT"

getent hosts "$MASTER_ADDR" || echo "getent could not resolve $MASTER_ADDR"

# Exec into the single container instance started once in first_training.sh
# (shared by every rank on this node) instead of starting a new container
# per rank.
singularity exec "instance://$SINGULARITY_INSTANCE_NAME" python -u $*

