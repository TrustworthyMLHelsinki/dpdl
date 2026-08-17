#!/usr/bin/env bash
#
# sweep_epsilon_hpo.sh: run an Optuna HPO study per epsilon value.
#
# Usage: ./tutorials/scripts/sweep_epsilon_hpo.sh
# Run from the repository root so that conf/, bin/, and logs/ resolve correctly.

set -euo pipefail

EPSILONS=(1 3 8)
LOG_DIR="logs/tutorial_hpo_sweep"

for EPSILON in "${EPSILONS[@]}"; do
    EXPERIMENT_NAME="eyepacs_hpo_eps${EPSILON}"

    echo "=== HPO study for epsilon=${EPSILON} ==="

    # --epochs is required up front even though "epochs" is also listed as a target hyperparameter below
    # Optuna overwrites it every trial, so this is just a placeholder
    sbatch first_training.sh optimize \
        --dataset-name bumbledeep/eyepacs \
        --dataset-label-field label_code \
        --model-name vit_tiny_patch16_224 \
        --metric-config conf/metrics/eyepacs.conf \
        --subset-size 0.05 \
        --target-epsilon "${EPSILON}" \
        --epochs 1 \
        --physical-batch-size 64 \
        --target-hypers learning_rate \
        --target-hypers batch_size \
        --target-hypers epochs \
        --target-hypers max_grad_norm \
        --optuna-config conf/optuna/eyepacs_hypers.conf \
        --optuna-manual-trials conf/optuna/eyepacs_trials.conf \
        --optuna-target-metric AUROC \
        --optuna-direction maximize \
        --n-trials 20 \
        --optuna-random-trials 5 \
        --device auto \
        --log-dir "${LOG_DIR}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --optuna-journal "${LOG_DIR}/${EXPERIMENT_NAME}.journal" \
        --overwrite-experiment
done

echo "=== Aggregating all studies ==="
python bin/aggregate-experiment-data-with-optuna-trials.py "${LOG_DIR}" -o eyepacs_hpo_sweep_summary.json
