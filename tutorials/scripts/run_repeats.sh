#!/usr/bin/env bash
#
# run_repeats.sh: run the same DPDL training configuration multiple times
# with different seeds, then report mean/std test AUROC.
#
# Usage: ./tutorials/scripts/run_repeats.sh
# Run from the repository root so that conf/ and logs/ resolve correctly.

set -euo pipefail

# --- configuration shared by every run -------------------------------------
DATASET_NAME="bumbledeep/eyepacs"
LABEL_FIELD="label_code"
MODEL_NAME="vit_tiny_patch16_224"
METRIC_CONF="conf/metrics/eyepacs.conf"
EPOCHS=5
BATCH_SIZE=256
PHYSICAL_BATCH_SIZE=64
LEARNING_RATE=1e-3
TARGET_EPSILON=8
DEVICE="auto"
LOG_DIR="logs/tutorial"
SEEDS=(0 1 2 3 4)           # one seed per repeat; must have NUM_RUNS entries

# --- run the repeats ---------------------------------------------------------
RESULT_DIRS=()

for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    EXPERIMENT_NAME="eyepacs_vit_tiny_eps${TARGET_EPSILON}_seed${SEED}"

    echo "=== Run $((i + 1))/${#SEEDS[@]}: seed=${SEED} ==="

    dpdl train \
        --dataset-name "${DATASET_NAME}" \
        --dataset-label-field "${LABEL_FIELD}" \
        --model-name "${MODEL_NAME}" \
        --metric-config "${METRIC_CONF}" \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --physical-batch-size "${PHYSICAL_BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --target-epsilon "${TARGET_EPSILON}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --log-dir "${LOG_DIR}" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --overwrite-experiment

    RESULT_DIRS+=("${LOG_DIR}/${EXPERIMENT_NAME}")
done

# --- aggregate: read test_metrics from every run and compute mean/std -------
python3 - "${RESULT_DIRS[@]}" <<'PYEOF'
import json
import statistics
import sys

dirs = sys.argv[1:]
aurocs = []
epsilons = []

for d in dirs:
    with open(f"{d}/test_metrics") as fh:
        metrics = json.load(fh)
    aurocs.append(metrics["AUROC"])

    with open(f"{d}/final_epsilon") as fh:
        epsilons.append(float(fh.read().strip()))

mean_auroc = statistics.mean(aurocs)
std_auroc = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
mean_eps = statistics.mean(epsilons)

print(f"\nRuns:          {len(aurocs)}")
print(f"AUROCs:        {[round(a, 4) for a in aurocs]}")
print(f"Mean AUROC:    {mean_auroc:.4f}")
print(f"Std AUROC:     {std_auroc:.4f}")
print(f"Mean actual epsilon spent: {mean_eps:.4f}")
PYEOF
