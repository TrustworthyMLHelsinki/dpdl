#!/usr/bin/env bash
#
# sweep_epsilon.sh — for each value of epsilon, run several repeats with
# different seeds and report mean/std test AUROC per epsilon.
#
# Usage: ./tutorials/scripts/sweep_epsilon.sh
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
DEVICE="auto"
LOG_DIR="logs/tutorial_sweep"

EPSILONS=(1 3 5 8)            # privacy budgets to sweep over
SEEDS=(0 1 2 3 4)            # repeats per epsilon value (same seeds reused
                              # for every epsilon, for an apples-to-apples
                              # comparison of initialization/data order)

# Where we collect every (epsilon, seed) result directory, so we can
# aggregate them in one pass at the end.
SUMMARY_FILE="$(mktemp)"
trap 'rm -f "${SUMMARY_FILE}"' EXIT

for EPSILON in "${EPSILONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXPERIMENT_NAME="eyepacs_vit_tiny_eps${EPSILON}_seed${SEED}"

        echo "=== epsilon=${EPSILON}, seed=${SEED} ==="

        dpdl train \
            --dataset-name "${DATASET_NAME}" \
            --dataset-label-field "${LABEL_FIELD}" \
            --model-name "${MODEL_NAME}" \
            --metric-config "${METRIC_CONF}" \
            --epochs "${EPOCHS}" \
            --batch-size "${BATCH_SIZE}" \
            --physical-batch-size "${PHYSICAL_BATCH_SIZE}" \
            --learning-rate "${LEARNING_RATE}" \
            --target-epsilon "${EPSILON}" \
            --seed "${SEED}" \
            --device "${DEVICE}" \
            --log-dir "${LOG_DIR}" \
            --experiment-name "${EXPERIMENT_NAME}" \
            --overwrite-experiment

        # Record which directory belongs to which requested epsilon, so the
        # aggregation step below can group results correctly.
        echo "${EPSILON} ${LOG_DIR}/${EXPERIMENT_NAME}" >> "${SUMMARY_FILE}"
    done
done

# --- aggregate: group runs by requested epsilon, compute mean/std per group -
python3 - "${SUMMARY_FILE}" <<'PYEOF'
import json
import statistics
import sys
from collections import defaultdict

summary_file = sys.argv[1]

# Map requested epsilon -> list of (auroc, actual_epsilon) tuples.
by_epsilon = defaultdict(list)

with open(summary_file) as fh:
    for line in fh:
        target_epsilon_str, result_dir = line.split(maxsplit=1)
        result_dir = result_dir.strip()

        with open(f"{result_dir}/test_metrics") as mfh:
            auroc = json.load(mfh)["AUROC"]

        with open(f"{result_dir}/final_epsilon") as efh:
            actual_epsilon = float(efh.read().strip())

        by_epsilon[target_epsilon_str].append((auroc, actual_epsilon))

print(f"{'target eps':>10} | {'n':>2} | {'mean auroc':>10} | {'std auroc':>9} | {'mean actual eps':>15}")
print("-" * 62)
for target_epsilon_str in sorted(by_epsilon, key=float):
    runs = by_epsilon[target_epsilon_str]
    aurocs = [a for a, _ in runs]
    actual_epsilons = [e for _, e in runs]

    mean_auroc = statistics.mean(aurocs)
    std_auroc = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0
    mean_eps = statistics.mean(actual_epsilons)

    print(f"{target_epsilon_str:>10} | {len(runs):>2} | {mean_auroc:>10.4f} | {std_auroc:>9.4f} | {mean_eps:>15.4f}")
PYEOF
