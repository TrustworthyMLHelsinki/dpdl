# Integration Testing Plan for DPDL

## Goals
- Cover the four primary execution paths: training with DP, training without DP, HPO with DP, HPO without DP.
- Add PEFT coverage (at least `head-only` and `film`) to validate these adapter paths.
- Validate prediction outputs from a saved model.
- Use deterministic “golden loss” checks on a tiny, fixed dataset/model so tests are stable and cheap.

## Non‑goals
- Exhaustive unit tests for every component.
- Performance benchmarking or accuracy targets on real datasets.
- GPU-only paths (these are separate, optional smoke tests already exist).

## Determinism Strategy
To make losses reproducible across runs and platforms:
- Use a small local dataset saved with `datasets.save_to_disk` and pass `--dataset-path <path>` (keep `--dataset-name` as a label only).
- Run on CPU: `--device cpu`.
- Fix all RNG seeds: `--seed 42` and `--split-seed 42` (or another constant). (Without --seed a random seed is drawn, so runs differ.)
- Disable non-determinism in tests when possible:
  - Set `torch.use_deterministic_algorithms(True)` inside tests (or use env vars if preferred).
  - Set `--num-workers 0` to avoid worker-level nondeterminism.
- Avoid network downloads: `--no-pretrained` and local datasets via `--dataset-path`.
- Keep steps tiny: `--use-steps --total-steps 2` (or 3) and small batch sizes.

Notes for DP runs:
- Keep Poisson sampling enabled; results should still be deterministic with fixed seeds. If they are not, that is a bug we want to catch.
- Use `--target-epsilon 8` (default) and avoid setting `--noise-multiplier` to prevent configuration conflicts. With fixed seeds, Poisson sampling and noise are deterministic, so golden losses are still meaningful.
- If you want explicit noise control, set `--noise-multiplier 0.1` and also set `--target-epsilon 0` to avoid the mutual‑exclusion validator, then compare loss with a small tolerance (e.g. `abs=1e-4`).

## Rationale for Common Flags
- `--device cpu`: avoids GPU hardware variability and makes CI/local runs consistent.
- `--dataset-path <path>`: removes network dependence and fixes dataset size/content.
- `--no-pretrained`: avoids downloading weights and makes initialization depend solely on the seed.
- `--seed 42`: fixes the seed; without it, each run draws (and records) a random one.
- `--split-seed 42`: fixes dataset sub‑sampling/splitting for determinism.
- `--num-workers 0`: avoids nondeterminism from dataloader workers.
- `--use-steps --total-steps 2`: bounds runtime and fixes the number of optimizer updates.
- `--batch-size 4 --physical-batch-size 4`: tiny batches for speed; no gradient accumulation required.
- `--log-dir <tmp>` + unique `--experiment-name`: isolates artifacts per test run.
- `--target-epsilon 8`: uses the default DP path while avoiding the noise‑multiplier/epsilon conflict.
- `--target-hypers learning_rate --n-trials 1` (HPO): single‑parameter, single‑trial to keep runtime deterministic and short.
- `--optuna-sampler RandomSampler` + fixed seed: simplest deterministic sampler for tiny tests.

## Test Fixtures
1) **Optuna config fixture** (for HPO tests)
- Add a minimal config (e.g. `tests/fixtures/optuna_hypers_small.conf`) with very small search space:
  - `learning_rate`: small log range, e.g. `[1e-4, 1e-3]`
  - `batch_size`: small categorical `[4, 8]`
  - `epochs`: fixed `1` (or omit from targets)
- This keeps trials deterministic and fast.
 - For full-batch mapping tests, use `tests/fixtures/optuna_hypers_full_batch.conf` to force the `-1` sentinel.

2) **Manual trials fixture** (optional, for fully deterministic HPO)
- Add `tests/fixtures/optuna_manual_trials.yaml` with a single trial so Optuna picks exactly one configuration.
- Run with `--n-trials 1 --optuna-manual-trials <file>` to avoid randomness entirely.

3) **Expected losses fixture**
- Store per-test golden losses in a JSON file, e.g. `tests/fixtures/expected_losses.json`:
  ```json
  {
    "train_non_dp": 1.234567,
    "train_dp": 1.345678,
    "peft_head_only": 1.456789,
    "peft_film": 1.567890,
    "hpo_non_dp_trial0": 1.111111,
    "hpo_dp_trial0": 1.222222,
    "predict_metrics_acc": 0.5
  }
  ```
- Populate these once by running the tests locally and recording the observed values.

4) **Expected HPO params fixture**
- Store expected best hyperparameters from HPO runs in `tests/fixtures/expected_hpo_params.json`, e.g.:
  ```json
  {
    "hpo_non_dp_trial0": {
      "learning_rate": 0.0005
    },
    "hpo_dp_trial0": {
      "learning_rate": 0.0005
    }
  }
  ```
- Populate after the first successful HPO run.

## Proposed Integration Tests
All tests should use the same harness pattern as the current smoke tests (run `run.py` via `torch.distributed.run` with `--nproc_per_node=1`), and write logs to `tmp_path`.

## Implementation Status
- [x] T1 — Train (Non‑DP) Deterministic Loss (tests/test_integration_train_non_dp.py)
- [x] T2 — Train (DP) Deterministic Loss (tests/test_integration_train_dp.py)
- [x] T2b — DP Epsilon Sanity Check (tests/test_integration_dp_epsilon_sanity.py)
- [x] T3 — PEFT (Head‑Only) (tests/test_integration_peft_head_only.py)
- [x] T4 — PEFT (FiLM) (tests/test_integration_peft_film.py)
- [x] T5 — Prediction (Saved Model) (tests/test_integration_train_predict.py)
- [x] T6 — HPO (Non‑DP) (tests/test_integration_hpo_non_dp.py)
- [x] T7 — HPO (DP) (tests/test_integration_hpo_dp.py)
- [x] T6b — HPO Full Batch Size Mapping (tests/test_integration_hpo_full_batch_size.py)

### T1 — Train (Non‑DP) Deterministic Loss
**Command (example):**
- `python -m torch.distributed.run --standalone --nproc_per_node=1 run.py train`
- `--device cpu --dataset-name local-image --dataset-path <path> --model-name resnet18 --no-pretrained`
- `--no-privacy --use-steps --total-steps 2`
- `--batch-size 4 --physical-batch-size 4 --num-workers 0`
- `--seed 42 --split-seed 42 --log-dir <tmp> --experiment-name train-non-dp`

**Assertions:**
- `<log_dir>/<exp>/test_metrics` exists and contains `loss`.
- `loss == expected_losses["train_non_dp"]` (exact or tight tolerance).
- `runtime` file exists.

### T2 — Train (DP) Deterministic Loss
**Command:** same as T1 but with DP enabled and fixed DP hyperparameters:
- `--privacy`
- `--target-epsilon 8`
- `--max-grad-norm 1.0`
- `--model-name vit_tiny_patch16_224.augreg_in21k` (avoid BatchNorm for Opacus)

**Assertions:**
- `<log_dir>/<exp>/test_metrics` exists with `loss`.
- `loss == expected_losses["train_dp"]` (exact or tolerance if small noise).

### T3 — PEFT (Head‑Only)
**Command:** same as T1, plus:
- `--peft head-only`

**Assertions:**
- `test_metrics` exists with deterministic loss equal to `expected_losses["peft_head_only"]`.
- (Optional) parse logs to confirm `Finetuning head only` message exists.

### T4 — PEFT (FiLM)
**Model selection:** pick a small model supported by FiLM config:
- Preferred: `vit_tiny_patch16_224` (small and FiLM-supported).
- Alternative: `resnetv2_50x1_bit.goog_in21k` (supported but heavier).

**Command:** same as T1, plus:
- `--model-name vit_tiny_patch16_224`
- `--peft film`

**Assertions:**
- `test_metrics` exists with deterministic loss equal to `expected_losses["peft_film"]`.
- (Optional) parse logs for `FiLM setup done`.

### T5 — Prediction (Saved Model)
Use the built‑in `train-predict` path so saving and prediction are covered together.

**Command:**
- `python -m torch.distributed.run --standalone --nproc_per_node=1 run.py train-predict`
- `--device cpu --dataset-name local-image --dataset-path <path> --model-name resnet18 --no-pretrained`
- `--no-privacy --use-steps --total-steps 2`
- `--batch-size 4 --physical-batch-size 4 --num-workers 0`
- `--predict-dataset-split test`
- `--seed 42 --split-seed 42 --log-dir <tmp> --experiment-name train-predict`

**Assertions:**
- `predictions_test.json` exists.
- JSON list length equals test split size (8 in the local fixture).
- `predict_metrics.json` exists.

### T6 — HPO (Non‑DP)
**Command:**
- `python -m torch.distributed.run --standalone --nproc_per_node=1 run.py optimize`
- `--device cpu --dataset-name local-image --dataset-path <path> --model-name resnet18 --no-pretrained`
- `--no-privacy --use-steps --total-steps 2`
- `--target-hypers learning_rate --n-trials 1`
- `--optuna-config tests/fixtures/optuna_hypers_small.conf`
- `--optuna-sampler RandomSampler` (or `TPESampler`)
- `--seed 42 --split-seed 42 --log-dir <tmp> --experiment-name hpo-non-dp`

**Assertions:**
- `hpo_metrics.json` exists and has one entry.
- `best-value`, `best-params.json`, `final-metrics` exist.
- First trial loss equals `expected_losses["hpo_non_dp_trial0"]`.

### T7 — HPO (DP)
**Command:** same as T6 plus DP flags:
- `--privacy`
- `--target-epsilon 8`
- `--max-grad-norm 1.0`

**Assertions:**
- Same as T6, loss equals `expected_losses["hpo_dp_trial0"]`.

### T6b — HPO Full Batch Size Mapping
**Goal:** Ensure Optuna’s `batch_size = -1` maps to the dataset size.

**Command:** run HPO with a config that forces ordered `batch_size` to only include `-1`:
- `--target-hypers batch_size`
- `--optuna-config tests/fixtures/optuna_hypers_full_batch.conf`
- `--n-trials 1`

**Assertions:**
- `best-params.json` has `batch_size == 20` (train size of the local dataset fixture).

## Implementation Notes
- Create a small test helper in `tests/` to:
  - Build command lists.
  - Run `subprocess.run` and capture stderr on failure.
  - Read JSON artifacts and return `loss` values.
- For local/CI runs, use `python -m torch.distributed.run --standalone --nproc_per_node=1` so `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` are set. The Slurm `run_wrapper.sh` is only needed in cluster jobs.
- Use `bin/run-tests.sh` for local runs; it checks for an active venv or uses `DPDL_VENV` to activate one.
- Mark new tests with `@pytest.mark.integration` so they can be excluded from quick local runs, while still available in CI for JOSS.
- Keep each test under ~5–10 seconds on CPU by using small steps and the local dataset fixture.

## Updating Golden Losses
1) Run the integration tests locally on CPU.
2) Capture `loss` values from `test_metrics` (or the first entry in `hpo_metrics.json`).
3) Write them to `tests/fixtures/expected_losses.json`.
4) If dependencies change and values drift, update this file intentionally and document the change in the PR.

## Optional Extensions (later)
- Add a GPU integration test variant (gated by marker and CUDA availability).
- Add LoRA path (if needed) with a tiny model that supports it.
- Add multi‑process DDP test (`--nproc_per_node=2`) if CI can support it.
### T2b — DP Epsilon Sanity Check
**Goal:** Verify that stronger privacy (lower epsilon) does not produce better loss than weaker privacy on the same deterministic setup.

**Command:** Run the DP training twice with identical settings except `--target-epsilon`:
- `--target-epsilon 8` (higher epsilon, weaker privacy)
- `--target-epsilon 2` (lower epsilon, stronger privacy)

**Assertion:**
- `loss(eps=2) >= loss(eps=8)` (with a tiny tolerance).
