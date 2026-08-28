from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import (
    assert_files_exist,
    assert_config_and_hyperparams,
    assert_test_metrics,
    assert_metrics,
    base_env,
    run_distributed,
)


# T1: the headline invariant of parallel evaluation -- evaluating a fixed model
# on N ranks (each over a disjoint shard, then reduced collectively) must yield
# the same loss and metrics as evaluating the whole split on a single rank.
#
# We evaluate a seed-initialised, *untrained* model (--epochs 0) so the two runs
# start from bit-identical weights. Training is deliberately excluded: DDP does
# not synchronise BatchNorm and the train sampler shards differently, so a
# trained model would diverge across world sizes for reasons unrelated to the
# evaluation path under test. The 8-example test split shards 4/4 with no
# DistributedSampler tail padding, so the aggregate is exact.


def _train_cmd(dataset_path: Path, log_dir: Path, experiment_name: str) -> list[str]:
    return [
        'run.py',
        'train',
        '--device', 'cpu',
        '--dataset-name', 'local-image',
        '--dataset-path', str(dataset_path),
        '--model-name', 'resnet18',
        '--no-pretrained',
        '--no-privacy',
        '--epochs', '0',
        '--batch-size', '4',
        '--physical-batch-size', '4',
        '--num-workers', '0',
        '--seed', '42',
        '--split-seed', '42',
        '--log-dir', str(log_dir),
        '--experiment-name', experiment_name,
    ]


def _assert_value_close(single, parallel, abs_tol: float, path: str) -> None:
    # bool is a subclass of int, so match it before the numeric branch.
    if isinstance(single, bool):
        assert single == parallel, f'{path}: {single} != {parallel}'
    elif isinstance(single, (int, float)):
        assert isinstance(parallel, (int, float)), f'{path}: type mismatch'
        assert single == pytest.approx(parallel, rel=0, abs=abs_tol), (
            f'{path}: {single} != {parallel}'
        )
    elif isinstance(single, list):
        assert isinstance(parallel, list) and len(single) == len(parallel), (
            f'{path}: list length mismatch'
        )
        for i, (a, b) in enumerate(zip(single, parallel)):
            _assert_value_close(a, b, abs_tol, f'{path}[{i}]')
    else:
        assert single == parallel, f'{path}: {single} != {parallel}'


def _assert_parallel_metrics_close(single: dict, parallel: dict, abs_tol: float = 1e-5) -> None:
    assert single.keys() == parallel.keys(), (
        f'Metric keys differ: {sorted(single)} vs {sorted(parallel)}'
    )
    for key in single:
        _assert_value_close(single[key], parallel[key], abs_tol, key)


@pytest.mark.integration
def test_parallel_evaluation_matches_single_rank(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    run_distributed(
        _train_cmd(image_dataset_path, tmp_path, 'eval-single-rank'),
        env,
        repo_root,
        nproc=1,
    )
    run_distributed(
        _train_cmd(image_dataset_path, tmp_path, 'eval-two-rank'),
        env,
        repo_root,
        nproc=2,
    )

    for experiment in ('eval-single-rank', 'eval-two-rank'):
        assert_config_and_hyperparams(
            tmp_path / experiment,
            expected_config={'privacy': False, 'use_steps': False, 'seed': 42},
            expected_hyperparams={'epochs': 0, 'total_steps': None, 'batch_size': 4},
        )

    single = assert_test_metrics(
        tmp_path / 'eval-single-rank', expected_keys={'MulticlassAccuracy'}
    )
    parallel = assert_test_metrics(
        tmp_path / 'eval-two-rank', expected_keys={'MulticlassAccuracy'}
    )

    # Sharded evaluation on two ranks must reproduce the same result as a single rank,
    # compare loss and every metric
    _assert_parallel_metrics_close(single, parallel)


@pytest.mark.integration
def test_two_rank_checkpoint_validation_smoke(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment = 'parallel-checkpoint-smoke'

    cmd = [
        'run.py',
        'train',
        '--device', 'cpu',
        '--dataset-name', 'local-image',
        '--dataset-path', str(image_dataset_path),
        '--model-name', 'resnet18',
        '--no-pretrained',
        '--no-privacy',
        '--use-steps',
        '--total-steps', '3',
        '--batch-size', '4',
        '--physical-batch-size', '4',
        '--num-workers', '0',
        '--validation-frequency', '0',
        '--checkpoint-step-interval', '2',
        '--skip-test',
        '--seed', '42',
        '--split-seed', '42',
        '--log-dir', str(tmp_path),
        '--experiment-name', experiment,
    ]

    run_distributed(
        cmd,
        base_env(),
        repo_root,
        nproc=2,
        timeout=180,
    )

    experiment_dir = tmp_path / experiment
    checkpoint_dir = experiment_dir / 'checkpoints'

    assert_files_exist(
        checkpoint_dir,
        (
            'checkpoint_step_2.pt',
            'checkpoint_step_2_metrics.json',
            'final_checkpoint_step_3.pt',
            'final_checkpoint_step_3_metrics.json',
        ),
    )

    for filename in (
        'checkpoints/checkpoint_step_2_metrics.json',
        'checkpoints/final_checkpoint_step_3_metrics.json',
    ):
        assert_metrics(
            experiment_dir,
            filename=filename,
            expected_keys={'avg_train_loss_since_last_checkpoint'},
        )
