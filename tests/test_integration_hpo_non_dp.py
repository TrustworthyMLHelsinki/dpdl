from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import (
    assert_config_and_hyperparams,
    assert_files_exist,
    assert_best_params,
    assert_hpo_metrics,
    assert_runtime,
    base_env,
    get_expected_hpo_params,
    get_expected_loss,
    run_distributed,
)


@pytest.mark.integration
def test_integration_hpo_non_dp(tmp_path: Path, image_dataset_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = base_env()

    cmd_args = [
        'run.py',
        'optimize',
        '--device',
        'cpu',
        '--dataset-name',
        'local-image',
        '--dataset-path',
        str(image_dataset_path),
        '--model-name',
        'resnet18',
        '--no-pretrained',
        '--no-privacy',
        '--use-steps',
        '--total-steps',
        '2',
        '--batch-size',
        '4',
        '--physical-batch-size',
        '4',
        '--num-workers',
        '0',
        '--seed',
        '42',
        '--split-seed',
        '42',
        '--target-hypers',
        'learning_rate',
        '--n-trials',
        '1',
        '--optuna-config',
        'tests/fixtures/optuna_hypers_small.conf',
        '--optuna-sampler',
        'RandomSampler',
        '--log-dir',
        str(tmp_path),
        '--experiment-name',
        'hpo-non-dp',
        '--optuna-journal',
        str(tmp_path / 'optuna-non-dp.journal'),
        '--save-model',
    ]

    run_distributed(cmd_args, env, repo_root)

    assert_config_and_hyperparams(
        tmp_path / 'hpo-non-dp',
        expected_config={
            'command': 'optimize',
            'device': 'cpu',
            'dataset_name': 'local-image',
            'dataset_path': str(image_dataset_path),
            'model_name': 'resnet18',
            'privacy': False,
            'use_steps': True,
            'target_hypers': ['learning_rate'],
            'n_trials': 1,
            'optuna_config': 'tests/fixtures/optuna_hypers_small.conf',
            'optuna_sampler': 'RandomSampler',
            'optuna_journal': str(tmp_path / 'optuna-non-dp.journal'),
            'log_dir': str(tmp_path),
            'experiment_name': 'hpo-non-dp',
            'seed': 42,
            'split_seed': 42,
        },
        expected_hyperparams={
            'epochs': None,
            'total_steps': 2,
            'batch_size': 4,
        },
    )

    expected_loss = get_expected_loss('hpo_non_dp_trial0')
    assert_hpo_metrics(tmp_path / 'hpo-non-dp', expected_loss=expected_loss)

    expected_params = get_expected_hpo_params('hpo_non_dp_trial0')
    assert_best_params(tmp_path / 'hpo-non-dp', expected_params=expected_params)
    assert_files_exist(
        tmp_path / 'hpo-non-dp',
        ('best-value', 'best-params.json', 'final-metrics', 'final_model.pt'),
    )

    assert_runtime(tmp_path / 'hpo-non-dp')
