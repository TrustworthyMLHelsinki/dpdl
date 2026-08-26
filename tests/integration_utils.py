import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_command(
    cmd: list[str],
    env: dict,
    cwd: Path,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f'Command timed out after {timeout}s:\n'
            f'{" ".join(cmd)}\n'
            f'{exc.stderr or ""}'
        )

    assert result.returncode == 0, result.stderr
    return result


def run_distributed(
    cmd_args: list[str],
    env: dict,
    cwd: Path,
    nproc: int = 1,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        f'--nproc_per_node={nproc}',
        *cmd_args,
    ]
    return run_command(cmd, env, cwd, timeout=timeout)


def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_experiment_config(log_dir: Path) -> tuple[dict, dict]:
    config_path = log_dir / 'configuration.json'
    hyperparams_path = log_dir / 'hyperparameters.json'
    assert config_path.exists(), f'Expected configuration.json at {config_path}'
    assert hyperparams_path.exists(), f'Expected hyperparameters.json at {hyperparams_path}'
    return load_json(config_path), load_json(hyperparams_path)


def _assert_expected(actual: dict, expected: dict, label: str) -> None:
    for key, expected_value in expected.items():
        assert key in actual, f'Missing {label} key: {key}'
        actual_value = actual[key]
        if isinstance(expected_value, float):
            assert actual_value == pytest.approx(expected_value, rel=0, abs=1e-8), (
                f'Unexpected {label} value for "{key}": {actual_value}'
            )
        else:
            assert actual_value == expected_value, (
                f'Unexpected {label} value for "{key}": {actual_value}'
            )


def assert_config_and_hyperparams(
    log_dir: Path,
    expected_config: dict | None = None,
    expected_hyperparams: dict | None = None,
) -> None:
    config, hyperparams = load_experiment_config(log_dir)

    if expected_config:
        _assert_expected(config, expected_config, 'configuration')

    if expected_hyperparams:
        _assert_expected(hyperparams, expected_hyperparams, 'hyperparameters')


def _iter_numbers(value):
    if isinstance(value, (int, float)):
        yield value
        return

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_numbers(item)
        return

    if isinstance(value, list):
        for item in value:
            yield from _iter_numbers(item)
        return


def assert_metrics(
    log_dir: Path,
    *,
    expected_keys: set[str] | None = None,
    filename: str = 'test_metrics',
    require_loss: bool = True,
    ensure_finite: bool = True,
) -> dict:
    metrics_path = log_dir / filename
    assert metrics_path.exists(), f'Expected metrics file at {metrics_path}'
    metrics = load_json(metrics_path)

    if require_loss:
        assert 'loss' in metrics, f'Missing loss in {metrics_path}'

    if expected_keys:
        for key in expected_keys:
            assert key in metrics, f'Missing metric "{key}" in {metrics_path}'

    if ensure_finite:
        for value in _iter_numbers(metrics):
            assert math.isfinite(value), 'Non-finite metric value found.'

    return metrics


def assert_test_metrics(
    log_dir: Path,
    *,
    expected_keys: set[str] | None = None,
    require_loss: bool = True,
    ensure_finite: bool = True,
) -> dict:
    return assert_metrics(
        log_dir,
        expected_keys=expected_keys,
        filename='test_metrics',
        require_loss=require_loss,
        ensure_finite=ensure_finite,
    )


def assert_predict_metrics(
    log_dir: Path,
    *,
    expected_keys: set[str] | None = None,
    ensure_finite: bool = True,
) -> dict:
    return assert_metrics(
        log_dir,
        expected_keys=expected_keys,
        filename='predict_metrics.json',
        require_loss=False,
        ensure_finite=ensure_finite,
    )


def assert_hpo_metrics(
    log_dir: Path,
    *,
    expected_loss: float | None = None,
    abs_tol: float = 1e-6,
    ensure_finite: bool = True,
) -> list[dict]:
    metrics_path = log_dir / 'hpo_metrics.json'
    assert metrics_path.exists(), f'Expected hpo_metrics.json at {metrics_path}'
    metrics = load_json(metrics_path)
    assert isinstance(metrics, list) and metrics, 'Expected non-empty hpo_metrics list.'
    assert 'loss' in metrics[0], 'Expected loss in first hpo_metrics entry.'

    if ensure_finite:
        for entry in metrics:
            for value in _iter_numbers(entry):
                assert math.isfinite(value), 'Non-finite HPO metric value found.'

    if expected_loss is not None:
        assert metrics[0]['loss'] == pytest.approx(expected_loss, rel=0, abs=abs_tol)

    return metrics


def assert_files_exist(log_dir: Path, filenames: tuple[str, ...]) -> None:
    for filename in filenames:
        path = log_dir / filename
        assert path.exists(), f'Expected {filename} at {path}'


def assert_best_params(
    log_dir: Path,
    *,
    expected_params: dict | None = None,
    filename: str = 'best-params.json',
    abs_tol: float = 1e-8,
) -> dict:
    path = log_dir / filename
    assert path.exists(), f'Expected {filename} at {path}'
    best_params = load_json(path)

    if expected_params:
        for key, expected_value in expected_params.items():
            assert key in best_params, f'Missing expected hyperparameter: {key}'
            if isinstance(expected_value, float):
                assert best_params[key] == pytest.approx(expected_value, rel=0, abs=abs_tol)
            else:
                assert best_params[key] == expected_value

    return best_params


def assert_predictions(
    log_dir: Path,
    *,
    split: str,
    expected_len: int | None = None,
) -> list:
    path = log_dir / f'predictions_{split}.json'
    assert path.exists(), f'Expected predictions file at {path}'
    preds = load_json(path)
    assert isinstance(preds, list), 'Expected predictions JSON to be a list.'
    if expected_len is not None:
        assert len(preds) == expected_len, f'Expected {expected_len} predictions.'
    return preds


def assert_runtime(log_dir: Path) -> None:
    runtime_path = log_dir / 'runtime'
    assert runtime_path.exists(), f'Expected runtime file at {runtime_path}'


def assert_final_epsilon(log_dir: Path) -> None:
    epsilon_path = log_dir / 'final_epsilon'
    assert epsilon_path.exists(), f'Expected final epsilon file at {epsilon_path}'


def assert_final_noise_multiplier(log_dir: Path) -> float:
    noise_path = log_dir / 'noise_multiplier'
    assert noise_path.exists(), f'Expected noise multiplier file at {noise_path}'
    noise_multiplier = float(noise_path.read_text().strip())

    # A target-epsilon run must add noise
    assert noise_multiplier > 0, f'Expected positive noise multiplier, got {noise_multiplier}'
    return noise_multiplier


def load_expected_losses() -> dict:
    fixture_path = Path(__file__).parent / 'fixtures' / 'expected_losses.json'
    if not fixture_path.exists():
        pytest.skip('Missing expected losses fixture file.')
    data = load_json(fixture_path)
    if not isinstance(data, dict):
        pytest.skip('Expected losses fixture file is not a JSON object.')
    return data


def get_expected_loss(key: str) -> float:
    data = load_expected_losses()
    if key not in data:
        pytest.skip(f'Missing expected loss for key: {key}')
    return data[key]


def load_expected_hpo_params() -> dict:
    fixture_path = Path(__file__).parent / 'fixtures' / 'expected_hpo_params.json'
    if not fixture_path.exists():
        pytest.skip('Missing expected HPO params fixture file.')
    data = load_json(fixture_path)
    if not isinstance(data, dict):
        pytest.skip('Expected HPO params fixture file is not a JSON object.')
    return data


def get_expected_hpo_params(key: str) -> dict:
    data = load_expected_hpo_params()
    if key not in data:
        pytest.skip(f'Missing expected HPO params for key: {key}')
    params = data[key]
    if not isinstance(params, dict):
        pytest.skip(f'Expected HPO params for key {key} is not a JSON object.')
    return params


def base_env() -> dict:
    env = os.environ.copy()
    env['_TYPER_STANDARD_TRACEBACK'] = '1'
    return env
