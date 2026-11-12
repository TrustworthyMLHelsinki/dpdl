"""Skipping final evaluation must retain training and run finalization."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.main import get_command

import dpdl.cli as cli_module
from dpdl.configurationmanager import ConfigurationManager
from integration_utils import assert_runtime, base_env, run_distributed


@pytest.mark.parametrize(
    "args,expected", [([], False), (["--skip-test"], True), (["--no-skip-test"], False)]
)
def test_skip_test_cli_and_configuration(args, expected):
    app = typer.Typer()
    app.command()(cli_module.cli)
    ctx = get_command(app).make_context(
        "run.py", ["train", "--no-privacy", "--epochs", "1", *args]
    )
    manager = ConfigurationManager(ctx.params)
    assert manager.configuration.skip_test is expected
    assert manager.configuration.model_dump()["skip_test"] is expected


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_run_train_retains_finalization_with_or_without_test(monkeypatch, skip, rank):
    manager = ConfigurationManager(
        {
            "command": "train",
            "privacy": False,
            "epochs": 1,
            "skip_test": skip,
            "noise_multiplier": None,
            "max_grad_norm": 1.0,
            "target_epsilon": None,
            "noise_batch_ratio": None,
        }
    )
    trainer = Mock()
    trainer.test.return_value = (0.25, {"MulticlassAccuracy": 0.9})
    monkeypatch.setattr(cli_module.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(cli_module, "seed_everything", Mock())
    monkeypatch.setattr(cli_module.TrainerFactory, "get_trainer", lambda _: trainer)
    runtime, metrics, epsilon = Mock(), Mock(), Mock()
    monkeypatch.setattr(cli_module, "log_runtime", runtime)
    monkeypatch.setattr(cli_module, "log_test_metrics", metrics)
    monkeypatch.setattr(cli_module, "log_final_epsilon", epsilon)

    assert cli_module.run_train(manager) is None

    trainer.fit.assert_called_once_with()
    assert trainer.test.call_count == metrics.call_count == (rank == 0 and not skip)
    assert runtime.call_count == (rank == 0)
    # This branch resolves final epsilon on every rank, including when skipping test.
    epsilon.assert_called_once_with(manager, trainer)


@pytest.mark.integration
def test_skip_test_preserves_runtime_and_resource_outputs(tmp_path, image_dataset_path):
    repo_root = Path(__file__).resolve().parents[1]
    run_distributed(
        [
            "run.py",
            "train",
            "--device",
            "cpu",
            "--dataset-name",
            "local-image",
            "--dataset-path",
            str(image_dataset_path),
            "--model-name",
            "resnet18",
            "--no-pretrained",
            "--no-privacy",
            "--use-steps",
            "--total-steps",
            "2",
            "--batch-size",
            "4",
            "--physical-batch-size",
            "4",
            "--num-workers",
            "0",
            "--validation-frequency",
            "0",
            "--skip-test",
            "--record-memory-usage",
            "--record-training-time",
            "--log-dir",
            str(tmp_path),
            "--experiment-name",
            "skip-test",
        ],
        base_env(),
        repo_root,
    )
    output = tmp_path / "skip-test"
    assert_runtime(output)
    assert json.loads((output / "configuration.json").read_text())["skip_test"] is True
    assert not (output / "test_metrics").exists()
    summary = json.loads((output / "resource_summary.json").read_text())
    assert summary["first_epoch"]["elapsed_seconds"] > 0
    rows = [
        json.loads(line)
        for line in (output / "resource_telemetry.jsonl").read_text().splitlines()
    ]
    assert sum(row["logical_steps"] for row in rows) == 2
