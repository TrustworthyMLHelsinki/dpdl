"""Test-first epoch resource contracts, with deterministic clocks/allocator reads.

The proposed resource module is resolved at test execution, so missing behavior
is an assertion failure rather than a test-collection/import error. Field names
below define the consumer-facing JSON contract; helper names are test seams.
"""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dpdl.callbacks.callback_factory import CallbackHandler
from dpdl.callbacks.base_callback import Callback
from dpdl.trainer import DifferentiallyPrivateTrainer, Trainer
from opacus.optimizers import DPOptimizer


def resources():
    assert (
        importlib.util.find_spec("dpdl.callbacks.resource_stats") is not None
    ), "DPDL must provide training resource recording and aggregation"
    return importlib.import_module("dpdl.callbacks.resource_stats")


def rank_segments(rank, durations, steps=4, allocated=100, reserved=200):
    return {
        "rank": rank,
        "segments": [
            {
                "elapsed_seconds": d,
                "logical_steps": steps // len(durations),
                "physical_batches": 6,
                "examples": 8,
                "start_allocated_bytes": 10,
                "start_reserved_bytes": 20,
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved,
            }
            for d in durations
        ],
    }


def epoch(number, durations, steps, allocated=100):
    return resources().summarize_epoch(
        [
            rank_segments(rank, values, steps, allocated)
            for rank, values in enumerate(durations)
        ],
        epoch=number,
        complete=True,
    )


def test_segment_maxima_are_summed_before_epoch_step_mean():
    row = epoch(0, [[9.0, 1.0], [1.0, 9.0]], 4)
    assert row["elapsed_seconds"] == 18.0
    assert row["logical_steps"] == 4
    assert row["mean_step_seconds"] == 4.5
    assert row["examples"] == 32  # Sum examples across ranks/segments.
    assert len(row["per_rank"]) == 2


def test_later_epoch_means_are_weighted_and_memory_uses_maximum():
    rows = [
        epoch(0, [[100.0]], 2),
        epoch(1, [[8.0]], 4, 150),
        epoch(2, [[12.0]], 2, 300),
    ]
    summary = resources().summarize_run(rows)
    assert summary["first_epoch"]["elapsed_seconds"] == 100.0
    assert summary["subsequent_epochs"]["logical_steps"] == 6
    assert summary["subsequent_epochs"]["mean_step_seconds"] == pytest.approx(20 / 6)
    assert summary["subsequent_epochs"]["peak_allocated_bytes"] == 300


def test_later_summary_sums_epoch_maxima_not_rank_totals():
    rows = [
        epoch(0, [[1.0], [1.0]], 2),
        epoch(1, [[9.0], [1.0]], 2),
        epoch(2, [[1.0], [9.0]], 2),
    ]
    assert (
        resources().summarize_run(rows)["subsequent_epochs"]["elapsed_seconds"] == 18.0
    )


def test_disagreeing_completed_update_counts_rejected():
    with pytest.raises(ValueError, match="step|count"):
        resources().summarize_epoch(
            [rank_segments(0, [1.0], 2), rank_segments(1, [1.0], 3)],
            epoch=0,
            complete=True,
        )


def test_no_later_epochs_and_zero_steps_are_explicitly_unavailable():
    row = epoch(0, [[0.0]], 0)
    assert row["mean_step_seconds"] is None
    assert resources().summarize_run([row])["subsequent_epochs"] is None


def test_partial_epoch_uses_actual_counts():
    row = resources().summarize_epoch(
        [rank_segments(0, [6.0], 3)], epoch=2, complete=False
    )
    assert row["complete"] is False
    assert row["logical_steps"] == 3
    assert row["mean_step_seconds"] == 2.0


def test_cpu_metrics_are_null_with_reason():
    row = resources().summarize_epoch(
        [rank_segments(0, [1.0], 2, None, None)], epoch=0, complete=True
    )
    assert row["peak_allocated_bytes"] is None
    assert row["peak_reserved_bytes"] is None
    assert row["memory_unavailable_reason"]


def test_recorder_excludes_validation_resets_peaks_and_writes_artifacts(
    tmp_path, monkeypatch
):
    module = resources()
    now = [0.0]
    memory = {
        "allocated": 10,
        "reserved": 30,
        "peak_allocated": 10,
        "peak_reserved": 30,
    }
    events = []

    def reset(device):
        events.append("reset")
        memory["peak_allocated"] = memory["allocated"]
        memory["peak_reserved"] = memory["reserved"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    for api, key in [
        ("memory_allocated", "allocated"),
        ("memory_reserved", "reserved"),
        ("max_memory_allocated", "peak_allocated"),
        ("max_memory_reserved", "peak_reserved"),
    ]:
        monkeypatch.setattr(torch.cuda, api, lambda device, key=key: memory[key])
    monkeypatch.setattr(
        torch.cuda,
        "empty_cache",
        lambda: pytest.fail("recording must not purge cached memory"),
    )
    recorder = module.RecordResourceStatsCallback(
        device=torch.device("cuda:0"), output_dir=tmp_path, clock=lambda: now[0]
    )
    recorder.start_epoch(0)
    now[0] += 2.0
    memory["peak_allocated"] = 100
    recorder.record_step(examples=4, physical_batches=2)
    recorder.end_epoch(complete=True)
    # Expensive validation, outside the training intervals.
    now[0] += 50.0
    memory.update(peak_allocated=9000, reserved=9000, peak_reserved=9000)
    recorder.start_epoch(1)
    now[0] += 3.0
    memory["peak_allocated"] = 120
    recorder.record_step(examples=4, physical_batches=2)
    recorder.end_epoch(complete=False)
    recorder.finish()
    import json

    rows = [
        json.loads(line)
        for line in (tmp_path / "resource_telemetry.jsonl").read_text().splitlines()
    ]
    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    assert [r["elapsed_seconds"] for r in rows] == [2.0, 3.0]
    assert [r["logical_steps"] for r in rows] == [1, 1]
    assert rows[1]["peak_allocated_bytes"] == 120
    assert rows[1]["peak_reserved_bytes"] == 9000  # Cache carryover is real.
    assert summary["subsequent_epochs"]["mean_step_seconds"] == 3.0
    assert events.count("reset") == 2
    assert events.count("sync") == 4  # Boundaries, never per-step.


def test_disabled_recorder_has_no_clock_device_collectives_or_output(
    tmp_path, monkeypatch
):
    module = resources()

    def forbidden(*args, **kwargs):
        pytest.fail("disabled instrumentation performed a resource operation")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.distributed, "all_gather_object", forbidden)
    recorder = module.RecordResourceStatsCallback(
        device=torch.device("cpu"), output_dir=tmp_path, clock=forbidden, enabled=False
    )
    recorder.start_epoch(0)
    recorder.record_step(examples=4, physical_batches=2)
    recorder.end_epoch(complete=True)
    recorder.finish()
    assert list(tmp_path.iterdir()) == []


def test_resource_counts_updates_without_changing_existing_batch_callbacks(tmp_path):
    # Preserve the old callback schedule, but measure actual optimizer updates.
    opt = DPOptimizer(
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=4,
    )
    actual_updates, reported = [], []
    opt.original_optimizer.register_step_post_hook(
        lambda *args: actual_updates.append(1)
    )
    trainer = object.__new__(DifferentiallyPrivateTrainer)
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = opt
    loader = DataLoader(TensorDataset(torch.arange(10), torch.zeros(10)), batch_size=4)
    trainer.datamodule = SimpleNamespace(get_dataloader=lambda split: loader)
    trainer.physical_batch_size = 3
    trainer.total_steps = 0
    recorder = resources().RecordResourceStatsCallback(
        device="cpu", output_dir=tmp_path
    )

    class ExistingBatchCallback(Callback):
        def on_train_batch_end(self, *args):
            reported.append(1)

    trainer.callback_handler = CallbackHandler([recorder, ExistingBatchCallback()])
    trainer._unwrap_model = lambda: SimpleNamespace(
        train_metrics=SimpleNamespace(compute=lambda: {}, reset=lambda: None)
    )

    def fit_physical(index, batch):
        opt.zero_grad()
        opt.params[0].grad_sample = torch.zeros(len(batch[0]), 2)
        opt.step()
        return 0.0

    trainer.fit_one_batch = fit_physical
    trainer.fit_one_epoch(0)
    assert len(actual_updates) == 3
    # With no worker prefetch, the existing post-step queue peek emits one
    # batch-end event for each of the five physical batches. Keep that behavior
    # out of scope for telemetry so older callbacks retain their schedule.
    assert len(reported) == 5
    assert recorder.rows[0]["physical_batches"] == 5
    assert recorder.rows[0]["logical_steps"] == len(actual_updates)


def test_resource_and_history_options_survive_configuration_validation():
    from dpdl.configurationmanager import Configuration

    fields = Configuration.model_fields
    assert "record_memory_usage" in fields
    assert "record_training_time" in fields
    assert (
        "noise_history_mode" in fields
    ), "configuration drops inverse-history selection"


@pytest.mark.parametrize("total_steps", [0, 4])
def test_real_training_loops_emit_resource_epochs(tmp_path, monkeypatch, total_steps):
    """Use real BMM/optimizer updates, not a synthetic resource-only loop."""
    import json
    from dpdl.callbacks.resource_stats import RecordResourceStatsCallback

    now = [0.0]
    trainer = object.__new__(DifferentiallyPrivateTrainer)
    trainer.model = torch.nn.Linear(1, 1)
    opt = DPOptimizer(
        optimizer=torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=4,
    )
    trainer.optimizer = opt
    loader = DataLoader(TensorDataset(torch.arange(10), torch.zeros(10)), batch_size=4)
    trainer.datamodule = SimpleNamespace(get_dataloader=lambda split: loader)
    trainer.physical_batch_size = 3
    trainer.callback_handler = SimpleNamespace(call=lambda *args: None)

    def compute_metrics():
        now[0] += 3.0
        return {}

    trainer._unwrap_model = lambda: SimpleNamespace(
        train_metrics=SimpleNamespace(compute=compute_metrics, reset=lambda: None)
    )
    trainer._calculate_steps_per_epoch = lambda: 3
    trainer.total_steps, trainer.epochs, trainer.validation_frequency = (
        total_steps,
        0 if total_steps else 2,
        1,
    )
    recorder = RecordResourceStatsCallback(
        device="cpu", output_dir=tmp_path, clock=lambda: now[0]
    )

    class EpochReporting(Callback):
        def on_train_epoch_start(self, trainer, epoch):
            now[0] += 100.0

        def on_train_epoch_end(self, trainer, epoch, metrics):
            now[0] += 1000.0

    trainer.callback_handler = CallbackHandler([recorder, EpochReporting()])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)

    def validate(*args):
        now[0] += 100.0

    trainer.validate = validate

    def fit(index, batch):
        opt.zero_grad()
        opt.params[0].grad_sample = torch.zeros(len(batch[0]), 2)
        opt.step()
        now[0] += 1.0
        return 0.0

    trainer.fit_one_batch = fit
    trainer.fit()
    path = tmp_path / "resource_telemetry.jsonl"
    assert path.exists(), "real training loops failed to emit resource records"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["logical_steps"] for r in rows] == ([3, 1] if total_steps else [3, 3])
    assert [r["elapsed_seconds"] for r in rows] == (
        [108.0, 105.0] if total_steps else [108.0, 108.0]
    )
    assert [r["complete"] for r in rows] == (
        [True, False] if total_steps else [True, True]
    )
    assert (tmp_path / "resource_summary.json").exists()


def test_configuration_rejects_ignored_history_selection():
    from pydantic import ValidationError
    from dpdl.configurationmanager import Configuration

    with pytest.raises(ValidationError, match="history.*inverse|inverse.*history"):
        Configuration(
            command="train", noise_mechanism="gaussian", noise_history_mode="stored"
        )


@pytest.mark.parametrize("memory,timing", [(True, False), (False, True), (True, True)])
def test_non_private_configuration_accepts_resource_recording(memory, timing):
    from dpdl.configurationmanager import Configuration

    Configuration(
        command="train",
        privacy=False,
        record_memory_usage=memory,
        record_training_time=timing,
    )


@pytest.mark.parametrize("total_steps", [0, 4])
@pytest.mark.parametrize("memory,timing", [(True, False), (False, True), (True, True)])
def test_non_private_training_records_completed_updates(
    tmp_path, total_steps, memory, timing
):
    import json

    now = [0.0]
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1)
    trainer.model.train_metrics = SimpleNamespace(
        compute=lambda: {}, reset=lambda: None
    )
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = None
    actual_updates = []

    def completed_update(*args):
        actual_updates.append(1)
        now[0] += 1.0

    trainer.optimizer.register_step_post_hook(completed_update)
    loader = DataLoader(
        TensorDataset(
            torch.arange(10, dtype=torch.float32).reshape(-1, 1), torch.zeros(10, 1)
        ),
        batch_size=4,
    )
    trainer.datamodule = SimpleNamespace(
        get_dataloader=lambda split: loader, batch_size=4
    )
    trainer.physical_batch_size = 3
    trainer.adapter = SimpleNamespace(
        move_to_device=lambda x, y: (x, y),
        iterate_physical_batches=lambda batch, size: zip(
            batch[0].split(size), batch[1].split(size)
        ),
        forward=lambda model, batch: model(batch[0]),
        compute_loss=lambda model, batch, output, normalize_by: (output - batch[1])
        .square()
        .mean()
        / normalize_by,
        update_metrics=lambda *args: None,
    )
    trainer.total_steps, trainer.epochs, trainer.validation_frequency = (
        total_steps,
        2,
        0,
    )
    recorder = resources().RecordResourceStatsCallback(
        device="cpu",
        output_dir=tmp_path,
        clock=lambda: now[0],
        record_memory_usage=memory,
        record_training_time=timing,
        configuration=SimpleNamespace(
            privacy=False,
            noise_mechanism="gaussian",
            physical_batch_size=3,
            record_memory_usage=memory,
            record_training_time=timing,
        ),
        hyperparams=SimpleNamespace(batch_size=4),
    )
    trainer.callback_handler = CallbackHandler([recorder])
    trainer.fit()
    rows = [
        json.loads(line)
        for line in (tmp_path / "resource_telemetry.jsonl").read_text().splitlines()
    ]
    counts = [3, 1] if total_steps else [3, 3]
    assert [row["logical_steps"] for row in rows] == counts
    assert sum(counts) == len(actual_updates)
    assert [row["complete"] for row in rows] == [True, not bool(total_steps)]
    for row, count in zip(rows, counts):
        assert row["elapsed_seconds"] == (float(count) if timing else None)
        assert row["mean_step_seconds"] == (1.0 if timing else None)
        assert row["metadata"]["method"] == "non_private"
        assert row["metadata"]["trainable_numel"] == 2
        assert row["metadata"]["noise_history_mode"] is None
        assert row["metadata"]["diagnostics"] is False
        assert row["max_retained_noise_history_bytes"] == (0 if memory else None)
        segment = row["per_rank"][0]["segments"][0]
        assert segment["physical_batches"] == (5 if count == 3 else 2)
        assert segment["examples"] == (10 if count == 3 else 4)
    assert (tmp_path / "resource_summary.json").exists()


def test_reused_output_directory_has_unambiguous_runs_and_preserved_summaries(tmp_path):
    import json
    from dpdl.callbacks.resource_stats import RecordResourceStatsCallback

    for _ in range(2):
        recorder = RecordResourceStatsCallback(device="cpu", output_dir=tmp_path)
        recorder.start_epoch(0)
        recorder.record_step(examples=4, physical_batches=1)
        recorder.end_epoch(complete=True)
        recorder.finish()
    rows = [
        json.loads(line)
        for line in (tmp_path / "resource_telemetry.jsonl").read_text().splitlines()
    ]
    assert len({row["run_id"] for row in rows}) == 2
    for row in rows:
        summary = json.loads(
            (tmp_path / "resource_summaries" / f"{row['run_id']}.json").read_text()
        )
        assert summary["run_id"] == summary["first_epoch"]["run_id"] == row["run_id"]
    latest = json.loads((tmp_path / "resource_summary.json").read_text())
    assert latest["run_id"] == rows[-1]["run_id"]


@pytest.mark.parametrize(
    "memory,timing", [(True, False), (False, True), (True, True), (False, False)]
)
def test_measurements_are_independently_optional(tmp_path, monkeypatch, memory, timing):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled measurement was collected")

    now = [0.0]
    noiser = SimpleNamespace(
        noise_state_inventory=(
            (lambda: {"history": {"cpu": 32}, "diagnostics": {}, "scratch": {}})
            if memory
            else forbidden
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        (lambda device: None) if memory else forbidden,
    )
    for name in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(
            torch.cuda, name, (lambda device: 100) if memory else forbidden
        )
    recorder = resources().RecordResourceStatsCallback(
        device="cuda:0",
        output_dir=tmp_path,
        mechanism=noiser,
        clock=(lambda: now[0]) if timing else forbidden,
        record_memory_usage=memory,
        record_training_time=timing,
    )
    for epoch_index in range(2):
        recorder.start_epoch(epoch_index)
        now[0] += 6.0
        recorder.record_step(examples=4, physical_batches=1)
        recorder.end_epoch(complete=True)
    recorder.finish()
    if not memory and not timing:
        assert list(tmp_path.iterdir()) == []
        return
    import json

    summary = json.loads((tmp_path / "resource_summary.json").read_text())
    for row in recorder.rows:
        assert row["elapsed_seconds"] == (6.0 if timing else None)
        assert row["mean_step_seconds"] == (6.0 if timing else None)
        assert row["peak_allocated_bytes"] == (100 if memory else None)
        assert row["max_retained_noise_history_bytes"] == (32 if memory else None)
        if not timing:
            assert row["timing_unavailable_reason"] == "disabled"
        if not memory:
            assert row["memory_unavailable_reason"] == "disabled"
            assert row["per_rank"][0]["segments"][0]["noise_state_start"] is None
    assert summary["subsequent_epochs"]["mean_step_seconds"] == (
        6.0 if timing else None
    )
    if not timing:
        assert summary["subsequent_epochs"]["timing_unavailable_reason"] == "disabled"
    if not memory:
        assert summary["subsequent_epochs"]["memory_unavailable_reason"] == "disabled"


@pytest.mark.parametrize("memory,timing", [(True, False), (False, True), (True, True)])
def test_cli_measurement_flags_and_train_predict_handoff(
    monkeypatch, tmp_path, memory, timing
):
    from pathlib import Path
    import dpdl.cli as cli_module
    import typer
    from typer.testing import CliRunner

    app = typer.Typer()
    app.command()(cli_module.cli)
    trained, predicted = [], []
    monkeypatch.setattr(
        cli_module,
        "run_train",
        lambda cm: trained.append(cm) or Path(tmp_path / "model.pt"),
    )
    monkeypatch.setattr(cli_module, "run_predict", lambda cm: predicted.append(cm))
    monkeypatch.setattr(cli_module, "synchronize_saved_model_path", lambda path: path)
    monkeypatch.setattr(cli_module, "start_experiment_logging", lambda *args: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    args = ["train-predict", "--epochs", "1", "--noise-multiplier", "1"]
    if memory:
        args.append("--record-memory-usage")
    if timing:
        args.append("--record-training-time")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert trained[0].configuration.record_memory_usage is memory
    assert trained[0].configuration.record_training_time is timing
    assert predicted[0].configuration.privacy is False
    assert predicted[0].configuration.record_memory_usage is False
    assert predicted[0].configuration.record_training_time is False


@pytest.mark.parametrize(
    "family,diagnostics",
    [
        ("bsr", True),
        ("bandmf", True),
        ("gaussian", False),
        ("bifr", False),
        ("bifr", True),
    ],
)
def test_runtime_metadata_matches_diagnostic_retention(family, diagnostics):
    from opacus.noise_mechanisms import (
        CorrelatedNoiseMechanism,
        GaussianNoiseMechanism,
        InverseBandNoiseMechanism,
    )

    module = resources()
    if family in ("bsr", "bandmf"):
        noiser = CorrelatedNoiseMechanism(coeffs=[1.0, 0.1], z_std=0.2)
    elif family == "bifr":
        noiser = InverseBandNoiseMechanism(
            inverse_coeffs=[1.0, -0.1], z_std=0.2, record_noise_diagnostics=diagnostics
        )
    else:
        noiser = GaussianNoiseMechanism()
    param = torch.nn.Parameter(torch.zeros(4))
    optimizer = DPOptimizer(
        optimizer=torch.optim.SGD([param], lr=0.1),
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
        noise_mechanism=noiser,
    )
    trainer = SimpleNamespace(
        optimizer=optimizer, device=torch.device("cpu"), total_steps=0
    )
    configuration = SimpleNamespace(
        privacy=True,
        record_memory_usage=True,
        record_training_time=False,
        noise_mechanism=family,
        physical_batch_size=2,
    )
    # Metadata is captured before training, when diagnostic tensors do not exist yet.
    metadata = module.runtime_metadata(
        trainer, configuration, SimpleNamespace(batch_size=2)
    )
    assert metadata["diagnostics"] is diagnostics
    param.grad_sample = torch.zeros(2, 4)
    optimizer.step()
    actual_bytes = sum(noiser.noise_state_inventory()["diagnostics"].values())
    assert bool(actual_bytes) is metadata["diagnostics"]


@pytest.mark.parametrize(
    "memory,timing", [(False, False), (True, False), (False, True), (True, True)]
)
def test_resource_recording_is_registered_by_callback_factory(tmp_path, memory, timing):
    from dpdl.callbacks.base_callback import Callback
    from dpdl.callbacks.callback_factory import CallbackFactory
    from dpdl.configurationmanager import Configuration

    callbacks = CallbackFactory.get_callbacks(
        Configuration(
            command="train",
            log_dir=str(tmp_path),
            record_memory_usage=memory,
            record_training_time=timing,
            record_loss_by_epoch=True,
        ),
        SimpleNamespace(batch_size=4),
        device=torch.device("cpu"),
    )
    resources = [
        cb for cb in callbacks if type(cb).__name__ == "RecordResourceStatsCallback"
    ]
    assert len(resources) == int(memory or timing)
    if resources:
        assert callbacks[0] is resources[0]
        assert isinstance(resources[0], Callback)
        assert resources[0].record_memory_usage is memory
        assert resources[0].record_training_time is timing


def test_resource_callback_initializes_from_trainer_and_writes_on_lifecycle(tmp_path):
    from dpdl.callbacks.callback_factory import CallbackFactory
    from dpdl.configurationmanager import Configuration

    config = Configuration(
        command="train",
        log_dir=str(tmp_path),
        experiment_name="callback",
        record_memory_usage=True,
        record_training_time=True,
    )
    callbacks = CallbackFactory.get_callbacks(
        config, SimpleNamespace(batch_size=2), device=torch.device("cpu")
    )
    callback = next(
        cb for cb in callbacks if type(cb).__name__ == "RecordResourceStatsCallback"
    )
    param = torch.nn.Parameter(torch.zeros(4))
    optimizer = DPOptimizer(
        optimizer=torch.optim.SGD([param], lr=0.1),
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        expected_batch_size=2,
    )
    trainer = SimpleNamespace(
        optimizer=optimizer, device=torch.device("cpu"), total_steps=0
    )
    handler = CallbackHandler([callback])
    handler.call("on_train_start", trainer)
    assert callback.mechanism is optimizer.noise_mechanism
    assert callback.metadata["record_memory_usage"] is True
    assert callback.metadata["record_training_time"] is True
    handler.call("on_train_epoch_start", trainer, 0)
    param.grad_sample = torch.zeros(2, 4)
    optimizer.step()
    handler.call(
        "on_train_physical_batch_end",
        trainer,
        0,
        (torch.zeros(2, 4), torch.zeros(2)),
        0.0,
    )
    handler.call(
        "on_train_batch_end", trainer, 0, (torch.zeros(2, 4), torch.zeros(2)), 0.0
    )
    handler.call("on_train_epoch_end", trainer, 0, {})
    handler.call("on_train_end", trainer)
    assert callback.rows[0]["logical_steps"] == 1
    assert (tmp_path / "callback" / "resource_summary.json").exists()
