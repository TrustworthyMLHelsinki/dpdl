"""Real rank aggregation and rank-zero output across epoch validation."""

import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

from dpdl.callbacks.resource_stats import RecordResourceStatsCallback
from dpdl.callbacks.callback_factory import CallbackHandler


def _worker(rank, rendezvous, directory, memory, timing):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        now = [0.0]
        recorder = RecordResourceStatsCallback(
            device="cpu",
            output_dir=directory,
            clock=lambda: now[0],
            record_memory_usage=memory,
            record_training_time=timing,
        )
        trainer = SimpleNamespace(
            callback_handler=CallbackHandler([recorder]),
            total_steps=3,
            _calculate_steps_per_epoch=lambda: 2,
        )
        for epoch in range(2):
            trainer.callback_handler.call("on_train_epoch_start", trainer, epoch)
            now[0] += rank + 1
            recorder.record_step(examples=4, physical_batches=2)
            if epoch == 0:
                now[0] += 3 - rank
                recorder.record_step(examples=4, physical_batches=2)
            trainer.callback_handler.call("on_train_epoch_end", trainer, epoch, {})
            if rank == 0:
                now[0] += 100  # Validation after the epoch is excluded.
            dist.barrier()
        trainer.callback_handler.call("on_train_end", trainer)
        run_ids = [None, None]
        dist.all_gather_object(run_ids, recorder.run_id)
        assert len(set(run_ids)) == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("memory,timing", [(True, True), (True, False), (False, True)])
def test_real_distributed_resource_epochs_and_rank_zero_output(
    tmp_path, memory, timing
):
    mp.spawn(
        _worker,
        args=(f"file://{tmp_path}/sync", str(tmp_path / "outputs"), memory, timing),
        nprocs=2,
        join=True,
    )
    output = tmp_path / "outputs"
    rows = [
        json.loads(line)
        for line in (output / "resource_telemetry.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2  # One writer, not one row per worker.
    assert [r["elapsed_seconds"] for r in rows] == (
        [4.0, 2.0] if timing else [None, None]
    )
    assert [r["logical_steps"] for r in rows] == [2, 1]
    assert [r["mean_step_seconds"] for r in rows] == (
        [2.0, 2.0] if timing else [None, None]
    )
    assert rows[0]["examples"] == 16
    assert len(rows[0]["per_rank"]) == 2
    assert rows[0]["run_id"] == rows[1]["run_id"]
    summary = json.loads((output / "resource_summary.json").read_text())
    assert summary["run_id"] == rows[0]["run_id"]
    assert summary["subsequent_epochs"]["mean_step_seconds"] == (
        2.0 if timing else None
    )
