"""Opt-in epoch measurements including epoch setup, microbatches and metric computation.

GPU bytes are PyTorch allocator statistics, not host RSS or total device usage.
Resetting peaks does not free live tensors or cached reservations. Noise payload
counts backing storage only, excluding Python objects and opaque live RNGs.
"""

import json
import time
import uuid
from pathlib import Path

import torch
from opacus.optimizers import DPOptimizer

from .base_callback import Callback


def _maximum(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def summarize_epoch(per_rank, *, epoch, complete):
    if not per_rank:
        raise ValueError("missing rank resource records")
    counts = [sum(s["logical_steps"] for s in r["segments"]) for r in per_rank]
    if len(set(counts)) != 1 or len({len(r["segments"]) for r in per_rank}) != 1:
        raise ValueError("inconsistent logical step counts or segments across ranks")
    timed = all(r.get("record_training_time", True) for r in per_rank)
    timed = timed and all(
        s["elapsed_seconds"] is not None for r in per_rank for s in r["segments"]
    )
    elapsed = 0.0 if timed else None
    for segments in zip(*(r["segments"] for r in per_rank)):
        if len({s["logical_steps"] for s in segments}) != 1:
            raise ValueError("inconsistent segment step counts")
        if timed:
            elapsed += max(s["elapsed_seconds"] for s in segments)
    segments = [s for r in per_rank for s in r["segments"]]
    row = dict(
        schema_version=1,
        epoch=epoch,
        complete=complete,
        per_rank=per_rank,
        elapsed_seconds=elapsed,
        logical_steps=counts[0],
        mean_step_seconds=elapsed / counts[0] if timed and counts[0] else None,
        examples=sum(s["examples"] for s in segments),
        physical_batches=sum(s["physical_batches"] for s in segments),
    )
    for key in (
        "start_allocated_bytes",
        "start_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        row[key] = _maximum(s.get(key) for s in segments)
    memory_enabled = any(r.get("record_memory_usage", True) for r in per_rank)
    row["timing_unavailable_reason"] = None if timed else "disabled"
    row["memory_unavailable_reason"] = (
        None
        if row["peak_allocated_bytes"] is not None
        else (
            "accelerator allocator metrics unavailable"
            if memory_enabled
            else "disabled"
        )
    )
    history_by_rank = [
        max(
            (
                sum((s.get("noise_state_high_water") or {}).get("history", {}).values())
                for s in rank["segments"]
            ),
            default=0,
        )
        for rank in per_rank
    ]
    row["max_retained_noise_history_bytes"] = (
        max(history_by_rank) if memory_enabled else None
    )
    row["sum_rank_retained_noise_history_bytes"] = (
        sum(history_by_rank) if memory_enabled else None
    )
    row["examples_per_second"] = row["examples"] / elapsed if elapsed else None
    return row


def summarize_run(epochs):
    later = None
    if len(epochs) > 1:
        steps = sum(r["logical_steps"] for r in epochs[1:])
        elapsed = (
            sum(r["elapsed_seconds"] for r in epochs[1:])
            if all(r["elapsed_seconds"] is not None for r in epochs[1:])
            else None
        )
        later = dict(
            logical_steps=steps,
            elapsed_seconds=elapsed,
            mean_step_seconds=(
                elapsed / steps if elapsed is not None and steps else None
            ),
        )
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            later[key] = _maximum(r.get(key) for r in epochs[1:])
        for key in ("timing_unavailable_reason", "memory_unavailable_reason"):
            later[key] = next((r.get(key) for r in epochs[1:] if r.get(key)), None)
    return dict(
        schema_version=1,
        first_epoch=epochs[0] if epochs else None,
        subsequent_epochs=later,
    )


class RecordResourceStatsCallback(Callback):
    """Collect independently selected memory and timing metrics at epoch boundaries.

    The callback factory registers this callback when either recording flag is enabled.
    Direct callers default to both metrics. Memory recording inventories retained
    noise state and resets allocator peaks; timing reads the monotonic clock.
    Both use device synchronization at segment boundaries, never per step.
    """

    def __init__(
        self,
        *,
        device,
        output_dir,
        clock=time.monotonic,
        enabled=True,
        mechanism=None,
        metadata=None,
        record_memory_usage=True,
        record_training_time=True,
        configuration=None,
        hyperparams=None,
    ):
        super().__init__()
        self.configuration, self.hyperparams = configuration, hyperparams
        self.device, self.output_dir = torch.device(device), Path(output_dir)
        self.record_memory_usage = record_memory_usage
        self.record_training_time = record_training_time
        self.clock = clock
        self.enabled = enabled and (record_memory_usage or record_training_time)
        self.mechanism, self.metadata = mechanism, metadata or {}
        self.rows, self.segments = [], []
        self.active = None
        self.run_id = None

    def on_train_start(self, trainer):
        super().on_train_start(trainer)
        if not self.enabled:
            return
        self.mechanism = getattr(trainer.optimizer, "noise_mechanism", None)
        if self.record_memory_usage and hasattr(
            self.mechanism, "record_noise_diagnostics"
        ):
            self.mechanism.record_noise_diagnostics = False
            self.mechanism.last_flat_z = self.mechanism.last_flat_u = None
        if self.configuration is not None:
            self.metadata = runtime_metadata(
                trainer, self.configuration, self.hyperparams
            )

    def on_train_epoch_start(self, trainer, epoch):
        self.start_epoch(epoch)

    def on_train_epoch_end(self, trainer, epoch, metrics):
        if not self.enabled:
            return
        steps = sum(segment["logical_steps"] for segment in self.segments)
        if self.active is not None:
            steps += self.active["logical_steps"]
        complete = (
            not trainer.total_steps or steps == trainer._calculate_steps_per_epoch()
        )
        self.end_epoch(complete=complete)

    def on_train_physical_batch_end(self, trainer, batch_idx, batch, loss):
        self.record_physical_batch(len(batch[1]))
        # DP optimizer.step() has already consumed the physical batch's skip
        # flag. Read its result without touching the queue or trainer counters.
        # Existing logical-batch callbacks can overcount updates in epoch mode.
        if isinstance(trainer.optimizer, DPOptimizer):
            if not trainer.optimizer._is_last_step_skipped:
                self.record_step()

    def on_train_batch_end(self, trainer, batch_idx, batch, loss):
        super().on_train_batch_end(trainer, batch_idx, batch, loss)
        # Non-DP training updates weights after all physical-batch callbacks,
        # so its completed update is counted at logical-batch end instead.
        if not isinstance(trainer.optimizer, DPOptimizer):
            self.record_step()

    def on_train_end(self, trainer):
        self.finish()

    def _distributed(self):
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _cuda(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _memory(self, start=False):
        prefix = "start" if start else "peak"
        if not self.record_memory_usage or not self._cuda():
            return {f"{prefix}_allocated_bytes": None, f"{prefix}_reserved_bytes": None}
        allocated = (
            torch.cuda.memory_allocated if start else torch.cuda.max_memory_allocated
        )
        reserved = (
            torch.cuda.memory_reserved if start else torch.cuda.max_memory_reserved
        )
        return {
            f"{prefix}_allocated_bytes": int(allocated(self.device)),
            f"{prefix}_reserved_bytes": int(reserved(self.device)),
        }

    def _inventory(self):
        if not self.record_memory_usage:
            return None
        read = getattr(self.mechanism, "noise_state_inventory", None)
        return read() if read else {"history": {}, "diagnostics": {}, "scratch": {}}

    def start_epoch(self, epoch):
        if not self.enabled:
            return
        if self.run_id is None:
            self.run_id = uuid.uuid4().hex
            if self._distributed():
                shared = [self.run_id if torch.distributed.get_rank() == 0 else None]
                torch.distributed.broadcast_object_list(shared, src=0)
                self.run_id = shared[0]
        self.epoch, self.segments = epoch, []
        self.start_segment()

    def start_segment(self):
        if not self.enabled:
            return
        if self.active is not None:
            raise RuntimeError("resource training segment already active")
        if self._distributed():
            torch.distributed.barrier()
        if self._cuda():
            torch.cuda.synchronize(self.device)
            if self.record_memory_usage:
                torch.cuda.reset_peak_memory_stats(self.device)
        initial = self._inventory()
        self.active = dict(
            logical_steps=0,
            physical_batches=0,
            examples=0,
            noise_state_start=initial,
            noise_state_high_water=initial,
            history_depth_start=getattr(self.mechanism, "state_depth", 0),
            **self._memory(start=True),
        )
        self.started = self.clock() if self.record_training_time else None

    def record_physical_batch(self, examples):
        if self.enabled:
            self.active["examples"] += examples
            self.active["physical_batches"] += 1

    def record_step(self, *, examples=0, physical_batches=0):
        if not self.enabled:
            return
        self.active["logical_steps"] += 1
        self.active["examples"] += examples
        self.active["physical_batches"] += physical_batches
        if not self.record_memory_usage:
            return
        inventory = self._inventory()
        high = self.active["noise_state_high_water"]
        self.active["noise_state_high_water"] = {
            category: {
                loc: max(high[category].get(loc, 0), inventory[category].get(loc, 0))
                for loc in high[category].keys() | inventory[category].keys()
            }
            for category in inventory
        }

    def end_segment(self):
        if not self.enabled or self.active is None:
            return
        if self._cuda():
            torch.cuda.synchronize(self.device)
        elapsed = self.clock() - self.started if self.record_training_time else None
        self.active.update(
            elapsed_seconds=elapsed,
            **self._memory(),
            noise_state_end=self._inventory(),
            history_depth_end=getattr(self.mechanism, "state_depth", 0),
            max_history_depth=getattr(self.mechanism, "max_state_depth", 0),
        )
        self.segments.append(self.active)
        self.active = None

    def end_epoch(self, *, complete):
        if not self.enabled:
            return
        self.end_segment()
        rank = torch.distributed.get_rank() if self._distributed() else 0
        local = {
            "rank": rank,
            "segments": self.segments,
            "record_memory_usage": self.record_memory_usage,
            "record_training_time": self.record_training_time,
        }
        per_rank = [local]
        if self._distributed():
            per_rank = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(per_rank, local)
        row = summarize_epoch(per_rank, epoch=self.epoch, complete=complete)
        row["metadata"] = self.metadata
        row["run_id"] = self.run_id
        self.rows.append(row)
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "resource_telemetry.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def finish(self):
        if not self.enabled:
            return
        if not self._distributed() or torch.distributed.get_rank() == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = summarize_run(self.rows)
            payload["metadata"] = self.metadata
            payload["run_id"] = self.run_id
            serialized = json.dumps(payload, indent=2, allow_nan=False)
            (self.output_dir / "resource_summary.json").write_text(serialized)
            if self.run_id is not None:
                summaries = self.output_dir / "resource_summaries"
                summaries.mkdir(exist_ok=True)
                (summaries / f"{self.run_id}.json").write_text(serialized)


def runtime_metadata(trainer, configuration, hyperparams):
    """Capture reproducibility context only when instrumentation is enabled."""
    import subprocess
    import opacus
    from opacus.noise_mechanisms import CorrelatedNoiseMechanism

    def revision(root):
        try:
            sha = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "-C", str(root), "status", "--porcelain"], text=True
                )
            )
            return {"revision": sha, "dirty": dirty}
        except (OSError, subprocess.CalledProcessError):
            return {"revision": None, "reason": "Git metadata unavailable"}

    model_params = [
        p
        for group in trainer.optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    ]
    noiser = getattr(trainer.optimizer, "noise_mechanism", None)
    return dict(
        record_memory_usage=configuration.record_memory_usage,
        record_training_time=configuration.record_training_time,
        method=(
            configuration.noise_mechanism if configuration.privacy else "non_private"
        ),
        noise_history_mode=getattr(noiser, "noise_history_mode", None),
        bandwidth=getattr(noiser, "bandwidth", None),
        buffers=getattr(noiser, "num_buffers", None),
        logical_batch_size=hyperparams.batch_size,
        physical_batch_size=configuration.physical_batch_size,
        trainable_numel=sum(p.numel() for p in model_params),
        dtypes=sorted({str(p.dtype) for p in model_params}),
        world_size=(
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        ),
        device=str(trainer.device),
        device_model=(
            torch.cuda.get_device_name(trainer.device)
            if trainer.device.type == "cuda"
            else "CPU"
        ),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        hip_version=torch.version.hip,
        dpdl=revision(Path(__file__).resolve().parents[2]),
        opacus=revision(Path(opacus.__file__).resolve().parents[1]),
        # Factor-side BSR/BandMF always retain diagnostic copies. Inverse
        # subclasses expose a flag, which takes precedence over that policy.
        diagnostics=getattr(
            noiser,
            "record_noise_diagnostics",
            isinstance(noiser, CorrelatedNoiseMechanism),
        ),
        scope="training epochs including epoch-start callbacks, metric computation and in-loop callbacks; epoch-end reporting and subsequent validation excluded; checkpointing must be disabled for comparisons",
        memory_scope="PyTorch allocated/reserved bytes; reserved cache can carry over from validation",
    )
