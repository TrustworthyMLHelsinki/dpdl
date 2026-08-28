import csv
import json
import logging
from pathlib import Path
from typing import Any

import torch

from .base_callback import Callback

log = logging.getLogger(__name__)


class RecordOptimizerStatsCallback(Callback):
    """
    Record gradient- and Adam-related statistics at every optimizer step.

    Notation:
      C : the clipping bound (max_grad_norm).
      g : per-example gradient norm ||g_i||, across all trainable
          parameters

    Columns:
      step                     - global optimizer step index
      gradient_norm_q50/90/95/99 - quantiles of g over the logical batch
      q99_over_c               - gradient_norm_q99 / C
      clipped_fraction         - fraction of examples with g > C
      retained_weight_mean     - mean of min(1, C/g)
      retained_weight_sq_mean  - mean of min(1, C/g)^2
      adam_lr_free_update_rms  - RMS(m_hat / (sqrt(v_hat) + adam_epsilon))
      adam_inv_denom_rms       - RMS(1 / (sqrt(v_hat) + adam_epsilon))
      noise_multiplier         - accountant-calibrated Gaussian multiplier
      effective_batch_size     - global expected logical batch size
      normalized_noise         - noise_multiplier / effective_batch_size
      adam_actual_update_rms   - RMS of the realized parameter update
      adam_expected_update_rms - RMS of the update reconstructed from Adam state
      adam_identity_relative_residual - relative RMS reconstruction error

    The two Adam fields are aggregated over all trainable parameters, where
    ``m_hat``/``v_hat`` are the first/second moments from the optimizer state after the step.
    Gradient statistics require per-example gradients (DP training); Adam statistics require an Adam-family optimizer.
    Fields that cannot be computed are recorded as NaN.

    Distributed training: gradient statistics are computed over a logical batch.
    Each rank owns a shard of the per-example gradients, so we all-gather the per-example norms
    across ranks before computing the quantiles and means
    """

    _FIELDNAMES = (
        'step',
        'gradient_norm_q50',
        'gradient_norm_q90',
        'gradient_norm_q95',
        'gradient_norm_q99',
        'q99_over_c',
        'clipped_fraction',
        'retained_weight_mean',
        'retained_weight_sq_mean',
        'adam_lr_free_update_rms',
        'adam_inv_denom_rms',
        'noise_multiplier',
        'expected_batch_size_per_worker',
        'effective_batch_size',
        'sample_rate',
        'normalized_noise',
        'adam_actual_update_rms',
        'adam_expected_update_rms',
        'adam_identity_relative_residual',
        'adam_identity_max_abs_residual',
    )

    # Quantiles reported for the per-example gradient norm g.
    _QUANTILES = (0.50, 0.90, 0.95, 0.99)

    def __init__(
        self,
        log_dir: str | Path,
        max_grad_norm: float,
        adam_identity_tolerance: float = 1e-5,
    ):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.C = float(max_grad_norm)
        self.adam_identity_tolerance = float(adam_identity_tolerance)

        # Rows are buffered in memory because a run has only one row per optimizer step.
        self._rows = []

        # Physical batches are pieces of one logical DP batch, so retain their norm vectors
        # until the logical optimizer step finishes.
        self._current_norms = []

        # Runtime-dependent values are initialized when the trainer and DP wrapper exist.
        self._world_size = 1
        self._device = torch.device('cpu')

        # Parameter snapshots are keyed by object identity to avoid relying on model names.
        self._params_before_step = {}
        self._optimizer_metadata = {}

    def on_train_start(self, trainer, *args, **kwargs):
        super().on_train_start(trainer, *args, **kwargs)

        self._world_size = self._distributed_world_size()
        self._device = trainer.device

        base_optimizer = getattr(trainer.optimizer, 'original_optimizer', trainer.optimizer)
        if not isinstance(base_optimizer, torch.optim.Adam):
            raise TypeError('RecordOptimizerStatsCallback requires torch.optim.Adam.')

        # Freeze optimizer identity and parameter membership before collecting result rows.
        self._optimizer_metadata = self._build_optimizer_metadata(trainer)
        self._snapshot_parameters(trainer.optimizer)

    def on_train_batch_start(self, trainer, *args, **kwargs):
        # A new logical batch starts a new norm accumulation and Adam identity snapshot.
        self._current_norms.clear()
        self._snapshot_parameters(trainer.optimizer)

    def on_train_physical_batch_end(self, trainer, *args, **kwargs):
        """Accumulate norm vectors until all physical pieces of a logical batch finish."""
        with torch.no_grad():
            sum_squares = None
            for grad_sample in trainer.optimizer.grad_samples:
                # Add each parameter block to the per-example squared gradient norm.
                flat = grad_sample.reshape(grad_sample.size(0), -1)
                block_squares = flat.pow(2).sum(dim=1)
                sum_squares = block_squares if sum_squares is None else sum_squares + block_squares

            self._current_norms.append(sum_squares.sqrt())

    def on_train_batch_end(self, trainer, batch_idx, batch, loss, *args, **kwargs):
        super().on_train_batch_end(trainer, batch_idx, batch, loss, *args, **kwargs)

        # Adam statistics are read after optimizer.step(); the snapshot is from batch start.
        row = {name: float('nan') for name in self._FIELDNAMES}
        row['step'] = self.global_step

        self._record_gradient_stats(row)

        # These values all describe the post-step optimizer state for the same row.
        self._record_adam_stats(row, trainer.optimizer)
        self._record_privacy_coordinates(row, trainer.optimizer)
        self._record_adam_identity(row, trainer.optimizer)

        self._rows.append(row)
        self._current_norms.clear()
        self._params_before_step.clear()

    def _record_gradient_stats(self, row: dict) -> None:
        with torch.no_grad():
            # Physical batches together form this worker's part of the logical batch.
            g_local = torch.cat(self._current_norms).float()

            # Quantiles must describe the global logical batch, not rank zero's shard.
            g = self._all_gather_norms(g_local)

            q = torch.tensor(self._QUANTILES, device=g.device, dtype=g.dtype)
            q50, q90, q95, q99 = torch.quantile(g, q).tolist()

            # Global clipping multiplies each per-example gradient by min(1, C / ||g||).
            retained = torch.clamp(self.C / g, max=1.0)

            row['gradient_norm_q50'] = q50
            row['gradient_norm_q90'] = q90
            row['gradient_norm_q95'] = q95
            row['gradient_norm_q99'] = q99
            row['q99_over_c'] = q99 / self.C if self.C > 0 else float('nan')
            row['clipped_fraction'] = (g > self.C).float().mean().item()
            row['retained_weight_mean'] = retained.mean().item()
            row['retained_weight_sq_mean'] = retained.pow(2).mean().item()

    def _all_gather_norms(self, g_local: torch.Tensor) -> torch.Tensor:
        """
        All-gather per-sample norm vectors across ranks and
        return their concatenation. Every rank receives the full vector.

        Must be called by all or no ranks.
        """
        if self._world_size == 1:
            return g_local

        device = g_local.device

        # Gather local batch sizes because Poisson sampling can give each worker a different count.
        local_size = torch.tensor([g_local.numel()], device=device, dtype=torch.long)
        size_list = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(self._world_size)]
        torch.distributed.all_gather(size_list, local_size)
        sizes = [int(s.item()) for s in size_list]

        max_size = max(sizes)
        # Pad to a common length so all_gather sees equal-sized tensors.
        padded = g_local.new_zeros(max_size)
        if g_local.numel() > 0:
            padded[: g_local.numel()] = g_local

        gathered = [g_local.new_zeros(max_size) for _ in range(self._world_size)]
        torch.distributed.all_gather(gathered, padded)

        # Strip communication padding before reconstructing the logical-batch vector.
        parts = [gathered[i][: sizes[i]] for i in range(self._world_size) if sizes[i] > 0]
        return torch.cat(parts)

    def _record_adam_stats(self, row: dict, optimizer) -> None:
        row['adam_lr_free_update_rms'], row['adam_inv_denom_rms'] = self._compute_adam_stats(optimizer)

    def _compute_adam_stats(self, optimizer) -> tuple[float, float]:
        sum_sq_update = 0.0
        sum_sq_inv_denom = 0.0
        count = 0

        # DDP synchronizes gradients and each worker advances identical Adam state.
        with torch.no_grad():
            for group in optimizer.param_groups:
                eps = group['eps']
                for parameter in group['params']:
                    if not parameter.requires_grad:
                        continue

                    m_hat, v_hat = self._bias_corrected_adam_state(optimizer, group, parameter)

                    # Separate Adam's normalized direction from the user-supplied LR.
                    denom = v_hat.sqrt() + eps
                    update = m_hat / denom
                    inv_denom = denom.reciprocal()

                    sum_sq_update += update.pow(2).sum().item()
                    sum_sq_inv_denom += inv_denom.pow(2).sum().item()
                    count += update.numel()

        # Weight by coordinate count so tensors of different shapes form one global RMS.
        lr_free_update_rms = (sum_sq_update / count) ** 0.5
        inv_denom_rms = (sum_sq_inv_denom / count) ** 0.5
        return lr_free_update_rms, inv_denom_rms

    @staticmethod
    def _bias_corrected_adam_state(optimizer, group: dict, parameter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Read ordinary Adam's post-step state using its exact bias corrections."""
        state = optimizer.state[parameter]
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        step = state['step']
        step = step.item() if isinstance(step, torch.Tensor) else step

        beta1, beta2 = group['betas']

        # PyTorch stores exponential moving sums; divide out startup bias explicitly.
        m_hat = exp_avg / (1.0 - beta1**step)

        # Apply the same startup correction to Adam's second-moment accumulator.
        v_hat = exp_avg_sq / (1.0 - beta2**step)
        return m_hat, v_hat

    def _record_privacy_coordinates(self, row: dict, optimizer) -> None:
        noise_multiplier = float(optimizer.noise_multiplier)
        expected_batch_size = float(optimizer.expected_batch_size)

        # Opacus stores expected batch per worker; the aggregate uses the global expected batch.
        effective_batch_size = expected_batch_size * self._world_size
        row['noise_multiplier'] = noise_multiplier
        row['expected_batch_size_per_worker'] = expected_batch_size
        row['effective_batch_size'] = effective_batch_size
        # Under normalized clipping, sigma / B_eff is the averaged-gradient noise scale.
        row['normalized_noise'] = noise_multiplier / effective_batch_size

        # The global Poisson sample rate is expected logical batch divided by dataset size.
        row['sample_rate'] = effective_batch_size / self._optimizer_metadata['training_dataset_size']

    def _snapshot_parameters(self, optimizer) -> None:
        # Adam identity requires the exact parameters immediately before optimizer.step().
        with torch.no_grad():
            self._params_before_step = {
                id(parameter): parameter.detach().clone() for parameter in self._optimizer_parameters(optimizer)
            }

    @staticmethod
    def _optimizer_parameters(optimizer):
        """Iterate parameters that Adam and Opacus update."""
        for group in optimizer.param_groups:
            for parameter in group['params']:
                # Frozen backbone parameters remain in the optimizer groups, but Opacus does
                # not attach per-sample gradients or create Adam state for them.
                if parameter.requires_grad:
                    yield parameter

    def _record_adam_identity(self, row: dict, optimizer) -> None:
        actual_sq = 0.0
        expected_sq = 0.0
        residual_sq = 0.0
        max_abs_residual = 0.0
        count = 0

        # Reconstruct each tensor's update from the moments that produced the actual step.
        with torch.no_grad():
            for group in optimizer.param_groups:
                learning_rate = float(group['lr'])
                adam_epsilon = float(group['eps'])

                for parameter in group['params']:
                    if not parameter.requires_grad:
                        continue

                    parameter_before = self._params_before_step[id(parameter)]
                    m_hat, v_hat = self._bias_corrected_adam_state(optimizer, group, parameter)

                    # Reconstruct ordinary Adam's update from its first and second moments.
                    expected_update = -learning_rate * m_hat / (v_hat.sqrt() + adam_epsilon)
                    actual_update = parameter.detach() - parameter_before
                    residual = actual_update - expected_update

                    # Accumulate in float64 so the diagnostic measures optimizer drift,
                    # rather than error introduced by the telemetry reduction itself.
                    actual_sq += actual_update.double().pow(2).sum().item()
                    expected_sq += expected_update.double().pow(2).sum().item()
                    residual_sq += residual.double().pow(2).sum().item()
                    max_abs_residual = max(max_abs_residual, residual.abs().max().item())
                    count += parameter.numel()

        if count == 0:
            return

        row['adam_actual_update_rms'] = (actual_sq / count) ** 0.5
        row['adam_expected_update_rms'] = (expected_sq / count) ** 0.5
        expected_scale = max(expected_sq, torch.finfo(torch.float64).tiny)
        row['adam_identity_relative_residual'] = (residual_sq / expected_scale) ** 0.5
        row['adam_identity_max_abs_residual'] = max_abs_residual

    def _build_optimizer_metadata(self, trainer) -> dict:
        optimizer = trainer.optimizer
        base_optimizer = getattr(optimizer, 'original_optimizer', optimizer)
        model = trainer._unwrap_model()
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        training_dataset = trainer.datamodule.get_dataloader('train').dataset
        training_dataset_size = len(training_dataset)

        inventory, occurrences = self._parameter_inventory(optimizer, parameter_names)

        # Coverage errors identify missing, duplicated, or accidentally frozen coordinates.
        coverage = self._parameter_coverage(model, parameter_names, occurrences)
        privacy = self._privacy_metadata(trainer, optimizer, training_dataset_size)

        return {
            'optimizer_wrapper_class': self._class_name(optimizer),
            'optimizer_class': self._class_name(base_optimizer),
            'parameter_groups': self._parameter_groups(optimizer),
            'parameter_inventory': inventory,
            'trainable_parameter_count': sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            'optimizer_parameter_count': sum(parameter.numel() for group in optimizer.param_groups for parameter in group['params']),
            **coverage,
            'world_size': self._world_size,
            'training_dataset_size': training_dataset_size,
            **privacy,
            'max_grad_norm': float(optimizer.max_grad_norm),
            'normalized_clipping': optimizer.normalize_clipping,
            'loss_reduction': optimizer.loss_reduction,
            'secure_mode': optimizer.secure_mode,
            'adam_identity_tolerance': self.adam_identity_tolerance,
        }

    def _parameter_inventory(self, optimizer, parameter_names: dict) -> tuple[list, dict]:
        """Describe every optimizer coordinate and count group membership."""
        inventory = []
        occurrences = {}

        # Preserve group and in-group positions so the complete optimizer routing is auditable.
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group['params']):
                parameter_id = id(parameter)
                occurrences[parameter_id] = occurrences.get(parameter_id, 0) + 1
                inventory.append(
                    {
                        'name': parameter_names.get(parameter_id, f'<unnamed:{parameter_id}>'),
                        'shape': list(parameter.shape),
                        'numel': parameter.numel(),
                        'dtype': str(parameter.dtype),
                        'requires_grad': bool(parameter.requires_grad),
                        'group_index': group_index,
                        'parameter_index': parameter_index,
                    }
                )
        return inventory, occurrences

    @staticmethod
    def _parameter_coverage(model, parameter_names: dict, occurrences: dict) -> dict:
        """Verify that optimizer groups partition the trainable parameters exactly once."""
        trainable = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}

        # Compare model ownership with optimizer ownership in both directions.
        missing = sorted(name for parameter_id, name in trainable.items() if occurrences.get(parameter_id, 0) == 0)
        duplicated = sorted(name for parameter_id, name in trainable.items() if occurrences.get(parameter_id, 0) > 1)
        non_trainable = sorted(
            parameter_names.get(parameter_id, f'<unnamed:{parameter_id}>')
            for parameter_id in occurrences
            if parameter_id not in trainable
        )
        return {
            'all_trainable_parameters_covered_once': not (missing or duplicated),
            'missing_trainable_parameters': missing,
            'duplicated_trainable_parameters': duplicated,
            'non_trainable_optimizer_parameters': non_trainable,
        }

    def _privacy_metadata(self, trainer, optimizer, training_dataset_size: int) -> dict:
        """Bind accountant noise to the global logical-batch coordinate used in analysis."""
        expected_batch_size = float(optimizer.expected_batch_size)

        # Opacus stores the expected batch for one worker; analyses use the global batch.
        effective_batch_size = expected_batch_size * self._world_size
        noise_multiplier = float(optimizer.noise_multiplier)
        return {
            'noise_multiplier': noise_multiplier,
            'expected_batch_size_per_worker': expected_batch_size,
            'effective_batch_size': effective_batch_size,
            'sample_rate': effective_batch_size / training_dataset_size,
            'normalized_noise': noise_multiplier / effective_batch_size,
            'target_epsilon': float(trainer.target_epsilon),
            'target_delta': float(trainer.target_delta),
        }

    def _parameter_groups(self, optimizer) -> list[dict]:
        """Serialize group hyperparameters while excluding live parameter tensors."""
        # The parameter inventory records tensors separately; groups retain only settings.
        return [{key: value for key, value in group.items() if key != 'params'} for group in optimizer.param_groups]

    @staticmethod
    def _class_name(value: Any) -> str:
        cls = type(value)
        return f'{cls.__module__}.{cls.__qualname__}'

    def _update_terminal_metadata(self, trainer) -> None:
        self._optimizer_metadata['completed_optimizer_steps'] = len(self._rows)
        maximum_residual = max(row['adam_identity_relative_residual'] for row in self._rows)
        self._optimizer_metadata['adam_identity_max_relative_residual'] = maximum_residual
        self._optimizer_metadata['adam_identity_passed'] = maximum_residual <= self.adam_identity_tolerance
        self._optimizer_metadata['achieved_epsilon'] = float(trainer.get_epsilon())

        allocated, reserved = self._peak_cuda_memory()
        self._optimizer_metadata['peak_cuda_memory_allocated_bytes'] = allocated
        self._optimizer_metadata['peak_cuda_memory_reserved_bytes'] = reserved

    def _peak_cuda_memory(self) -> tuple[int | None, int | None]:
        if self._device.type != 'cuda' or not torch.cuda.is_available():
            return None, None

        # Report the largest worker peak because rank-zero memory alone can hide imbalance.
        values = torch.tensor(
            [
                torch.cuda.max_memory_allocated(self._device),
                torch.cuda.max_memory_reserved(self._device),
            ],
            dtype=torch.int64,
            device=self._device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
        return int(values[0].item()), int(values[1].item())

    def on_train_end(self, trainer, *args, **kwargs):
        if not self._rows:
            return

        # All ranks participate in peak-memory reduction; only rank zero writes artifacts.
        self._update_terminal_metadata(trainer)
        if not self._is_global_zero():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.log_dir / 'optimizer_stats.csv'

        # The CSV is the per-step analysis surface; metadata binds run-level semantics.
        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            writer.writerows(self._rows)

        metadata_path = self.log_dir / 'optimizer_metadata.json'
        with metadata_path.open('w') as f:
            json.dump(self._optimizer_metadata, f, indent=2, sort_keys=True)
            f.write('\n')

        log.info(f'Optimizer stats written to {out_path}')
        log.info(f'Optimizer metadata written to {metadata_path}')
