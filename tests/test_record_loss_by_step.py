import types

import pytest

torch = pytest.importorskip('torch')

from dpdl.callbacks.record_losses import RecordTrainLossByStepCallback


def _fake_trainer():
    return types.SimpleNamespace(device=torch.device('cpu'))


def _patch_distributed(monkeypatch, world_size, all_reduce):
    monkeypatch.setattr(torch.distributed, 'is_available', lambda: True)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda: world_size)
    monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)


def test_mean_across_ranks(monkeypatch):
    # Simulate 2-rank group and make peer contribute 5.0 to the summed loss
    _patch_distributed(monkeypatch, world_size=2, all_reduce=lambda t, *a, **k: t.add_(5.0))

    result = RecordTrainLossByStepCallback._mean_across_ranks(1.0, torch.device('cpu'))

    # result must be the mean sum / world_size
    assert result == pytest.approx((1.0 + 5.0) / 2)


def test_mean_across_ranks_single_process(monkeypatch):
    # For a single rank, make sure that all_reduce is not called and the input is returned unchanged
    def _fail(*args, **kwargs):
        raise AssertionError('all_reduce must not run for world_size == 1')

    _patch_distributed(monkeypatch, world_size=1, all_reduce=_fail)

    assert RecordTrainLossByStepCallback._mean_across_ranks(2.0, torch.device('cpu')) == 2.0


def test_callback_reduced_loss_per_step(monkeypatch, tmp_path):
    # The callback must store the global loss and advance the step counter once per logical batch.
    _patch_distributed(monkeypatch, world_size=2, all_reduce=lambda t, *a, **k: t.add_(4.0))

    callback = RecordTrainLossByStepCallback(log_dir=str(tmp_path))
    trainer = _fake_trainer()

    callback.on_train_start(trainer)
    for batch_idx, loss in enumerate([1.0, 2.0]):
        callback.on_train_batch_end(trainer, batch_idx, None, loss)

    assert callback.train_losses == [
        {'step': 1, 'train_loss': pytest.approx((1.0 + 4.0) / 2)},
        {'step': 2, 'train_loss': pytest.approx((2.0 + 4.0) / 2)},
    ]
