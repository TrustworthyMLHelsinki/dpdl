import pytest

torch = pytest.importorskip('torch')

from dpdl.callbacks.callback_factory import CallbackHandler
from dpdl.trainer import Trainer

class _FakeMetrics:
    def __init__(self, result):
        self._result = result
        self.reset_calls = 0
        self.compute_calls = 0
        self.updates = []

    def reset(self):
        self.reset_calls += 1

    def compute(self):
        self.compute_calls += 1
        return self._result

    def update(self, forward_output, target):
        self.updates.append((forward_output, target))


class _FakeModel:
    def __init__(self, metrics):
        self.valid_metrics = metrics
        self.test_metrics = metrics
        self.train_metrics = metrics
        self.mode = None

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'


class _FakeDataModule:
    def __init__(self, batches):
        self._batches = {'valid': batches, 'test': batches, 'train_eval': batches}

    def get_dataloader(self, name):
        return self._batches[name]


class _ScriptedAdapter:

    def __init__(self):
        self.processed = []

    def move_to_device(self, X, y=None):
        return (X, y)

    def forward(self, model, batch):
        return None

    def compute_loss(self, model, batch, forward_output, normalize_by=None):
        loss_value, _ = batch
        return torch.tensor(float(loss_value))

    def num_loss_items(self, model, batch, forward_output):
        _, weight = batch
        return int(weight)

    def update_metrics(self, model, batch, forward_output, metrics=None):
        self.processed.append(batch)
        if metrics is not None:
            metrics.update(forward_output, batch[1])


class _RecordingCallback:

    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.events.append((name, args, kwargs))

        return _record


def _make_trainer(batches, metrics_result=None):
    metrics = _FakeMetrics({'metric': 1.0} if metrics_result is None else metrics_result)
    recorder = _RecordingCallback()

    trainer = Trainer.__new__(Trainer)
    trainer.model = _FakeModel(metrics)
    trainer.adapter = _ScriptedAdapter()
    trainer.datamodule = _FakeDataModule(batches)
    trainer.callback_handler = CallbackHandler([recorder])
    trainer.device = torch.device('cpu')

    return trainer, metrics, recorder


def test_evaluate_reduces_to_per_item_mean():
    # Unequal weights differentiate the per-item mean from the mean of means
    batches = [(1.0, 2), (2.0, 3), (10.0, 5)]
    trainer, metrics, _ = _make_trainer(batches)

    loss, out_metrics = trainer.validate(enable_callbacks=False)

    expected_per_item = (1.0 * 2 + 2.0 * 3 + 10.0 * 5) / (2 + 3 + 5)
    assert loss == pytest.approx(expected_per_item)  # 5.8

    assert out_metrics == {'metric': 1.0}
    assert metrics.reset_calls == 1
    assert metrics.compute_calls == 1


def test_evaluate_matches_mean_of_means_when_weights_are_equal():
    # Sanity check: equal weights produce the same result as a mean of batch means
    batches = [(1.0, 4), (2.0, 4), (9.0, 4)]
    trainer, _, _ = _make_trainer(batches)

    loss, _ = trainer.validate(enable_callbacks=False)

    assert loss == pytest.approx((1.0 + 2.0 + 9.0) / 3)


def test_evaluate_returns_zero_loss_for_empty_dataloader():
    # sample_count = 0 shouldn't cause division by 0
    trainer, _, _ = _make_trainer(batches=[])

    loss, _ = trainer.validate(enable_callbacks=False)

    assert loss == 0.0


def test_evaluate_test_mode_uses_test_split():
    validation_batches = [(100.0, 1)]
    test_batches = [(4.0, 1), (6.0, 1)]

    trainer, validation_metrics, _ = _make_trainer(validation_batches)

    test_metrics = _FakeMetrics({'split': 'test'})
    trainer.datamodule._batches['test'] = test_batches
    trainer.model.test_metrics = test_metrics

    loss, out_metrics = trainer.test()

    assert loss == pytest.approx(5.0)
    assert out_metrics == {'split': 'test'}
    assert trainer.adapter.processed == test_batches

    assert test_metrics.reset_calls == 1
    assert test_metrics.compute_calls == 1
    assert validation_metrics.reset_calls == 0
    assert validation_metrics.compute_calls == 0


def test_non_zero_rank_no_callbacks(monkeypatch):
    batches = [(1.0, 2), (3.0, 2)]
    trainer, metrics, recorder = _make_trainer(batches)
    monkeypatch.setattr('dpdl.trainer.is_global_zero', lambda: False)

    loss, _ = trainer._evaluate('validation', epoch=0, enable_callbacks=True)

    assert loss == pytest.approx((1.0 * 2 + 3.0 * 2) / 4)  # 2.0

    assert len(trainer.adapter.processed) == 2
    assert len(metrics.updates) == 2
    assert metrics.reset_calls == 1
    assert metrics.compute_calls == 1

    # Callbacks are on rank 0, none is expected
    assert recorder.events == []


def test_rank_zero_callbacks_and_reduced_loss(monkeypatch):
    batches = [(1.0, 2), (3.0, 2)]
    trainer, _, recorder = _make_trainer(batches)
    monkeypatch.setattr('dpdl.trainer.is_global_zero', lambda: True)

    trainer._evaluate('validation', epoch=7, enable_callbacks=True)

    event_names = [name for name, _, _ in recorder.events]
    assert event_names == [
        'on_validation_epoch_start',
        'on_validation_batch_start',
        'on_validation_batch_end',
        'on_validation_batch_start',
        'on_validation_batch_end',
        'on_validation_epoch_end',
    ]

    # on_validation_epoch_end receives the reduced loss
    _, args, _ = recorder.events[-1]
    _, epoch, _, passed_loss = args
    assert epoch == 7
    assert passed_loss == pytest.approx(2.0)


def test_disabling_callbacks(monkeypatch):
    batches = [(2.0, 1)]
    trainer, _, recorder = _make_trainer(batches)
    monkeypatch.setattr('dpdl.trainer.is_global_zero', lambda: True)

    trainer._evaluate('validation', epoch=0, enable_callbacks=False)

    assert recorder.events == []
