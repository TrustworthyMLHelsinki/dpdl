import torch

from dpdl.callbacks.checkpoint import CheckpointCallback


class _Metrics:
    def compute(self):
        return {}

    def reset(self):
        pass


class _Trainer:
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        self.model.valid_metrics = _Metrics()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.validation_calls = 0

    def _unwrap_model(self):
        return self.model

    def validate(self, enable_callbacks=False):
        assert enable_callbacks is False
        self.validation_calls += 1
        return 0.0, self.model.valid_metrics.compute()


def test_saves_model_and_optimizer_at_exact_steps(tmp_path, monkeypatch):
    trainer = _Trainer()
    callback = CheckpointCallback(
        log_dir=tmp_path,
        checkpoint_steps=[0, 1, 13, 52, 104],
        device=torch.device('cpu'),
    )
    monkeypatch.setattr(callback, '_is_global_zero', lambda: True)
    callback.on_train_start(trainer)

    for step in range(1, 105):
        trainer.optimizer.zero_grad()
        trainer.model(torch.ones(1, 2)).sum().backward()
        trainer.optimizer.step()
        callback.on_train_batch_end(trainer, step - 1, None, 1.0)

    callback.on_train_end(trainer)

    checkpoint_dir = tmp_path / 'checkpoints'
    assert sorted(path.name for path in checkpoint_dir.glob('checkpoint_step_*.pt')) == [
        'checkpoint_step_0.pt',
        'checkpoint_step_1.pt',
        'checkpoint_step_104.pt',
        'checkpoint_step_13.pt',
        'checkpoint_step_52.pt',
    ]
    assert not list(checkpoint_dir.glob('final_checkpoint_step_*.pt'))
    assert trainer.validation_calls == 1

    initial_checkpoint = torch.load(checkpoint_dir / 'checkpoint_step_0.pt', weights_only=True)
    assert initial_checkpoint['optimizer_state_dict']['state'] == {}
    checkpoint = torch.load(checkpoint_dir / 'checkpoint_step_52.pt', weights_only=True)
    assert checkpoint['step'] == 52
    assert set(checkpoint) == {'step', 'model_state_dict', 'optimizer_state_dict'}
    assert checkpoint['optimizer_state_dict']['state']


def test_non_zero_rank_checkpoint_validation(tmp_path, monkeypatch):
    trainer = _Trainer()
    callback = CheckpointCallback(
        log_dir=tmp_path,
        checkpoint_steps=[0, 1, 13, 52, 104],
        device=torch.device('cpu'),
    )
    monkeypatch.setattr(callback, '_is_global_zero', lambda: False)
    callback.on_train_start(trainer)

    for step in range(1, 105):
        callback.on_train_batch_end(trainer, step - 1, None, 1.0)

    callback.on_train_end(trainer)

    checkpoint_dir = tmp_path / 'checkpoints'

    assert callback.global_step == 104

    # Validation happens once, also for non-zero ranks
    assert trainer.validation_calls == 1

    # Only rank 0 writes output files
    assert not list(checkpoint_dir.glob('*.pt'))
    assert not list(checkpoint_dir.glob('*_metrics.json'))


def test_non_zero_rank_interval_and_final(tmp_path, monkeypatch):
    trainer = _Trainer()
    callback = CheckpointCallback(
        log_dir=tmp_path,
        checkpoint_step_interval=10,
        device=torch.device('cpu'),
    )
    monkeypatch.setattr(callback, '_is_global_zero', lambda: False)
    callback.on_train_start(trainer)

    for step in range(1, 26):
        callback.on_train_batch_end(trainer, step - 1, None, 1.0)

    callback.on_train_end(trainer)

    checkpoint_dir = tmp_path / 'checkpoints'
    assert callback.global_step == 25

    # Two interval boundaries (steps 10 and 20) plus the final validate():
    # three validations for all ranks, no files, due to non-zero rank
    assert trainer.validation_calls == 3
    assert not list(checkpoint_dir.glob('*.pt'))
    assert not list(checkpoint_dir.glob('*_metrics.json'))
