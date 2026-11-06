import inspect
import sys
from pathlib import Path

import pytest

pytest.importorskip('torch')

from integration_utils import base_env, run_distributed

# The CLI only writes out rank 0's results, so the check has to run inside each rank
# resnet18 has BatchNorm buffers, so this also covers syncing before collective evaluation

EPOCHS = 2

@pytest.mark.integration
def test_non_private_rank_sync(
    tmp_path: Path, image_dataset_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    run_distributed(
        ['tests/test_integration_ddp_training.py', str(image_dataset_path), str(tmp_path)],
        base_env(),
        repo_root,
        nproc=2,
        timeout=180,
    )


def _check_ranks_in_sync(dataset_path: str, log_dir: str) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from dpdl.cli import cli
    from dpdl.configurationmanager import ConfigurationManager
    from dpdl.trainer import TrainerFactory

    torch.distributed.init_process_group('gloo')

    # start from the CLI defaults, like typer passes them to ConfigurationManager
    params = {
        name: param.default
        for name, param in inspect.signature(cli).parameters.items()
        if name != 'ctx'
    }
    params.update(
        command='train',
        device='cpu',
        dataset_name='local-image',
        dataset_path=dataset_path,
        model_name='resnet18',
        pretrained=False,
        privacy=False,
        epochs=EPOCHS,
        batch_size=4,
        physical_batch_size=4,
        num_workers=0,
        seed=42,
        split_seed=42,
        log_dir=log_dir,
        experiment_name='ddp-training-check',
    )

    trainer = TrainerFactory.get_trainer(ConfigurationManager(params))
    trainer.fit()

    # every rank must hold the same weights and buffers
    state = trainer._unwrap_model().model.state_dict()
    local = {name: tensor.double().sum().item() for name, tensor in state.items()}
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)

    diverged = [name for name in local if len({rank[name] for rank in gathered}) > 1]
    assert not diverged, f'Ranks diverged in {len(diverged)} tensors, e.g. {diverged[:5]}'

    sampler = trainer.get_dataloader('train').sampler
    assert sampler.epoch == EPOCHS - 1, (
        f'Train sampler epoch is {sampler.epoch}, expected {EPOCHS - 1}'
    )

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    _check_ranks_in_sync(*sys.argv[1:])
