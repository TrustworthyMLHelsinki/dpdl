import contextlib
import logging
import os
import pathlib
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def is_global_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def seed_everything(seed) -> None:
    if not seed:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tensor_to_python_type(data):
    """
    Recursively converts all tensors in a dictionary or list to Python native types.
    """
    if isinstance(data, torch.Tensor):
        return data.item() if data.numel() == 1 else data.tolist()
    elif isinstance(data, dict):
        return {key: tensor_to_python_type(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [tensor_to_python_type(item) for item in data]
    else:
        return data

@contextlib.contextmanager
def safe_open(path: pathlib.Path, mode: str = 'w', **kwargs):
    """
    An `open` call with error checking that guarantees that the file handle
    is synced, flushed, and closed or error is thrown.

    NB: We had a couple silent failure when writing experiment files to lustre.
    """
    if not mode.startswith(('w','x')):
        raise ValueError(f"safe_open only supports write modes, got {mode!r}")

    try:
        fh = open(path, mode, **kwargs)
        yield fh
        fh.flush()
        os.fsync(fh.fileno())
    except OSError as e:
        log.error(f'Failed to write {path}: {e}')
        raise
    finally:
        fh.close()

def shift_and_flatten(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels_flat = shift_labels.view(-1)

    return shift_logits_flat, shift_labels_flat
