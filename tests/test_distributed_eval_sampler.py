import pytest

torch = pytest.importorskip('torch')

from dpdl.datamodules import DistributedEvalSampler


def _shards(dataset_len, world_size):
    dataset = list(range(dataset_len))
    return [
        list(DistributedEvalSampler(dataset, num_replicas=world_size, rank=rank))
        for rank in range(world_size)
    ]


@pytest.mark.parametrize(
    'dataset_len, world_size',
    [
        (8, 2),    # evenly divisible
        (7, 2),    # non-divisible: 4 / 3
        (10, 3),   # non-divisible: 4 / 3 / 3
        (3, 8),    # fewer samples than ranks: some ranks get nothing
        (0, 4),    # empty dataset
        (1, 1),    # single process
    ],
)
def test_partition_is_exact_and_disjoint(dataset_len, world_size):
    shards = _shards(dataset_len, world_size)

    # Every sample is covered exactly once: no duplication, no dropping
    covered = [idx for shard in shards for idx in shard]
    assert sorted(covered) == list(range(dataset_len))
    assert len(covered) == dataset_len  # no padding-induced duplicates

    # Shards are disjoint
    seen = set()
    for shard in shards:
        assert not (seen & set(shard))
        seen.update(shard)

    # Load is balanced: shard sizes differ by at most one
    sizes = [len(shard) for shard in shards]
    assert max(sizes) - min(sizes) <= 1

    # __len__ agrees with what iteration yields
    for rank in range(world_size):
        sampler = DistributedEvalSampler(list(range(dataset_len)), num_replicas=world_size, rank=rank)
        assert len(sampler) == len(list(sampler))


def test_fewer_samples_than_ranks_leaves_tail_ranks_empty():
    # 3 samples, 8 ranks: ranks 0-2 get one each, ranks 3-7 get none
    shards = _shards(dataset_len=3, world_size=8)
    assert shards == [[0], [1], [2], [], [], [], [], []]


def test_defaults_to_single_process_without_process_group():
    # No torch.distributed init -> world_size 1, rank 0, whole dataset on one shard
    sampler = DistributedEvalSampler(list(range(5)))
    assert list(sampler) == [0, 1, 2, 3, 4]
