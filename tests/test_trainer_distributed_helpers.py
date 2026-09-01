from __future__ import annotations

import pytest
import torch

from effdock.training.trainer import (
    DistributedSizeAwareSampler,
    _all_ranks_finite,
    _balanced_shard_indices,
    _require_finite_metrics,
)


class _CostDataset:
    def __init__(self, costs: list[float]) -> None:
        self.sample_costs = costs

    def __len__(self) -> int:
        return len(self.sample_costs)


def test_balanced_shard_indices_are_disjoint_complete_and_cost_balanced() -> None:
    dataset = _CostDataset([20.0, 15.0, 10.0, 8.0, 7.0, 5.0, 3.0, 2.0])
    shards = [
        _balanced_shard_indices(dataset, n_items=8, num_replicas=4, rank=rank) for rank in range(4)
    ]

    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(8))
    assert len(flattened) == len(set(flattened))
    totals = [sum(dataset.sample_costs[index] for index in shard) for shard in shards]
    assert max(totals) - min(totals) <= max(dataset.sample_costs)


def test_balanced_shard_indices_falls_back_to_stride_without_costs() -> None:
    dataset = object()
    assert _balanced_shard_indices(dataset, n_items=7, num_replicas=3, rank=1) == [1, 4]


def test_all_ranks_finite_single_process() -> None:
    assert _all_ranks_finite(torch.tensor(1.0), world_size=1)
    assert not _all_ranks_finite(torch.tensor(float("nan")), world_size=1)


def test_all_ranks_finite_uses_global_min(monkeypatch) -> None:
    def mark_remote_nonfinite(flag: torch.Tensor, *, op) -> None:
        assert op == torch.distributed.ReduceOp.MIN
        flag.zero_()

    monkeypatch.setattr(torch.distributed, "all_reduce", mark_remote_nonfinite)
    assert not _all_ranks_finite(torch.tensor(1.0), world_size=4)


def test_evaluation_metrics_fail_closed_on_nan() -> None:
    with pytest.raises(FloatingPointError, match="rollout smoke"):
        _require_finite_metrics(
            {"rollout/success_2A": 0.5, "rollout/rmsd_mean": float("nan")},
            context="rollout smoke",
            device=torch.device("cpu"),
            world_size=1,
        )


def test_size_aware_sampler_can_retain_full_split_with_rotating_padding() -> None:
    dataset = _CostDataset([float(index + 1) for index in range(67)])
    shards = []
    for rank in range(4):
        sampler = DistributedSizeAwareSampler(
            dataset,
            batch_size=4,
            num_replicas=4,
            rank=rank,
            seed=42,
            drop_last=False,
        )
        shards.append(list(sampler))

    flattened = [index for shard in shards for index in shard]
    assert set(flattened) == set(range(67))
    assert len(flattened) == 80
    assert {len(shard) for shard in shards} == {20}
