import torch
from torch import nn

from effdock.training.losses import flow_matching_loss


def test_cpu_training_contract_smoke() -> None:
    head = nn.Linear(8, 6)
    output = head(torch.randn(4, 8))
    losses = flow_matching_loss(
        output[:, :3],
        output[:, 3:],
        torch.randn(4, 3),
        torch.randn(4, 3),
        torch.tensor([3, 1, 4, 2]),
    )
    losses["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in head.parameters()
    )
