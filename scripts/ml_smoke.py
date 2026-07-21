"""CPU-only loss/backward/checkpoint smoke for EFF-Dock."""

import tempfile
from pathlib import Path

import torch
from torch import nn

from effdock.checkpoint import load_checkpoint_file, load_portable_model_state
from effdock.training.losses import flow_matching_loss


class SmokeHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 6)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.projection(features)
        return output[:, :3], output[:, 3:]


def main() -> int:
    torch.manual_seed(42)
    model = SmokeHead()
    features = torch.randn(4, 8)
    v_pred, omega_pred = model(features)
    losses = flow_matching_loss(
        v_pred,
        omega_pred,
        torch.randn_like(v_pred),
        torch.randn_like(omega_pred),
        torch.tensor([3, 1, 4, 2]),
    )
    losses["loss"].backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        checkpoint = load_checkpoint_file(path)
        restored = SmokeHead()
        load_portable_model_state(restored, checkpoint["model_state_dict"])
    print("ml_smoke: forward/loss/backward/safe-checkpoint ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
