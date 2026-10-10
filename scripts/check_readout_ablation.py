"""GPU check: both readouts are rotation-equivariant on real validation batches."""

import copy
import sys

import torch
import yaml

from effdock.geometry.se3 import matrix_to_quaternion, quaternion_multiply
from effdock.models.effdock import EFFDock
from effdock.training.trainer import Trainer


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    cfg["logging"]["use_wandb"] = False
    cfg["logging"]["output_dir"] = "outputs/readout-ablation/check"
    cfg["data"]["num_workers"] = 0
    trainer = Trainer(copy.deepcopy(cfg))
    batch = next(iter(trainer.val_loader))
    device = trainer.device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    torch.manual_seed(0)
    Q = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    Q = Q.to(device=device, dtype=batch["node_coords"].dtype)
    rotated = dict(batch)
    rotated["node_coords"] = batch["node_coords"] @ Q.T
    rotated["T_frag"] = batch["T_frag"] @ Q.T
    q_Q = matrix_to_quaternion(Q.unsqueeze(0)).expand_as(batch["q_frag"])
    rotated["q_frag"] = quaternion_multiply(q_Q, batch["q_frag"])
    for readout in ("newton_euler", "fragment_direct"):
        kwargs = {k: v for k, v in cfg["model"].items() if k not in ("model_type", "readout")}
        model = EFFDock(**kwargs, readout=readout).to(device).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.02 * torch.randn_like(p))
            a, b = model(batch), model(rotated)
        n_params = sum(p.numel() for p in model.parameters())
        for key in ("v_pred", "omega_pred"):
            err = (b[key] - a[key] @ Q.T).norm() / a[key].norm().clamp(min=1e-8)
            print(f"{readout} params={n_params} {key} rel_err={err.item():.2e} norm={a[key].norm().item():.3e}")
            assert err < 1e-3, (readout, key, err.item())
    print("PASS")


if __name__ == "__main__":
    main(sys.argv[1])
