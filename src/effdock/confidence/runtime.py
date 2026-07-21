"""Runtime helpers for pose-confidence inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from effdock.checkpoint import load_portable_model_state
from effdock.confidence.features import extract_t1_ligand_irreps
from effdock.confidence.model import DockingGraphPoseConfidence


def _arg(args: dict[str, Any], key: str, default: Any) -> Any:
    value = args.get(key, default)
    return default if value is None else value


def load_pose_confidence_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    with torch.serialization.safe_globals([type(Path())]):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_type = ckpt.get("model_type", "ligand_protein_pose_confidence")
    if model_type != "docking_graph_pose_confidence":
        raise RuntimeError(
            f"unsupported confidence checkpoint model_type={model_type!r}; "
            "use the pocket+ligand-hidden DockingGraphPoseConfidence checkpoint"
        )
    model_cfg = dict(ckpt["model_cfg"])
    train_args = ckpt.get("args", {})
    model = DockingGraphPoseConfidence(
        **model_cfg,
        n_rbf=int(_arg(train_args, "n_rbf", 32)),
        contact_cutoff=float(_arg(train_args, "contact_cutoff", 5.0)),
        hidden=int(_arg(train_args, "hidden", 512)),
        num_layers=int(_arg(train_args, "num_layers", 4)),
        sh_lmax=int(_arg(train_args, "sh_lmax", 2)),
        cond_dim=int(_arg(train_args, "cond_dim", 128)),
        dropout=float(_arg(train_args, "dropout", 0.1)),
        pose_readout=str(_arg(train_args, "pose_readout", "global_pool")),
    ).to(device)
    load_portable_model_state(model, ckpt["state_dict"])
    model.eval()
    return model, ckpt


def sample_sigmas(results: list[dict[str, torch.Tensor]], default_sigma: float) -> torch.Tensor:
    return torch.tensor(
        [float(result.get("sigma", default_sigma)) for result in results], dtype=torch.float32
    )


def score_poses_with_confidence(
    confidence_model: torch.nn.Module,
    docking_model: torch.nn.Module,
    graph: dict[str, torch.Tensor],
    lig_data: dict[str, torch.Tensor],
    meta: dict[str, Any],
    poses: list[torch.Tensor],
    *,
    sigma: float | torch.Tensor,
    device: torch.device,
    hidden_dtype: torch.dtype = torch.float32,
) -> list[dict[str, float]]:
    pose_tensor = torch.stack([pose.detach().cpu().to(torch.float32) for pose in poses], dim=0)
    feats = extract_t1_ligand_irreps(
        docking_model,
        graph,
        lig_data,
        meta,
        pose_tensor,
        sigma=sigma,
        device=device,
        hidden_dtype=hidden_dtype,
    )
    graph_centered = {
        key: value.detach().cpu() for key, value in graph.items() if torch.is_tensor(value)
    }
    graph_centered["node_coords"] = (
        graph_centered["node_coords"].to(torch.float32)
        - meta["pocket_center"].to(torch.float32).cpu()
    )
    item = {
        "graph": graph_centered,
        "pose_atom_coords": pose_tensor,
        "h_lig_node": feats["h_lig_node"].to(torch.float32),
        "lig_node_type": feats["lig_node_type"].to(torch.long),
        "fragment_id": lig_data["fragment_id"].cpu().to(torch.long),
        "frag_sizes": lig_data["frag_sizes"].cpu().to(torch.long),
        "pocket_center_used": meta["pocket_center"].to(torch.float32).cpu(),
    }
    with torch.no_grad():
        out = confidence_model.forward_complex(item)
    pred_rmsd = out["pose_rmsd"].detach().cpu().to(torch.float32)
    pred_success_logit = out["pose_success_logit"].detach().cpu().to(torch.float32)
    pred_success = torch.sigmoid(pred_success_logit)
    atom_disp = torch.expm1(
        out["atom_disp_log1p"].detach().cpu().to(torch.float32).clamp(-2.0, 5.0)
    ).clamp_min(0.0)
    atom_ok = torch.sigmoid(out["atom_ok_logit"].detach().cpu().to(torch.float32))
    atom_rmsd = torch.sqrt(atom_disp.square().mean(dim=1).clamp_min(0.0))
    atom_q90 = torch.quantile(atom_disp, 0.9, dim=1)
    atom_ok_mean = atom_ok.mean(dim=1)
    return [
        {
            "confidence_rmsd": float(pred_rmsd[i]),
            "confidence_success_logit": float(pred_success_logit[i]),
            "confidence_success": float(pred_success[i]),
            "confidence_atom_rmsd": float(atom_rmsd[i]),
            "confidence_atom_q90": float(atom_q90[i]),
            "confidence_atom_ok": float(atom_ok_mean[i]),
        }
        for i in range(pose_tensor.shape[0])
    ]


__all__ = [
    "load_pose_confidence_model",
    "sample_sigmas",
    "score_poses_with_confidence",
]
