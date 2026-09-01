#!/usr/bin/env python3
"""Run one confidence forward/backward for a sealed large-graph bank sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from effdock.confidence.dataset import to_device
from effdock.confidence.losses import pose_confidence_loss
from effdock.confidence.runtime import load_pose_confidence_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-key", required=True)
    parser.add_argument("--num-poses", type=int, required=True)
    args = parser.parse_args()
    if args.num_poses < 1:
        raise ValueError("num-poses must be positive")

    manifest = json.loads(args.bank_manifest.read_text())
    record = next(
        row for row in manifest["records"] if row["sample_key"] == args.sample_key
    )
    shard = torch.load(record["pt_path"], map_location="cpu", weights_only=True)
    pose_count = min(args.num_poses, int(shard["pose_atom_coords"].shape[0]))
    indices = torch.arange(pose_count, dtype=torch.long)
    item = {
        "pid": args.sample_key,
        "graph": shard["graph_centered"],
        "pose_atom_coords": shard["pose_atom_coords"].index_select(0, indices),
        "h_lig_node": shard["h_lig_node"].index_select(0, indices),
        "lig_node_type": shard["lig_node_type"],
        "fragment_id": shard["fragment_id"],
        "frag_sizes": shard["frag_sizes"],
        "atom_disp": shard["atom_disp"].index_select(0, indices),
        "pose_rmsd": shard["pose_rmsd"].index_select(0, indices),
    }

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, checkpoint = load_pose_confidence_model(args.checkpoint, device)
    model.train()
    item = to_device(item, device)
    output = model(item)
    loss_kwargs = dict(checkpoint.get("args", {}).get("loss") or {})
    losses = pose_confidence_loss(output, item, **loss_kwargs)
    losses["loss"].backward()
    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "sample_key": args.sample_key,
                "num_poses": pose_count,
                "graph_nodes": int(item["graph"]["node_coords"].shape[0]),
                "graph_edges": int(item["graph"]["edge_index"].shape[1]),
                "loss": float(losses["loss"].detach()),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
