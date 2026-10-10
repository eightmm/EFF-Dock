"""Per-complex internal-validation rollout RMSD for a readout-ablation checkpoint.

Repeats the trainer's rollout exactly (EMA model, same seeds and settings) and
writes one RMSD per validation complex so that arms can be compared pairwise.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml

from effdock.evaluation.metrics import ligand_rmsd
from effdock.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg["logging"]["use_wandb"] = False
    cfg["logging"]["output_dir"] = str(args.output.parent / "rollout_tmp")
    trainer = Trainer(copy.deepcopy(cfg))
    trainer.load_checkpoint(str(args.checkpoint))
    model = trainer._eval_model()
    model.train(False)
    lcfg, dcfg = cfg["logging"], cfg["data"]
    val_ds = trainer.val_loader.dataset
    rows = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            data = val_ds[i]
            _, pred, true = trainer._rollout_single_unified(
                model, data, sigma=dcfg.get("prior_sigma", 5.0),
                num_steps=lcfg.get("rollout_steps", 20),
                time_schedule=lcfg.get("rollout_time_schedule", "uniform"),
                schedule_power=lcfg.get("rollout_schedule_power", 3.0),
                seed=cfg["training"].get("seed", 42) + i)
            rows.append({"index": i, "id": str(data.get("id", data.get("sample_id", i))),
                         "rmsd": ligand_rmsd(pred, true).item()})
    rmsd = torch.tensor([r["rmsd"] for r in rows])
    summary = {"checkpoint": str(args.checkpoint), "step": trainer.global_step, "n": len(rows),
               "success_2A": (rmsd < 2).float().mean().item(), "median": rmsd.median().item(),
               "rows": rows}
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
