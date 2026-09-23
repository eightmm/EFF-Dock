"""Count released model parameters on CPU and summarize existing throughput records."""

import hashlib
import json
from pathlib import Path

import torch

from effdock.confidence import load_pose_confidence_model
from effdock.inference.docking import load_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/results/paper"


def main():
    device = torch.device("cpu")
    docking = ROOT / "weights/effdock_docking_early_time_t0p10_50k.pt"
    confidence = ROOT / "weights/effdock_confidence_s50_raw_refined_u70k.pt"
    model, _, _ = load_model(ROOT / "configs/train.yaml", docking, device)
    selector, _ = load_pose_confidence_model(confidence, device)
    models = [
        dict(
            name=name,
            parameters=sum(p.numel() for p in instance.parameters()),
            checkpoint=path.relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for name, instance, path in (
            ("Docking", model, docking),
            ("Confidence", selector, confidence),
        )
    ]
    timing = DATA / "runtime_comparison.json"
    rows = [
        dict(
            dataset=r["dataset"],
            seconds_per_complex=r["pipeline_mean_s"],
            repeat_sd_seconds=r["pipeline_repeat_sd_s"],
            amortized_poses_per_second=r["poses_per_complex"] / r["pipeline_mean_s"],
            devices=r["devices"],
        )
        for r in json.loads(timing.read_text())
        if r["arm"] in ("unguided_n100_s10", "temporal_n100_s10")
    ]
    result = dict(
        models=models,
        throughput=rows,
        timing_sha256=hashlib.sha256(timing.read_bytes()).hexdigest(),
        training_wall_time_seconds=None,
        training_wall_time_note="A complete, interruption-aware wall-time ledger for all precursor and fine-tuning runs was not audited. Step counts are not elapsed training time.",
        throughput_note="Inverse of existing mean per-complex pipeline time, multiplied by 100 poses. Includes sampling, refinement and confidence evaluation of both banks; excludes separate official PB assessment. Not a measured saturated service throughput or a new timing experiment.",
    )
    (DATA / "evidence/model_cost.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
