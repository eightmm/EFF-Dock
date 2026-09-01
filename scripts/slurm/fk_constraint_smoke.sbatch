#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-smoke
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:a5000:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SMOKE_ROOT:?EFFDOCK_FK_SMOKE_ROOT is required}
output_prefix=outputs/benchmarks/fk_constraint_smoke_runs
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe FK smoke output root: $output_root" >&2
  exit 2
fi
mkdir -p "$output_prefix"
if ! mkdir "$output_root"; then
  echo "refusing to reuse FK smoke output root: $output_root" >&2
  exit 2
fi

for required in \
  .venv/bin/eff-dock \
  configs/train.yaml \
  weights/effdock_geometry_ft_100k_best.pt \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  data/external_test/astex_reference_pocket_centers.json \
  data/external_benchmarks/data/astex_diverse_set; do
  if [[ ! -e "$required" ]]; then
    echo "missing FK smoke input: $required" >&2
    exit 2
  fi
done

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "FK smoke requires exactly one visible CUDA device" >&2
  exit 2
fi
printf 'protocol=EFFDOCK-FK-CONSTRAINT-SMOKE-V1 output_root=%s\n' "$output_root"
printf 'gpu=%s\n' "$gpu_inventory"

export PYTHONHASHSEED=0
common_args=(
  --dataset astex
  --data-dir data/external_benchmarks/data/astex_diverse_set
  --external-dir data/external_test
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --require-full-ligand-atom-mapping
  --pocket-centers data/external_test/astex_reference_pocket_centers.json
  --checkpoint weights/effdock_geometry_ft_100k_best.pt
  --config configs/train.yaml
  --device cuda
  --protocol-id EFFDOCK-FK-CONSTRAINT-SMOKE-V1
  --num-samples 16
  --num-steps 10
  --prior-pool-size 16
  --sigma 0.5
  --time-schedule uniform
  --schedule-power 1.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
  --seed 42
  --only-id 1jje
  --expected-discovered-count 85
  --require-complete-success
  --no-confidence
  --save-selected-poses
  --refine none
  --unified-guidance-receptor-policy geometry_only
)

.venv/bin/eff-dock evaluate \
  "${common_args[@]}" \
  --output-dir "$output_root/baseline" \
  --run-name fk-constraint-smoke-baseline

.venv/bin/eff-dock evaluate \
  "${common_args[@]}" \
  --output-dir "$output_root/fk" \
  --run-name fk-constraint-smoke-beta0p01 \
  --fk-constraint-beta 0.01 \
  --fk-resample-times 0.3,0.6,0.8 \
  --fk-resample-method systematic

.venv/bin/python - "$output_root" <<'PY'
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem


root = Path(sys.argv[1])
arms = {
    "baseline": (root / "baseline", "fk-constraint-smoke-baseline"),
    "fk": (root / "fk", "fk-constraint-smoke-beta0p01"),
}


def load_arm(name: str) -> tuple[dict, dict[str, str]]:
    output_dir, run_name = arms[name]
    summary_path = output_dir / f"{run_name}.summary.json"
    csv_path = output_dir / f"{run_name}.csv"
    summary = json.loads(summary_path.read_text())
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if summary.get("num_success") != 1 or summary.get("num_failed") != 0:
        raise RuntimeError(f"{name} did not complete exactly one successful complex")
    if len(rows) != 1 or rows[0].get("id") != "1jje":
        raise RuntimeError(f"{name} output must contain exactly the 1jje row")
    if int(summary.get("model_pose_step_budget", -1)) != 160:
        raise RuntimeError(f"{name} model-pose-step budget must be 160")
    return summary, rows[0]


baseline_summary, baseline = load_arm("baseline")
fk_summary, fk_row = load_arm("fk")
if baseline.get("prior_pool_sha256") != fk_row.get("prior_pool_sha256"):
    raise RuntimeError("baseline and FK prior pools differ")
if baseline.get("guidance_mode") != "none":
    raise RuntimeError("baseline guidance mode must be none")
if fk_row.get("guidance_mode") != "feynman_kac_constraint_resampling":
    raise RuntimeError("FK row did not use the FK constraint-resampling mode")

diagnostics = json.loads(fk_row["fk_diagnostics_json"])
events = diagnostics.get("events", [])
if diagnostics.get("num_resampling_events") != 3 or len(events) != 3:
    raise RuntimeError("FK smoke must record exactly three resampling events")
requested = (0.3, 0.6, 0.8)
ess_fractions: list[float] = []
for index, (event, expected_time) in enumerate(zip(events, requested)):
    if event.get("event_index") != index:
        raise RuntimeError("FK event indices are not contiguous")
    if not math.isclose(
        float(event.get("requested_time")),
        expected_time,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise RuntimeError("FK requested resampling schedule drifted")
    actual_time = float(event.get("actual_time"))
    if not math.isclose(actual_time, expected_time, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("uniform-grid FK event did not run at its requested time")
    ess_fraction = float(event.get("ess_fraction"))
    if not math.isfinite(ess_fraction) or not 0.0 < ess_fraction <= 1.0:
        raise RuntimeError("FK ESS fraction is invalid")
    ess_fractions.append(ess_fraction)
    for key in (
        "potential_min",
        "potential_median",
        "potential_max",
        "delta_min",
        "delta_median",
        "delta_max",
        "max_group_weight",
    ):
        if not math.isfinite(float(event.get(key))):
            raise RuntimeError(f"FK event field {key} is non-finite")
    if not event.get("source_index_sha256"):
        raise RuntimeError("FK event is missing its source-index identity")

final_unique = int(diagnostics.get("final_unique_initial_ancestors"))
if not 1 <= final_unique <= 16:
    raise RuntimeError("FK final genealogy is outside the particle budget")


def load_sdf(path_text: str) -> list[Chem.Mol]:
    molecules = [mol for mol in Chem.SDMolSupplier(path_text, removeHs=False) if mol]
    if len(molecules) != 16:
        raise RuntimeError(f"expected 16 poses in {path_text}, got {len(molecules)}")
    return molecules


baseline_mols = load_sdf(baseline["all_poses_sdf"])
fk_mols = load_sdf(fk_row["all_poses_sdf"])
coordinate_errors: list[float] = []
ancestors: list[int] = []
for fk_mol in fk_mols:
    if not fk_mol.HasProp("fk_initial_sample_index"):
        raise RuntimeError("FK SDF record is missing initial-particle ancestry")
    ancestor = int(fk_mol.GetProp("fk_initial_sample_index"))
    if not 0 <= ancestor < 16:
        raise RuntimeError("FK SDF ancestry is out of range")
    ancestors.append(ancestor)
    fk_coords = np.asarray(fk_mol.GetConformer().GetPositions(), dtype=np.float64)
    baseline_coords = np.asarray(
        baseline_mols[ancestor].GetConformer().GetPositions(),
        dtype=np.float64,
    )
    if fk_coords.shape != baseline_coords.shape:
        raise RuntimeError("baseline/FK atom shapes differ")
    coordinate_errors.append(float(np.linalg.norm(fk_coords - baseline_coords, axis=1).max()))

if len(set(ancestors)) != final_unique:
    raise RuntimeError("SDF ancestry and FK diagnostics disagree")
max_coordinate_error = max(coordinate_errors)
if not math.isfinite(max_coordinate_error) or max_coordinate_error > 5.0e-3:
    raise RuntimeError(
        "no-jitter FK outputs are not a numerical subset of baseline trajectories: "
        f"max atom error={max_coordinate_error:.6g} A"
    )

metrics = {
    "status": "passed",
    "num_resampling_events": 3,
    "min_ess_fraction": min(ess_fractions),
    "max_ess_fraction": max(ess_fractions),
    "final_unique_initial_ancestors": final_unique,
    "final_unique_initial_ancestor_fraction": final_unique / 16.0,
    "max_baseline_subset_atom_error_angstrom": max_coordinate_error,
    "prior_pool_hash_equal": True,
}
(root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(json.dumps(metrics, sort_keys=True))
PY
