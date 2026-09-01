#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-pb-smoke
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
# shellcheck source=guidance_eta_sweep_confidence_standalone_profile.sh
source scripts/slurm/guidance_eta_sweep_confidence_standalone_profile.sh

verify_manifest() {
  local label=$1
  local manifest=$2
  local expected_sha256=$3
  local actual_sha256
  if [[ ! -f "$manifest" ]]; then
    echo "missing frozen $label manifest: $manifest" >&2
    exit 2
  fi
  read -r actual_sha256 _ < <(sha256sum "$manifest")
  if [[ "$actual_sha256" != "$expected_sha256" ]] \
    || ! sha256sum --check --quiet "$manifest"; then
    echo "frozen $label manifest verification failed" >&2
    exit 2
  fi
}

execution_manifest=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST:?EFFDOCK_STANDALONE_EXECUTION_MANIFEST is required}
execution_manifest_sha256=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256:?EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256 is required}
frozen_inputs_manifest=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST:?EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST is required}
frozen_inputs_manifest_sha256=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256:?EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256 is required}
verify_manifest execution "$execution_manifest" "$execution_manifest_sha256"
verify_manifest inputs "$frozen_inputs_manifest" "$frozen_inputs_manifest_sha256"

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe standalone output root: $output_root" >&2
  exit 2
fi
sampling_dir="$output_root/smoke/raw"
official_root="$output_root/smoke/posebusters_official"
audit="$output_root/audit/smoke.json"
run_name="${run_name_prefix}-posebusters-n100-s10-${eta_tags[$posebusters_smoke_eta_index]}"

if [[ ! -f "$audit" ]]; then
  echo "missing passed standalone smoke audit: $audit" >&2
  exit 2
fi
.venv/bin/python -c \
  'import json,sys; a=json.load(open(sys.argv[1])); ok=a.get("protocol_id")==sys.argv[2] and a.get("audit_contract")==sys.argv[3] and a.get("schema_version")==sys.argv[4] and a.get("mode")=="fresh_one_pass_characterization" and a.get("run_scope")=="smoke" and a.get("status")=="passed" and a.get("parent_compared") is False and a.get("deterministic_replay_claim") is False; raise SystemExit(0 if ok else "standalone smoke audit did not pass the frozen parent-free contract")' \
  "$audit" "$protocol_id" "$audit_contract" "$audit_schema_version"

mkdir -p "$output_root/.task_reservations"
if ! mkdir "$output_root/.task_reservations/posebusters-smoke"; then
  echo "refusing to rerun or overwrite standalone PoseBusters smoke" >&2
  exit 2
fi

echo "protocol=$protocol_id run=$run_name selectors=confidence,confidence_filter"
for selector in confidence confidence_filter; do
  pose_dir="$sampling_dir/poses/$run_name/posebusters/$selector"
  output_dir="$official_root/$selector/$run_name"
  if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite standalone PoseBusters smoke output: $output_dir" >&2
    exit 2
  fi
  .venv/bin/python -m effdock.workflows.posebusters_report \
    --input-dir "$sampling_dir" \
    --num-shards 1 \
    --shard-index 0 \
    --run-name "$run_name" \
    --pose-dir "$pose_dir" \
    --output-dir "$output_dir" \
    --selector "$selector" \
    --expected-discovered-count 1 \
    --require-input-hashes \
    --require-complete-success
done

.venv/bin/python - "$sampling_dir" "$official_root" "$run_name" <<'PY'
import json
import math
import sys
from pathlib import Path

from effdock.workflows.guidance_budget_posebusters_report import (
    POSEBUSTERS_CONFIG,
    POSEBUSTERS_VERSION,
    VALIDITY_CHECKS,
    _read_csv_rows,
    _resolve_cell_csv,
)
from effdock.workflows.posebusters_report import (
    load_rows,
    require_posebusters_runtime_version,
    verify_posebusters_input_hashes,
)

sampling_dir = Path(sys.argv[1])
official_root = Path(sys.argv[2])
run_name = sys.argv[3]
selectors = ("confidence", "confidence_filter")
expected_id = "7b2c_tp7"
expected_rmsd_check = "rmsd_≤_2å"
tag = "shard-000-of-001"

if POSEBUSTERS_VERSION != "0.6.5" or require_posebusters_runtime_version() != "0.6.5":
    raise RuntimeError("PoseBusters smoke requires runtime version 0.6.5")
if POSEBUSTERS_CONFIG != "redock":
    raise RuntimeError("PoseBusters smoke requires the redock configuration")
if len(VALIDITY_CHECKS) != 27 or any(check.startswith("rmsd_") for check in VALIDITY_CHECKS):
    raise RuntimeError("frozen PoseBusters validity schema must contain 27 non-RMSD checks")

sampling_rows = load_rows(sampling_dir, run_name)
if len(sampling_rows) != 1 or sampling_rows[0].get("id") != expected_id:
    raise RuntimeError("PoseBusters smoke sampling input must be exactly 7b2c_tp7")
sampling_row = sampling_rows[0]

observed_rmsd_check = None
for selector in selectors:
    pose_dir = sampling_dir / "poses" / run_name / "posebusters" / selector
    verify_posebusters_input_hashes(
        sampling_row,
        pose_dir / f"{expected_id}.sdf",
        selector,
    )

    output_dir = official_root / selector / run_name
    summary_path = output_dir / f"{tag}.summary.json"
    expected_csv = output_dir / f"{tag}.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing PoseBusters smoke summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path}: summary must be a JSON object")

    exact = {
        "posebusters_version": "0.6.5",
        "config": "redock",
        "selector": selector,
        "input_hashes_verified": True,
        "num_input_hashes_verified": 1,
        "expected_discovered_count": 1,
        "require_complete_success": True,
        "num_discovered_total": 1,
        "num_assigned": 1,
        "num_success": 1,
        "num_failed": 0,
        "failures": [],
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{summary_path}: {key} must be {expected!r}, got {summary.get(key)!r}"
            )

    csv_path = _resolve_cell_csv(summary_path, summary.get("csv"), expected_csv)
    rows, rmsd_check = _read_csv_rows(
        csv_path,
        expected_rmsd_check=observed_rmsd_check,
    )
    if rmsd_check != expected_rmsd_check:
        raise ValueError(
            f"{csv_path}: excluded RMSD check must be {expected_rmsd_check!r}, "
            f"got {rmsd_check!r}"
        )
    observed_rmsd_check = rmsd_check
    if len(rows) != 1 or rows[0]["id"] != expected_id:
        raise ValueError(f"{csv_path}: expected exactly the 7b2c_tp7 smoke row")
    expected_valid_pct = 100.0 if rows[0]["posebusters_valid"] else 0.0
    reported_valid_pct = summary.get("posebusters_valid_pct")
    if reported_valid_pct is None or not math.isclose(
        float(reported_valid_pct),
        expected_valid_pct,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{summary_path}: posebusters_valid_pct disagrees with current CSV")

print(
    "standalone PoseBusters smoke passed: "
    "version=0.6.5 config=redock selectors=2 rows=1 validity_checks=27 "
    f"excluded_rmsd_check={observed_rmsd_check}"
)
PY
