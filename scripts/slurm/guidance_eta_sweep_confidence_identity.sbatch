#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-id
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

code_manifest=${EFFDOCK_CONFIDENCE_CODE_MANIFEST:?EFFDOCK_CONFIDENCE_CODE_MANIFEST is required}
expected_code_manifest_sha256=${EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256:?EFFDOCK_CONFIDENCE_CODE_MANIFEST_SHA256 is required}
read -r actual_code_manifest_sha256 _ < <(sha256sum "$code_manifest")
if [[ "$actual_code_manifest_sha256" != "$expected_code_manifest_sha256" ]] \
  || ! sha256sum --check --quiet "$code_manifest"; then
  echo "frozen confidence execution manifest verification failed" >&2
  exit 2
fi

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
extension_root="$output_root/confidence_selector_replay"
mode_args=()
if [[ "${EFFDOCK_CONFIDENCE_SMOKE_GRID:-0}" == 1 ]]; then
  parent_dir="$output_root/smoke/raw"
  confidence_dir="$extension_root/smoke/raw"
  output="$extension_root/identity_smoke.json"
  mode_args=(--smoke)
else
  parent_dir="$output_root/raw"
  confidence_dir="$extension_root/raw"
  output="$extension_root/identity_full.json"
fi

.venv/bin/python -m effdock.workflows.guidance_eta_sweep_confidence_identity \
  --parent-dir "$parent_dir" \
  --confidence-dir "$confidence_dir" \
  --output "$output" \
  "${mode_args[@]}"
