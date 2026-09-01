#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

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
cohort_audit=${EFFDOCK_COHORT_MANIFEST:?EFFDOCK_COHORT_MANIFEST is required}
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe standalone output root: $output_root" >&2
  exit 2
fi
mode=full
sampling_dir="$output_root/raw"
output="$output_root/audit/full.json"
mode_args=()
if [[ "${EFFDOCK_STANDALONE_SMOKE_GRID:-0}" == 1 ]]; then
  mode=smoke
  sampling_dir="$output_root/smoke/raw"
  output="$output_root/audit/smoke.json"
  mode_args=(--smoke)
fi

mkdir -p "$output_root/.task_reservations" "$output_root/audit"
if ! mkdir "$output_root/.task_reservations/audit-$mode"; then
  echo "refusing to rerun or overwrite standalone $mode audit" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite standalone audit: $output" >&2
  exit 2
fi

.venv/bin/python -m effdock.workflows.guidance_eta_sweep_confidence_standalone_audit \
  --sampling-dir "$sampling_dir" \
  --cohort-audit "$cohort_audit" \
  --output "$output" \
  --profile "$standalone_profile" \
  "${mode_args[@]}"
