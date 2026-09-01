#!/usr/bin/env bash
#SBATCH --job-name=effdock-high-eta-stress-audit
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
if [[ "$standalone_profile" != steric_high_eta_v1 ]]; then
  echo "stress audit is defined only for steric_high_eta_v1" >&2
  exit 2
fi

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

execution_manifest=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST:?required}
execution_manifest_sha256=${EFFDOCK_STANDALONE_EXECUTION_MANIFEST_SHA256:?required}
frozen_inputs_manifest=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST:?required}
frozen_inputs_manifest_sha256=${EFFDOCK_STANDALONE_FROZEN_INPUTS_MANIFEST_SHA256:?required}
verify_manifest execution "$execution_manifest" "$execution_manifest_sha256"
verify_manifest inputs "$frozen_inputs_manifest" "$frozen_inputs_manifest_sha256"

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
cohort_audit=${EFFDOCK_COHORT_MANIFEST:?EFFDOCK_COHORT_MANIFEST is required}
output_prefix_slash="$output_prefix/"
run_component=${output_root#"$output_prefix_slash"}
if [[ "$run_component" == "$output_root" || "$run_component" == */* ]] \
  || [[ ! "$run_component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "unsafe high-eta output root: $output_root" >&2
  exit 2
fi

output="$output_root/audit/stress.json"
mkdir -p "$output_root/.task_reservations" "$output_root/audit"
if ! mkdir "$output_root/.task_reservations/audit-stress"; then
  echo "refusing to rerun or overwrite high-eta stress audit" >&2
  exit 2
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite high-eta stress audit: $output" >&2
  exit 2
fi

.venv/bin/python -m effdock.workflows.guidance_steric_high_eta_stress_audit \
  --sampling-dir "$output_root/stress/raw" \
  --cohort-audit "$cohort_audit" \
  --output "$output"
