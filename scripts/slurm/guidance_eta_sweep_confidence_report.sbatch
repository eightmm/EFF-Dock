#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-conf-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

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
cohort_audit=${EFFDOCK_COHORT_MANIFEST:-$output_root/audit/combined.json}
extension_root="$output_root/confidence_selector_replay"
.venv/bin/python -m effdock.workflows.guidance_eta_sweep_confidence_report \
  --input-dir "$extension_root/posebusters_official" \
  --sampling-dir "$extension_root/raw" \
  --cohort-audit "$cohort_audit" \
  --identity-audit "$extension_root/identity_full.json" \
  --output "$extension_root/aggregate.json"
