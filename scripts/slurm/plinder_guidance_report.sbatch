#!/usr/bin/env bash
#SBATCH --job-name=effdock-plinder-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  echo "EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required" >&2
  exit 2
fi
cd "$repo_dir"
# shellcheck source=plinder_guidance_validation_profile.sh
source scripts/slurm/plinder_guidance_validation_profile.sh
verify_plinder_execution_contract
verify_plinder_raw_gate

if [[ "${SLURM_JOB_PARTITION:-}" != "cpu_only" ]]; then
  echo "PLINDER aggregate must run on cpu_only" >&2
  exit 2
fi
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
raw_root=${EFFDOCK_PLINDER_RAW_ROOT:?EFFDOCK_PLINDER_RAW_ROOT is required}
raw_manifest=${EFFDOCK_PLINDER_RAW_MANIFEST:?EFFDOCK_PLINDER_RAW_MANIFEST is required}
raw_gate=${EFFDOCK_PLINDER_RAW_GATE:?EFFDOCK_PLINDER_RAW_GATE is required}
raw_gate_sidecar=${EFFDOCK_PLINDER_RAW_GATE_SIDECAR:?EFFDOCK_PLINDER_RAW_GATE_SIDECAR is required}
require_safe_plinder_output_root "$output_root"
audit="$output_root/audit/full.json"
if [[ ! -f "$audit" ]]; then
  echo "missing passed PLINDER full audit: $audit" >&2
  exit 2
fi
output="$output_root/aggregate.json"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite PLINDER aggregate: $output" >&2
  exit 2
fi

.venv/bin/python scripts/aggregate_plinder_guidance_validation.py \
  --sampling-root "$output_root/sampling" \
  --posebusters-root "$output_root/posebusters" \
  --audit "$audit" \
  --split-file data/splits/plinder.json \
  --raw-manifest "$raw_manifest" \
  --raw-gate "$raw_gate" \
  --raw-gate-sidecar "$raw_gate_sidecar" \
  --raw-root "$raw_root" \
  --output "$output"
