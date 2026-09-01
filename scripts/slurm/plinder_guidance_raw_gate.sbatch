#!/usr/bin/env bash
#SBATCH --job-name=effdock-plinder-raw-gate
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

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
verify_plinder_raw_manifest

if [[ "${SLURM_JOB_PARTITION:-}" != "cpu_only" ]]; then
  echo "PLINDER raw integrity gate must run on cpu_only" >&2
  exit 2
fi
if [[ "${SLURM_CPUS_PER_TASK:-}" != 8 ]]; then
  echo "PLINDER raw integrity gate requires exactly 8 CPUs" >&2
  exit 2
fi
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
raw_root=${EFFDOCK_PLINDER_RAW_ROOT:?EFFDOCK_PLINDER_RAW_ROOT is required}
raw_manifest=${EFFDOCK_PLINDER_RAW_MANIFEST:?EFFDOCK_PLINDER_RAW_MANIFEST is required}
raw_gate=${EFFDOCK_PLINDER_RAW_GATE:?EFFDOCK_PLINDER_RAW_GATE is required}
raw_gate_sidecar=${EFFDOCK_PLINDER_RAW_GATE_SIDECAR:?EFFDOCK_PLINDER_RAW_GATE_SIDECAR is required}
require_safe_plinder_output_root "$output_root"
raw_gate_dir=$(dirname "$raw_gate")
if [[ "$raw_gate_sidecar" != "$raw_gate_dir/verified.sha256" \
  || "$raw_gate" != "$raw_gate_dir/verified.json" ]]; then
  echo "raw gate paths must share the frozen final directory/layout" >&2
  exit 2
fi
if [[ -e "$raw_gate_dir" ]]; then
  echo "refusing to overwrite an existing PLINDER raw gate" >&2
  exit 2
fi

attempt_root="$output_root/.raw_gate.incomplete"
mkdir -p "$attempt_root"
attempt_dir=$(mktemp -d "$attempt_root/attempt-XXXXXXXX")
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
.venv/bin/python scripts/preflight_plinder_guidance_raw.py \
  --split-file data/splits/plinder.json \
  --raw-manifest "$raw_manifest" \
  --raw-root "$raw_root" \
  --output "$attempt_dir/verified.json" \
  --sidecar "$attempt_dir/verified.sha256" \
  --published-output "$raw_gate" \
  --workers "$SLURM_CPUS_PER_TASK"
if ! mv "$attempt_dir" "$raw_gate_dir"; then
  echo "atomic PLINDER raw-gate publish failed; attempt retained at $attempt_dir" >&2
  exit 2
fi
