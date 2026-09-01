#!/usr/bin/env bash
#SBATCH --job-name=effdock-sigma-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
mode=${EFFDOCK_SIGMA_SWEEP_AUDIT_MODE:?EFFDOCK_SIGMA_SWEEP_AUDIT_MODE is required}
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
.venv/bin/python scripts/audit_guidance_sigma_sweep_eta2.py \
  --output-root "$output_root" \
  --mode "$mode" \
  --input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --output "$output_root/audit/${mode}.json"
