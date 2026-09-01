#!/usr/bin/env bash
#SBATCH --job-name=effdock-term-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
mode=${EFFDOCK_TERM_COEFF_AUDIT_MODE:?EFFDOCK_TERM_COEFF_AUDIT_MODE is required}
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
.venv/bin/python scripts/audit_guidance_term_coefficient_sweep.py \
  --output-root "$output_root" \
  --mode "$mode" \
  --input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --output "$output_root/audit/${mode}.json"
