#!/usr/bin/env bash
#SBATCH --job-name=effdock-sigma-report
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
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
reference_root=${EFFDOCK_SIGMA_REFERENCE_ROOT:?EFFDOCK_SIGMA_REFERENCE_ROOT is required}
.venv/bin/python scripts/report_guidance_sigma_sweep_eta2.py \
  --output-root "$output_root" \
  --reference-root "$reference_root" \
  --output-json "$output_root/report/aggregate.json" \
  --output-md "$output_root/report/RESULTS.md"
