#!/usr/bin/env bash
#SBATCH --job-name=effdock-sigma-pb-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only,test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
sweep_root=${EFFDOCK_SIGMA_SWEEP_ROOT:?EFFDOCK_SIGMA_SWEEP_ROOT is required}
reference_root=${EFFDOCK_SIGMA_REFERENCE_ROOT:?EFFDOCK_SIGMA_REFERENCE_ROOT is required}
.venv/bin/python -m effdock.workflows.guidance_sigma_sweep_posebusters_report \
  --sweep-root "$sweep_root" \
  --reference-root "$reference_root" \
  --official-root "$sweep_root/sigmadock_posebusters/official" \
  --input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --full-audit "$sweep_root/audit/full.json" \
  --reference-audit "$sweep_root/audit/reference.json" \
  --output-json "$sweep_root/sigmadock_posebusters/aggregate.json" \
  --output-md "$sweep_root/sigmadock_posebusters/RESULTS.md"
