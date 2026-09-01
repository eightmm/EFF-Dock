#!/usr/bin/env bash
#SBATCH --job-name=effdock-sigma-pb-smoke-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only,test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
sweep_root=${EFFDOCK_SIGMA_SWEEP_ROOT:?EFFDOCK_SIGMA_SWEEP_ROOT is required}
.venv/bin/python scripts/audit_guidance_sigma_sweep_posebusters_smoke.py \
  --sweep-root "$sweep_root" \
  --output "$sweep_root/sigmadock_posebusters/audit/smoke.json"
