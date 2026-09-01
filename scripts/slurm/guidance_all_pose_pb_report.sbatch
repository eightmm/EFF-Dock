#!/usr/bin/env bash
#SBATCH --job-name=effdock-allpose-report
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=test,cpu_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
.venv/bin/python scripts/report_guidance_all_pose_posebusters.py \
  --output-root "$output_root/posebusters" \
  --audit "$output_root/audit/full.json" \
  --num-shards 64 \
  --output-json "$output_root/report/aggregate.json" \
  --output-md "$output_root/report/RESULTS.md"
