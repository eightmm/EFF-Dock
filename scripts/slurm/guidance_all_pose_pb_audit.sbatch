#!/usr/bin/env bash
#SBATCH --job-name=effdock-allpose-audit
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
mode=${EFFDOCK_ALL_POSE_PB_MODE:?EFFDOCK_ALL_POSE_PB_MODE is required}
case "$mode" in smoke) num_shards=14 ;; full) num_shards=64 ;; *) exit 2 ;; esac
read -r manifest_sha _ < "$output_root/manifest.sha256"
.venv/bin/python scripts/audit_guidance_all_pose_posebusters.py \
  --manifest "$output_root/manifest.json" \
  --manifest-sha256 "$manifest_sha" \
  --output-root "$output_root/posebusters" \
  --mode "$mode" \
  --num-shards "$num_shards" \
  --output "$output_root/audit/${mode}.json"
