#!/usr/bin/env bash
#SBATCH --job-name=effdock-allpose-manifest
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=test,cpu_only
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
reference_root=${EFFDOCK_ALL_POSE_REFERENCE_ROOT:?EFFDOCK_ALL_POSE_REFERENCE_ROOT is required}
extension_root=${EFFDOCK_ALL_POSE_EXTENSION_ROOT:?EFFDOCK_ALL_POSE_EXTENSION_ROOT is required}
.venv/bin/python scripts/build_guidance_all_pose_pb_manifest.py \
  --reference-root "$reference_root" \
  --extension-root "$extension_root" \
  --output "$output_root/manifest.json"
sha256sum "$output_root/manifest.json" > "$output_root/manifest.sha256"
