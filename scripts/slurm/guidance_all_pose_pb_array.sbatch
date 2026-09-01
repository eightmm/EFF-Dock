#!/usr/bin/env bash
#SBATCH --job-name=effdock-allpose-pb
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=test,cpu_only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-63%8
set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
mode=${EFFDOCK_ALL_POSE_PB_MODE:?EFFDOCK_ALL_POSE_PB_MODE is required}
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
case "$mode" in
  smoke) num_shards=14; [[ "$task_count" -eq 14 ]] || { echo "smoke requires 14 tasks" >&2; exit 2; } ;;
  full) num_shards=64; [[ "$task_count" -eq 64 ]] || { echo "full requires 64 tasks" >&2; exit 2; } ;;
  *) echo "mode must be smoke or full" >&2; exit 2 ;;
esac
[[ "${SLURM_CPUS_PER_TASK:-}" == 4 ]] || { echo "exactly four CPUs required" >&2; exit 2; }
case "${SLURM_JOB_PARTITION:-}" in test|cpu_only) ;; *) echo "unexpected partition" >&2; exit 2 ;; esac
read -r manifest_sha _ < "$output_root/manifest.sha256"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "protocol=EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1 mode=$mode shard=$task_id/$num_shards workers=4"
.venv/bin/python scripts/run_guidance_all_pose_posebusters.py \
  --manifest "$output_root/manifest.json" \
  --manifest-sha256 "$manifest_sha" \
  --output-root "$output_root/posebusters" \
  --mode "$mode" \
  --num-shards "$num_shards" \
  --shard-index "$task_id" \
  --workers 4
