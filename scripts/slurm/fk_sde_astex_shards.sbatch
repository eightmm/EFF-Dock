#!/usr/bin/env bash
# Run all four frozen arms sequentially per shard so each paired complex uses
# one physical GPU, while the eight Astex shards remain parallel.
#SBATCH --job-name=effdock-fk-sde-paired
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=test,6000ada,heavy
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%8

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

shard_index=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if [[ "$shard_index" -lt 0 || "$shard_index" -ge 8 ]]; then
  echo "paired FK-SDE wrapper requires shard index 0..7" >&2
  exit 2
fi

for arm_offset in 0 8 16 24; do
  child_task_id=$((arm_offset + shard_index))
  printf 'paired_shard=%s child_task=%s\n' "$shard_index" "$child_task_id"
  SLURM_ARRAY_TASK_ID=$child_task_id \
    bash scripts/slurm/fk_sde_astex_array.sbatch
done
