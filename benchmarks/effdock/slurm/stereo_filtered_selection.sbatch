#!/usr/bin/env bash
#SBATCH --job-name=effdock-stereo-filter
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-47%16
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
num_shards=16
task_id=${SLURM_ARRAY_TASK_ID:?}
repeat_index=$((task_id / num_shards))
shard_index=$((task_id % num_shards))
condition_root="$EFFDOCK_OUTPUT_ROOT/cutoff_10/repeat_$repeat_index"
"$runtime_venv/bin/python" benchmarks/effdock/evaluate_stereo_filtered_selection.py \
  --condition-root "$condition_root" --cutoff 10 --repeat-index "$repeat_index" \
  --num-shards "$num_shards" --shard-index "$shard_index"
