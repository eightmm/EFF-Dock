#!/usr/bin/env bash
#SBATCH --job-name=effdock-cut-pb
#SBATCH --partition=cpu_only
#SBATCH --qos=short
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
cutoffs=(6 8 10 12)
num_shards=16
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 192 ]] || { echo "PB stage requires 192 tasks" >&2; exit 2; }
shard_index=$((task_id % num_shards))
repeat_index=$(((task_id / num_shards) % 3))
cutoff=${cutoffs[$((task_id / (num_shards * 3)))]}
condition_root="$EFFDOCK_OUTPUT_ROOT/cutoff_$(printf '%02d' "$cutoff")/repeat_$repeat_index"
"$runtime_venv/bin/python" benchmarks/effdock/evaluate_pocket_cutoff_selected_posebusters.py \
  --condition-root "$condition_root" --cutoff "$cutoff" \
  --repeat-index "$repeat_index" --num-shards "$num_shards" \
  --shard-index "$shard_index"
