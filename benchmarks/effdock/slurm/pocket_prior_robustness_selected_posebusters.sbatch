#!/usr/bin/env bash
#SBATCH --job-name=effdock-pocket-prior-pb
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
source benchmarks/effdock/slurm/pocket_prior_robustness_common.sh
num_shards=16
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 432 ]] || { echo "PB stage requires 432 tasks" >&2; exit 2; }
shard_index=$((task_id % num_shards))
condition_index=$((task_id / num_shards))
effdock_resolve_pocket_prior_condition "$condition_index"
condition_root=$(effdock_pocket_prior_condition_root "$EFFDOCK_OUTPUT_ROOT")
"$runtime_venv/bin/python" benchmarks/effdock/evaluate_pocket_cutoff_selected_posebusters.py \
  --condition-root "$condition_root" --cutoff "$EFFDOCK_CONDITION_CUTOFF" \
  --sigma "$EFFDOCK_CONDITION_SIGMA" --jitter "$EFFDOCK_CONDITION_JITTER" \
  --repeat-index "$EFFDOCK_CONDITION_REPEAT" --num-shards "$num_shards" \
  --shard-index "$shard_index" --extension
