#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-pb
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

task_id=${SLURM_ARRAY_TASK_ID:?}
task_count=${SLURM_ARRAY_TASK_COUNT:?}
[[ "$task_count" -eq 72 ]] || { echo "PoseBusters requires 72 tasks" >&2; exit 2; }
if (( task_id < 13 )); then
  dataset=phibench; num_shards=13; shard_index=$task_id
elif (( task_id < 18 )); then
  dataset=foldbench; num_shards=5; shard_index=$((task_id - 13))
else
  dataset=openbind; num_shards=54; shard_index=$((task_id - 18))
fi

command=(
  .venv/bin/python scripts/evaluate_external_temporal_posebusters_shard.py
  --dataset "$dataset"
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/posebusters-${dataset}-shard${shard_index}.jsonl" \
  --note "protein-ligand and official PoseBusters validity on refined selected poses" \
  --no-gate --reason "immutable execution capsule passed submission-time focused gates" \
  -- "${command[@]}"
