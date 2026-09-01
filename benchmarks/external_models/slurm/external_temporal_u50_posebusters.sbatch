#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-u50-pb
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_EVALUATOR_SHA256:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

read -r evaluator_sha _ < <(sha256sum scripts/evaluate_external_temporal_posebusters_shard.py)
[[ "$evaluator_sha" == "$EFFDOCK_EVALUATOR_SHA256" ]] || { echo "evaluator SHA mismatch" >&2; exit 2; }

task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 72 ]] || { echo "expected 72 tasks" >&2; exit 2; }
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
  --expected-confidence-sha256 fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/posebusters-${dataset}-shard${shard_index}.jsonl" \
  --note "U50 selected-pose PoseBusters evaluation" \
  --no-gate --reason "U50 confidence summaries passed hash and completion gates" \
  -- "${command[@]}"
