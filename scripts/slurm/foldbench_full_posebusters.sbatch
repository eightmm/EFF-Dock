#!/usr/bin/env bash
#SBATCH --job-name=effdock-fb558-pb
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail
: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_EVALUATOR_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
thread_count=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$thread_count" MKL_NUM_THREADS="$thread_count" OPENBLAS_NUM_THREADS="$thread_count"

evaluator=scripts/evaluate_external_temporal_posebusters_shard.py
read -r actual _ < <(sha256sum "$evaluator")
[[ "$actual" == "$EFFDOCK_EVALUATOR_SHA" ]] || { echo "evaluator SHA mismatch" >&2; exit 2; }
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 44 ]] || { echo "expected 44 tasks" >&2; exit 2; }

command=(
  .venv/bin/python "$evaluator"
  --dataset foldbench
  --num-shards 44
  --shard-index "$task_id"
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --expected-confidence-sha256 ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
  --protocol-id EFFDOCK-FOLDBENCH-POCKET-558-V1
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/posebusters-foldbench-shard${task_id}.jsonl" \
  --note "FoldBench-Pocket-558 selected-pose PoseBusters" \
  --no-gate --reason "dependent evaluation of frozen generated poses" \
  -- "${command[@]}"
