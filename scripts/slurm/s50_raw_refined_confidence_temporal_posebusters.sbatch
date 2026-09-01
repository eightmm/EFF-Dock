#!/usr/bin/env bash
#SBATCH --job-name=effdock-rrc-temp-pb
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_EVALUATOR_SHA:?}"
: "${EFFDOCK_EVALUATOR_HELPER_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

read -r evaluator_sha _ < <(sha256sum scripts/evaluate_s50_raw_refined_confidence_temporal_posebusters_shard.py)
[[ "$evaluator_sha" == "$EFFDOCK_EVALUATOR_SHA" ]] || { echo "evaluator SHA mismatch" >&2; exit 2; }
read -r helper_sha _ < <(sha256sum scripts/evaluate_external_temporal_posebusters_shard.py)
[[ "$helper_sha" == "$EFFDOCK_EVALUATOR_HELPER_SHA" ]] || { echo "helper SHA mismatch" >&2; exit 2; }

arms=(u070000 u100000)
hashes=(
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
  2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8
)
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 144 ]] || { echo "expected 144 tasks" >&2; exit 2; }
arm_index=$((task_id / 72))
local_task=$((task_id % 72))
if (( local_task < 13 )); then
  dataset=phibench; num_shards=13; shard_index=$local_task
elif (( local_task < 18 )); then
  dataset=foldbench; num_shards=5; shard_index=$((local_task - 13))
else
  dataset=openbind; num_shards=54; shard_index=$((local_task - 18))
fi
arm=${arms[$arm_index]}
checkpoint_sha=${hashes[$arm_index]}

command=(
  .venv/bin/python scripts/evaluate_s50_raw_refined_confidence_temporal_posebusters_shard.py
  --dataset "$dataset"
  --arm "$arm"
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --expected-confidence-sha256 "$checkpoint_sha"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/posebusters-${arm}-${dataset}-shard${shard_index}.jsonl" \
  --note "U70k/U100k temporal-external selected-pose PoseBusters" \
  --no-gate --reason "confidence shards passed immutable-input completion gates" \
  -- "${command[@]}"
