#!/usr/bin/env bash
#SBATCH --job-name=effdock-nfe-pb
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
: "${EFFDOCK_CONTROL_CAPSULE:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_RUN_MODE:?}"
: "${EFFDOCK_SOURCE_REPORT:?}"
: "${EFFDOCK_SOURCE_REPORT_SHA256:?}"
: "${EFFDOCK_N100_SCORES_ROOT:?}"
: "${EFFDOCK_N40_SCORES_ROOT:?}"
: "${EFFDOCK_EVALUATOR_SHA256:?}"
: "${EFFDOCK_PROTOCOL_SHA256:?}"
cd "$EFFDOCK_CONTROL_CAPSULE"
read -r evaluator_sha _ < <(sha256sum scripts/evaluate_fixed_nfe_step_pose_pb.py)
read -r protocol_sha _ < <(sha256sum docs/FIXED_NFE_STEP_POSE_PB_PROTOCOL.md)
[[ "$evaluator_sha" == "$EFFDOCK_EVALUATOR_SHA256" ]]
[[ "$protocol_sha" == "$EFFDOCK_PROTOCOL_SHA256" ]]
case "$EFFDOCK_RUN_MODE" in
  smoke) num_shards=2 ;;
  full) num_shards=32 ;;
  *) exit 2 ;;
esac
workers=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE"
command=(
  .venv/bin/python scripts/evaluate_fixed_nfe_step_pose_pb.py shard
  --source-report "$EFFDOCK_SOURCE_REPORT"
  --expected-source-sha256 "$EFFDOCK_SOURCE_REPORT_SHA256"
  --n100-scores-root "$EFFDOCK_N100_SCORES_ROOT"
  --n40-scores-root "$EFFDOCK_N40_SCORES_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE/shards"
  --mode "$EFFDOCK_RUN_MODE"
  --num-shards "$num_shards"
  --shard-index "${SLURM_ARRAY_TASK_ID:?}"
  --workers "$workers"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE/pb-${SLURM_ARRAY_TASK_ID}.jsonl" \
  --note "fixed-NFE frozen-U50 selected-pose PoseBusters $EFFDOCK_RUN_MODE shard" \
  --no-gate --reason "user-authorized repeated-use descriptive validity comparison" \
  -- "${command[@]}"
