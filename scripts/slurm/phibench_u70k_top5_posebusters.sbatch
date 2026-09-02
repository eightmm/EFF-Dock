#!/usr/bin/env bash
#SBATCH --job-name=effdock-phi-top5-pb
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

die() { echo "$*" >&2; exit 2; }
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing regular file: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "SHA mismatch: $path"
}

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_SOURCE_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_STAGE:?}"
: "${EFFDOCK_PROTOCOL_SHA:?}"
: "${EFFDOCK_INPUTS_SHA:?}"
: "${EFFDOCK_EVALUATOR_SHA:?}"
: "${EFFDOCK_HELPER_SHA:?}"
: "${EFFDOCK_SRC_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

verify_sha docs/PHIBENCH_U70K_TOP5_PROTOCOL.md "$EFFDOCK_PROTOCOL_SHA"
verify_sha docs/PHIBENCH_U70K_TOP5_INPUTS.json "$EFFDOCK_INPUTS_SHA"
verify_sha scripts/evaluate_phibench_u70k_top5_posebusters_shard.py "$EFFDOCK_EVALUATOR_SHA"
verify_sha scripts/evaluate_external_temporal_posebusters_shard.py "$EFFDOCK_HELPER_SHA"
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$src_sha" == "$EFFDOCK_SRC_SHA" ]] || die "source tree SHA mismatch"

task_id=${SLURM_ARRAY_TASK_ID:?}
max_args=()
case "$EFFDOCK_STAGE" in
  smoke)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 1 && "$task_id" -eq 0 ]] \
      || die "smoke requires task zero only"
    max_args=(--max-records 1)
    ;;
  full)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 13 ]] || die "full requires 13 tasks"
    ;;
  *) die "invalid stage: $EFFDOCK_STAGE" ;;
esac

command=(
  .venv/bin/python scripts/evaluate_phibench_u70k_top5_posebusters_shard.py
  --source-root "$EFFDOCK_SOURCE_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --num-shards 13
  --shard-index "$task_id"
  --stage "$EFFDOCK_STAGE"
  "${max_args[@]}"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-shard${task_id}.jsonl" \
  --note "PhiBench U70k confidence-ranked Top-5 PoseBusters evaluation" \
  --no-gate --reason "user-authorized descriptive endpoint-aligned reanalysis" \
  -- "${command[@]}"
