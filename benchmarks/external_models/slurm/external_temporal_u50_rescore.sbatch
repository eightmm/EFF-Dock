#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-u50
#SBATCH --partition=6000ada,heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_SOURCE_ROOT:?}"
: "${EFFDOCK_STAGE:?}"
: "${EFFDOCK_RUNNER_SHA256:?}"
: "${EFFDOCK_SCORER_SHA256:?}"
: "${EFFDOCK_SRC_SHA256:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

die() { echo "$*" >&2; exit 2; }
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing regular file: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "SHA mismatch: $path"
}

verify_sha scripts/rescore_external_temporal_u50_shard.py "$EFFDOCK_RUNNER_SHA256"
verify_sha scripts/score_guidance_sdf_post_refinement_confidence.py "$EFFDOCK_SCORER_SHA256"
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
actual_src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$actual_src_sha" == "$EFFDOCK_SRC_SHA256" ]] || die "source tree SHA mismatch"

docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
confidence=outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence" fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || die "exactly one visible GPU is required"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 47000 ]] \
  || die "at least 47000 MiB GPU memory is required"

task_id=${SLURM_ARRAY_TASK_ID:?}
max_args=()
case "$EFFDOCK_STAGE" in
  smoke)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 3 ]] || die "smoke requires three tasks"
    case "$task_id" in
      0) dataset=phibench; num_shards=13 ;;
      1) dataset=foldbench; num_shards=5 ;;
      2) dataset=openbind; num_shards=54 ;;
      *) die "invalid smoke task" ;;
    esac
    shard_index=0
    max_args=(--max-records 1)
    ;;
  full)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 72 ]] || die "full requires 72 tasks"
    if (( task_id < 13 )); then
      dataset=phibench; num_shards=13; shard_index=$task_id
    elif (( task_id < 18 )); then
      dataset=foldbench; num_shards=5; shard_index=$((task_id - 13))
    else
      dataset=openbind; num_shards=54; shard_index=$((task_id - 18))
    fi
    ;;
  *) die "invalid stage" ;;
esac

command=(
  .venv/bin/python scripts/rescore_external_temporal_u50_shard.py
  --dataset "$dataset"
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --stage "$EFFDOCK_STAGE"
  --source-root "$EFFDOCK_SOURCE_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --external-dir data/external_test
  --config configs/train.yaml
  --docking-checkpoint "$docking"
  --confidence-checkpoint "$confidence"
  --expected-docking-sha256 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
  --expected-confidence-sha256 fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
  "${max_args[@]}"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-${dataset}-shard${shard_index}.jsonl" \
  --note "U50 confidence-only rescore of immutable external temporal pose bank" \
  --no-gate --reason "user-selected reporting convention; no sampling or refinement mutation" \
  -- "${command[@]}"
