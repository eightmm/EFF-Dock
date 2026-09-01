#!/usr/bin/env bash
#SBATCH --job-name=effdock-rrc-temp
#SBATCH --partition=6000ada,heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
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
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_SOURCE_ROOT:?}"
: "${EFFDOCK_STAGE:?}"
: "${EFFDOCK_PROTOCOL_SHA:?}"
: "${EFFDOCK_RUNNER_SHA:?}"
: "${EFFDOCK_U50_HELPER_SHA:?}"
: "${EFFDOCK_SCORER_SHA:?}"
: "${EFFDOCK_SRC_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

verify_sha docs/S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_PROTOCOL.md "$EFFDOCK_PROTOCOL_SHA"
verify_sha scripts/run_s50_raw_refined_confidence_temporal_external_shard.py "$EFFDOCK_RUNNER_SHA"
verify_sha scripts/rescore_external_temporal_u50_shard.py "$EFFDOCK_U50_HELPER_SHA"
verify_sha scripts/score_guidance_sdf_post_refinement_confidence.py "$EFFDOCK_SCORER_SHA"
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$src_sha" == "$EFFDOCK_SRC_SHA" ]] || die "source tree SHA mismatch"

docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
checkpoint_root=outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || die "exactly one visible GPU is required"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 47000 ]] \
  || die "at least 47000 MiB GPU memory is required"

arms=(u070000 u100000)
checkpoints=("$checkpoint_root/best.pt" "$checkpoint_root/latest.pt")
hashes=(
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
  2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8
)
task_id=${SLURM_ARRAY_TASK_ID:?}
max_args=()
case "$EFFDOCK_STAGE" in
  smoke)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 6 ]] || die "smoke requires six tasks"
    arm_index=$((task_id / 3))
    dataset_index=$((task_id % 3))
    shard_index=0
    max_args=(--max-records 1)
    ;;
  full)
    [[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 144 ]] || die "full requires 144 tasks"
    arm_index=$((task_id / 72))
    local_task=$((task_id % 72))
    if (( local_task < 13 )); then
      dataset_index=0; shard_index=$local_task
    elif (( local_task < 18 )); then
      dataset_index=1; shard_index=$((local_task - 13))
    else
      dataset_index=2; shard_index=$((local_task - 18))
    fi
    ;;
  *) die "invalid stage" ;;
esac
case "$dataset_index" in
  0) dataset=phibench; num_shards=13 ;;
  1) dataset=foldbench; num_shards=5 ;;
  2) dataset=openbind; num_shards=54 ;;
  *) die "invalid dataset index" ;;
esac
arm=${arms[$arm_index]}
checkpoint=${checkpoints[$arm_index]}
checkpoint_sha=${hashes[$arm_index]}
verify_sha "$checkpoint" "$checkpoint_sha"

command=(
  .venv/bin/python scripts/run_s50_raw_refined_confidence_temporal_external_shard.py
  --dataset "$dataset"
  --num-shards "$num_shards"
  --shard-index "$shard_index"
  --stage "$EFFDOCK_STAGE"
  --source-root "$EFFDOCK_SOURCE_ROOT"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --external-dir data/external_test
  --config configs/train.yaml
  --docking-checkpoint "$docking"
  --confidence-checkpoint "$checkpoint"
  --arm "$arm"
  --expected-docking-sha256 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
  --expected-confidence-sha256 "$checkpoint_sha"
  "${max_args[@]}"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-${arm}-${dataset}-shard${shard_index}.jsonl" \
  --note "U70k/U100k confidence-only temporal-external rescore" \
  --no-gate --reason "user-authorized descriptive comparison on immutable poses" \
  -- "${command[@]}"
