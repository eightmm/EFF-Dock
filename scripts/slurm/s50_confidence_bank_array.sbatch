#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-bank
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1

set -euo pipefail

die() {
  echo "$*" >&2
  exit 2
}

verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label expected SHA-256 is invalid"
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

read_sha_sidecar() {
  local label=$1 path=$2 sidecar=$3 expected declared
  [[ -f "$sidecar" ]] || die "missing $label SHA sidecar: $sidecar"
  read -r expected declared < "$sidecar"
  verify_sha256 "$label" "$path" "$expected"
  printf '%s' "$expected"
}

verify_runtime_environment() {
  local expected=${EFFDOCK_ENVIRONMENT_SHA256:?EFFDOCK_ENVIRONMENT_SHA256 is required}
  local snapshot="$run_root/RUNTIME_ENVIRONMENT.json"
  local probe=scripts/capture_s50_runtime_environment.py
  local actual
  verify_sha256 runtime-environment "$snapshot" "$expected"
  [[ -x .venv/bin/python ]] || die "missing capsule .venv/bin/python"
  [[ -f "$probe" ]] || die "missing runtime-environment probe: $probe"
  read -r actual _ < <(.venv/bin/python "$probe" | sha256sum)
  [[ "$actual" == "$expected" ]] \
    || die "linked .venv runtime changed: expected=$expected actual=$actual"
}

repo_dir=${EFFDOCK_REPO_DIR:?EFFDOCK_REPO_DIR must name the execution capsule}
repo_dir=$(cd "$repo_dir" && pwd -P)
cd "$repo_dir"

scope=${EFFDOCK_BANK_SCOPE:?EFFDOCK_BANK_SCOPE must be smoke or full}
split=${EFFDOCK_BANK_SPLIT:?EFFDOCK_BANK_SPLIT must be train or val}
case "$scope:$split" in
  smoke:train|smoke:val|full:train|full:val) ;;
  *) die "invalid bank scope/split: $scope/$split" ;;
esac

run_root=${EFFDOCK_RUN_ROOT:?EFFDOCK_RUN_ROOT is required}
run_root=$(readlink -f "$run_root")
content_sha256=${EFFDOCK_CONTENT_SHA256:?EFFDOCK_CONTENT_SHA256 is required}
read -r recorded_content_sha256 < "$run_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_sha256" == "$content_sha256" ]] || die "content identity mismatch"
verify_runtime_environment

builder=scripts/prepare_s50_confidence_training_bank.py
builder_sha256=${EFFDOCK_BUILDER_SHA256:?EFFDOCK_BUILDER_SHA256 is required}
config=configs/train.yaml
checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
checkpoint_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
pose_tag=s50_n100_s10_sig2_latep3_pc10_rdkitseed0

verify_sha256 builder "$builder" "$builder_sha256"
verify_sha256 docking-config "$config" "$config_sha256"
verify_sha256 s50-checkpoint "$checkpoint" "$checkpoint_sha256"

input_manifest="$run_root/frozen/input_manifest.json"
input_sha256=$(read_sha_sidecar \
  input-manifest "$input_manifest" "$input_manifest.sha256")

task_id=${SLURM_ARRAY_TASK_ID:?this script requires a Slurm array}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
case "$scope:$split" in
  smoke:train|smoke:val)
    expected_tasks=1
    num_shards=1
    subset_args=(--max-records 2)
    ;;
  full:train)
    expected_tasks=128
    num_shards=128
    subset_args=()
    ;;
  full:val)
    expected_tasks=8
    num_shards=8
    subset_args=()
    ;;
esac
[[ "$task_count" -eq "$expected_tasks" && "$task_id" -ge 0 && "$task_id" -lt "$num_shards" ]] \
  || die "$scope/$split requires exactly $expected_tasks array tasks"

[[ ${SLURM_JOB_PARTITION:-} == 6000ada ]] \
  || die "bank generation requires partition 6000ada"
gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 1 ]] || die "exactly one visible GPU is required"
IFS=, read -r gpu_name gpu_memory_mib <<< "$gpu_inventory"
gpu_memory_mib=${gpu_memory_mib//[[:space:]]/}
[[ "$gpu_name" == *"RTX 6000 Ada"* ]] || die "expected RTX 6000 Ada; got $gpu_name"
[[ "$gpu_memory_mib" =~ ^[0-9]+$ && "$gpu_memory_mib" -ge 45000 ]] \
  || die "at least 45000 MiB GPU memory is required"

if [[ "$scope" == full ]]; then
  smoke_metrics="$run_root/smoke/training_smoke/metrics.json"
  read_sha_sidecar smoke-training-metrics \
    "$smoke_metrics" "$smoke_metrics.sha256" >/dev/null
fi

reservation_root="$run_root/.task_reservations/$scope/$split"
mkdir -p "$reservation_root"
reservation=$(printf '%s/shard-%03d-of-%03d' \
  "$reservation_root" "$task_id" "$num_shards")
mkdir "$reservation" || die "refusing duplicate task reservation: $reservation"

bank_root="$run_root/$scope/bank"
mkdir -p "$bank_root" "$run_root/ledgers"
ledger=$(printf '%s/ledgers/bank-%s-%s-shard-%03d.jsonl' \
  "$run_root" "$scope" "$split" "$task_id")

export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS

command=(
  .venv/bin/python "$builder" generate-shard
  --input-manifest "$input_manifest"
  --expected-input-manifest-sha256 "$input_sha256"
  --checkpoint "$checkpoint"
  --expected-checkpoint-sha256 "$checkpoint_sha256"
  --config "$config"
  --expected-config-sha256 "$config_sha256"
  --expected-builder-sha256 "$builder_sha256"
  --output-root "$bank_root"
  --split "$split"
  --num-shards "$num_shards"
  --shard-index "$task_id"
  --device cuda
  --pose-tag "$pose_tag"
  --hidden-chunk-size 20
  "${subset_args[@]}"
)

oms run-ledger \
  --note "S50 matched confidence $scope $split bank shard $task_id/$num_shards; fixed N100/S10/sigma2/late-p3" \
  --file "$ledger" \
  --no-gate \
  --reason "the capsule and frozen-input manifest were already verified; the shard command is fail-closed" \
  -- "${command[@]}"

printf 'scope=%s split=%s shard=%d/%d gpu=%s content_sha256=%s\n' \
  "$scope" "$split" "$task_id" "$num_shards" "$gpu_name" "$content_sha256"
