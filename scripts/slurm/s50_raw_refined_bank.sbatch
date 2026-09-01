#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-mix-bank
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1

set -euo pipefail
die() { echo "$*" >&2; exit 2; }
verify() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/nonregular $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
stage=${EFFDOCK_MIX_BANK_STAGE:?stage must be smoke or full}
case "$stage" in smoke|full) ;; *) die "invalid stage: $stage" ;; esac
worker=${EFFDOCK_MIX_BANK_WORKER:?}
builder=${EFFDOCK_MIX_BUILDER:?}
raw_bank=${EFFDOCK_MIX_RAW_BANK:?}
input_manifest=${EFFDOCK_MIX_INPUT_MANIFEST:?}
refined_manifest=${EFFDOCK_MIX_REFINED_MANIFEST:?}
output_root=${EFFDOCK_MIX_BANK_OUTPUT_ROOT:?}
checkpoint=${EFFDOCK_MIX_SAMPLER_CHECKPOINT:?}
config=${EFFDOCK_MIX_SAMPLER_CONFIG:?}
verify worker "$worker" "${EFFDOCK_MIX_BANK_WORKER_SHA:?}"
verify builder "$builder" "${EFFDOCK_MIX_BUILDER_SHA:?}"
verify raw-bank "$raw_bank" "${EFFDOCK_MIX_RAW_BANK_SHA:?}"
verify input-manifest "$input_manifest" "${EFFDOCK_MIX_INPUT_MANIFEST_SHA:?}"
verify sampler-checkpoint "$checkpoint" "${EFFDOCK_MIX_SAMPLER_CHECKPOINT_SHA:?}"
verify sampler-config "$config" "${EFFDOCK_MIX_SAMPLER_CONFIG_SHA:?}"
[[ -f "$refined_manifest.sha256" ]] || die "missing refined-manifest sidecar"
read -r refined_sha refined_path < "$refined_manifest.sha256"
[[ "$refined_path" == "$refined_manifest" ]] || die "refined sidecar path mismatch"
verify refined-manifest "$refined_manifest" "$refined_sha"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 1 ]] \
  || die "materialization requires one visible GPU"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0

run_one() {
  local root=$1 split=$2 shard_index=$3 num_shards=$4
  shift 4
  .venv/bin/python "$worker" generate-shard \
    --raw-bank-manifest "$raw_bank" --refined-manifest "$refined_manifest" \
    --input-manifest "$input_manifest" --output-root "$root" \
    --processed-root "$repo_dir/data/plinder_processed" \
    --checkpoint "$checkpoint" --checkpoint-sha256 "$EFFDOCK_MIX_SAMPLER_CHECKPOINT_SHA" \
    --config "$config" --config-sha256 "$EFFDOCK_MIX_SAMPLER_CONFIG_SHA" \
    --split "$split" --shard-index "$shard_index" --num-shards "$num_shards" \
    --device cuda "$@"
}

if [[ "$stage" == smoke ]]; then
  [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || die "smoke must not be an array"
  [[ ! -e "$output_root/smoke" ]] || die "refusing to reuse smoke output"
  run_one "$output_root/smoke" train 0 128 --limit 2
  run_one "$output_root/smoke" val 0 8 --limit 2
  printf 'complete\n' > "$output_root/smoke/SMOKE_COMPLETE.tmp"
  ln "$output_root/smoke/SMOKE_COMPLETE.tmp" "$output_root/smoke/SMOKE_COMPLETE"
  rm -f "$output_root/smoke/SMOKE_COMPLETE.tmp"
  exit 0
fi

[[ -f "$output_root/smoke/SMOKE_COMPLETE" ]] || die "missing materialization smoke"
task_id=${SLURM_ARRAY_TASK_ID:?full stage requires array task ID}
if (( task_id < 128 )); then
  split=train; shard_index=$task_id; num_shards=128
elif (( task_id < 136 )); then
  split=val; shard_index=$((task_id - 128)); num_shards=8
else
  die "array task must be 0..135"
fi
mkdir -p "$output_root/full/.reservations"
mkdir "$output_root/full/.reservations/task-$(printf '%03d' "$task_id")" \
  || die "task reservation already exists"
run_one "$output_root/full" "$split" "$shard_index" "$num_shards"
