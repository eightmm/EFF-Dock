#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refconf-bank
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
verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
scope=${EFFDOCK_REFCONF_SCOPE:?EFFDOCK_REFCONF_SCOPE must be smoke or full}
case "$scope" in smoke|full) ;; *) die "invalid materialization scope: $scope" ;; esac

source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
refined_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/207611a7fed8805b7f52cdf4648da2ea03e7d1e699c7d47106b632215634fdc2/full"
worker="$repo_dir/scripts/materialize_s50_refined_confidence_bank.py"
builder="$repo_dir/scripts/prepare_s50_confidence_training_bank.py"
checkpoint="$repo_dir/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt"
config="$repo_dir/configs/train.yaml"
verify_sha256 worker "$worker" 07f45445e0d2035d895535dadf1ee73c64a85d1193b36a971c8b4a68a1013768
verify_sha256 builder "$builder" 0d1fb50e98345d0712862ff8a262fb8e08d8a2d231d2b0288facb6053ae19d51
verify_sha256 sampler-checkpoint "$checkpoint" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha256 sampler-config "$config" 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
[[ -f "$refined_root/manifest.json" ]] || die "missing complete refined pose manifest"
[[ -x .venv/bin/python ]] || die "missing .venv Python"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0
raw_bank="$source_root/full/bank_manifest.json"
input_manifest="$source_root/frozen/input_manifest.json"

run_one() {
  local output_root=$1 split=$2 shard_index=$3 num_shards=$4
  shift 4
  .venv/bin/python "$worker" generate-shard \
    --raw-bank-manifest "$raw_bank" \
    --refined-manifest "$refined_root/manifest.json" \
    --input-manifest "$input_manifest" \
    --output-root "$output_root" \
    --processed-root "$repo_dir/data/plinder_processed" \
    --checkpoint "$checkpoint" \
    --checkpoint-sha256 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 \
    --config "$config" \
    --config-sha256 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec \
    --split "$split" --shard-index "$shard_index" --num-shards "$num_shards" \
    --device cuda "$@"
}

if [[ "$scope" == smoke ]]; then
  smoke_root="$refined_root/confidence_smoke"
  [[ ! -f "$smoke_root/SMOKE_COMPLETE" ]] || die "refusing to reuse refined confidence smoke"
  run_one "$smoke_root" train 0 128 --limit 2
  run_one "$smoke_root" val 0 8 --limit 2
  printf 'complete\n' > "$smoke_root/SMOKE_COMPLETE.tmp"
  ln "$smoke_root/SMOKE_COMPLETE.tmp" "$smoke_root/SMOKE_COMPLETE"
  rm -f "$smoke_root/SMOKE_COMPLETE.tmp"
  exit 0
fi

[[ -f "$refined_root/confidence_smoke/SMOKE_COMPLETE" ]] \
  || die "missing refined confidence materialization smoke"
task_id=${SLURM_ARRAY_TASK_ID:?full materialization requires array task ID}
if (( task_id < 128 )); then
  split=train; shard_index=$task_id; num_shards=128
elif (( task_id < 136 )); then
  split=val; shard_index=$((task_id - 128)); num_shards=8
else
  die "array task outside 0..135"
fi
run_one "$refined_root/confidence_bank" "$split" "$shard_index" "$num_shards"
