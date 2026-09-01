#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-bank-ctl
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00

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
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || die "$label sidecar SHA-256 is invalid"
  verify_sha256 "$label" "$path" "$expected"
  printf '%s' "$expected"
}

seal_sha_sidecar() {
  local path=$1 sidecar=$2 digest temporary
  [[ -f "$path" ]] || die "cannot seal missing file: $path"
  [[ ! -e "$sidecar" ]] || die "refusing to overwrite SHA sidecar: $sidecar"
  read -r digest _ < <(sha256sum "$path")
  temporary="${sidecar}.tmp.${SLURM_JOB_ID:-$$}"
  printf '%s  %s\n' "$digest" "$path" > "$temporary"
  if ! ln "$temporary" "$sidecar" 2>/dev/null; then
    rm -f "$temporary"
    die "failed to seal no-overwrite SHA sidecar: $sidecar"
  fi
  rm -f "$temporary"
}

verify_runtime_environment() {
  local expected=${EFFDOCK_ENVIRONMENT_SHA256:?EFFDOCK_ENVIRONMENT_SHA256 is required}
  local snapshot="$run_root/RUNTIME_ENVIRONMENT.json"
  local probe=scripts/capture_s50_runtime_environment.py
  local actual
  verify_sha256 runtime-environment "$snapshot" "$expected"
  [[ -f "$probe" ]] || die "missing runtime-environment probe: $probe"
  read -r actual _ < <(.venv/bin/python "$probe" | sha256sum)
  [[ "$actual" == "$expected" ]] \
    || die "linked .venv runtime changed: expected=$expected actual=$actual"
}

repo_dir=${EFFDOCK_REPO_DIR:?EFFDOCK_REPO_DIR must name the execution capsule}
repo_dir=$(cd "$repo_dir" && pwd -P)
cd "$repo_dir"

stage=${EFFDOCK_STAGE:?EFFDOCK_STAGE must be freeze, aggregate-smoke, or aggregate-full}
case "$stage" in
  freeze|aggregate-smoke|aggregate-full) ;;
  *) die "invalid EFFDOCK_STAGE=$stage" ;;
esac

run_root=${EFFDOCK_RUN_ROOT:?EFFDOCK_RUN_ROOT is required}
run_root=$(readlink -f "$run_root")
[[ -d "$run_root" ]] || die "missing content root: $run_root"
content_sha256=${EFFDOCK_CONTENT_SHA256:?EFFDOCK_CONTENT_SHA256 is required}
[[ "$content_sha256" =~ ^[0-9a-f]{64}$ ]] || die "invalid content identity"
read -r recorded_content_sha256 < "$run_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_sha256" == "$content_sha256" ]] || die "content identity mismatch"

builder=scripts/prepare_s50_confidence_training_bank.py
builder_sha256=${EFFDOCK_BUILDER_SHA256:?EFFDOCK_BUILDER_SHA256 is required}
split=data/splits/plinder.json
pool=data/plinder_pool.parquet
processed=data/plinder_processed
docking_config=configs/train.yaml
s50_checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
val_bank=outputs/benchmarks/early_time_sampler_s50_confidence_runs/frozen_inputs/label_free_bank.v2.json

split_sha256=3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b
pool_sha256=0ff455da77ce5540b839918cccb96f45414e91efff6272d7da3a65337ab1fe91
docking_config_sha256=39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
s50_checkpoint_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
val_bank_sha256=928b7219ed1ef8375c1ee52470f6ef606b8fca4d5bf4ea5c51355e8332e29a4b
pose_tag=s50_n100_s10_sig2_latep3_pc10_rdkitseed0

verify_sha256 builder "$builder" "$builder_sha256"
verify_sha256 split "$split" "$split_sha256"
verify_sha256 canonical-ligand-pool "$pool" "$pool_sha256"
verify_sha256 docking-config "$docking_config" "$docking_config_sha256"
verify_sha256 s50-checkpoint "$s50_checkpoint" "$s50_checkpoint_sha256"
verify_sha256 validation-bank "$val_bank" "$val_bank_sha256"
[[ -d "$processed" ]] || die "missing processed PLINDER root: $processed"
[[ -x .venv/bin/python ]] || die "missing capsule .venv/bin/python"
command -v oms >/dev/null || die "missing oms run-ledger"
verify_runtime_environment

mkdir -p "$run_root/ledgers"
input_manifest="$run_root/frozen/input_manifest.json"
input_manifest_sha="$input_manifest.sha256"

export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

case "$stage" in
  freeze)
    [[ ! -e "$run_root/frozen" ]] || die "refusing to reuse frozen-input directory"
    mkdir "$run_root/frozen"
    command=(
      .venv/bin/python "$builder" freeze-inputs
      --split-file "$split"
      --expected-split-sha256 "$split_sha256"
      --pool-parquet "$pool"
      --expected-pool-sha256 "$pool_sha256"
      --processed-root "$processed"
      --checkpoint "$s50_checkpoint"
      --expected-checkpoint-sha256 "$s50_checkpoint_sha256"
      --config "$docking_config"
      --expected-config-sha256 "$docking_config_sha256"
      --val-bank-manifest "$val_bank"
      --expected-val-bank-manifest-sha256 "$val_bank_sha256"
      --expected-builder-sha256 "$builder_sha256"
      --output "$input_manifest"
      --workers "${SLURM_CPUS_PER_TASK:-8}"
    )
    oms run-ledger \
      --note "S50 matched confidence complete-split input freeze; 47310 train and 1076 val attempted" \
      --file "$run_root/ledgers/freeze-inputs.jsonl" \
      --no-gate \
      --reason "the reviewed execution capsule and this fail-closed input preflight are the stage gate" \
      -- "${command[@]}"
    seal_sha_sidecar "$input_manifest" "$input_manifest_sha"
    ;;

  aggregate-smoke|aggregate-full)
    input_sha256=$(read_sha_sidecar input-manifest "$input_manifest" "$input_manifest_sha")
    if [[ "$stage" == aggregate-smoke ]]; then
      scope=smoke
      train_shards=1
      val_shards=1
      subset_args=(--allow-smoke-subset)
    else
      scope=full
      train_shards=128
      val_shards=8
      subset_args=()
      smoke_manifest="$run_root/smoke/bank_manifest.json"
      smoke_manifest_sha="$smoke_manifest.sha256"
      read_sha_sidecar smoke-bank-manifest "$smoke_manifest" "$smoke_manifest_sha" >/dev/null
    fi
    bank_root="$run_root/$scope/bank"
    output_manifest="$run_root/$scope/bank_manifest.json"
    filtered_split="$run_root/$scope/filtered_split.json"
    [[ ! -e "$output_manifest" && ! -e "$filtered_split" ]] \
      || die "refusing to overwrite $scope aggregate outputs"
    command=(
      .venv/bin/python "$builder" aggregate
      --input-manifest "$input_manifest"
      --expected-input-manifest-sha256 "$input_sha256"
      --expected-builder-sha256 "$builder_sha256"
      --output-root "$bank_root"
      --num-train-shards "$train_shards"
      --num-val-shards "$val_shards"
      --output-manifest "$output_manifest"
      --filtered-split-output "$filtered_split"
      --pose-tag "$pose_tag"
      "${subset_args[@]}"
    )
    oms run-ledger \
      --note "S50 matched confidence $scope bank aggregate and complete attempted-denominator audit" \
      --file "$run_root/ledgers/aggregate-$scope.jsonl" \
      --no-gate \
      --reason "all shard hashes and split accounting are revalidated by the aggregate command" \
      -- "${command[@]}"
    seal_sha_sidecar "$output_manifest" "$output_manifest.sha256"
    seal_sha_sidecar "$filtered_split" "$filtered_split.sha256"
    ;;
esac

printf 'stage=%s run_root=%s content_sha256=%s\n' "$stage" "$run_root" "$content_sha256"
