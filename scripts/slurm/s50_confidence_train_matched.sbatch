#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-train
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=heavy
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-23:59:00
#SBATCH --gres=gpu:4

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

scope=${EFFDOCK_TRAIN_SCOPE:?EFFDOCK_TRAIN_SCOPE must be smoke or full}
case "$scope" in smoke|full) ;; *) die "invalid training scope: $scope" ;; esac
world_size=${EFFDOCK_TRAIN_WORLD_SIZE:?EFFDOCK_TRAIN_WORLD_SIZE must be 4}
[[ "$world_size" -eq 4 ]] || die "S50 matched training requires four GPUs"

run_root=${EFFDOCK_RUN_ROOT:?EFFDOCK_RUN_ROOT is required}
run_root=$(readlink -f "$run_root")
content_sha256=${EFFDOCK_CONTENT_SHA256:?EFFDOCK_CONTENT_SHA256 is required}
read -r recorded_content_sha256 < "$run_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_sha256" == "$content_sha256" ]] || die "content identity mismatch"

config=configs/train_confidence_s50_matched.yaml
config_sha256=${EFFDOCK_TRAIN_CONFIG_SHA256:?EFFDOCK_TRAIN_CONFIG_SHA256 is required}
init_checkpoint=weights/effdock_confidence_extmatch_n80_s25_step42500.pt
init_checkpoint_sha256=e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f
pose_tag=s50_n100_s10_sig2_latep3_pc10_rdkitseed0
verify_sha256 training-config "$config" "$config_sha256"
verify_sha256 confidence-warm-start "$init_checkpoint" "$init_checkpoint_sha256"
[[ -x .venv/bin/torchrun && -x .venv/bin/eff-dock ]] \
  || die "missing capsule torchrun/eff-dock entry points"
command -v oms >/dev/null || die "missing oms run-ledger"
verify_runtime_environment

bank_root="$run_root/$scope/bank"
bank_manifest="$run_root/$scope/bank_manifest.json"
filtered_split="$run_root/$scope/filtered_split.json"
bank_manifest_sha256=$(read_sha_sidecar \
  "$scope bank manifest" "$bank_manifest" "$bank_manifest.sha256")
read_sha_sidecar "$scope filtered split" \
  "$filtered_split" "$filtered_split.sha256" >/dev/null

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
visible_gpus=$(wc -l <<< "$gpu_inventory")
[[ "$visible_gpus" -eq "$world_size" ]] \
  || die "world_size=$world_size but $visible_gpus GPUs are visible"
[[ ${SLURM_JOB_PARTITION:-} == heavy ]] \
  || die "four-GPU training requires the verified heavy partition"
gradient_accumulation_steps=1
if [[ "$scope" == smoke ]]; then
  out_dir="$run_root/smoke/training_smoke"
  [[ ! -e "$out_dir" ]] || die "refusing to overwrite smoke training output"
  mkdir "$out_dir"
  train_mode=(
    --init_checkpoint "$init_checkpoint"
    --init_checkpoint_sha256 "$init_checkpoint_sha256"
  )
  scope_args=(
    --smoke
    --total_steps 1
    --train_limit 2
    --val_limit 2
    --eval_every 1
    --eval_complexes 2
    --save_every 1
    --num_workers 0
    --max_train_poses_per_complex 80
    --max_val_poses_per_complex 100
    --no-use_wandb
  )
else
  out_dir="$run_root/training"
  mkdir -p "$out_dir"
  exec 9>"$out_dir/.run.lock"
  flock -n 9 || die "another training process holds $out_dir/.run.lock"
  if [[ -f "$out_dir/latest.pt" ]]; then
    train_mode=(--resume "$out_dir/latest.pt" --no-eval-on-start)
  elif [[ -f "$out_dir/best.pt" ]]; then
    train_mode=(--resume "$out_dir/best.pt" --no-eval-on-start)
  elif find "$out_dir" -mindepth 1 -maxdepth 1 \
      ! -name '.run.lock' ! -name '*.out' ! -name '*.err' | read -r _; then
    die "non-empty training root has no resumable checkpoint"
  else
    train_mode=(
      --init_checkpoint "$init_checkpoint"
      --init_checkpoint_sha256 "$init_checkpoint_sha256"
    )
  fi
  scope_args=()
fi

mkdir -p "$run_root/ledgers"
ledger="$run_root/ledgers/train-$scope.jsonl"
export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export CONFIDENCE_DDP_TIMEOUT_MIN=180

command=(
  .venv/bin/torchrun --standalone --nproc_per_node="$world_size"
  .venv/bin/eff-dock confidence train
  --config "$config"
  --split_file "$filtered_split"
  --processed_dir "$bank_root"
  --pose_tag "$pose_tag"
  --bank_manifest "$bank_manifest"
  --bank_manifest_sha256 "$bank_manifest_sha256"
  --out_dir "$out_dir"
  --run_name "effdock-confidence-s50-matched-$content_sha256"
  --gradient_accumulation_steps "$gradient_accumulation_steps"
  "${train_mode[@]}"
  "${scope_args[@]}"
)

oms run-ledger \
  --note "S50 matched confidence $scope training; U0 warm start, effective global batch four for full" \
  --file "$ledger" \
  --no-gate \
  --reason "bank aggregate, capsule identity, config, checkpoint, and GPU contract are verified above" \
  -- "${command[@]}"

metrics="$out_dir/metrics.json"
seal_sha_sidecar "$metrics" "$metrics.sha256"
printf 'scope=%s world_size=%s accumulation=%s out_dir=%s content_sha256=%s\n' \
  "$scope" "$world_size" "$gradient_accumulation_steps" "$out_dir" "$content_sha256"
