#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-n64
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

die() { echo "$*" >&2; exit 2; }

verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] \
    || die "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path"
}

seal_sha_sidecar() {
  local path=$1 sidecar=$2 digest temporary
  [[ -f "$path" ]] || die "cannot seal missing file: $path"
  [[ ! -e "$sidecar" ]] || die "refusing to overwrite SHA sidecar: $sidecar"
  read -r digest _ < <(sha256sum "$path")
  temporary="${sidecar}.tmp.${SLURM_JOB_ID:-$$}"
  printf '%s  %s\n' "$digest" "$path" > "$temporary"
  ln "$temporary" "$sidecar" || die "failed to seal $sidecar"
  rm -f "$temporary"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
source_run="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
execution_root="$repo_dir/.effdock_execution_capsules/EFFDOCK-S50-MATCHED-CONFIDENCE-TRAIN-VAL-V1/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
scope=${EFFDOCK_N64_SCOPE:?EFFDOCK_N64_SCOPE must be smoke or full}
case "$scope" in smoke|full) ;; *) die "invalid N64 scope: $scope" ;; esac

content_sha256=14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b
config_sha256=2f0b9d6b4805d20b7fafe937023c81890338a09908eab04fd40cb850bbae10a8
bank_manifest_sha256=d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
filtered_split_sha256=1f23b50ef3a5eff73fd8cad683c1f3adfe4e1ab235199274d0dbc220c1d22507
init_checkpoint_sha256=e31fde6f351284205c78f7a1510002779c43312e94d9f82003d47a14d72bc78f

read -r recorded_content < "$source_run/CONTENT_IDENTITY.sha256"
[[ "$recorded_content" == "$content_sha256" ]] || die "source content identity mismatch"

config="$execution_root/configs/train_confidence_s50_matched.yaml"
bank_root="$source_run/full/bank"
bank_manifest="$source_run/full/bank_manifest.json"
filtered_split="$source_run/full/filtered_split.json"
init_checkpoint="$execution_root/weights/effdock_confidence_extmatch_n80_s25_step42500.pt"
verify_sha256 training-config "$config" "$config_sha256"
verify_sha256 bank-manifest "$bank_manifest" "$bank_manifest_sha256"
verify_sha256 filtered-split "$filtered_split" "$filtered_split_sha256"
verify_sha256 confidence-warm-start "$init_checkpoint" "$init_checkpoint_sha256"
[[ -x "$execution_root/.venv/bin/torchrun" && -x "$execution_root/.venv/bin/eff-dock" ]] \
  || die "missing execution-capsule entry points"

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 4 ]] || die "N64 training requires exactly four visible GPUs"
[[ ${SLURM_JOB_PARTITION:-} == heavy ]] || die "N64 training requires heavy partition"

if [[ "$scope" == smoke ]]; then
  out_dir="$source_run/smoke/training_n64_v1_smoke_v2"
  [[ ! -e "$out_dir" ]] || die "refusing to overwrite N64 smoke output: $out_dir"
  mkdir "$out_dir"
  train_mode=(
    --init_checkpoint "$init_checkpoint"
    --init_checkpoint_sha256 "$init_checkpoint_sha256"
  )
  scope_args=(
    --total_steps 1
    --train_limit 8
    --val_limit 2
    --eval_every 1
    --eval_complexes 2
    --save_every 1
    --num_workers 0
  )
else
  out_dir="$source_run/training_n64_v1"
  mkdir -p "$out_dir"
  exec 9>"$out_dir/.run.lock"
  flock -n 9 || die "another process holds $out_dir/.run.lock"
  if [[ -f "$out_dir/latest.pt" ]]; then
    train_mode=(--resume "$out_dir/latest.pt" --no-eval-on-start)
  elif [[ -f "$out_dir/best.pt" ]]; then
    train_mode=(--resume "$out_dir/best.pt" --no-eval-on-start)
  elif find "$out_dir" -mindepth 1 -maxdepth 1 \
      ! -name '.run.lock' ! -name '*.out' ! -name '*.err' | read -r _; then
    die "non-empty N64 training root has no resumable checkpoint"
  else
    train_mode=(
      --init_checkpoint "$init_checkpoint"
      --init_checkpoint_sha256 "$init_checkpoint_sha256"
    )
  fi
  scope_args=(--eval_every 5000 --save_every 500)
fi

mkdir -p "$source_run/ledgers"
ledger="$source_run/ledgers/train-n64-v1-$scope.jsonl"
export PYTHONPATH="$execution_root/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0
export CONFIDENCE_DDP_TIMEOUT_MIN=180
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "$execution_root/.venv/bin/torchrun" --standalone --nproc_per_node=4
  "$execution_root/.venv/bin/eff-dock" confidence train
  --config "$config"
  --split_file "$filtered_split"
  --processed_dir "$bank_root"
  --pose_tag s50_n100_s10_sig2_latep3_pc10_rdkitseed0
  --bank_manifest "$bank_manifest"
  --bank_manifest_sha256 "$bank_manifest_sha256"
  --out_dir "$out_dir"
  --run_name "effdock-confidence-s50-n64-v1-$content_sha256"
  --gradient_accumulation_steps 1
  --max_train_poses_per_complex 64
  --max_val_poses_per_complex 100
  --no-use_wandb
  "${train_mode[@]}"
  "${scope_args[@]}"
)

cd "$execution_root"
oms run-ledger \
  --note "S50 confidence N64 v1 $scope; old step-42500 weights, fresh optimizer; latest every 500 updates in full" \
  --file "$ledger" \
  --no-gate \
  --reason "sealed N100 bank/config/checkpoint verified; N64 is the user-authorized OOM recovery" \
  -- "${command[@]}"

metrics="$out_dir/metrics.json"
seal_sha_sidecar "$metrics" "$metrics.sha256"
printf 'scope=%s train_poses=64 val_poses=100 out_dir=%s\n' "$scope" "$out_dir"
