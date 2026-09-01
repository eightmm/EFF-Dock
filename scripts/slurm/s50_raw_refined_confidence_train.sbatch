#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-mix-train
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=heavy
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-23:59:00
#SBATCH --gres=gpu:4

set -euo pipefail
die() { echo "$*" >&2; exit 2; }
verify() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" && ! -L "$path" ]] || die "missing/nonregular $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch"
}
seal() {
  local path=$1 sidecar=$2 sha
  [[ -f "$path" && ! -e "$sidecar" ]] || die "cannot seal $sidecar"
  sha=$(sha256sum "$path" | cut -d' ' -f1)
  printf '%s  %s\n' "$sha" "$path" > "$sidecar.tmp"
  ln "$sidecar.tmp" "$sidecar"
  rm -f "$sidecar.tmp"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
stage=${EFFDOCK_MIX_TRAIN_STAGE:?stage must be smoke or full}
case "$stage" in smoke|full) ;; *) die "invalid stage: $stage" ;; esac
train_nproc=${EFFDOCK_MIX_TRAIN_NPROC:-4}
case "$stage:$train_nproc" in
  smoke:2|smoke:4|full:2|full:4) ;;
  *) die "invalid stage/GPU count: $stage/$train_nproc" ;;
esac
config=${EFFDOCK_MIX_TRAIN_CONFIG:?}
trainer="$repo_dir/src/effdock/workflows/train_confidence.py"
dataset="$repo_dir/src/effdock/confidence/dataset.py"
protocol=${EFFDOCK_MIX_PROTOCOL:-$repo_dir/docs/S50_RAW_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md}
verify config "$config" "${EFFDOCK_MIX_TRAIN_CONFIG_SHA:?}"
verify trainer "$trainer" "${EFFDOCK_MIX_TRAINER_SHA:?}"
verify dataset "$dataset" "${EFFDOCK_MIX_DATASET_SHA:?}"
verify protocol "$protocol" "${EFFDOCK_MIX_PROTOCOL_SHA:?}"

refined_bank=${EFFDOCK_MIX_BANK_OUTPUT_ROOT:?}/full/bank_manifest.json
refined_target=${EFFDOCK_MIX_LABEL_OUTPUT_ROOT:?}/full/manifest.json
for path in "$refined_bank" "$refined_target"; do
  [[ -f "$path.sha256" ]] || die "missing sidecar for $path"
done
read -r refined_bank_sha refined_bank_path < "$refined_bank.sha256"
read -r refined_target_sha refined_target_path < "$refined_target.sha256"
[[ "$refined_bank_path" == "$refined_bank" ]] || die "refined bank sidecar path mismatch"
[[ "$refined_target_path" == "$refined_target" ]] \
  || die "refined target sidecar path mismatch"
verify refined-bank "$refined_bank" "$refined_bank_sha"
verify refined-target "$refined_target" "$refined_target_sha"
verify raw-bank "${EFFDOCK_MIX_RAW_BANK:?}" "${EFFDOCK_MIX_RAW_BANK_SHA:?}"
verify raw-target "${EFFDOCK_MIX_RAW_TARGET:?}" "${EFFDOCK_MIX_RAW_TARGET_SHA:?}"
verify init-checkpoint "${EFFDOCK_MIX_INIT_CHECKPOINT:?}" \
  "${EFFDOCK_MIX_INIT_CHECKPOINT_SHA:?}"
.venv/bin/python -c \
  'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert int(x.get("step",-1)) == 50000, x.get("step")' \
  "$EFFDOCK_MIX_INIT_CHECKPOINT" \
  || die "initial confidence checkpoint is not the frozen U50,000 endpoint"
verify filtered-split "${EFFDOCK_MIX_FILTERED_SPLIT:?}" \
  "${EFFDOCK_MIX_FILTERED_SPLIT_SHA:?}"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq "$train_nproc" ]] \
  || die "paired confidence training visible GPU count mismatch"

if [[ "$stage" == smoke ]]; then
  smoke_name=${EFFDOCK_MIX_SMOKE_NAME:-smoke}
  [[ "$smoke_name" =~ ^[a-zA-Z0-9._-]+$ ]] || die "invalid smoke output name"
  smoke_steps=${EFFDOCK_MIX_SMOKE_STEPS:-1}
  smoke_train_limit=${EFFDOCK_MIX_SMOKE_TRAIN_LIMIT:-8}
  smoke_val_limit=${EFFDOCK_MIX_SMOKE_VAL_LIMIT:-2}
  [[ "$smoke_steps" =~ ^[1-9][0-9]*$ && "$smoke_steps" -le 100 ]] \
    || die "smoke steps must be an integer in [1, 100]"
  [[ "$smoke_train_limit" =~ ^[0-9]+$ ]] || die "invalid smoke train limit"
  [[ "$smoke_val_limit" =~ ^[1-9][0-9]*$ ]] || die "invalid smoke val limit"
  out_dir=${EFFDOCK_MIX_TRAIN_OUTPUT_ROOT:?}/$smoke_name
  [[ ! -e "$out_dir" ]] || die "refusing to overwrite training smoke"
  mkdir -p "$out_dir"
  train_mode=(--init_checkpoint "$EFFDOCK_MIX_INIT_CHECKPOINT" \
    --init_checkpoint_sha256 "$EFFDOCK_MIX_INIT_CHECKPOINT_SHA")
  scope_args=(--total_steps "$smoke_steps" --val_limit "$smoke_val_limit" \
    --eval_every "$smoke_steps" --eval_complexes "$smoke_val_limit" \
    --save_every "$smoke_steps" --num_workers 0)
  if (( smoke_train_limit > 0 )); then
    scope_args+=(--train_limit "$smoke_train_limit")
  fi
else
  full_steps=${EFFDOCK_MIX_TOTAL_STEPS:-10000}
  eval_every=${EFFDOCK_MIX_EVAL_EVERY:-2000}
  save_every=${EFFDOCK_MIX_SAVE_EVERY:-500}
  for value in "$full_steps" "$eval_every" "$save_every"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "training step controls must be positive integers"
  done
  out_dir=${EFFDOCK_MIX_TRAIN_OUTPUT_ROOT:?}/full
  mkdir -p "$out_dir"
  exec 9>"$out_dir/.run.lock"
  flock -n 9 || die "another paired-training process holds the lock"
  if [[ -f "$out_dir/latest.pt" ]]; then
    train_mode=(--resume "$out_dir/latest.pt" --no-eval-on-start)
  elif find "$out_dir" -mindepth 1 -maxdepth 1 \
      ! -name '.run.lock' ! -name '*.out' ! -name '*.err' | read -r _; then
    die "non-empty training root has no latest.pt"
  else
    train_mode=(--init_checkpoint "$EFFDOCK_MIX_INIT_CHECKPOINT" \
      --init_checkpoint_sha256 "$EFFDOCK_MIX_INIT_CHECKPOINT_SHA")
  fi
  scope_args=(--total_steps "$full_steps" --eval_every "$eval_every" \
    --save_every "$save_every")
fi

run_name=${EFFDOCK_MIX_RUN_NAME:-effdock-confidence-s50-raw-refined-10k-v2}
[[ "$run_name" =~ ^[a-zA-Z0-9._-]+$ ]] || die "invalid training run name"

export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1
export CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0
# Full validation is intentionally rank-0-only and currently takes about
# 11 minutes for 1,035 complexes while the peer waits at the DDP barrier.
export CONFIDENCE_DDP_TIMEOUT_MIN="${CONFIDENCE_DDP_TIMEOUT_MIN:-20}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv/bin/torchrun --standalone --nproc_per_node="$train_nproc" \
  .venv/bin/eff-dock confidence train --config "$config" \
  --split_file "$EFFDOCK_MIX_FILTERED_SPLIT" \
  --processed_dir "${EFFDOCK_MIX_BANK_OUTPUT_ROOT}/full" \
  --pose_tag s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100 \
  --bank_manifest "$refined_bank" --bank_manifest_sha256 "$refined_bank_sha" \
  --training_target pose_rmsd_symmetry_no_align \
  --train_pose_target_manifest "$refined_target" \
  --train_pose_target_manifest_sha256 "$refined_target_sha" \
  --train_aux_bank_manifest "$EFFDOCK_MIX_RAW_BANK" \
  --train_aux_bank_manifest_sha256 "$EFFDOCK_MIX_RAW_BANK_SHA" \
  --train_aux_pose_tag s50_n100_s10_sig2_latep3_pc10_rdkitseed0 \
  --train_aux_pose_target_manifest "$EFFDOCK_MIX_RAW_TARGET" \
  --train_aux_pose_target_manifest_sha256 "$EFFDOCK_MIX_RAW_TARGET_SHA" \
  --eval_target pose_rmsd_symmetry_no_align --out_dir "$out_dir" \
  --run_name "$run_name" \
  --max_train_poses_per_complex 64 \
  --max_val_poses_per_complex 100 --no-use_wandb \
  "${train_mode[@]}" "${scope_args[@]}"

[[ -f "$out_dir/latest.pt" && -f "$out_dir/best.pt" ]] \
  || die "training did not publish latest.pt and best.pt"
seal "$out_dir/metrics.json" "$out_dir/metrics.json.sha256"
