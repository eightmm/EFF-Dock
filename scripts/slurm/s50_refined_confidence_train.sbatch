#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refconf-train
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
verify_sha256() {
  local label=$1 path=$2 expected=$3 actual
  [[ -f "$path" ]] || die "missing $label: $path"
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || die "$label SHA mismatch"
}
seal_sha_sidecar() {
  local path=$1 sidecar=$2 digest tmp
  [[ -f "$path" && ! -e "$sidecar" ]] || die "cannot seal $sidecar"
  read -r digest _ < <(sha256sum "$path")
  tmp="${sidecar}.tmp.${SLURM_JOB_ID:-$$}"
  printf '%s  %s\n' "$digest" "$path" > "$tmp"
  ln "$tmp" "$sidecar" || die "failed to publish $sidecar"
  rm -f "$tmp"
}

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
scope=${EFFDOCK_REFCONF_TRAIN_SCOPE:?EFFDOCK_REFCONF_TRAIN_SCOPE must be smoke or full}
case "$scope" in smoke|full) ;; *) die "invalid training scope: $scope" ;; esac

source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
refined_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/207611a7fed8805b7f52cdf4648da2ea03e7d1e699c7d47106b632215634fdc2/full"
config="$repo_dir/configs/train_confidence_s50_refined_10k.yaml"
protocol="$repo_dir/docs/S50_REFINED_CONFIDENCE_FINETUNE_PROTOCOL.md"
trainer="$repo_dir/src/effdock/workflows/train_confidence.py"
verify_sha256 config "$config" 5f2228c3d699cca66ef9ede62c8703c0fdb62ec668ed0cbd5b0072da1d60b259
verify_sha256 protocol "$protocol" db6a63aab9b14162e234c836edd1e5fb1c444e5bf4b0a654398f6d70b0794dcf
verify_sha256 trainer "$trainer" b1d4ed441a1097c8ea451419ba8bc4cd4845ac04e28570f310e18e6ec2391203
bank_manifest="$refined_root/confidence_bank/bank_manifest.json"
bank_sidecar="$refined_root/confidence_bank/bank_manifest.json.sha256"
[[ -f "$bank_manifest" && -f "$bank_sidecar" ]] || die "missing refined bank aggregate"
read -r bank_sha declared < "$bank_sidecar"
verify_sha256 refined-bank "$bank_manifest" "$bank_sha"
target_manifest="$refined_root/symmetry_labels_v1/full/manifest.json"
target_sidecar="$target_manifest.sha256"
[[ -f "$target_manifest" && -f "$target_sidecar" ]] \
  || die "missing refined symmetry target aggregate"
read -r target_sha target_path < "$target_sidecar"
[[ "$target_path" == "$target_manifest" ]] || die "symmetry target sidecar path mismatch"
verify_sha256 refined-symmetry-target "$target_manifest" "$target_sha"
filtered_split="$source_root/full/filtered_split.json"
verify_sha256 filtered-split "$filtered_split" 1f23b50ef3a5eff73fd8cad683c1f3adfe4e1ab235199274d0dbc220c1d22507

raw_training="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full"
raw_latest="$raw_training/latest.pt"
[[ -f "$raw_training/metrics.json" && -f "$raw_training/metrics.json.sha256" ]] \
  || die "raw N64 50k metrics are not sealed"
read -r raw_metrics_sha _ < "$raw_training/metrics.json.sha256"
verify_sha256 raw-metrics "$raw_training/metrics.json" "$raw_metrics_sha"
[[ -f "$raw_latest" ]] || die "missing raw N64 latest.pt"
raw_latest_sha=$(sha256sum "$raw_latest" | cut -d' ' -f1)
.venv/bin/python -c \
  'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert int(p["step"]) == 50000' \
  "$raw_latest" || die "raw latest.pt is not U50000"

if [[ "$scope" == smoke ]]; then
  out_dir="$refined_root/training_smoke_n64_10k_v1"
  [[ ! -e "$out_dir" ]] || die "refusing to overwrite refined training smoke"
  mkdir "$out_dir"
  train_mode=(--init_checkpoint "$raw_latest" --init_checkpoint_sha256 "$raw_latest_sha")
  scope_args=(
    --total_steps 1 --train_limit 8 --val_limit 2
    --eval_every 1 --eval_complexes 2 --save_every 1 --num_workers 0
  )
else
  out_dir="$refined_root/training_n64_10k_v1"
  mkdir -p "$out_dir"
  exec 9>"$out_dir/.run.lock"
  flock -n 9 || die "another refined training process holds the run lock"
  if [[ -f "$out_dir/latest.pt" ]]; then
    train_mode=(--resume "$out_dir/latest.pt" --no-eval-on-start)
  elif [[ -f "$out_dir/best.pt" ]]; then
    train_mode=(--resume "$out_dir/best.pt" --no-eval-on-start)
  elif find "$out_dir" -mindepth 1 -maxdepth 1 \
      ! -name '.run.lock' ! -name '*.out' ! -name '*.err' | read -r _; then
    die "non-empty refined training root has no resumable checkpoint"
  else
    train_mode=(--init_checkpoint "$raw_latest" --init_checkpoint_sha256 "$raw_latest_sha")
  fi
  scope_args=(--eval_every 2000 --save_every 500)
fi

export PYTHONPATH="$repo_dir:$repo_dir/src"
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
  .venv/bin/torchrun --standalone --nproc_per_node=4
  .venv/bin/eff-dock confidence train
  --config "$config"
  --split_file "$filtered_split"
  --processed_dir "$refined_root/confidence_bank"
  --pose_tag s50_n100_s10_sig2_latep3_pc10_rdkitseed0_refine100
  --bank_manifest "$bank_manifest"
  --bank_manifest_sha256 "$bank_sha"
  --training_target pose_rmsd_symmetry_no_align
  --train_pose_target_manifest "$target_manifest"
  --train_pose_target_manifest_sha256 "$target_sha"
  --eval_target pose_rmsd_symmetry_no_align
  --out_dir "$out_dir"
  --run_name effdock-confidence-s50-refined-n64-10k-v1
  --gradient_accumulation_steps 1
  --max_train_poses_per_complex 64
  --max_val_poses_per_complex 100
  --no-use_wandb
  "${train_mode[@]}"
  "${scope_args[@]}"
)
mkdir -p "$refined_root/ledgers"
oms run-ledger \
  --note "S50 refined confidence N64 10k $scope; symmetry-label U50000 weights and refined symmetry targets, fresh optimizer" \
  --file "$refined_root/ledgers/train-$scope.jsonl" \
  --no-gate --reason "raw 50k and refined bank dependencies completed and hashes verified" \
  -- "${command[@]}"
seal_sha_sidecar "$out_dir/metrics.json" "$out_dir/metrics.json.sha256"
