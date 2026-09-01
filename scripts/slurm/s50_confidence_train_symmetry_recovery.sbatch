#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-conf-sym-r4
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
  [[ -f "$path" && ! -L "$path" ]] || die "missing/non-regular $label: $path"
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
content_sha256=88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97
recovery_runtime_sha256=e0484955d44159ea0ed0b5b8a53f7853ee7a09a23bf34319616e7ea21338a9eb
execution_root="$repo_dir/.effdock_execution_capsules/EFFDOCK-S50-SYMMETRY-CONFIDENCE-RECOVERY-V4/$recovery_runtime_sha256"
source_run="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
symmetry_root="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-training/$content_sha256"
out_dir="$symmetry_root/full"
resume_anchor="$symmetry_root/recovery/anchors/u001500_job56496_latest.pt"

config="$execution_root/configs/train_confidence_s50_symmetry.yaml"
protocol="$execution_root/docs/S50_SYMMETRY_CONFIDENCE_TRAINING_PROTOCOL.md"
bank_root="$source_run/full/bank"
bank_manifest="$source_run/full/bank_manifest.json"
filtered_split="$source_run/full/filtered_split.json"
target_manifest="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-labels/41c5ff349ba95e64b45ac43356b970dc037ebf31898c6533e61a2360880f031f/full/manifest.json"

verify_sha256 config "$config" 906f3723bcc65d2ce1a95b60b4957120ec69338204e85c7e0c30c723630dd684
verify_sha256 protocol "$protocol" d28a5b76e039cf758995f14579f9699754a93a5843c28876cb99717f65fab1f7
verify_sha256 bank-manifest "$bank_manifest" d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
verify_sha256 filtered-split "$filtered_split" 1f23b50ef3a5eff73fd8cad683c1f3adfe4e1ab235199274d0dbc220c1d22507
verify_sha256 symmetry-target-manifest "$target_manifest" 89ea40b02b121387228e3d47461a68a30bd749143b6281fe4d5ea9ab89056981
verify_sha256 dataset-code "$execution_root/src/effdock/confidence/dataset.py" 980c7ca1f680b781bffbdca84c9751c731f11f472a36af9a6a4aa5ea9de838fe
verify_sha256 trainer-code "$execution_root/src/effdock/workflows/train_confidence.py" 9033d716048b214408d897e9913df61038482888e06b253a155f36ba47c6d7c0
verify_sha256 loss-code "$execution_root/src/effdock/confidence/losses.py" 904d9b084af57f448028a6fd8085d0bcf3e57352cc1d385e51fa5a4d35c21159
verify_sha256 resume-anchor "$resume_anchor" 2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15

[[ -x "$execution_root/.venv/bin/torchrun" && -x "$execution_root/.venv/bin/eff-dock" ]] \
  || die "missing execution-capsule entry points"
[[ ${SLURM_JOB_PARTITION:-} == heavy ]] || die "symmetry recovery requires heavy partition"
gpu_inventory=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
[[ $(wc -l <<< "$gpu_inventory") -eq 4 ]] || die "symmetry recovery requires four visible GPUs"

smoke_metrics="$symmetry_root/smoke/metrics.json"
smoke_sidecar="$smoke_metrics.sha256"
[[ -f "$smoke_sidecar" ]] || die "missing completed symmetry smoke sidecar"
read -r smoke_sha smoke_path < "$smoke_sidecar"
[[ "$smoke_path" == "$smoke_metrics" ]] || die "symmetry smoke sidecar path mismatch"
verify_sha256 symmetry-smoke-metrics "$smoke_metrics" "$smoke_sha"

[[ -d "$out_dir" ]] || die "missing original full output root: $out_dir"
[[ ! -e "$out_dir/metrics.json.sha256" ]] || die "full run is already sealed complete"
exec 9>"$out_dir/.run.lock"
flock -n 9 || die "another process holds $out_dir/.run.lock"

mkdir -p "$symmetry_root/ledgers"
ledger="$symmetry_root/ledgers/full-recovery-large-graph-v4.jsonl"
export PYTHONPATH="$execution_root/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0
export CONFIDENCE_DDP_TIMEOUT_MIN=15
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

command=(
  "$execution_root/.venv/bin/torchrun" --standalone --nproc_per_node=4
  "$execution_root/.venv/bin/eff-dock" confidence train
  --config "$config"
  --split_file "$filtered_split"
  --processed_dir "$bank_root"
  --pose_tag s50_n100_s10_sig2_latep3_pc10_rdkitseed0
  --bank_manifest "$bank_manifest"
  --bank_manifest_sha256 d45e36f3f2d75fb8ba9553715e1ec45031e4ad31881631d8632b6c19999f2d2b
  --training_target pose_rmsd_symmetry_no_align
  --train_pose_target_manifest "$target_manifest"
  --train_pose_target_manifest_sha256 89ea40b02b121387228e3d47461a68a30bd749143b6281fe4d5ea9ab89056981
  --eval_target pose_rmsd_symmetry_no_align
  --out_dir "$out_dir"
  --run_name "effdock-s50-confidence-symmetry-$content_sha256-recovery-large-graph-v4"
  --gradient_accumulation_steps 1
  --max_train_poses_per_complex 64
  --max_train_pose_node_product 64000
  --large_train_graph_node_threshold 1200
  --large_train_graph_max_poses 8
  --max_val_poses_per_complex 100
  --num_workers 0
  --resume "$resume_anchor"
  --no-eval-on-start
  --eval_every 5000
  --save_every 500
  --no-use_wandb
)

printf 'recovery_from_jobs=56496,56630,56632 resume_update=1500 num_workers=0 max_train_pose_node_product=64000 large_graph_threshold=1200 large_graph_max_poses=8 runtime=%s\n%s\n' "$recovery_runtime_sha256" "$gpu_inventory"
cd "$execution_root"
oms run-ledger \
  --note "S50 symmetry-confidence recovery V4 from U1500; measured large-graph cap" \
  --file "$ledger" \
  --no-gate \
  --reason "8qfe 1892-node graph exhausted rank0; exact probe measured N8 at 31.1 GiB reserved, so retain N64 normally and use N8 only above 1200 nodes" \
  -- "${command[@]}"

metrics="$out_dir/metrics.json"
seal_sha_sidecar "$metrics" "$metrics.sha256"
printf 'recovery_complete=1 out_dir=%s\n' "$out_dir"
