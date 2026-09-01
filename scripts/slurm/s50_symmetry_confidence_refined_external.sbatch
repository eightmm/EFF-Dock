#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50c-ext-ref
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
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
: "${EFFDOCK_STAGE:?}"
[[ "$EFFDOCK_STAGE" == smoke || "$EFFDOCK_STAGE" == full ]] || die "invalid stage"
cd "$EFFDOCK_EXECUTION_ROOT"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3

verify_sha scripts/run_s50_symmetry_confidence_refined_external_shard.py 1942152b59000880bdc2ad6bb0c05d6f2e2af36f46aa5e1346209c218a55290a
verify_sha scripts/score_guidance_sdf_post_refinement_confidence.py d586b6bc9205e9c00d2b983f5c2802b03105e96c55a63b908ad543e3d471f808
verify_sha docs/S50_SYMMETRY_CONFIDENCE_REFINED_EXTERNAL_PROTOCOL.md 91bb6dd6092731afd501ba07ed38ce219a75ab70e48786e5de73105471412bed
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json 99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$src_sha" == ffb5d1190064fad81ff45a538105d5c20cf50a455aca3d813fadd6e3ca5fd17e ]] \
  || die "source tree SHA mismatch"

source_root="$EFFDOCK_EXECUTION_ROOT/outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full"
manifest="$EFFDOCK_EXECUTION_ROOT/outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json"
docking="$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt"
verify_sha "$manifest" 9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6

arms=(u001500 u025000 u050000)
checkpoints=(
  "$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/recovery/anchors/u001500_job56496_latest.pt"
  "$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/best.pt"
  "$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt"
)
hashes=(
  2af26bf66bec53676b8344e811911bbf47ee85aa6550610f35c3812b7a7f9d15
  1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030
  fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
)

task_id=${SLURM_ARRAY_TASK_ID:?}
if [[ "$EFFDOCK_STAGE" == smoke ]]; then
  (( task_id >= 0 && task_id < 3 )) || die "smoke task must be 0..2"
  arm_index=$task_id
  shard_index=0
  output_root="$EFFDOCK_OUTPUT_ROOT/smoke"
  smoke_arg=(--smoke)
else
  (( task_id >= 0 && task_id < 96 )) || die "full task must be 0..95"
  arm_index=$((task_id / 32))
  shard_index=$((task_id % 32))
  output_root="$EFFDOCK_OUTPUT_ROOT/full"
  smoke_arg=()
fi
arm=${arms[$arm_index]}
checkpoint=${checkpoints[$arm_index]}
checkpoint_sha=${hashes[$arm_index]}
verify_sha "$checkpoint" "$checkpoint_sha"

mkdir -p "$EFFDOCK_OUTPUT_ROOT/ledgers" outputs/slurm
command=(
  .venv/bin/python scripts/run_s50_symmetry_confidence_refined_external_shard.py
  --manifest "$manifest"
  --refinement-root "$source_root"
  --output-root "$output_root"
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --external-dir data/external_test
  --config configs/train.yaml
  --docking-checkpoint "$docking"
  --confidence-checkpoint "$checkpoint"
  --arm "$arm"
  --expected-confidence-sha256 "$checkpoint_sha"
  --expected-docking-sha256 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
  --num-shards 32
  --shard-index "$shard_index"
  "${smoke_arg[@]}"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-${arm}-shard${shard_index}.jsonl" \
  --note "S50 symmetry confidence on frozen sigma2 refined external bank" \
  --no-gate --reason "user-authorized repeated-use descriptive score-only comparison" \
  -- "${command[@]}"
