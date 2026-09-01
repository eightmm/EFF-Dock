#!/usr/bin/env bash
#SBATCH --job-name=effdock-rrc-ext
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
: "${EFFDOCK_PROTOCOL_SHA:?}"
: "${EFFDOCK_RUNNER_SHA:?}"
: "${EFFDOCK_HELPER_RUNNER_SHA:?}"
: "${EFFDOCK_SCORER_SHA:?}"
: "${EFFDOCK_SRC_SHA:?}"
[[ "$EFFDOCK_STAGE" == smoke || "$EFFDOCK_STAGE" == full ]] || die "invalid stage"

cd "$EFFDOCK_EXECUTION_ROOT"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3

verify_sha docs/S50_RAW_REFINED_CONFIDENCE_EXTERNAL_PROTOCOL.md "$EFFDOCK_PROTOCOL_SHA"
verify_sha scripts/run_s50_raw_refined_confidence_external_shard.py "$EFFDOCK_RUNNER_SHA"
verify_sha scripts/run_s50_symmetry_confidence_refined_external_shard.py "$EFFDOCK_HELPER_RUNNER_SHA"
verify_sha scripts/score_guidance_sdf_post_refinement_confidence.py "$EFFDOCK_SCORER_SHA"
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json 99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$src_sha" == "$EFFDOCK_SRC_SHA" ]] || die "source tree SHA mismatch"

source_root="$EFFDOCK_EXECUTION_ROOT/outputs/benchmarks/guidance_sdf_post_refinement_runs/sigma2-eta2-adaptive-20260819T062833Z/full"
manifest="$EFFDOCK_EXECUTION_ROOT/outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json"
docking="$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt"
verify_sha "$manifest" 9e8be4d47dba8e346a6900b6bf02f5b853a93141f571f5c65b4c719de632d695
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6

arms=(u070000 u100000)
checkpoints=(
  "$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/best.pt"
  "$EFFDOCK_EXECUTION_ROOT/outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/latest.pt"
)
hashes=(
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
  2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8
)

task_id=${SLURM_ARRAY_TASK_ID:?}
if [[ "$EFFDOCK_STAGE" == smoke ]]; then
  (( task_id >= 0 && task_id < 2 )) || die "smoke task must be 0..1"
  arm_index=$task_id
  shard_index=0
  output_root="$EFFDOCK_OUTPUT_ROOT/smoke"
  smoke_arg=(--smoke)
else
  (( task_id >= 0 && task_id < 64 )) || die "full task must be 0..63"
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
  .venv/bin/python scripts/run_s50_raw_refined_confidence_external_shard.py
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
  --note "U70k/U100k confidence on frozen sigma2 raw+refined external bank" \
  --no-gate --reason "user-authorized repeated-use descriptive score-only comparison" \
  -- "${command[@]}"
