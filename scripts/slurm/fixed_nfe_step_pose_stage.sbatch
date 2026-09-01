#!/usr/bin/env bash
#SBATCH --job-name=effdock-nfe-stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
die() { echo "$*" >&2; exit 2; }
: "${EFFDOCK_CONTROL_CAPSULE:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_RUN_MODE:?}"
: "${EFFDOCK_STAGE:?}"
: "${EFFDOCK_REFINEMENT_CAPSULE:?}"
: "${EFFDOCK_CONFIDENCE_CAPSULE:?}"

[[ "$EFFDOCK_RUN_MODE" == smoke || "$EFFDOCK_RUN_MODE" == full ]] || die "invalid run mode"
[[ "$EFFDOCK_STAGE" == refinement || "$EFFDOCK_STAGE" == confidence ]] || die "invalid stage"
task_id=${SLURM_ARRAY_TASK_ID:?}
task_count=${SLURM_ARRAY_TASK_COUNT:?}
if [[ "$EFFDOCK_RUN_MODE" == smoke ]]; then
  [[ "$task_count" -eq 2 ]] || die "smoke stage requires two tasks"
else
  [[ "$task_count" -eq 32 ]] || die "full stage requires 32 tasks"
fi

ref_capsule_sha=$(sha256sum "$EFFDOCK_REFINEMENT_CAPSULE/execution_capsule.json" | cut -d' ' -f1)
[[ "$ref_capsule_sha" == 4891267ff04d52915be1f3c39a9a78ffa82a19e87589c28971f3cf7d63becb75 ]] \
  || die "refinement capsule identity mismatch"
confidence_src_sha=$(cd "$EFFDOCK_CONFIDENCE_CAPSULE" && find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
[[ "$confidence_src_sha" == ffb5d1190064fad81ff45a538105d5c20cf50a455aca3d813fadd6e3ca5fd17e ]] \
  || die "confidence source identity mismatch"
[[ $(sha256sum "$EFFDOCK_DOCKING_CHECKPOINT" | cut -d' ' -f1) == 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 ]] \
  || die "confidence feature-backbone checkpoint mismatch"
[[ $(sha256sum "$EFFDOCK_CONFIDENCE_CHECKPOINT" | cut -d' ' -f1) == fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469 ]] \
  || die "U50 confidence checkpoint mismatch"

cd "$EFFDOCK_CONTROL_CAPSULE"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE"
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
command=(
  .venv/bin/python scripts/run_fixed_nfe_step_pose_stage.py
  --repo-root "$EFFDOCK_CONTROL_CAPSULE"
  --manifest "$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE/manifest.json"
  --output-root "$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE"
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --external-dir data/external_test
  --protocol-file docs/FIXED_NFE_STEP_POSE_PROTOCOL.md
  --refinement-capsule "$EFFDOCK_REFINEMENT_CAPSULE"
  --confidence-capsule "$EFFDOCK_CONFIDENCE_CAPSULE"
  --config "$EFFDOCK_CONFIDENCE_CAPSULE/configs/train.yaml"
  --docking-checkpoint "$EFFDOCK_DOCKING_CHECKPOINT"
  --confidence-checkpoint "$EFFDOCK_CONFIDENCE_CHECKPOINT"
  --stage "$EFFDOCK_STAGE"
  --num-shards "$task_count"
  --shard-index "$task_id"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE/${EFFDOCK_STAGE}-${task_id}.jsonl" \
  --note "fixed-NFE $EFFDOCK_STAGE shard" \
  --no-gate --reason "user-authorized repeated-use descriptive fixed-NFE comparison" \
  -- "${command[@]}"
