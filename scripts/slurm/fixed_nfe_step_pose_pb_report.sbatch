#!/usr/bin/env bash
#SBATCH --job-name=effdock-nfe-pb-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err

set -euo pipefail
: "${EFFDOCK_CONTROL_CAPSULE:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_SOURCE_REPORT:?}"
: "${EFFDOCK_SOURCE_REPORT_SHA256:?}"
: "${EFFDOCK_EVALUATOR_SHA256:?}"
: "${EFFDOCK_PROTOCOL_SHA256:?}"
cd "$EFFDOCK_CONTROL_CAPSULE"
read -r evaluator_sha _ < <(sha256sum scripts/evaluate_fixed_nfe_step_pose_pb.py)
read -r protocol_sha _ < <(sha256sum docs/FIXED_NFE_STEP_POSE_PB_PROTOCOL.md)
[[ "$evaluator_sha" == "$EFFDOCK_EVALUATOR_SHA256" ]]
[[ "$protocol_sha" == "$EFFDOCK_PROTOCOL_SHA256" ]]
workers=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$workers" MKL_NUM_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/ledgers/full"
command=(
  .venv/bin/python scripts/evaluate_fixed_nfe_step_pose_pb.py report
  --source-report "$EFFDOCK_SOURCE_REPORT"
  --expected-source-sha256 "$EFFDOCK_SOURCE_REPORT_SHA256"
  --shards-root "$EFFDOCK_OUTPUT_ROOT/full/shards"
  --num-shards 32
  --output-root "$EFFDOCK_OUTPUT_ROOT/report"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/full/report.jsonl" \
  --note "aggregate fixed-NFE frozen-U50 selected-pose PoseBusters comparison" \
  --no-gate --reason "user-authorized repeated-use descriptive validity comparison" \
  -- "${command[@]}"
