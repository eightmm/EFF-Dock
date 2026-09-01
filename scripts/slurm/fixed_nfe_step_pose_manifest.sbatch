#!/usr/bin/env bash
#SBATCH --job-name=effdock-nfe-manifest
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err

set -euo pipefail
: "${EFFDOCK_CONTROL_CAPSULE:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_RUN_MODE:?}"
cd "$EFFDOCK_CONTROL_CAPSULE"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE" "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE"
command=(
  .venv/bin/python scripts/build_fixed_nfe_step_pose_manifest.py
  --sampling-root "$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE/sampling"
  --baseline-manifest "$EFFDOCK_N100_MANIFEST"
  --mode "$EFFDOCK_RUN_MODE"
  --output "$EFFDOCK_OUTPUT_ROOT/$EFFDOCK_RUN_MODE/manifest.json"
)
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/$EFFDOCK_RUN_MODE/manifest.jsonl" \
  --note "validate fixed-NFE prior pairing and freeze manifest" \
  --no-gate --reason "mechanical integrity gate for user-authorized descriptive run" \
  -- "${command[@]}"
