#!/usr/bin/env bash
#SBATCH --job-name=effdock-nfe-report
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
cd "$EFFDOCK_CONTROL_CAPSULE"
mkdir -p "$EFFDOCK_OUTPUT_ROOT/ledgers/full"
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/effdock-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
report_python=${EFFDOCK_REPORT_PYTHON:-python3}
command=(
  "$report_python" scripts/report_fixed_nfe_step_pose.py
  --n100-manifest "$EFFDOCK_N100_MANIFEST"
  --n40-manifest "$EFFDOCK_OUTPUT_ROOT/full/manifest.json"
  --n100-scores-root "$EFFDOCK_N100_SCORES_ROOT"
  --n40-scores-root "$EFFDOCK_OUTPUT_ROOT/full/confidence/u050000"
  --baseline-report "$EFFDOCK_N100_REPORT"
  --output-root "$EFFDOCK_OUTPUT_ROOT/report"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/full/report.jsonl" \
  --note "aggregate fixed-NFE cumulative RMSD success curves" \
  --no-gate --reason "user-authorized repeated-use descriptive fixed-NFE comparison" \
  -- "${command[@]}"
