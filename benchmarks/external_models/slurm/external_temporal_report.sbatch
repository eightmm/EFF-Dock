#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

command=(
  .venv/bin/python scripts/report_external_temporal_benchmark.py
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-dir "$EFFDOCK_OUTPUT_ROOT/report"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/report.jsonl" \
  --metrics "$EFFDOCK_OUTPUT_ROOT/report/summary.json" \
  --note "coverage-enforcing external temporal benchmark report" \
  --no-gate --reason "immutable execution capsule passed submission-time focused gates" \
  -- "${command[@]}"
