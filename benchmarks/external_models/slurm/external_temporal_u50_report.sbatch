#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-u50-report
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
: "${EFFDOCK_REPORTER_SHA256:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

read -r reporter_sha _ < <(sha256sum scripts/report_external_temporal_benchmark.py)
[[ "$reporter_sha" == "$EFFDOCK_REPORTER_SHA256" ]] || { echo "reporter SHA mismatch" >&2; exit 2; }
command=(
  .venv/bin/python scripts/report_external_temporal_benchmark.py
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-dir "$EFFDOCK_OUTPUT_ROOT/report"
  --selector-label "U50k symmetry-confidence stable argmin predicted RMSD"
  --selector-checkpoint-sha256 fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/report.jsonl" \
  --metrics "$EFFDOCK_OUTPUT_ROOT/report/summary.json" \
  --note "U50-selected external temporal benchmark report" \
  --no-gate --reason "all U50 confidence and selected-pose validity shards completed" \
  -- "${command[@]}"
