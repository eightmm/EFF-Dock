#!/usr/bin/env bash
#SBATCH --job-name=effdock-rrc-temp-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_REPORTER_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

read -r reporter_sha _ < <(sha256sum scripts/report_s50_raw_refined_confidence_temporal_external.py)
[[ "$reporter_sha" == "$EFFDOCK_REPORTER_SHA" ]] || { echo "reporter SHA mismatch" >&2; exit 2; }
command=(
  .venv/bin/python scripts/report_s50_raw_refined_confidence_temporal_external.py
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-dir "$EFFDOCK_OUTPUT_ROOT/report"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/report.jsonl" \
  --metrics "$EFFDOCK_OUTPUT_ROOT/report/summary.json" \
  --note "paired U70k/U100k temporal-external report" \
  --no-gate --reason "all confidence and selected-pose validity shards completed" \
  -- "${command[@]}"
