#!/usr/bin/env bash
#SBATCH --job-name=eff-ob-top25-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_SOURCE_RUN:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_NUM_SHARDS:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

command=(
  .venv/bin/python scripts/evaluate_openbind_official_topn.py report
  --source-run "$EFFDOCK_SOURCE_RUN"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --num-shards "$EFFDOCK_NUM_SHARDS"
  --top-n 25
)
if [[ -n "${EFFDOCK_CONFIDENCE_SHA256:-}" ]]; then
  command+=(--confidence-checkpoint-sha256 "$EFFDOCK_CONFIDENCE_SHA256")
fi
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/report.jsonl" \
  --note "Aggregate official-style OpenBind Top-1/5/25 metrics over N=802" \
  --no-gate --reason "all PoseBusters and OpenStructure shards completed" \
  -- "${command[@]}"
