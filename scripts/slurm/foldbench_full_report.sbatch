#!/usr/bin/env bash
#SBATCH --job-name=effdock-fb558-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
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

reporter=scripts/report_foldbench_full.py
read -r actual _ < <(sha256sum "$reporter")
[[ "$actual" == "$EFFDOCK_REPORTER_SHA" ]] || { echo "reporter SHA mismatch" >&2; exit 2; }
command=(
  .venv/bin/python "$reporter"
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --manifest data/external_benchmarks/foldbench_full/manifests/foldbench.json
  --output-dir "$EFFDOCK_OUTPUT_ROOT/report"
  --selector-checkpoint-sha256 ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/report.jsonl" \
  --note "FoldBench-Pocket-558 final report" \
  --no-gate --reason "mechanical aggregation after complete dependent evaluation" \
  -- "${command[@]}"
