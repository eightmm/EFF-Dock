#!/usr/bin/env bash
#SBATCH --job-name=effdock-phi-top5-report
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

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
: "${EFFDOCK_REPORTER_SHA:?}"
: "${EFFDOCK_EVALUATOR_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

verify_sha scripts/report_phibench_u70k_top5.py "$EFFDOCK_REPORTER_SHA"
verify_sha scripts/evaluate_phibench_u70k_top5_posebusters_shard.py "$EFFDOCK_EVALUATOR_SHA"
command=(
  .venv/bin/python scripts/report_phibench_u70k_top5.py
  --input-root "$EFFDOCK_OUTPUT_ROOT"
  --output-dir "$EFFDOCK_OUTPUT_ROOT/report"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/report.jsonl" \
  --note "Aggregate PhiBench U70k Top-5 results" \
  --no-gate --reason "mechanical aggregation of frozen descriptive results" \
  -- "${command[@]}"
