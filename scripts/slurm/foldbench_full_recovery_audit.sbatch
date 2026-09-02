#!/usr/bin/env bash
#SBATCH --job-name=effdock-fb558-audit
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_RECOVERY_ARCHIVE:?}"
: "${EFFDOCK_AUDITOR_SHA:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONPATH="$EFFDOCK_EXECUTION_ROOT/src:$EFFDOCK_EXECUTION_ROOT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

auditor=scripts/audit_foldbench_full_recovery.py
read -r actual _ < <(sha256sum "$auditor")
[[ "$actual" == "$EFFDOCK_AUDITOR_SHA" ]] || { echo "auditor SHA mismatch" >&2; exit 2; }
command=(
  .venv/bin/python "$auditor"
  --run-root "$EFFDOCK_OUTPUT_ROOT"
  --archive-dir "$EFFDOCK_RECOVERY_ARCHIVE"
  --output "$EFFDOCK_OUTPUT_ROOT/recovery/audit.json"
)
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/recovery-audit.jsonl" \
  --note "FoldBench-Pocket-558 recovery identity audit" \
  --no-gate --reason "mechanical verification of two failed-shard replacements" \
  -- "${command[@]}"
