#!/usr/bin/env bash
#SBATCH --job-name=eff-ob-ost25
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

: "${EFFDOCK_EXECUTION_ROOT:?}"
: "${EFFDOCK_SOURCE_RUN:?}"
: "${EFFDOCK_OUTPUT_ROOT:?}"
: "${EFFDOCK_STAGE:?}"
: "${EFFDOCK_NUM_SHARDS:?}"
cd "$EFFDOCK_EXECUTION_ROOT"
mkdir -p outputs/slurm "$EFFDOCK_OUTPUT_ROOT/ledgers"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS" OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

shard_index=${SLURM_ARRAY_TASK_ID:-0}
command=(
  .venvs/openstructure/bin/python scripts/evaluate_openbind_official_topn.py openstructure-shard
  --source-run "$EFFDOCK_SOURCE_RUN"
  --output-root "$EFFDOCK_OUTPUT_ROOT"
  --stage "$EFFDOCK_STAGE"
  --num-shards "$EFFDOCK_NUM_SHARDS"
  --shard-index "$shard_index"
  --top-n 25
  --ost-bin .venvs/openstructure/bin/ost
)
if [[ -n "${EFFDOCK_MAX_COMPLEXES:-}" ]]; then
  command+=(--max-complexes "$EFFDOCK_MAX_COMPLEXES")
fi
if [[ -n "${EFFDOCK_CONFIDENCE_SHA256:-}" ]]; then
  command+=(--confidence-checkpoint-sha256 "$EFFDOCK_CONFIDENCE_SHA256")
fi
oms run-ledger \
  --file "$EFFDOCK_OUTPUT_ROOT/ledgers/${EFFDOCK_STAGE}-openstructure-shard${shard_index}.jsonl" \
  --note "OpenBind official-style Top-25 BiSyRMSD and LDDT-PLI evaluation" \
  --no-gate --reason "PoseBusters prerequisite and isolated OpenStructure runtime are frozen" \
  -- "${command[@]}"
