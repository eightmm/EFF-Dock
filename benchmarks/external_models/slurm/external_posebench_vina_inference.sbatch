#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-vina
#SBATCH --partition=cpu_only
#SBATCH --qos=long
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
if test -z "${INPUT_CSV:-}"; then
  : "${SHARD_INPUT_ROOT:?Set INPUT_CSV or SHARD_INPUT_ROOT}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT for an array run}"
  : "${SLURM_ARRAY_TASK_ID:?Array task ID is required with SHARD_INPUT_ROOT}"
  printf -v SHARD_ID '%03d' "$SLURM_ARRAY_TASK_ID"
  INPUT_CSV="$SHARD_INPUT_ROOT/shard_$SHARD_ID/vina_inputs.csv"
  OUTPUT_DIR="$OUTPUT_ROOT/seed_${SEED:-0}/shard_$SHARD_ID"
fi
: "${DATASET:?Set DATASET}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

INPUT_CSV="$(readlink -f "$INPUT_CSV")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

EXTERNAL_ROOT="${EXTERNAL_ROOT:-$SLURM_SUBMIT_DIR/external_models}"
POSEBENCH_ROOT="$EXTERNAL_ROOT/src/posebench"
POSEBENCH_PYTHON="$EXTERNAL_ROOT/envs/posebench/bin/python"
ADFR_ROOT="${ADFR_ROOT:-$EXTERNAL_ROOT/envs/posebench-adfr-clean}"
ADFR_PYTHON="${ADFR_PYTHON:-$ADFR_ROOT/bin/python2.7}"
PREPARE_RECEPTOR="${PREPARE_RECEPTOR:-$ADFR_ROOT/CCSBpckgs/AutoDockTools/Utilities24/prepare_receptor4.py}"
SEED="${SEED:-1}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-32}"
NUM_MODES="${NUM_MODES:-40}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"
CPU_WORKERS="${SLURM_CPUS_PER_TASK:-1}"

if test "$SEED" -eq 0; then
  echo "Vina seed 0 requests a random seed; use an explicit nonzero SEED" >&2
  exit 2
fi

test -x "$POSEBENCH_PYTHON"
test -x "$ADFR_PYTHON"
test -f "$PREPARE_RECEPTOR"
test -f "$INPUT_CSV"
"$ADFR_PYTHON" -c 'import numpy; from MolKit import Read; print("ADFR receptor environment OK")'
mkdir -p external_models/logs "$OUTPUT_DIR"
export PATH="$EXTERNAL_ROOT/envs/posebench/bin:$PATH"
export EFFDOCK_VINA_RECEPTOR_WRITER="posebench_adfr_prepare_receptor4"

run_args=(
  --posebench-root "$POSEBENCH_ROOT"
  --posebench-python "$POSEBENCH_PYTHON"
  --adfr-python "$ADFR_PYTHON"
  --prepare-receptor "$PREPARE_RECEPTOR"
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --dataset "$DATASET"
  --seed "$SEED"
  --exhaustiveness "$EXHAUSTIVENESS"
  --num-modes "$NUM_MODES"
  --cpu-workers "$CPU_WORKERS"
)
if test "$FAIL_ON_INCOMPLETE" = 1; then
  run_args+=(--fail-on-incomplete)
fi
"$POSEBENCH_PYTHON" scripts/external_models/run_posebench_vina.py "${run_args[@]}"
