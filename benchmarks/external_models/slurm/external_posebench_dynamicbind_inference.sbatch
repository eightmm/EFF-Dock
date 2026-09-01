#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-dynbind
#SBATCH --partition=test
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
if test -z "${LIGAND_CSV_DIR:-}"; then
  : "${SHARD_INPUT_ROOT:?Set LIGAND_CSV_DIR or SHARD_INPUT_ROOT}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT for an array run}"
  : "${SLURM_ARRAY_TASK_ID:?Array task ID is required with SHARD_INPUT_ROOT}"
  printf -v SHARD_ID '%03d' "$SLURM_ARRAY_TASK_ID"
  SHARD_ROOT="$SHARD_INPUT_ROOT/shard_$SHARD_ID"
  LIGAND_CSV_DIR="$SHARD_ROOT/dynamicbind_inputs"
  PROTEIN_DIR="$SHARD_ROOT/proteins"
  OUTPUT_DIR="$OUTPUT_ROOT/seed_${SEED:-42}/shard_$SHARD_ID"
fi
: "${DATASET:?Set DATASET}"
: "${PROTEIN_DIR:?Set PROTEIN_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

LIGAND_CSV_DIR="$(readlink -f "$LIGAND_CSV_DIR")"
PROTEIN_DIR="$(readlink -f "$PROTEIN_DIR")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

EXTERNAL_ROOT="${EXTERNAL_ROOT:-$SLURM_SUBMIT_DIR/external_models}"
DYNAMICBIND_ROOT="$EXTERNAL_ROOT/src/posebench/forks/DynamicBind"
DYNAMICBIND_PYTHON="$DYNAMICBIND_ROOT/DynamicBind/bin/python3"
RUNTIME_INFERENCE_ROOT="$SLURM_SUBMIT_DIR/outputs/external_models/runs/posebench_dynamicbind/runtime"
SEED="${SEED:-42}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"
ONLY_TARGET="${ONLY_TARGET:-}"
RUN_TAG="${RUN_TAG:-${DATASET}_s${INFERENCE_STEPS}_n${SAMPLES_PER_COMPLEX}_seed${SEED}}"

test -x "$DYNAMICBIND_PYTHON"
test -f "$DYNAMICBIND_ROOT/run_single_protein_inference.py"
test -d "$LIGAND_CSV_DIR"
test -d "$PROTEIN_DIR"
mkdir -p external_models/logs "$OUTPUT_DIR" "$RUNTIME_INFERENCE_ROOT"
if test -e "$DYNAMICBIND_ROOT/inference" || test -L "$DYNAMICBIND_ROOT/inference"; then
  test "$(readlink -f "$DYNAMICBIND_ROOT/inference")" = "$(readlink -f "$RUNTIME_INFERENCE_ROOT")"
else
  ln -s "$RUNTIME_INFERENCE_ROOT" "$DYNAMICBIND_ROOT/inference"
fi

run_args=(
  --dynamicbind-root "$DYNAMICBIND_ROOT"
  --python "$DYNAMICBIND_PYTHON"
  --protein-dir "$PROTEIN_DIR"
  --ligand-csv-dir "$LIGAND_CSV_DIR"
  --output-dir "$OUTPUT_DIR"
  --run-tag "$RUN_TAG"
  --seed "$SEED"
  --inference-steps "$INFERENCE_STEPS"
  --samples-per-complex "$SAMPLES_PER_COMPLEX"
  --batch-size "$BATCH_SIZE"
)
if test "$FAIL_ON_INCOMPLETE" = 1; then
  run_args+=(--fail-on-incomplete)
fi
if test -n "$ONLY_TARGET"; then
  run_args+=(--target-id "$ONLY_TARGET")
fi

"$DYNAMICBIND_PYTHON" scripts/external_models/run_posebench_dynamicbind.py "${run_args[@]}"
