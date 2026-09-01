#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-rlpp-cpu
#SBATCH --partition=cpu_only
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=external_models/logs/%x-%A_%a.out
#SBATCH --error=external_models/logs/%x-%A_%a.err

set -euo pipefail

SEED="${SEED:-0}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-40}"
RECEPTOR_COMPAT_POLICY="${RECEPTOR_COMPAT_POLICY:-fail}"
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
if test -z "${INPUT_CSV:-}"; then
  : "${SHARD_INPUT_DIR:?Set INPUT_CSV or SHARD_INPUT_DIR}"
  : "${DATASET:?Set DATASET for an array run}"
  : "${RAW_OUTPUT_ROOT:?Set RAW_OUTPUT_ROOT for an array run}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT for an array run}"
  : "${SLURM_ARRAY_TASK_ID:?Array task ID is required with SHARD_INPUT_DIR}"
  printf -v SHARD_ID '%03d' "$SLURM_ARRAY_TASK_ID"
  INPUT_CSV="$SHARD_INPUT_DIR/${DATASET}_shard_${SHARD_ID}.csv"
  RAW_OUTPUT_DIR="$RAW_OUTPUT_ROOT/seed_${SEED}/shard_${SHARD_ID}"
  OUTPUT_DIR="$OUTPUT_ROOT/seed_${SEED}/shard_${SHARD_ID}"
fi
: "${RAW_OUTPUT_DIR:?Set RAW_OUTPUT_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

EXTERNAL_ROOT="${EXTERNAL_ROOT:-$SLURM_SUBMIT_DIR/external_models}"
RLDIFF_ROOT="$EXTERNAL_ROOT/src/DiffDock-Pocket/RLDiff"
RLDIFF_PYTHON="$EXTERNAL_ROOT/envs/rldiff/bin/python"
SMINA_PATH="$EXTERNAL_ROOT/envs/rldiff/bin/smina"
GNINA_PATH="$EXTERNAL_ROOT/bin/gnina-rldiff-v1.0"

test -x "$RLDIFF_PYTHON"
test -x "$SMINA_PATH"
test -x "$GNINA_PATH"
test -f "$INPUT_CSV"
test -d "$RAW_OUTPUT_DIR"

mkdir -p external_models/logs "$OUTPUT_DIR"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_rlpp_${SLURM_JOB_ID}_XXXXXX")"
cp scripts/external_models/postprocess_rldiff_rlpp.py "$RUNTIME_DIR/postprocess_rldiff_rlpp.py"
cp scripts/external_models/gnina_cpu_wrapper.sh "$RUNTIME_DIR/gnina_cpu_wrapper.sh"
chmod +x "$RUNTIME_DIR/gnina_cpu_wrapper.sh"

export INPUT_CSV RAW_OUTPUT_DIR OUTPUT_DIR SEED SAMPLES_PER_COMPLEX
export RECEPTOR_COMPAT_POLICY
export GNINA_REAL_PATH="$GNINA_PATH"
export CUDA_VISIBLE_DEVICES=""
RUN_METADATA="$OUTPUT_DIR/run_metadata.json"
export RUN_METADATA
"$RLDIFF_PYTHON" -c '
import json, os, pathlib
output = pathlib.Path(os.environ["RUN_METADATA"])
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "raw_output_dir": str(pathlib.Path(os.environ["RAW_OUTPUT_DIR"]).resolve()),
    "output_dir": str(pathlib.Path(os.environ["OUTPUT_DIR"]).resolve()),
    "seed": int(os.environ["SEED"]),
    "samples_per_complex": int(os.environ["SAMPLES_PER_COMPLEX"]),
    "receptor_compat_policy": os.environ["RECEPTOR_COMPAT_POLICY"],
    "variant": "rlpp_cpu_postprocess",
    "gnina_backend": "cpu (--no_gpu)",
    "partition": os.environ.get("SLURM_JOB_PARTITION"),
}
output.write_text(json.dumps(payload, indent=2) + "\n")
'

"$RLDIFF_PYTHON" "$RUNTIME_DIR/postprocess_rldiff_rlpp.py" \
  --input-csv "$INPUT_CSV" \
  --raw-output-dir "$RAW_OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --rldiff-root "$RLDIFF_ROOT" \
  --smina-path "$SMINA_PATH" \
  --gnina-path "$RUNTIME_DIR/gnina_cpu_wrapper.sh" \
  --samples-per-complex "$SAMPLES_PER_COMPLEX" \
  --receptor-compat-policy "$RECEPTOR_COMPAT_POLICY"
