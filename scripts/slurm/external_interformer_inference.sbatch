#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-interf
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=others/logs/%x-%j.out
#SBATCH --error=others/logs/%x-%j.err

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
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

INPUT_CSV="$(readlink -f "$INPUT_CSV")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

INTERFORMER_MODEL_ROOT="${INTERFORMER_MODEL_ROOT:-$SLURM_SUBMIT_DIR/others/interformer}"
INTERFORMER_ROOT="$INTERFORMER_MODEL_ROOT/upstream"
INTERFORMER_TORCH_LIB="$INTERFORMER_MODEL_ROOT/.venv/lib/python3.12/site-packages/torch/lib"
REDUCE_BIN="$INTERFORMER_MODEL_ROOT/bin/reduce"
MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"
SEED="${SEED:-0}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-20}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"
WORK_DIR="$OUTPUT_DIR/work"
ENERGY_DIR="$OUTPUT_DIR/energy"

test -x "$INTERFORMER_MODEL_ROOT/.venv/bin/python"
test -x "$MODEL_RUNNER"
test -f "$INTERFORMER_TORCH_LIB/libc10.so"
test -x "$REDUCE_BIN"
test -f "$INTERFORMER_ROOT/inference.py"
test -d "$INTERFORMER_ROOT/checkpoints/v0.2_energy_model"
test -f "$INPUT_CSV"
mkdir -p others/logs "$OUTPUT_DIR"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_interformer_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$RUNTIME_DIR/tmp"
export TMPDIR="$RUNTIME_DIR/tmp"
export LD_LIBRARY_PATH="$INTERFORMER_TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED="$SEED"
export INPUT_CSV OUTPUT_DIR SEED SAMPLES_PER_COMPLEX FAIL_ON_INCOMPLETE
export INTERFORMER_MODEL_ROOT INTERFORMER_ROOT WORK_DIR ENERGY_DIR

"$MODEL_RUNNER" interformer python scripts/external_models/prepare_interformer_runtime.py \
  --input-csv "$INPUT_CSV" \
  --interformer-root "$INTERFORMER_ROOT" \
  --output-dir "$WORK_DIR" \
  --reduce-bin "$REDUCE_BIN"

"$MODEL_RUNNER" interformer python -c '
import json, os, pathlib, subprocess
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "output_dir": str(output_dir.resolve()),
    "seed": int(os.environ["SEED"]),
    "monte_carlo_repeats": 64,
    "steps_per_monte_carlo_repeat": 2000,
    "samples_per_complex": int(os.environ["SAMPLES_PER_COMPLEX"]),
    "fail_on_incomplete": os.environ["FAIL_ON_INCOMPLETE"] == "1",
    "interformer_commit": subprocess.check_output(
        ["git", "-C", os.environ["INTERFORMER_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "pose_generator": "Interformer learned energy + native PyVina Monte Carlo/BFGS",
    "pose_ranker": "native learned Interformer energy",
    "refinement": "native PyVina BFGS",
    "site_policy": "reference-ligand-supplied pocket",
    "receptor_policy": "holo-aligned predicted receptor",
    "initial_conformer": "upstream ETKDGv3/UFF",
    "pyvina_seed_policy": "seed+i over the scheduler-provided CPU affinity set",
    "appended_input_pose_policy": "excluded from pose count and evaluation",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
(output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
'

(
  cd "$INTERFORMER_ROOT"
  PYTHONPATH="$INTERFORMER_ROOT/interformer" "$MODEL_RUNNER" interformer python inference.py \
    -test_csv "$WORK_DIR/query.csv" \
    -work_path "$WORK_DIR" \
    -ensemble checkpoints/v0.2_energy_model \
    -gpus 1 \
    -batch_size 1 \
    -posfix '*val_loss*' \
    -energy_output_folder "$ENERGY_DIR" \
    -seed "$SEED" \
    -reload \
    -debug

  "$MODEL_RUNNER" interformer python "$SLURM_SUBMIT_DIR/scripts/external_models/run_seeded_interformer.py" \
    --upstream-script "$INTERFORMER_ROOT/docking/reconstruct_ligands.py" \
    --seed "$SEED" \
    --normalize-db-markers \
    -y -s \
    --cwd "$ENERGY_DIR" \
    --find_all \
    --uff_folder uff \
    --num_output_poses "$SAMPLES_PER_COMPLEX" \
    --continue_dock_index 0 \
    find
  "$MODEL_RUNNER" interformer python docking/reconstruct_ligands.py \
    --cwd "$ENERGY_DIR" \
    --find_all \
    --uff_folder uff \
    stat
)

summary_args=(
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --expected-poses "$SAMPLES_PER_COMPLEX"
)
if test "$FAIL_ON_INCOMPLETE" = 1; then
  summary_args+=(--fail-on-incomplete)
fi
"$MODEL_RUNNER" interformer python scripts/external_models/summarize_interformer.py "${summary_args[@]}"
