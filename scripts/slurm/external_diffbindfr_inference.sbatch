#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-dbindfr
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
  INPUT_CSV="$SHARD_INPUT_ROOT/shard_$SHARD_ID/diffbindfr_inputs.csv"
  OUTPUT_DIR="$OUTPUT_ROOT/seed_${SEED:-0}/shard_$SHARD_ID"
fi
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

INPUT_CSV="$(readlink -f "$INPUT_CSV")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

DIFFBINDFR_MODEL_ROOT="${DIFFBINDFR_MODEL_ROOT:-$SLURM_SUBMIT_DIR/others/diffbindfr}"
DIFFBINDFR_ROOT="$DIFFBINDFR_MODEL_ROOT/upstream"
MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"
SEED="${SEED:-0}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-40}"
EXPERIMENT_NAME="native_s20_n${SAMPLES_PER_COMPLEX}_seed${SEED}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"

test -x "$DIFFBINDFR_MODEL_ROOT/.venv/bin/python"
test -x "$MODEL_RUNNER"
test -f "$DIFFBINDFR_ROOT/DiffBindFR/app/predict.py"
test -f "$DIFFBINDFR_ROOT/DiffBindFR/weights/diffbindfr_paper.pth"
test -f "$DIFFBINDFR_ROOT/DiffBindFR/weights/mdn_paper.pt"
test -f "$INPUT_CSV"
mkdir -p others/logs "$OUTPUT_DIR"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_diffbindfr_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$RUNTIME_DIR/tmp"
export TMPDIR="$RUNTIME_DIR/tmp"
export PYTHONPATH="$DIFFBINDFR_ROOT"
export PYTHONHASHSEED="$SEED"
export INPUT_CSV OUTPUT_DIR SEED SAMPLES_PER_COMPLEX EXPERIMENT_NAME
export FAIL_ON_INCOMPLETE DIFFBINDFR_MODEL_ROOT DIFFBINDFR_ROOT

"$MODEL_RUNNER" diffbindfr python -c '
import json, os, pathlib, subprocess
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "output_dir": str(output_dir.resolve()),
    "seed": int(os.environ["SEED"]),
    "declared_inference_steps": 22,
    "actual_inference_steps": 20,
    "samples_per_complex": int(os.environ["SAMPLES_PER_COMPLEX"]),
    "fail_on_incomplete": os.environ["FAIL_ON_INCOMPLETE"] == "1",
    "diffbindfr_commit": subprocess.check_output(
        ["git", "-C", os.environ["DIFFBINDFR_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "pose_generator": "DiffBindFR flexible diffusion",
    "pose_ranker": "DiffBindFR MDN",
    "refinement": "none (--no_error_correction)",
    "site_policy": "reference-ligand-supplied 12A pocket",
    "receptor_policy": "holo-aligned predicted receptor",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
(output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
'

(
  cd "$DIFFBINDFR_ROOT"
  "$MODEL_RUNNER" diffbindfr python \
    "$SLURM_SUBMIT_DIR/scripts/external_models/run_seeded_upstream.py" \
    --upstream-script "$DIFFBINDFR_ROOT/DiffBindFR/app/predict.py" \
    --upstream-cwd "$DIFFBINDFR_ROOT" \
    --seed "$SEED" \
    --stub-diffbindfr-pymol \
    --input_csv "$INPUT_CSV" \
    --export_dir "$OUTPUT_DIR" \
    --num_poses "$SAMPLES_PER_COMPLEX" \
    --gpu_id 0 \
    --num_workers 1 \
    --batch_size 8 \
    --experiment_name "$EXPERIMENT_NAME" \
    --seed "$SEED" \
    --evaluation \
    --no_error_correction
)

summary_args=(
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --experiment-name "$EXPERIMENT_NAME"
  --expected-poses "$SAMPLES_PER_COMPLEX"
)
if test "$FAIL_ON_INCOMPLETE" = 1; then
  summary_args+=(--fail-on-incomplete)
fi
"$MODEL_RUNNER" diffbindfr python scripts/external_models/summarize_diffbindfr.py "${summary_args[@]}"
