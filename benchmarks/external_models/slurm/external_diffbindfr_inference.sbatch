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
BATCH_SIZE="${BATCH_SIZE:-16}"
ERROR_CORRECTION="${ERROR_CORRECTION:-1}"
RECEPTOR_POLICY="${RECEPTOR_POLICY:-fixed holo receptor (DiffBindFR paper redocking protocol)}"
EXPERIMENT_NAME="native_s20_n${SAMPLES_PER_COMPLEX}_seed${SEED}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"
REUSE_RAW_OUTPUT="${REUSE_RAW_OUTPUT:-0}"

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
export INPUT_CSV OUTPUT_DIR SEED SAMPLES_PER_COMPLEX BATCH_SIZE ERROR_CORRECTION
export RECEPTOR_POLICY EXPERIMENT_NAME
export FAIL_ON_INCOMPLETE REUSE_RAW_OUTPUT DIFFBINDFR_MODEL_ROOT DIFFBINDFR_ROOT

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
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "error_correction": os.environ["ERROR_CORRECTION"] == "1",
    "reuse_raw_output": os.environ["REUSE_RAW_OUTPUT"] == "1",
    "refinement": "official smina error correction" if os.environ["ERROR_CORRECTION"] == "1" else "none",
    "site_policy": "reference-ligand-supplied 12A pocket",
    "receptor_policy": os.environ["RECEPTOR_POLICY"],
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
(output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
'

predict_args=(
  --input_csv "$INPUT_CSV"
  --export_dir "$OUTPUT_DIR"
  --num_poses "$SAMPLES_PER_COMPLEX"
  --gpu_id 0
  --num_workers 1
  --batch_size "$BATCH_SIZE"
  --experiment_name "$EXPERIMENT_NAME"
  --seed "$SEED"
  --evaluation
)

run_predict() {
  (
    cd "$DIFFBINDFR_ROOT"
    "$MODEL_RUNNER" diffbindfr python \
      "$SLURM_SUBMIT_DIR/scripts/external_models/run_seeded_upstream.py" \
      --upstream-script "$DIFFBINDFR_ROOT/DiffBindFR/app/predict.py" \
      --upstream-cwd "$DIFFBINDFR_ROOT" \
      --seed "$SEED" \
      --stub-diffbindfr-pymol \
      "${predict_args[@]}" "$@"
  )
}

RESULTS_CSV="$OUTPUT_DIR/$EXPERIMENT_NAME/results/results.csv"
# Export/reload raw poses first.  MDN scoring is deliberately delayed until
# the fixed-width PDB compatibility pass has repaired any >26-chain exports.
# A failed compatibility/scoring pass may explicitly reuse an already complete
# raw export; this never reuses selector scores or a partially generated pose.
if test "$REUSE_RAW_OUTPUT" = 1; then
  test -s "$RESULTS_CSV"
  test -d "$OUTPUT_DIR/$EXPERIMENT_NAME/structures"
  printf 'Reusing complete raw DiffBindFR export: %s\n' "$RESULTS_CSV"
else
  run_predict --no_mdn_scoring --no_error_correction
fi
"$MODEL_RUNNER" diffbindfr python \
  scripts/external_models/repair_diffbindfr_output_pdb.py \
  --results-csv "$RESULTS_CSV"
if test "$ERROR_CORRECTION" = 1; then
  run_predict
  SUMMARY_RESULTS_NAME=results_ec.csv
else
  run_predict --no_error_correction
  SUMMARY_RESULTS_NAME=results.csv
fi

summary_args=(
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --experiment-name "$EXPERIMENT_NAME"
  --expected-poses "$SAMPLES_PER_COMPLEX"
  --results-name "$SUMMARY_RESULTS_NAME"
)
if test "$FAIL_ON_INCOMPLETE" = 1; then
  summary_args+=(--fail-on-incomplete)
fi
"$MODEL_RUNNER" diffbindfr python scripts/external_models/summarize_diffbindfr.py "${summary_args[@]}"
