#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-surfdock
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

SURFDOCK_MODEL_ROOT="${SURFDOCK_MODEL_ROOT:-$SLURM_SUBMIT_DIR/others/surfdock}"
SURFDOCK_ROOT="$SURFDOCK_MODEL_ROOT/upstream"
MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"
SEED="${SEED:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-40}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"
PREPROCESS_DIR="$OUTPUT_DIR/preprocessed"

test -x "$SURFDOCK_MODEL_ROOT/.venv/bin/python"
test -x "$MODEL_RUNNER"
test -f "$SURFDOCK_ROOT/inference_accelerate.py"
test -f "$SURFDOCK_ROOT/model_weights/docking/best_ema_inference_epoch_model.pt"
test -f "$SURFDOCK_ROOT/model_weights/posepredict/best_model.pt"
test -f "$INPUT_CSV"
mkdir -p others/logs "$OUTPUT_DIR" "$SURFDOCK_MODEL_ROOT/.cache/torch" \
  "$SURFDOCK_MODEL_ROOT/.cache/precomputed_arrays"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_surfdock_${SLURM_JOB_ID}_XXXXXX")"
mkdir -p "$RUNTIME_DIR/tmp"
export TMPDIR="$RUNTIME_DIR/tmp"
export PYTHONPATH="$SURFDOCK_ROOT"
export PYTHONHASHSEED="$SEED"
export precomputed_arrays="$SURFDOCK_MODEL_ROOT/.cache/precomputed_arrays"

"$MODEL_RUNNER" surfdock python scripts/external_models/prepare_surfdock_runtime.py \
  --input-csv "$INPUT_CSV" \
  --surfdock-root "$SURFDOCK_ROOT" \
  --output-dir "$PREPROCESS_DIR" \
  --distance-threshold 8

export INPUT_CSV OUTPUT_DIR SEED INFERENCE_STEPS SAMPLES_PER_COMPLEX FAIL_ON_INCOMPLETE
export SURFDOCK_MODEL_ROOT SURFDOCK_ROOT PREPROCESS_DIR
"$MODEL_RUNNER" surfdock python -c '
import json, os, pathlib, subprocess
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "output_dir": str(output_dir.resolve()),
    "seed": int(os.environ["SEED"]),
    "inference_steps": int(os.environ["INFERENCE_STEPS"]),
    "samples_per_complex": int(os.environ["SAMPLES_PER_COMPLEX"]),
    "fail_on_incomplete": os.environ["FAIL_ON_INCOMPLETE"] == "1",
    "surfdock_commit": subprocess.check_output(
        ["git", "-C", os.environ["SURFDOCK_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "pose_generator": "SurfDock surface diffusion",
    "pose_ranker": "SurfDock posepredict MDN",
    "refinement": "none",
    "site_policy": "reference-ligand-supplied 8A surface pocket",
    "receptor_policy": "holo-aligned predicted receptor",
    "esm_policy": "ESM2-650M full-chain embeddings; local 1022-residue windows only for longer chains",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
(output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
'

"$MODEL_RUNNER" surfdock python scripts/external_models/run_seeded_upstream.py \
  --upstream-script "$SURFDOCK_ROOT/inference_accelerate.py" \
  --seed "$SEED" \
  --stub-surfdock-force-optimize \
  --data_csv "$PREPROCESS_DIR/surfdock_inputs.csv" \
  --model_dir "$SURFDOCK_ROOT/model_weights/docking" \
  --ckpt best_ema_inference_epoch_model.pt \
  --confidence_model_dir "$SURFDOCK_ROOT/model_weights/posepredict" \
  --confidence_ckpt best_model.pt \
  --save_docking_result \
  --mdn_dist_threshold_test 3.0 \
  --esm_embeddings_path "$PREPROCESS_DIR/esm/esm2_t33_650M_pocket_embeddings.pt" \
  --run_name "native_s${INFERENCE_STEPS}_n${SAMPLES_PER_COMPLEX}_seed${SEED}" \
  --project effdock_external_surfdock \
  --out_dir "$OUTPUT_DIR" \
  --batch_size "$SAMPLES_PER_COMPLEX" \
  --batch_size_molecule 1 \
  --samples_per_complex "$SAMPLES_PER_COMPLEX" \
  --save_docking_result_number "$SAMPLES_PER_COMPLEX" \
  --head_index 0 \
  --tail_index 1000000 \
  --inference_steps "$INFERENCE_STEPS" \
  --random_seed "$SEED" \
  --num_workers 1 \
  --inference_mode evaluate

summary_args=(
  --input-csv "$INPUT_CSV"
  --output-dir "$OUTPUT_DIR"
  --expected-poses "$SAMPLES_PER_COMPLEX"
)
if test "$FAIL_ON_INCOMPLETE" = 1; then
  summary_args+=(--fail-on-incomplete)
fi
"$MODEL_RUNNER" surfdock python scripts/external_models/summarize_surfdock.py "${summary_args[@]}"
