#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-sigmadock
#SBATCH --partition=6000ada
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=others/logs/%x-%j.out
#SBATCH --error=others/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${INPUT_CSV:?Set INPUT_CSV to a prepared SigmaDock PDB/SDF CSV}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
: "${RUN_TAG:?Set RUN_TAG}"
: "${EXPECTED_TARGETS:?Set EXPECTED_TARGETS}"

SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
INFERENCE_STEPS="${INFERENCE_STEPS:-25}"
POCKET_CUTOFF="${POCKET_CUTOFF:-5.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODEL_ID="${MODEL_ID:-sigmadock_v0p1_beta_s25}"
ENSEMBLE_SIZE="${ENSEMBLE_SIZE:-40}"
RECEPTOR_POLICY="${RECEPTOR_POLICY:-caller-declared input CSV receptor}"
SITE_POLICY="${SITE_POLICY:-cognate crystal ligand supplied pocket}"
MODEL_ROOT="${SIGMADOCK_MODEL_ROOT:-$SLURM_SUBMIT_DIR/others/sigmadock}"
UPSTREAM="$MODEL_ROOT/upstream"
MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"
CHECKPOINT="$MODEL_ROOT/weights/checkpoint.ckpt"

INPUT_CSV="$(readlink -f "$INPUT_CSV")"
case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$SLURM_SUBMIT_DIR/$OUTPUT_ROOT" ;;
esac
PREDICTION_DIR="$OUTPUT_ROOT/results/$RUN_TAG/$MODEL_ID/seed_$SEED"
PREDICTION_PATH="$PREDICTION_DIR/predictions.pt"

test -x "$MODEL_ROOT/.venv/bin/python"
test -x "$MODEL_ROOT/bin/gnina"
test -x "$MODEL_RUNNER"
test -f "$UPSTREAM/scripts/sample.py"
test -f "$CHECKPOINT"
test -f "$INPUT_CSV"
mkdir -p others/logs "$OUTPUT_ROOT" "$MODEL_ROOT/.cache/so3"
export PATH="$MODEL_ROOT/bin:$PATH"

# Fail before GPU sampling if the pinned scorer cannot start on this node.
# SigmaDock catches GNINA failures internally, which would otherwise leave a
# seemingly complete prediction file with no usable Vinardo scores.
"$MODEL_RUNNER" sigmadock bash -lc '
  command -v gnina
  gnina --version
'

if test -e "$PREDICTION_DIR" && ! test -f "$PREDICTION_DIR/coverage.json"; then
  echo "Refusing to overwrite incomplete SigmaDock seed directory: $PREDICTION_DIR" >&2
  exit 2
fi

if ! test -f "$PREDICTION_DIR/coverage.json"; then
  "$MODEL_RUNNER" sigmadock python \
    scripts/external_models/run_sigmadock_sample.py \
    "ckpt=$CHECKPOINT" \
    "inference.inference_datafront=$INPUT_CSV" \
    "output_dir=$OUTPUT_ROOT" \
    "run_tag=$RUN_TAG" \
    "seed=$SEED" \
    "model.model_id=$MODEL_ID" \
    "model.cached_s03_dir=$MODEL_ROOT/.cache/so3" \
    "num_seeds=1" \
    "graph.pocket_distance_cutoff=$POCKET_CUTOFF" \
    "diffusion.num_steps=$INFERENCE_STEPS" \
    "data.batch_size=$BATCH_SIZE" \
    "hardware.accelerator=gpu" \
    "hardware.devices=[0]" \
    "postprocessing.scoring=vinardo" \
    "postprocessing.bust_config=null"
fi

export INPUT_CSV OUTPUT_ROOT RUN_TAG SEED INFERENCE_STEPS POCKET_CUTOFF
export BATCH_SIZE MODEL_ID MODEL_ROOT UPSTREAM CHECKPOINT PREDICTION_DIR
export ENSEMBLE_SIZE RECEPTOR_POLICY SITE_POLICY
"$MODEL_RUNNER" sigmadock python -c '
import hashlib, json, os, pathlib, subprocess
checkpoint = pathlib.Path(os.environ["CHECKPOINT"])
hasher = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "output_root": str(pathlib.Path(os.environ["OUTPUT_ROOT"]).resolve()),
    "seed": int(os.environ["SEED"]),
    "inference_steps": int(os.environ["INFERENCE_STEPS"]),
    "samples_per_complex": 1,
    "ensemble_size": int(os.environ["ENSEMBLE_SIZE"]),
    "pocket_cutoff": float(os.environ["POCKET_CUTOFF"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "model_id": os.environ["MODEL_ID"],
    "source_commit": subprocess.check_output(
        ["git", "-C", os.environ["UPSTREAM"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "checkpoint_release": "v0.1.0-beta",
    "checkpoint_sha256": digest,
    "pose_generator": "SigmaDock fragmented SE(3) diffusion",
    "pose_ranker": "Vinardo (GNINA 1.3.2, CPU score-only)",
    "receptor_policy": os.environ["RECEPTOR_POLICY"],
    "site_policy": os.environ["SITE_POLICY"],
    "protein_parser_compat": "fallback removes only nonstandard inter-residue RDKit proximity bonds; complete parses remove only whole PDB residue groups lacking CA before SigmaDock graph construction",
    "ligand_parser_compat": "restores missing SDF double-bond stereochemistry from existing 3D coordinates before independent conformer generation",
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
path = pathlib.Path(os.environ["PREDICTION_DIR"]) / "run_metadata.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
'

"$MODEL_RUNNER" sigmadock python \
  scripts/external_models/summarize_sigmadock.py \
  --input-csv "$INPUT_CSV" \
  --prediction-path "$PREDICTION_PATH" \
  --output-json "$PREDICTION_DIR/coverage.json" \
  --expected-targets "$EXPECTED_TARGETS" \
  --require-vinardo
