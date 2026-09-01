#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-rldiff
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
if test -z "${INPUT_CSV:-}"; then
  : "${SHARD_INPUT_DIR:?Set INPUT_CSV or SHARD_INPUT_DIR}"
  : "${DATASET:?Set DATASET for an array run}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT for an array run}"
  : "${SLURM_ARRAY_TASK_ID:?Array task ID is required with SHARD_INPUT_DIR}"
  printf -v SHARD_ID '%03d' "$SLURM_ARRAY_TASK_ID"
  INPUT_CSV="$SHARD_INPUT_DIR/${DATASET}_shard_${SHARD_ID}.csv"
  OUTPUT_DIR="$OUTPUT_ROOT/seed_${SEED:-0}/shard_${SHARD_ID}"
fi
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the run output directory}"
INPUT_CSV="$(readlink -f "$INPUT_CSV")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

EXTERNAL_ROOT="${EXTERNAL_ROOT:-$SLURM_SUBMIT_DIR/external_models}"
RLDIFF_ROOT="$EXTERNAL_ROOT/src/DiffDock-Pocket/RLDiff"
RLDIFF_PYTHON="$EXTERNAL_ROOT/envs/rldiff/bin/python"
STATE_DICT="$EXTERNAL_ROOT/weights/rldiff/DD_Pocket_RL_score_model/DD_Pocket_RL_score_model.pt"
FILTERING_MODEL_DIR="$EXTERNAL_ROOT/weights/rldiff/v1.0.0/confidence_model"
SMINA_PATH="$EXTERNAL_ROOT/envs/rldiff/bin/smina"
GNINA_PATH="$EXTERNAL_ROOT/bin/gnina-rldiff-v1.0"
SEED="${SEED:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-40}"
BATCH_SIZE="${BATCH_SIZE:-5}"
RUN_VARIANT="${RUN_VARIANT:-native}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"

test "$RUN_VARIANT" = native || test "$RUN_VARIANT" = rlpp || test "$RUN_VARIANT" = rlpp_raw
test -x "$RLDIFF_PYTHON"
test -f "$RLDIFF_ROOT/inference.py"
test -f "$INPUT_CSV"
test -f "$STATE_DICT"
test -f "$FILTERING_MODEL_DIR/best_model.pt"
if test "$RUN_VARIANT" = rlpp; then
  test -x "$SMINA_PATH"
  test -x "$GNINA_PATH"
fi

mkdir -p external_models/logs "$OUTPUT_DIR" "$OUTPUT_DIR/cache"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_rldiff_${SLURM_JOB_ID}_XXXXXX")"
cp scripts/external_models/run_seeded_upstream.py "$RUNTIME_DIR/run_seeded_upstream.py"

export INPUT_CSV OUTPUT_DIR SEED INFERENCE_STEPS SAMPLES_PER_COMPLEX BATCH_SIZE
export RUN_VARIANT RLDIFF_ROOT EXTERNAL_ROOT PYTHONHASHSEED="$SEED"
export FAIL_ON_INCOMPLETE

RUN_METADATA="$OUTPUT_DIR/run_metadata.json"
export RUN_METADATA
"$RLDIFF_PYTHON" -c '
import json, os, pathlib, subprocess
output = pathlib.Path(os.environ["RUN_METADATA"])
external_root = pathlib.Path(os.environ["EXTERNAL_ROOT"])
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "output_dir": str(pathlib.Path(os.environ["OUTPUT_DIR"]).resolve()),
    "seed": int(os.environ["SEED"]),
    "inference_steps": int(os.environ["INFERENCE_STEPS"]),
    "samples_per_complex": int(os.environ["SAMPLES_PER_COMPLEX"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "variant": os.environ["RUN_VARIANT"],
    "fail_on_incomplete": os.environ["FAIL_ON_INCOMPLETE"] == "1",
    "rldiff_commit": subprocess.check_output(
        ["git", "-C", str(external_root / "src/rldiff"), "rev-parse", "HEAD"], text=True
    ).strip(),
    "diffdock_pocket_commit": subprocess.check_output(
        ["git", "-C", str(external_root / "src/diffdock-pocket"), "rev-parse", "HEAD"], text=True
    ).strip(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
output.write_text(json.dumps(payload, indent=2) + "\n")
'

wrapper_args=(
  --upstream-script "$RLDIFF_ROOT/inference.py"
  --seed "$SEED"
)
run_args=(
  --protein_ligand_csv "$INPUT_CSV"
  --out_dir "$OUTPUT_DIR"
  --samples_per_complex "$SAMPLES_PER_COMPLEX"
  --batch_size "$BATCH_SIZE"
  --num_workers 1
  --inference_steps "$INFERENCE_STEPS"
  --state_dict "$STATE_DICT"
  --filtering_model_dir "$FILTERING_MODEL_DIR"
  --filtering_ckpt best_model.pt
  --cache_path "$OUTPUT_DIR/cache"
  --skip_existing
)
if test "$RUN_VARIANT" = rlpp || test "$RUN_VARIANT" = rlpp_raw; then
  run_args+=(
    --minimize_and_rerank
  )
fi
if test "$RUN_VARIANT" = rlpp; then
  run_args+=(
    --minimize_workers 1
    --smina_path "$SMINA_PATH"
    --gnina_path "$GNINA_PATH"
  )
fi
if test "$RUN_VARIANT" = rlpp_raw; then
  wrapper_args+=(--stub-rldiff-postprocess)
fi

"$RLDIFF_PYTHON" "$RUNTIME_DIR/run_seeded_upstream.py" \
  "${wrapper_args[@]}" \
  "${run_args[@]}"

"$RLDIFF_PYTHON" -c '
import csv, json, os, pathlib, re
input_csv = pathlib.Path(os.environ["INPUT_CSV"])
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
expected_poses = int(os.environ["SAMPLES_PER_COMPLEX"])
variant = os.environ["RUN_VARIANT"]
with input_csv.open(newline="") as handle:
    target_ids = [row["complex_name"] for row in csv.DictReader(handle)]
coverage = {}
for target_id in target_ids:
    raw_dirs = list(output_dir.glob(f"index*___{target_id}"))
    raw_poses = []
    for raw_dir in raw_dirs:
        raw_poses.extend(
            path for path in raw_dir.glob("rank*.sdf")
            if re.fullmatch(r"rank\d+\.sdf", path.name)
        )
    if variant == "rlpp":
        selected_poses = list((output_dir / "minimized_poses" / target_id).glob("rank*_gnina*.sdf"))
    else:
        selected_poses = raw_poses
    coverage[target_id] = {
        "raw_pose_count": len(raw_poses),
        "selected_pose_count": len(selected_poses),
    }
summary = {
    "schema_version": 1,
    "expected_targets": len(target_ids),
    "targets_with_any_pose": sum(v["selected_pose_count"] > 0 for v in coverage.values()),
    "targets_with_expected_pose_count": sum(
        v["selected_pose_count"] == expected_poses for v in coverage.values()
    ),
    "coverage": coverage,
}
(output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
if (
    os.environ["FAIL_ON_INCOMPLETE"] == "1"
    and summary["targets_with_expected_pose_count"] != summary["expected_targets"]
):
    completed = summary["targets_with_expected_pose_count"]
    expected = summary["expected_targets"]
    raise SystemExit(
        f"Incomplete RLDiff coverage: {completed}/{expected} targets "
        "have the expected pose count"
    )
'
