#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-ddpocket
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
DIFFDOCK_POCKET_ROOT="$EXTERNAL_ROOT/src/DiffDock-Pocket"
DIFFDOCK_POCKET_PYTHON="$EXTERNAL_ROOT/envs/rldiff/bin/python"
MODEL_DIR="$EXTERNAL_ROOT/weights/rldiff/v1.0.0/score_model"
FILTERING_MODEL_DIR="$EXTERNAL_ROOT/weights/rldiff/v1.0.0/confidence_model"
COMPAT_CONFIG="$SLURM_SUBMIT_DIR/configs/external_diffdock_pocket_compat.yaml"
SEED="${SEED:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-30}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-40}"
BATCH_SIZE="${BATCH_SIZE:-5}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"

test -x "$DIFFDOCK_POCKET_PYTHON"
test -f "$DIFFDOCK_POCKET_ROOT/inference.py"
test -f "$INPUT_CSV"
test -f "$MODEL_DIR/best_ema_inference_epoch_model.pt"
test -f "$FILTERING_MODEL_DIR/best_model.pt"
test -f "$COMPAT_CONFIG"

mkdir -p external_models/logs "$OUTPUT_DIR" "$OUTPUT_DIR/cache"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_ddpocket_${SLURM_JOB_ID}_XXXXXX")"
cp scripts/external_models/run_seeded_upstream.py "$RUNTIME_DIR/run_seeded_upstream.py"

export INPUT_CSV OUTPUT_DIR SEED INFERENCE_STEPS SAMPLES_PER_COMPLEX BATCH_SIZE
export FAIL_ON_INCOMPLETE
export DIFFDOCK_POCKET_ROOT EXTERNAL_ROOT PYTHONHASHSEED="$SEED"
RUN_METADATA="$OUTPUT_DIR/run_metadata.json"
export RUN_METADATA
"$DIFFDOCK_POCKET_PYTHON" -c '
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
    "fail_on_incomplete": os.environ["FAIL_ON_INCOMPLETE"] == "1",
    "diffdock_pocket_commit": subprocess.check_output(
        ["git", "-C", str(external_root / "src/diffdock-pocket"), "rev-parse", "HEAD"], text=True
    ).strip(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "runtime_compatibility_alias": "filtering_dir -> filtering_model_dir",
    "optional_relaxation_import_stubbed": True,
}
output.write_text(json.dumps(payload, indent=2) + "\n")
'

"$DIFFDOCK_POCKET_PYTHON" "$RUNTIME_DIR/run_seeded_upstream.py" \
  --upstream-script "$DIFFDOCK_POCKET_ROOT/inference.py" \
  --seed "$SEED" \
  --stub-optional-posebusters-em \
  --config "$COMPAT_CONFIG" \
  --protein_ligand_csv "$INPUT_CSV" \
  --out_dir "$OUTPUT_DIR" \
  --samples_per_complex "$SAMPLES_PER_COMPLEX" \
  --batch_size "$BATCH_SIZE" \
  --num_workers 1 \
  --inference_steps "$INFERENCE_STEPS" \
  --model_dir "$MODEL_DIR" \
  --filtering_model_dir "$FILTERING_MODEL_DIR" \
  --filtering_ckpt best_model.pt \
  --cache_path "$OUTPUT_DIR/cache" \
  --keep_local_structures \
  --skip_existing

"$DIFFDOCK_POCKET_PYTHON" -c '
import csv, json, os, pathlib, re
input_csv = pathlib.Path(os.environ["INPUT_CSV"])
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
expected_poses = int(os.environ["SAMPLES_PER_COMPLEX"])
with input_csv.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
coverage = {}
for row in rows:
    target_id = row["complex_name"]
    protein_name = pathlib.Path(row["experimental_protein"]).name
    ligand_name = pathlib.Path(row["ligand"]).name
    raw_dirs = list(output_dir.glob(f"index*___{protein_name}___{ligand_name}"))
    poses = []
    for raw_dir in raw_dirs:
        poses.extend(
            path for path in raw_dir.glob("rank*_confidence*.sdf")
            if re.fullmatch(r"rank\d+_confidence[-\d.]+\.sdf", path.name)
        )
    coverage[target_id] = {"pose_count": len(poses)}
summary = {
    "schema_version": 1,
    "expected_targets": len(rows),
    "targets_with_any_pose": sum(v["pose_count"] > 0 for v in coverage.values()),
    "targets_with_expected_pose_count": sum(
        v["pose_count"] == expected_poses for v in coverage.values()
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
        f"Incomplete DiffDock-Pocket coverage: {completed}/{expected} targets "
        "have the expected pose count"
    )
'
