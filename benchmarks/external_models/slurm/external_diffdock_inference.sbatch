#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-ddock
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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

# The pinned PoseBench entry point changes into the upstream DiffDock checkout.
# Keep scheduler-provided relative paths valid across that cwd change.
INPUT_CSV="$(readlink -f "$INPUT_CSV")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

POSEBENCH_ROOT="${POSEBENCH_ROOT:-$SLURM_SUBMIT_DIR/external_models/src/posebench}"
DIFFDOCK_ROOT="${DIFFDOCK_ROOT:-$POSEBENCH_ROOT/forks/DiffDock}"
DIFFDOCK_PYTHON="${DIFFDOCK_PYTHON:-$DIFFDOCK_ROOT/DiffDock/bin/python3}"
SEED="${SEED:-0}"
INFERENCE_STEPS="${INFERENCE_STEPS:-20}"
ACTUAL_STEPS="${ACTUAL_STEPS:-19}"
SAMPLES_PER_COMPLEX="${SAMPLES_PER_COMPLEX:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"
TRACE_TARGET_FAILURES="${TRACE_TARGET_FAILURES:-0}"
ISOLATE_TARGETS="${ISOLATE_TARGETS:-1}"
ALLOW_NATIVE_UNSUPPORTED="${ALLOW_NATIVE_UNSUPPORTED:-1}"

test -x "$DIFFDOCK_PYTHON"
test -f "$INPUT_CSV"
test -f "$DIFFDOCK_ROOT/inference.py"
test -f "$DIFFDOCK_ROOT/default_inference_args.yaml"
test -f "$DIFFDOCK_ROOT/workdir/v1.1/score_model/best_ema_inference_epoch_model.pt"
test -f "$DIFFDOCK_ROOT/workdir/v1.1/confidence_model/best_model_epoch75.pt"

mkdir -p external_models/logs "$OUTPUT_DIR"
RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" "effdock_external_diffdock_${SLURM_JOB_ID}_XXXXXX")"
cp scripts/external_models/run_seeded_diffdock.py "$RUNTIME_DIR/run_seeded_diffdock.py"

RUN_METADATA="$OUTPUT_DIR/run_metadata.json"
export RUN_METADATA INPUT_CSV OUTPUT_DIR SEED INFERENCE_STEPS ACTUAL_STEPS
export SAMPLES_PER_COMPLEX BATCH_SIZE POSEBENCH_ROOT
export FAIL_ON_INCOMPLETE
export TRACE_TARGET_FAILURES
export ISOLATE_TARGETS ALLOW_NATIVE_UNSUPPORTED
export PYTHONHASHSEED="$SEED"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

"$DIFFDOCK_PYTHON" -c '
import json, os, pathlib, subprocess
output = pathlib.Path(os.environ["RUN_METADATA"])
payload = {
    "schema_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "input_csv": str(pathlib.Path(os.environ["INPUT_CSV"]).resolve()),
    "output_dir": str(pathlib.Path(os.environ["OUTPUT_DIR"]).resolve()),
    "seed": int(os.environ["SEED"]),
    "inference_steps": int(os.environ["INFERENCE_STEPS"]),
    "actual_steps": int(os.environ["ACTUAL_STEPS"]),
    "samples_per_complex": int(os.environ["SAMPLES_PER_COMPLEX"]),
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "fail_on_incomplete": os.environ["FAIL_ON_INCOMPLETE"] == "1",
    "isolate_targets": os.environ["ISOLATE_TARGETS"] == "1",
    "allow_native_unsupported": os.environ["ALLOW_NATIVE_UNSUPPORTED"] == "1",
    "compatibility": {
        "empty_torsion_edge_output": "use requested rotatable-bond out_nodes",
        "target_process_isolation": os.environ["ISOLATE_TARGETS"] == "1",
        "isolation_seed_scope": (
            "reset base repeat seed per target"
            if os.environ["ISOLATE_TARGETS"] == "1"
            else "single manifest RNG stream"
        ),
        "native_unsupported_policy": (
            "retain in denominator as zero-pose terminal outcome"
            if os.environ["ALLOW_NATIVE_UNSUPPORTED"] == "1"
            else "fail shard"
        ),
    },
    "posebench_commit": subprocess.check_output(
        ["git", "-C", os.environ["POSEBENCH_ROOT"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
output.write_text(json.dumps(payload, indent=2) + "\n")
'

runner_args=(
  --upstream-script "$DIFFDOCK_ROOT/inference.py"
  --seed "$SEED"
  --failure-log "$OUTPUT_DIR/target_failures.json"
)
if test "$TRACE_TARGET_FAILURES" = 1; then
  runner_args+=(--trace-target-failures)
fi
if test "$ISOLATE_TARGETS" = 1; then
  runner_args+=(--isolate-targets)
fi

"$DIFFDOCK_PYTHON" "$RUNTIME_DIR/run_seeded_diffdock.py" "${runner_args[@]}" \
  --protein_ligand_csv "$INPUT_CSV" \
  --out_dir "$OUTPUT_DIR" \
  --inference_steps "$INFERENCE_STEPS" \
  --samples_per_complex "$SAMPLES_PER_COMPLEX" \
  --batch_size "$BATCH_SIZE" \
  --actual_steps "$ACTUAL_STEPS" \
  --no_final_step_noise \
  --cuda_device_index 0 \
  --model_dir "$DIFFDOCK_ROOT/workdir/v1.1/score_model" \
  --confidence_model_dir "$DIFFDOCK_ROOT/workdir/v1.1/confidence_model" \
  --config "$DIFFDOCK_ROOT/default_inference_args.yaml" \
  --skip_existing

"$DIFFDOCK_PYTHON" -c '
import csv, json, os, pathlib, re
input_csv = pathlib.Path(os.environ["INPUT_CSV"])
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
failure_log = output_dir / "target_failures.json"
failures = json.loads(failure_log.read_text()).get("failures", {}) if failure_log.is_file() else {}
with input_csv.open(newline="") as handle:
    target_ids = [row["complex_name"] for row in csv.DictReader(handle)]
coverage = {}
for target_id in target_ids:
    target_dir = output_dir / target_id
    if not target_dir.is_dir():
        # The pinned DiffDock occasionally truncates an all-numeric CCD suffix
        # and writes e.g. 1OWE_675 under 1OWE. Dataset PDB IDs are unique.
        bare_target_dir = output_dir / target_id.split("_", 1)[0]
        if bare_target_dir.is_dir():
            target_dir = bare_target_dir
    poses = sorted(target_dir.glob("rank*_confidence*.sdf")) if target_dir.is_dir() else []
    rank_indices = set()
    for pose in poses:
        match = re.match(r"rank(\d+)_confidence", pose.name)
        if match:
            rank_indices.add(int(match.group(1)))
    failure = failures.get(target_id)
    complete = len(rank_indices) == int(os.environ["SAMPLES_PER_COMPLEX"])
    native_unsupported = bool(failure and failure.get("kind") == "native_unsupported")
    coverage[target_id] = {
        "pose_count": len(rank_indices),
        "pose_file_count": len(poses),
        "rank1_exists": any(target_dir.glob("rank1*.sdf")) if target_dir.is_dir() else False,
        "status": (
            "complete" if complete else
            "native_unsupported" if native_unsupported else
            "failed" if failure else
            "missing"
        ),
        "failure": failure,
    }
expected_pose_count = int(os.environ["SAMPLES_PER_COMPLEX"])
allow_native_unsupported = os.environ["ALLOW_NATIVE_UNSUPPORTED"] == "1"
summary = {
    "schema_version": 1,
    "expected_targets": len(target_ids),
    "targets_with_rank1": sum(v["rank1_exists"] for v in coverage.values()),
    "targets_with_expected_pose_count": sum(
        v["pose_count"] == expected_pose_count
        for v in coverage.values()
    ),
    "targets_native_unsupported": sum(
        v["status"] == "native_unsupported" for v in coverage.values()
    ),
    "targets_with_terminal_outcome": sum(
        v["pose_count"] == expected_pose_count
        or (allow_native_unsupported and v["status"] == "native_unsupported")
        for v in coverage.values()
    ),
    "coverage": coverage,
}
(output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
if (
    os.environ["FAIL_ON_INCOMPLETE"] == "1"
    and summary["targets_with_terminal_outcome"] != summary["expected_targets"]
):
    completed = summary["targets_with_terminal_outcome"]
    expected = summary["expected_targets"]
    raise SystemExit(
        f"Incomplete DiffDock coverage: {completed}/{expected} targets have "
        "either the expected pose count or an explicitly accepted native limitation"
    )
'
