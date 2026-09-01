#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-fabind
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=external_models/logs/%x-%j.out
#SBATCH --error=external_models/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
if test -z "${INPUT_CSV:-}"; then
  : "${SHARD_INPUT_ROOT:?Set INPUT_CSV or SHARD_INPUT_ROOT}"
  : "${OUTPUT_ROOT:?Set OUTPUT_ROOT for an array run}"
  : "${SLURM_ARRAY_TASK_ID:?Array task ID is required with SHARD_INPUT_ROOT}"
  printf -v SHARD_ID '%03d' "$SLURM_ARRAY_TASK_ID"
  SHARD_ROOT="$SHARD_INPUT_ROOT/shard_$SHARD_ID"
  INPUT_CSV="$SHARD_ROOT/fabind_inputs.csv"
  PROTEIN_DIR="$SHARD_ROOT/proteins"
  OUTPUT_DIR="$OUTPUT_ROOT/seed_${SEED:-128}/shard_$SHARD_ID"
fi
: "${DATASET:?Set DATASET}"
: "${PROTEIN_DIR:?Set PROTEIN_DIR}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

INPUT_CSV="$(readlink -f "$INPUT_CSV")"
PROTEIN_DIR="$(readlink -f "$PROTEIN_DIR")"
case "$OUTPUT_DIR" in
  /*) ;;
  *) OUTPUT_DIR="$SLURM_SUBMIT_DIR/$OUTPUT_DIR" ;;
esac

EXTERNAL_ROOT="${EXTERNAL_ROOT:-$SLURM_SUBMIT_DIR/external_models}"
POSEBENCH_ROOT="$EXTERNAL_ROOT/src/posebench"
FABIND_PYTHON="$POSEBENCH_ROOT/forks/FABind/FABind/bin/python3"
FABIND_EXEC_DIR="$POSEBENCH_ROOT/forks/FABind/fabind"
FABIND_CHECKPOINT="$POSEBENCH_ROOT/forks/FABind/ckpt/best_model.bin"
SEED="${SEED:-128}"
FAIL_ON_INCOMPLETE="${FAIL_ON_INCOMPLETE:-0}"

test -x "$FABIND_PYTHON"
test -f "$INPUT_CSV"
test -d "$PROTEIN_DIR"
test -f "$FABIND_CHECKPOINT"
mkdir -p external_models/logs "$OUTPUT_DIR" "$OUTPUT_DIR/preprocess/mol" "$OUTPUT_DIR/predictions"

export PYTHONHASHSEED="$SEED"
export INPUT_CSV PROTEIN_DIR OUTPUT_DIR DATASET SEED FAIL_ON_INCOMPLETE

"$FABIND_PYTHON" scripts/external_models/run_posebench_fabind.py \
  --python "$FABIND_PYTHON" \
  --fabind-dir "$FABIND_EXEC_DIR" \
  --input-csv "$INPUT_CSV" \
  --protein-dir "$PROTEIN_DIR" \
  --save-mols-dir "$OUTPUT_DIR/preprocess/mol" \
  --save-pt-dir "$OUTPUT_DIR/preprocess" \
  --checkpoint "$FABIND_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/predictions" \
  --seed "$SEED" \
  --cuda-device-index 0

"$FABIND_PYTHON" -c '
import csv, json, os, pathlib
input_csv = pathlib.Path(os.environ["INPUT_CSV"])
output_dir = pathlib.Path(os.environ["OUTPUT_DIR"])
with input_csv.open(newline="") as handle:
    expected = [row["pdb_id"] for row in csv.DictReader(handle)]
index_path = output_dir / "predictions" / "uid_smiles_sdfname.csv"
emitted = {}
if index_path.is_file():
    with index_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            target_id = row["uid"]
            emitted[target_id] = (output_dir / "predictions" / row["idx_sdf_name"]).is_file()
coverage = {target_id: {"pose_count": int(emitted.get(target_id, False))} for target_id in expected}
summary = {
    "schema_version": 1,
    "dataset": os.environ["DATASET"],
    "seed": int(os.environ["SEED"]),
    "official_seed": 128,
    "seed_policy": "PoseBench official seed 128; repeats vary only this seed",
    "expected_targets": len(expected),
    "targets_with_any_pose": sum(v["pose_count"] > 0 for v in coverage.values()),
    "targets_with_expected_pose_count": sum(v["pose_count"] == 1 for v in coverage.values()),
    "coverage": coverage,
}
(output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
if os.environ["FAIL_ON_INCOMPLETE"] == "1" and summary["targets_with_expected_pose_count"] != len(expected):
    completed = summary["targets_with_expected_pose_count"]
    raise SystemExit(f"Incomplete FABind coverage: {completed}/{len(expected)}")
'
