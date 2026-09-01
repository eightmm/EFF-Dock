#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-interf-score
#SBATCH --partition=test
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=others/logs/%x-%A_%a.out
#SBATCH --error=others/logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${SOURCE_ROOT:?Set SOURCE_ROOT to the Interformer seed_0 directory}"
: "${SCORE_ROOT:?Set SCORE_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as an array}"
printf -v SHARD_ID '%03d' "$SLURM_ARRAY_TASK_ID"
SOURCE_SHARD="$SOURCE_ROOT/shard_$SHARD_ID"
SCORE_DIR="$SCORE_ROOT/shard_$SHARD_ID"

INTERFORMER_MODEL_ROOT="${INTERFORMER_MODEL_ROOT:-$SLURM_SUBMIT_DIR/others/interformer}"
INTERFORMER_ROOT="$INTERFORMER_MODEL_ROOT/upstream"
MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"
TORCH_LIB="$INTERFORMER_MODEL_ROOT/.venv/lib/python3.12/site-packages/torch/lib"
EXPECTED_POSES="${EXPECTED_POSES:-20}"

test -d "$SOURCE_SHARD"
test -x "$MODEL_RUNNER"
test -f "$INTERFORMER_ROOT/inference.py"
test -d "$INTERFORMER_ROOT/checkpoints/v0.2_affinity_model"
test -f "$TORCH_LIB/libc10.so"
mkdir -p others/logs "$SCORE_DIR"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

"$MODEL_RUNNER" interformer python scripts/external_models/prepare_interformer_pose_score.py \
  --source-shard "$SOURCE_SHARD" \
  --score-dir "$SCORE_DIR" \
  --expected-poses "$EXPECTED_POSES"

(
  cd "$SCORE_DIR"
  PYTHONPATH="$INTERFORMER_ROOT/interformer" "$MODEL_RUNNER" interformer python \
    "$INTERFORMER_ROOT/inference.py" \
    -test_csv "$SCORE_DIR/work/query.round0.csv" \
    -work_path "$SCORE_DIR/work" \
    -ligand_folder infer \
    -ensemble \
      "$INTERFORMER_ROOT/checkpoints/v0.2_affinity_model/model0" \
      "$INTERFORMER_ROOT/checkpoints/v0.2_affinity_model/model1" \
      "$INTERFORMER_ROOT/checkpoints/v0.2_affinity_model/model2" \
      "$INTERFORMER_ROOT/checkpoints/v0.2_affinity_model/model3" \
    -gpus 1 \
    -batch_size 20 \
    -posfix '*val_pose_selection*' \
    -debug
)

RESULT_CSV="$SCORE_DIR/result/query.round0_ensemble.csv"
test -s "$RESULT_CSV"
export RESULT_CSV EXPECTED_POSES
"$MODEL_RUNNER" interformer python -c '
import json, os, pathlib, pandas as pd
import numpy as np
result = pathlib.Path(os.environ["RESULT_CSV"])
expected = int(os.environ["EXPECTED_POSES"])
df = pd.read_csv(result)
required = {"Target", "pose_rank", "pred_pose"}
missing = required.difference(df.columns)
assert not missing, f"missing columns: {sorted(missing)}"
assert (df["pose_rank"].astype(int) < expected).all(), "input conformer leaked into PoseScore"
assert np.isfinite(df["pred_pose"].to_numpy()).all(), "non-finite PoseScore values"
counts = df.groupby("Target")["pose_rank"].nunique()
assert (counts == expected).all(), counts[counts != expected].to_dict()
payload = {
    "schema_version": 1,
    "result_csv": str(result.resolve()),
    "targets": int(df["Target"].nunique()),
    "poses": int(len(df)),
    "poses_per_target": expected,
    "ranker": "Interformer v0.2 four-checkpoint PoseScore ensemble",
    "checkpoint_selector": "*val_pose_selection*",
    "top1_policy": "maximum pred_pose within target",
}
(result.parent.parent / "score_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
'
