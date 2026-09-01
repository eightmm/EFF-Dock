#!/usr/bin/env bash
#SBATCH --job-name=effdock-ext-sigma-eval
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=others/logs/%x-%j.out
#SBATCH --error=others/logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
: "${INPUT_CSV:?Set INPUT_CSV}"
: "${MODEL_ROOT:?Set MODEL_ROOT}"
: "${EXPECTED_TARGETS:?Set EXPECTED_TARGETS}"
: "${EVAL_ROOT:?Set EVAL_ROOT}"
: "${DATASET_LABEL:?Set DATASET_LABEL}"

MODEL_RUNNER="$SLURM_SUBMIT_DIR/scripts/others/run_model.sh"
for repeat in 0 1 2; do
  seed_start=$((repeat * 40))
  "$MODEL_RUNNER" sigmadock python \
    scripts/external_models/evaluate_sigmadock_official.py \
    --input-csv "$INPUT_CSV" \
    --model-root "$MODEL_ROOT" \
    --seed-start "$seed_start" \
    --pool-size 40 \
    --expected-targets "$EXPECTED_TARGETS" \
    --output-dir "$EVAL_ROOT/repeat_$repeat"
done

python3 scripts/external_models/aggregate_repeat_metrics.py \
  --input "$EVAL_ROOT/repeat_0/summary.json" \
  --input "$EVAL_ROOT/repeat_1/summary.json" \
  --input "$EVAL_ROOT/repeat_2/summary.json" \
  --output-json "$EVAL_ROOT/mean_std.json" \
  --output-md "$EVAL_ROOT/RESULTS.md" \
  --label "SigmaDock official $DATASET_LABEL"
