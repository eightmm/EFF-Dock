#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT_ROOT="$PWD"
TAG="recovery_20260828_v3"
VINA_BASE="$PROJECT_ROOT/outputs/external_models/runs/posebench_vina/$TAG"

job_id() {
  local submitted
  submitted="$(sbatch --parsable "$@")"
  printf '%s' "${submitted%%;*}"
}

VINA_GATE="$(job_id \
  --partition=cpu_only --qos=short --time=12:00:00 \
  --export=ALL,DATASET=astex_diverse,INPUT_CSV="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/vina_astex_diverse_smoke_inputs.csv",OUTPUT_DIR="$VINA_BASE/gate",SEED=0,EXHAUSTIVENESS=8,NUM_MODES=1,FAIL_ON_INCOMPLETE=1 \
  scripts/slurm/external_posebench_vina_inference.sbatch)"
VINA_ASTEX="$(job_id \
  --dependency="afterok:$VINA_GATE" --partition=cpu_only --qos=long --time=3-00:00:00 --array=0-3%1 \
  --export=ALL,DATASET=astex_diverse,SHARD_INPUT_ROOT="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/shards",OUTPUT_ROOT="$VINA_BASE/full/astex_diverse",SEED=0,EXHAUSTIVENESS=32,NUM_MODES=40,FAIL_ON_INCOMPLETE=0 \
  scripts/slurm/external_posebench_vina_inference.sbatch)"
VINA_PB="$(job_id \
  --dependency="afterok:$VINA_GATE" --partition=cpu_only --qos=long --time=3-00:00:00 --array=0-11%1 \
  --export=ALL,DATASET=posebusters_benchmark,SHARD_INPUT_ROOT="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/posebusters_benchmark/shards",OUTPUT_ROOT="$VINA_BASE/full/posebusters_benchmark",SEED=0,EXHAUSTIVENESS=32,NUM_MODES=40,FAIL_ON_INCOMPLETE=0 \
  scripts/slurm/external_posebench_vina_inference.sbatch)"
VINA_ASTEX_AGG="$(job_id \
  --dependency="afterany:$VINA_ASTEX" \
  --export=ALL,EXPECTED_CSV="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/vina_astex_diverse_inputs.csv",RUN_ROOT="$VINA_BASE/full/astex_diverse",OUTPUT_JSON="$VINA_BASE/full/astex_diverse/aggregate.json",STRICT=0 \
  scripts/slurm/external_coverage_aggregate.sbatch)"
VINA_PB_AGG="$(job_id \
  --dependency="afterany:$VINA_PB" \
  --export=ALL,EXPECTED_CSV="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/posebusters_benchmark/vina_posebusters_benchmark_inputs.csv",RUN_ROOT="$VINA_BASE/full/posebusters_benchmark",OUTPUT_JSON="$VINA_BASE/full/posebusters_benchmark/aggregate.json",STRICT=0 \
  scripts/slurm/external_coverage_aggregate.sbatch)"

# Cancel only the descendants of the failed v2 Vina gate after the replacement
# chain has been accepted. The failed gate record and logs remain intact.
if ! scancel 60405 60406 60407 60408; then
  printf 'warning: one or more failed-gate descendants had already left the queue\n' >&2
fi

printf 'vina_gate=%s astex=%s posebusters=%s aggregate=%s,%s\n' \
  "$VINA_GATE" "$VINA_ASTEX" "$VINA_PB" "$VINA_ASTEX_AGG" "$VINA_PB_AGG"
