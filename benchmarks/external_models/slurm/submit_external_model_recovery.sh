#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT_ROOT="$PWD"
TAG="recovery_20260828_v1"
INPUT_ROOT="$PROJECT_ROOT/outputs/external_models/inputs/$TAG"
mkdir -p "$INPUT_ROOT"

filter_manifest() {
  local source_csv="$1"
  local output_csv="$2"
  shift 2
  local args=()
  local target_id
  for target_id in "$@"; do
    args+=(--target-id "$target_id")
  done
  .venv/bin/python scripts/external_models/filter_target_manifest.py \
    --input-csv "$source_csv" \
    --output-csv "$output_csv" \
    "${args[@]}"
}

filter_manifest \
  outputs/external_models/inputs/posebench_native/astex_diverse/vina_astex_diverse_inputs.csv \
  "$INPUT_ROOT/surfdock_astex_missing.csv" \
  1GKC_NFH 1HWW_SWA
filter_manifest \
  outputs/external_models/inputs/posebench_native/posebusters_benchmark/vina_posebusters_benchmark_inputs.csv \
  "$INPUT_ROOT/surfdock_posebusters_missing.csv" \
  6YT6_PKE 6Z2C_Q5E 7BMI_U4B 7JMV_4NC 7M31_TDR 7NPL_UKZ \
  7QHG_T3B 7TYP_KUR 7XRL_FWK 7ZDY_6MJ 7ZXV_45D 8DKO_TFB
filter_manifest \
  outputs/external_models/inputs/posebench_diffdock/astex_diverse_full.csv \
  "$INPUT_ROOT/diffdock_astex_missing.csv" \
  1G9V_RQ3 1MMV_3AR 1OF6_DTY 1P2Y_NCT 1PMN_984 1Q4G_BFL 1R9O_FLP 1T9B_1CS
filter_manifest \
  outputs/external_models/inputs/posebench_diffdock/posebusters_benchmark_full.csv \
  "$INPUT_ROOT/diffdock_posebusters_missing.csv" \
  5SAK_ZRY 6XCT_478 6YJA_2BA 7B2C_TP7 7BKA_4JC 7CL8_TES 7FRX_O88 \
  7JY3_VUD 7KQU_YOF 7M31_TDR 7N6F_0I1 7NFB_GEN 7P5T_5YG 7Q25_8J9 \
  7RKW_5TV 7SUC_COM 7TS6_KMI 7TSF_H4B 7V43_C4O 7WCF_ACP 7WY1_D0L \
  7XJN_NSD 7XRL_FWK 8D39_QDB 8DSC_NCA 8EXL_799 8F4J_PHO 8HO0_3ZI

job_id() {
  local submitted
  submitted="$(sbatch --parsable "$@")"
  printf '%s' "${submitted%%;*}"
}

DBINDFR_JOB="$(job_id \
  --partition=6000ada --qos=normal --time=12:00:00 \
  --export=ALL,INPUT_CSV="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/posebusters_benchmark/shards/shard_005/diffbindfr_inputs.csv",OUTPUT_DIR="$PROJECT_ROOT/outputs/external_models/runs/diffbindfr/posebusters_native_s20_n40_seed0/seed_0/shard_005",SEED=0,SAMPLES_PER_COMPLEX=40,FAIL_ON_INCOMPLETE=1 \
  scripts/slurm/external_diffbindfr_inference.sbatch)"

SURF_ASTEX_JOB="$(job_id \
  --partition=6000ada --qos=normal --time=1-00:00:00 \
  --export=ALL,INPUT_CSV="$INPUT_ROOT/surfdock_astex_missing.csv",OUTPUT_DIR="$PROJECT_ROOT/outputs/external_models/runs/surfdock/$TAG/astex_diverse",SEED=0,INFERENCE_STEPS=20,SAMPLES_PER_COMPLEX=40,FAIL_ON_INCOMPLETE=1 \
  scripts/slurm/external_surfdock_inference.sbatch)"
SURF_PB_JOB="$(job_id \
  --partition=6000ada --qos=normal --time=1-00:00:00 \
  --export=ALL,INPUT_CSV="$INPUT_ROOT/surfdock_posebusters_missing.csv",OUTPUT_DIR="$PROJECT_ROOT/outputs/external_models/runs/surfdock/$TAG/posebusters_benchmark",SEED=0,INFERENCE_STEPS=20,SAMPLES_PER_COMPLEX=40,FAIL_ON_INCOMPLETE=1 \
  scripts/slurm/external_surfdock_inference.sbatch)"

DDOCK_ASTEX_JOB="$(job_id \
  --partition=6000ada --qos=normal --time=12:00:00 \
  --export=ALL,INPUT_CSV="$INPUT_ROOT/diffdock_astex_missing.csv",OUTPUT_DIR="$PROJECT_ROOT/outputs/external_models/runs/posebench_diffdock/$TAG/astex_diverse",SEED=0,INFERENCE_STEPS=20,ACTUAL_STEPS=19,SAMPLES_PER_COMPLEX=5,BATCH_SIZE=1,FAIL_ON_INCOMPLETE=0 \
  scripts/slurm/external_diffdock_inference.sbatch)"
DDOCK_PB_JOB="$(job_id \
  --partition=6000ada --qos=normal --time=1-00:00:00 \
  --export=ALL,INPUT_CSV="$INPUT_ROOT/diffdock_posebusters_missing.csv",OUTPUT_DIR="$PROJECT_ROOT/outputs/external_models/runs/posebench_diffdock/$TAG/posebusters_benchmark",SEED=0,INFERENCE_STEPS=20,ACTUAL_STEPS=19,SAMPLES_PER_COMPLEX=5,BATCH_SIZE=1,FAIL_ON_INCOMPLETE=0 \
  scripts/slurm/external_diffdock_inference.sbatch)"

DYN_BASE="$PROJECT_ROOT/outputs/external_models/runs/posebench_dynamicbind/$TAG"
DYN_GATE="$(job_id \
  --partition=test --qos=veryshort --time=04:00:00 \
  --export=ALL,DATASET=astex_diverse,LIGAND_CSV_DIR="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/dynamicbind_astex_diverse_smoke_inputs",PROTEIN_DIR="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/astex_diverse_smoke_proteins",OUTPUT_DIR="$DYN_BASE/gate",SEED=0,INFERENCE_STEPS=2,SAMPLES_PER_COMPLEX=1,BATCH_SIZE=1,FAIL_ON_INCOMPLETE=1 \
  scripts/slurm/external_posebench_dynamicbind_inference.sbatch)"
DYN_ASTEX="$(job_id \
  --dependency="afterok:$DYN_GATE" --partition=6000ada --qos=long --time=2-00:00:00 --array=0-3%2 \
  --export=ALL,DATASET=astex_diverse,SHARD_INPUT_ROOT="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/shards",OUTPUT_ROOT="$DYN_BASE/full/astex_diverse",SEED=0,INFERENCE_STEPS=20,SAMPLES_PER_COMPLEX=40,BATCH_SIZE=5,FAIL_ON_INCOMPLETE=0 \
  scripts/slurm/external_posebench_dynamicbind_inference.sbatch)"
DYN_PB="$(job_id \
  --dependency="afterok:$DYN_GATE" --partition=6000ada --qos=long --time=2-00:00:00 --array=0-11%2 \
  --export=ALL,DATASET=posebusters_benchmark,SHARD_INPUT_ROOT="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/posebusters_benchmark/shards",OUTPUT_ROOT="$DYN_BASE/full/posebusters_benchmark",SEED=0,INFERENCE_STEPS=20,SAMPLES_PER_COMPLEX=40,BATCH_SIZE=5,FAIL_ON_INCOMPLETE=0 \
  scripts/slurm/external_posebench_dynamicbind_inference.sbatch)"
DYN_ASTEX_AGG="$(job_id \
  --dependency="afterany:$DYN_ASTEX" \
  --export=ALL,EXPECTED_CSV="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/astex_diverse/vina_astex_diverse_inputs.csv",RUN_ROOT="$DYN_BASE/full/astex_diverse",OUTPUT_JSON="$DYN_BASE/full/astex_diverse/aggregate.json",STRICT=0 \
  scripts/slurm/external_coverage_aggregate.sbatch)"
DYN_PB_AGG="$(job_id \
  --dependency="afterany:$DYN_PB" \
  --export=ALL,EXPECTED_CSV="$PROJECT_ROOT/outputs/external_models/inputs/posebench_native/posebusters_benchmark/vina_posebusters_benchmark_inputs.csv",RUN_ROOT="$DYN_BASE/full/posebusters_benchmark",OUTPUT_JSON="$DYN_BASE/full/posebusters_benchmark/aggregate.json",STRICT=0 \
  scripts/slurm/external_coverage_aggregate.sbatch)"

VINA_BASE="$PROJECT_ROOT/outputs/external_models/runs/posebench_vina/$TAG"
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

# These exact chains are permanently blocked on failed/superseded gates.  New
# jobs above are accepted before they are cancelled.
if ! scancel 59953 59956 59957 59958 59959 59960 \
  59964 59965 59966 59967 59968 59969 59970 59971 59972; then
  printf 'warning: one or more superseded jobs had already left the queue\n' >&2
fi

printf 'diffbindfr_retry=%s\n' "$DBINDFR_JOB"
printf 'surfdock_astex_retry=%s\n' "$SURF_ASTEX_JOB"
printf 'surfdock_posebusters_retry=%s\n' "$SURF_PB_JOB"
printf 'diffdock_astex_retry=%s\n' "$DDOCK_ASTEX_JOB"
printf 'diffdock_posebusters_retry=%s\n' "$DDOCK_PB_JOB"
printf 'dynamicbind_gate=%s astex=%s posebusters=%s aggregate=%s,%s\n' \
  "$DYN_GATE" "$DYN_ASTEX" "$DYN_PB" "$DYN_ASTEX_AGG" "$DYN_PB_AGG"
printf 'vina_gate=%s astex=%s posebusters=%s aggregate=%s,%s\n' \
  "$VINA_GATE" "$VINA_ASTEX" "$VINA_PB" "$VINA_ASTEX_AGG" "$VINA_PB_AGG"
