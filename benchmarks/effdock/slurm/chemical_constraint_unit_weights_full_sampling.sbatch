#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-full-sample
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=6000ada,heavy
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-15%8

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"

task_id=${SLURM_ARRAY_TASK_ID:?}
dataset_index=$((task_id / 8))
shard_index=$((task_id % 8))
case "$dataset_index" in
  0)
    dataset=astex
    expected_count=85
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  1)
    dataset=posebusters
    expected_count=308
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
  *) exit 2 ;;
esac

checkpoint=weights/effdock_docking_early_time_t0p10_50k.pt
confidence=weights/effdock_confidence_s50_raw_refined_u70k.pt
audit="$EFFDOCK_OUTPUT_ROOT/audit/combined.json"
for path in .venv/bin/eff-dock "$checkpoint" "$confidence" "$audit" \
  configs/train.yaml docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json; do
  [[ -e "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done
read -r docking_sha _ < <(sha256sum "$checkpoint")
read -r confidence_sha _ < <(sha256sum "$confidence")
[[ "$docking_sha" == 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 ]]
[[ "$confidence_sha" == ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638 ]]
.venv/bin/python -c \
  'import sys; from pathlib import Path; from effdock.workflows.guidance_budget_full_report import validate_full_cohort_audit_for_dataset as v; v(Path(sys.argv[1]), sys.argv[2])' \
  "$audit" "$dataset"

mkdir -p "$EFFDOCK_OUTPUT_ROOT/raw" outputs/benchmarks/logs
common_args=(
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --require-full-ligand-atom-mapping
  --pocket-centers "$centers"
  --checkpoint "$checkpoint"
  --confidence-checkpoint "$confidence"
  --selector-profile confidence_cluster_free
  --config configs/train.yaml
  --device cuda
  --output-dir "$EFFDOCK_OUTPUT_ROOT/raw"
  --protocol-id EFFDOCK-CHEMICAL-CONSTRAINT-UNIT-WEIGHTS-FULL-V1
  --num-samples 100
  --num-steps 10
  --prior-pool-size 100
  --sigma 2
  --time-schedule late
  --schedule-power 3
  --pocket-cutoff 10
  --center-jitter-sigma 0
  --ligand-conformer-seed 0
  --refine none
  --unified-guidance-mode normalized_drift
  --unified-guidance-receptor-policy geometry_only
  --unified-guidance-start-t 0.5
  --unified-guidance-ramp-power 1
  --unified-guidance-max-force 20
  --unified-guidance-max-velocity 5
  --unified-guidance-max-angular-velocity 5
  --unified-guidance-max-atom-displacement 0.25
  --unified-guidance-max-backtracks 8
  --unified-guidance-protein-shell 18
  --unified-guidance-chemical-constraint-ramp-power 1
  --unified-guidance-chemical-constraint-max-atom-displacement 0.10
  --seed 42
  --num-shards 8
  --shard-index "$shard_index"
  --expected-discovered-count "$expected_count"
  --require-complete-success
  --save-selected-poses
  --eligibility-manifest "$audit"
)

arms=(physical_interaction_1 all_three_1)
chemical_strengths=(0 1)
for arm_index in "${!arms[@]}"; do
  arm=${arms[$arm_index]}
  chemical_strength=${chemical_strengths[$arm_index]}
  run_name="effdock-chemical-unit1-full-${dataset}-${arm}"
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --run-name "$run_name" \
    --unified-guidance-scale 1 \
    --unified-guidance-chemical-constraint-strength "$chemical_strength"
done
