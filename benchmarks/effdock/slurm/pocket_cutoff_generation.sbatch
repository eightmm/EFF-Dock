#!/usr/bin/env bash
#SBATCH --job-name=effdock-cut-gen
#SBATCH --partition=6000ada,heavy,test
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}

cutoffs=(6 8 10 12)
num_shards=8
task_id=${SLURM_ARRAY_TASK_ID:?}
task_count=${SLURM_ARRAY_TASK_COUNT:?}
smoke=${EFFDOCK_SMOKE:-0}
if [[ "$smoke" == 1 ]]; then
  [[ "$task_count" -eq 2 ]] || { echo "smoke requires two tasks" >&2; exit 2; }
  cutoff_index=$task_id
  cutoff=${cutoffs[$((cutoff_index * 3))]}
  repeat_index=0
  dataset=astex
  shard_index=0
  active_shards=1
  only_id_args=(--only-id 1jje)
  condition_root="$EFFDOCK_OUTPUT_ROOT/smoke/cutoff_$(printf '%02d' "$cutoff")"
else
  [[ "$task_count" -eq 192 ]] || { echo "full generation requires 192 tasks" >&2; exit 2; }
  shard_index=$((task_id % num_shards))
  dataset_index=$(((task_id / num_shards) % 2))
  repeat_index=$(((task_id / (num_shards * 2)) % 3))
  cutoff_index=$((task_id / (num_shards * 2 * 3)))
  cutoff=${cutoffs[$cutoff_index]}
  case "$dataset_index" in 0) dataset=astex ;; 1) dataset=posebusters ;; *) exit 2 ;; esac
  active_shards=$num_shards
  only_id_args=()
  condition_root="$EFFDOCK_OUTPUT_ROOT/cutoff_$(printf '%02d' "$cutoff")/repeat_$repeat_index"
fi

case "$dataset" in
  astex)
    expected_count=85
    data_dir=data/external_benchmarks/data/astex_diverse_set
    centers=data/external_test/astex_reference_pocket_centers.json
    ;;
  posebusters)
    expected_count=308
    data_dir=data/external_benchmarks/data/posebusters_benchmark_set
    centers=data/external_test/posebusters_reference_pocket_centers.json
    ;;
esac

checkpoint=weights/effdock_docking_early_time_t0p10_50k.pt
confidence=weights/effdock_confidence_s50_raw_refined_u70k.pt
eligibility=outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json
for path in "$checkpoint" "$confidence" "$eligibility" configs/train.yaml docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json; do
  [[ -f "$path" ]] || { echo "missing input $path" >&2; exit 2; }
done

base_seed=$((42 + repeat_index * 100000))
run_name="effdock-pocket-cutoff-v1-${dataset}-c$(printf '%02d' "$cutoff")-r${repeat_index}-n100-s10"
mkdir -p "$condition_root/raw" outputs/benchmarks/logs
"$runtime_venv/bin/eff-dock" evaluate \
  --dataset "$dataset" --data-dir "$data_dir" --external-dir data/external_test \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --require-full-ligand-atom-mapping --pocket-centers "$centers" \
  --checkpoint "$checkpoint" --no-confidence \
  --selector-profile candidate_only --config configs/train.yaml \
  --device cuda --output-dir "$condition_root/raw" \
  --protocol-id EFFDOCK-POCKET-CUTOFF-ROBUSTNESS-V1 --run-name "$run_name" \
  --num-samples 100 --num-steps 10 --prior-pool-size 100 --sigma 2 \
  --time-schedule late --schedule-power 3 --pocket-cutoff "$cutoff" \
  --center-jitter-sigma 0 --refine none \
  --unified-guidance-mode normalized_drift --unified-guidance-scale 2 \
  --unified-guidance-receptor-policy geometry_only \
  --unified-guidance-start-t 0.5 --unified-guidance-ramp-power 1 \
  --unified-guidance-max-force 20 --unified-guidance-max-velocity 5 \
  --unified-guidance-max-angular-velocity 5 \
  --unified-guidance-max-atom-displacement 0.25 \
  --unified-guidance-max-backtracks 8 --unified-guidance-protein-shell 18 \
  --seed "$base_seed" --ligand-conformer-seed 0 \
  --num-shards "$active_shards" --shard-index "$shard_index" \
  --expected-discovered-count "$expected_count" --require-complete-success \
  --save-selected-poses --eligibility-manifest "$eligibility" \
  "${only_id_args[@]}"
