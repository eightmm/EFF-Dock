#!/usr/bin/env bash
#SBATCH --job-name=effdock-chemical-unit1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --partition=heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

protocol_id=EFFDOCK-CHEMICAL-CONSTRAINT-UNIT-WEIGHTS-SMOKE-V1
checkpoint=weights/effdock_docking_early_time_t0p10_50k.pt
config=configs/train.yaml
benchmark_input_manifest=docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
expected_input_sha256=99f15f557644cc51c3dd1f559b0dd97dd4259c1de3e1403fb761b7c7e079f668
output_root=${EFFDOCK_OUTPUT_ROOT:-outputs/benchmarks/chemical_constraint_unit_weights_v1/raw}
dataset=posebusters
data_dir=data/external_benchmarks/data/posebusters_benchmark_set
centers=data/external_test/posebusters_reference_pocket_centers.json
complex_ids=(7sdd_4ip 6xht_v2v 8f4j_pho)
complex_id=${complex_ids[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}

for required in .venv/bin/eff-dock "$checkpoint" "$config" "$centers" \
  "$benchmark_input_manifest"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done
read -r actual_input_sha256 _ < <(sha256sum "$benchmark_input_manifest")
if [[ "$actual_input_sha256" != "$expected_input_sha256" ]]; then
  echo "frozen benchmark-input manifest SHA-256 mismatch" >&2
  exit 2
fi

echo "protocol=$protocol_id complex_id=$complex_id"
git rev-parse HEAD
git diff --no-ext-diff | sha256sum
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

common_args=(
  --dataset "$dataset"
  --data-dir "$data_dir"
  --external-dir data/external_test
  --benchmark-input-manifest "$benchmark_input_manifest"
  --require-full-ligand-atom-mapping
  --pocket-centers "$centers"
  --checkpoint "$checkpoint"
  --config "$config"
  --no-confidence
  --output-dir "$output_root"
  --protocol-id "$protocol_id"
  --num-samples 100
  --num-steps 10
  --prior-pool-size 100
  --sigma 2.0
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
  --ligand-conformer-seed 0
  --unified-guidance-mode normalized_drift
  --unified-guidance-receptor-policy geometry_only
  --unified-guidance-start-t 0.5
  --unified-guidance-ramp-power 1.0
  --unified-guidance-max-force 20.0
  --unified-guidance-max-velocity 5.0
  --unified-guidance-max-angular-velocity 5.0
  --unified-guidance-max-atom-displacement 0.25
  --unified-guidance-max-backtracks 8
  --unified-guidance-protein-shell 18.0
  --unified-guidance-chemical-constraint-ramp-power 1.0
  --unified-guidance-chemical-constraint-max-atom-displacement 0.10
  --seed 42
  --num-shards 1
  --shard-index 0
  --expected-discovered-count 308
  --require-complete-success
  --save-selected-poses
  --only-id "$complex_id"
)

arms=(physical_interaction_1 all_three_1)
chemical_strengths=(0.0 1.0)
for arm_index in "${!arms[@]}"; do
  arm=${arms[$arm_index]}
  chemical_strength=${chemical_strengths[$arm_index]}
  run_name="effdock-chemical-unit1-${complex_id}-${arm}"
  echo "run=$run_name physical_interaction_strength=1.0 chemical_strength=$chemical_strength"
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --run-name "$run_name" \
    --unified-guidance-scale 1.0 \
    --unified-guidance-chemical-constraint-strength "$chemical_strength"
done
