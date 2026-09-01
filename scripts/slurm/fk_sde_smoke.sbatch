#!/usr/bin/env bash
#SBATCH --job-name=effdock-fk-sde-smoke
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=test,6000ada,heavy
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_FK_SDE_OUTPUT_ROOT:?EFFDOCK_FK_SDE_OUTPUT_ROOT is required}
smoke_root="$output_root/smoke"
if ! mkdir "$smoke_root"; then
  echo "refusing to reuse FK-SDE smoke root: $smoke_root" >&2
  exit 2
fi

for required in \
  .venv/bin/eff-dock \
  configs/train.yaml \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  docs/GUIDANCE_BUDGET1000_FULL_COHORT.json \
  data/external_test/astex_reference_pocket_centers.json \
  data/external_benchmarks/data/astex_diverse_set; do
  if [[ ! -e "$required" ]]; then
    echo "missing frozen FK-SDE smoke input: $required" >&2
    exit 2
  fi
done

gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
if [[ -z "$gpu_inventory" || $(wc -l <<< "$gpu_inventory") -ne 1 ]]; then
  echo "FK-SDE smoke requires exactly one visible CUDA device" >&2
  exit 2
fi
printf 'protocol=%s output_root=%s\n' EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1 "$smoke_root"
printf 'gpu=%s\n' "$gpu_inventory"

export PYTHONHASHSEED=0
common_args=(
  --dataset astex
  --data-dir data/external_benchmarks/data/astex_diverse_set
  --external-dir data/external_test
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --eligibility-manifest docs/GUIDANCE_BUDGET1000_FULL_COHORT.json
  --require-full-ligand-atom-mapping
  --pocket-centers data/external_test/astex_reference_pocket_centers.json
  --checkpoint weights/effdock_geometry_ft_100k_best.pt
  --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt
  --selector-profile confidence_cluster_free
  --config configs/train.yaml
  --device cuda
  --protocol-id EFFDOCK-FK-TRANSLATION-SDE-ASTEX-V1
  --num-samples 40
  --num-steps 25
  --prior-pool-size 100
  --sigma 0.5
  --time-schedule late
  --schedule-power 3.0
  --pocket-cutoff 10.0
  --center-jitter-sigma 0.0
  --seed 42
  --only-id 1jje
  --expected-discovered-count 85
  --require-complete-success
  --save-selected-poses
  --refine none
  --unified-guidance-receptor-policy geometry_only
)

run_arm() {
  local arm=$1
  local slug=$2
  shift 2
  .venv/bin/eff-dock evaluate \
    "${common_args[@]}" \
    --output-dir "$smoke_root/$arm" \
    --run-name "effdock-fk-sde-astex-v1-smoke-$slug" \
    "$@"
}

run_arm ode ode
run_arm sde sde --translation-sde-base-sigma 0.3
run_arm fk_ode fk-ode \
  --fk-constraint-beta 0.01 \
  --fk-resample-times 0.3,0.6,0.8 \
  --fk-resample-method systematic
run_arm fk_sde fk-sde \
  --translation-sde-base-sigma 0.3 \
  --fk-constraint-beta 0.01 \
  --fk-resample-times 0.3,0.6,0.8 \
  --fk-resample-method systematic

.venv/bin/python -m effdock.workflows.fk_sde_astex_report smoke \
  --input-root "$smoke_root" \
  --output "$smoke_root/technical_audit.json"
