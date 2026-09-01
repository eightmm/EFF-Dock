#!/usr/bin/env bash
#SBATCH --job-name=effdock-cut-smoke
#SBATCH --partition=6000ada,heavy,test
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
cd "${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
: "${EFFDOCK_OUTPUT_ROOT:?set EFFDOCK_OUTPUT_ROOT}"
runtime_venv=${EFFDOCK_RUNTIME_VENV:-.venv}
task_id=${SLURM_ARRAY_TASK_ID:?}
[[ "${SLURM_ARRAY_TASK_COUNT:?}" -eq 2 ]] || { echo "smoke requires two tasks" >&2; exit 2; }
cutoffs=(6 12)
cutoff=${cutoffs[$task_id]}
condition_root="$EFFDOCK_OUTPUT_ROOT/smoke/cutoff_$(printf '%02d' "$cutoff")"

EFFDOCK_SMOKE=1 bash benchmarks/effdock/slurm/pocket_cutoff_generation.sbatch
"$runtime_venv/bin/python" benchmarks/effdock/build_pocket_cutoff_manifest.py \
  --condition-root "$condition_root" --cutoff "$cutoff" --repeat-index 0 \
  --output "$condition_root/manifest.json" --smoke
"$runtime_venv/bin/python" scripts/run_guidance_sdf_post_refinement.py \
  --manifest "$condition_root/manifest.json" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --external-dir data/external_test \
  --pocket-centers data/external_test/astex_reference_pocket_centers.json \
  --protocol-file docs/GUIDANCE_SIGMA2_ETA2_REFINEMENT_PROTOCOL.md \
  --dataset astex --eta 2 --complex-id 1jje \
  --output-dir "$condition_root/full/refinement/astex/1jje" \
  --device cuda --steps 100 --save-every 25 --batch-size 10 \
  --pocket-cutoff 10 --energy-convergence-absolute-kcal-mol 0.02 \
  --energy-convergence-relative 0.001 --energy-convergence-patience 5 \
  --energy-convergence-min-steps 25
"$runtime_venv/bin/python" scripts/score_guidance_sdf_post_refinement_confidence.py \
  --refinement-summary "$condition_root/full/refinement/astex/1jje/summary.json" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --external-dir data/external_test --config configs/train.yaml \
  --docking-checkpoint weights/effdock_docking_early_time_t0p10_50k.pt \
  --confidence-checkpoint weights/effdock_confidence_s50_raw_refined_u70k.pt \
  --output-dir "$condition_root/full/confidence_chunk20_fresh/astex/1jje" \
  --sigma 2 --pocket-cutoff 10 --pose-batch-size 20 --device cuda
"$runtime_venv/bin/python" benchmarks/effdock/evaluate_pocket_cutoff_selected_posebusters.py \
  --condition-root "$condition_root" --cutoff "$cutoff" --repeat-index 0 \
  --num-shards 1 --shard-index 0 --smoke
