#!/usr/bin/env bash
#SBATCH --job-name=effdock-s2-ref-cal
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=6000ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
dataset=${EFFDOCK_DATASET:-posebusters}
complex_id=${EFFDOCK_COMPLEX_ID:-7b2c_tp7}
energy_absolute=${EFFDOCK_ENERGY_CONVERGENCE_ABSOLUTE:-1e-4}
energy_relative=${EFFDOCK_ENERGY_CONVERGENCE_RELATIVE:-1e-6}
energy_patience=${EFFDOCK_ENERGY_CONVERGENCE_PATIENCE:-5}
energy_min_steps=${EFFDOCK_ENERGY_CONVERGENCE_MIN_STEPS:-25}
run_fixed=${EFFDOCK_RUN_FIXED:-1}
case "$dataset" in
  astex) pocket_centers=data/external_test/astex_reference_pocket_centers.json ;;
  posebusters) pocket_centers=data/external_test/posebusters_reference_pocket_centers.json ;;
  *) echo "EFFDOCK_DATASET must be astex or posebusters" >&2; exit 2 ;;
esac
mkdir -p outputs/benchmarks/logs "$output_root"
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

common_args=(
  --manifest outputs/benchmarks/guidance_sigma2_eta2_refinement_runs/20260819T060555Z/manifest.json
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --external-dir data/external_test
  --pocket-centers "$pocket_centers"
  --protocol-file docs/GUIDANCE_SIGMA2_ETA2_REFINEMENT_PROTOCOL.md
  --dataset "$dataset"
  --eta 2
  --complex-id "$complex_id"
  --device cuda
  --steps 100
  --save-every 25
  --batch-size 10
)

# Adaptive runs first so any filesystem cache benefit favors the fixed-step
# baseline and the measured adaptive speedup is conservative.
/usr/bin/time -f 'adaptive_elapsed_seconds=%e adaptive_max_rss_kb=%M' \
  .venv/bin/python scripts/run_guidance_sdf_post_refinement.py \
  "${common_args[@]}" --output-dir "$output_root/adaptive" \
  --energy-convergence-absolute-kcal-mol "$energy_absolute" \
  --energy-convergence-relative "$energy_relative" \
  --energy-convergence-patience "$energy_patience" \
  --energy-convergence-min-steps "$energy_min_steps"
if [[ "$run_fixed" == 1 ]]; then
  /usr/bin/time -f 'fixed_elapsed_seconds=%e fixed_max_rss_kb=%M' \
    .venv/bin/python scripts/run_guidance_sdf_post_refinement.py \
    "${common_args[@]}" --output-dir "$output_root/fixed"
fi
