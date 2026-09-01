#!/usr/bin/env bash
#SBATCH --job-name=effdock-refine-bench
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
  --manifest outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/manifest.json
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  --external-dir data/external_test
  --pocket-centers "$pocket_centers"
  --protocol-file docs/GUIDANCE_SDF_POST_REFINEMENT_PROTOCOL.md
  --dataset "$dataset"
  --eta 0
  --complex-id "$complex_id"
  --device cuda
  --steps 100
  --save-every 25
  --batch-size 10
)

# Run the optimized path first so any filesystem cache benefit favors the
# legacy baseline; this makes the reported speedup conservative.
/usr/bin/time -f 'optimized_elapsed_seconds=%e optimized_max_rss_kb=%M' \
  .venv/bin/python scripts/run_guidance_sdf_post_refinement.py \
  "${common_args[@]}" --output-dir "$output_root/optimized"
/usr/bin/time -f 'dense_elapsed_seconds=%e dense_max_rss_kb=%M' \
  .venv/bin/python scripts/run_guidance_sdf_post_refinement.py \
  "${common_args[@]}" --output-dir "$output_root/dense" --dense-diagnostics
