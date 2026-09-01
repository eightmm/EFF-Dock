#!/usr/bin/env bash
#SBATCH --job-name=effdock-sdf-pbfull
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --array=0-31%16
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err

set -euo pipefail
repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
cd "${repo_dir:?EFFDOCK_REPO_DIR or SLURM_SUBMIT_DIR is required}"
: "${EFFDOCK_INPUT_ROOT:?set EFFDOCK_INPUT_ROOT}"
: "${EFFDOCK_PB_OUTPUT_ROOT:?set EFFDOCK_PB_OUTPUT_ROOT}"
task_id=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
task_count=${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}
[[ "$task_count" -eq 32 ]] || { echo "full PB run requires 32 shards" >&2; exit 2; }
task_cpus=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="$task_cpus" MKL_NUM_THREADS="$task_cpus" OPENBLAS_NUM_THREADS="$task_cpus"
source_manifest=${EFFDOCK_SOURCE_MANIFEST:-outputs/benchmarks/guidance_all_pose_pb_eta_runs/20260811T042451Z/manifest.json}
source_eta=${EFFDOCK_SOURCE_ETA:-0}
.venv/bin/python scripts/run_guidance_sdf_post_refinement_posebusters_shard.py \
  --manifest "$source_manifest" \
  --input-root "$EFFDOCK_INPUT_ROOT" \
  --output-root "$EFFDOCK_PB_OUTPUT_ROOT" \
  --num-shards 32 --shard-index "$task_id" --workers "$task_cpus" \
  --eta "$source_eta"
