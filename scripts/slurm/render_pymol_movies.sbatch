#!/usr/bin/env bash
#SBATCH --job-name=effdock-pymol
#SBATCH --output=outputs/presentation/docking_movies/logs/%x-%A_%a.out
#SBATCH --error=outputs/presentation/docking_movies/logs/%x-%A_%a.err
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-2

set -euo pipefail

repo_dir=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
cd "$repo_dir"

datasets=(astex posebusters posebusters posebusters posebusters)
complex_ids=(1n46 7xg5_plp 6xbo_5mc 8b8h_ojq 7cns_pmv)
task_index=${SLURM_ARRAY_TASK_ID:?missing array task ID}
dataset=${datasets[$task_index]}
complex_id=${complex_ids[$task_index]}
output_root=outputs/presentation/docking_movies
output_suffix=${EFFDOCK_OUTPUT_SUFFIX:-}
reference_args=()
if [[ "${EFFDOCK_SHOW_REFERENCE:-0}" == 1 ]]; then
  reference_args=(--show-reference)
fi

mkdir -p "$output_root/logs" "$output_root/pymol_gifs"

.venv/bin/python scripts/render_docking_pymol.py \
  "$output_root/trajectories/${complex_id}.pt" \
  "$output_root/pymol_gifs/${dataset}_${complex_id}${output_suffix}.gif" \
  --pymol outputs/.pymol-venv/bin/pymol \
  --width 960 \
  --height 540 \
  "${reference_args[@]}"
