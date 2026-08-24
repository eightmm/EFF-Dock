#!/usr/bin/env bash
#SBATCH --job-name=effdock-movie
#SBATCH --output=outputs/presentation/docking_movies/logs/%x-%A_%a.out
#SBATCH --error=outputs/presentation/docking_movies/logs/%x-%A_%a.err
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-2

set -euo pipefail

repo_dir=${SLURM_SUBMIT_DIR:?submit this job from the repository root}
cd "$repo_dir"

datasets=(
  astex posebusters posebusters posebusters posebusters posebusters
  posebusters astex posebusters
)
complex_ids=(
  1n46 6ysp_pal 7uj5_dgl 7xg5_plp 6xbo_5mc 8b8h_ojq
  7frx_o88 1oq5 7cns_pmv
)
data_dirs=(
  data/external_benchmarks/data/astex_diverse_set
  data/external_benchmarks/data/posebusters_benchmark_set
  data/external_benchmarks/data/posebusters_benchmark_set
  data/external_benchmarks/data/posebusters_benchmark_set
  data/external_benchmarks/data/posebusters_benchmark_set
  data/external_benchmarks/data/posebusters_benchmark_set
  data/external_benchmarks/data/posebusters_benchmark_set
  data/external_benchmarks/data/astex_diverse_set
  data/external_benchmarks/data/posebusters_benchmark_set
)
center_files=(
  data/external_test/astex_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
  data/external_test/astex_reference_pocket_centers.json
  data/external_test/posebusters_reference_pocket_centers.json
)

task_index=${SLURM_ARRAY_TASK_ID:?missing array task ID}
dataset=${datasets[$task_index]}
complex_id=${complex_ids[$task_index]}
data_dir=${data_dirs[$task_index]}
centers=${center_files[$task_index]}
output_root=outputs/presentation/docking_movies

mkdir -p "$output_root/logs" "$output_root/trajectories" "$output_root/pymol_gifs" \
  "$output_root/evaluation"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
printf 'dataset=%s complex_id=%s seed_policy=42+global_index\n' "$dataset" "$complex_id"

uv run eff-dock evaluate \
  --dataset "$dataset" \
  --data-dir "$data_dir" \
  --external-dir data/external_test \
  --pocket-centers "$centers" \
  --checkpoint weights/effdock_geometry_ft_100k_best.pt \
  --confidence-checkpoint weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  --config configs/train.yaml \
  --output-dir "$output_root/evaluation" \
  --run-name "movie_${dataset}_${complex_id}" \
  --protocol-id EFFDOCK-CONFIDENCE-SELECTION-V1 \
  --num-samples 80 \
  --num-steps 25 \
  --sigma 0.5 \
  --time-schedule late \
  --schedule-power 3.0 \
  --pocket-cutoff 10.0 \
  --seed 42 \
  --only-id "$complex_id" \
  --trajectory-dir "$output_root/trajectories"

.venv/bin/python scripts/render_docking_pymol.py \
  "$output_root/trajectories/${complex_id}.pt" \
  "$output_root/pymol_gifs/${dataset}_${complex_id}.gif" \
  --pymol outputs/.pymol-venv/bin/pymol
