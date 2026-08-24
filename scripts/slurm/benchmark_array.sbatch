#!/usr/bin/env bash
#SBATCH --job-name=effdock-bench
#SBATCH --output=outputs/benchmarks/logs/%x-%A_%a.out
#SBATCH --error=outputs/benchmarks/logs/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

: "${EFFDOCK_DATASET:?set EFFDOCK_DATASET}"
: "${EFFDOCK_DATA_DIR:?set EFFDOCK_DATA_DIR}"
: "${EFFDOCK_CENTERS:?set EFFDOCK_CENTERS}"
: "${EFFDOCK_NUM_SHARDS:?set EFFDOCK_NUM_SHARDS}"
: "${EFFDOCK_RUN_NAME:?set EFFDOCK_RUN_NAME}"

checkpoint=${EFFDOCK_CHECKPOINT:-weights/effdock_geometry_ft_100k_best.pt}
config=${EFFDOCK_CONFIG:-configs/train.yaml}
num_samples=${EFFDOCK_NUM_SAMPLES:-100}
num_steps=${EFFDOCK_NUM_STEPS:-10}
sigma=${EFFDOCK_SIGMA:-2.0}
pocket_cutoff=${EFFDOCK_POCKET_CUTOFF:-10.0}
protocol_id=${EFFDOCK_PROTOCOL_ID:-}
vina_guidance_scale=${EFFDOCK_VINA_GUIDANCE_SCALE:-0.0}
vina_guidance_start_t=${EFFDOCK_VINA_GUIDANCE_START_T:-0.5}
vina_guidance_ramp_power=${EFFDOCK_VINA_GUIDANCE_RAMP_POWER:-1.0}
vina_guidance_max_force=${EFFDOCK_VINA_GUIDANCE_MAX_FORCE:-10.0}
vina_guidance_max_velocity=${EFFDOCK_VINA_GUIDANCE_MAX_VELOCITY:-5.0}
vina_guidance_max_angular_velocity=${EFFDOCK_VINA_GUIDANCE_MAX_ANGULAR_VELOCITY:-5.0}
vina_guidance_protein_shell=${EFFDOCK_VINA_GUIDANCE_PROTEIN_SHELL:-18.0}
vina_guidance_w_strain=${EFFDOCK_VINA_GUIDANCE_W_STRAIN:-1.0}
confidence_args=()
if [[ -n "${EFFDOCK_CONFIDENCE_CHECKPOINT:-}" ]]; then
  confidence_args=(--confidence-checkpoint "$EFFDOCK_CONFIDENCE_CHECKPOINT")
fi

limit_args=()
if [[ -n "${EFFDOCK_LIMIT:-}" ]]; then
  limit_args=(--limit "$EFFDOCK_LIMIT")
fi
only_id_args=()
if [[ -n "${EFFDOCK_ONLY_ID:-}" ]]; then
  only_id_args=(--only-id "$EFFDOCK_ONLY_ID")
fi

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run eff-dock evaluate \
  --dataset "$EFFDOCK_DATASET" \
  --data-dir "$EFFDOCK_DATA_DIR" \
  --external-dir data/external_test \
  --pocket-centers "$EFFDOCK_CENTERS" \
  --checkpoint "$checkpoint" \
  --config "$config" \
  "${confidence_args[@]}" \
  --output-dir outputs/benchmarks/raw \
  --run-name "$EFFDOCK_RUN_NAME" \
  --protocol-id "$protocol_id" \
  --num-samples "$num_samples" \
  --num-steps "$num_steps" \
  --sigma "$sigma" \
  --time-schedule late \
  --schedule-power 3.0 \
  --pocket-cutoff "$pocket_cutoff" \
  --vina-guidance-scale "$vina_guidance_scale" \
  --vina-guidance-start-t "$vina_guidance_start_t" \
  --vina-guidance-ramp-power "$vina_guidance_ramp_power" \
  --vina-guidance-max-force "$vina_guidance_max_force" \
  --vina-guidance-max-velocity "$vina_guidance_max_velocity" \
  --vina-guidance-max-angular-velocity "$vina_guidance_max_angular_velocity" \
  --vina-guidance-protein-shell "$vina_guidance_protein_shell" \
  --vina-guidance-w-strain "$vina_guidance_w_strain" \
  --seed 42 \
  --num-shards "$EFFDOCK_NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID" \
  "${limit_args[@]}" \
  "${only_id_args[@]}"
