#!/usr/bin/env bash
#SBATCH --job-name=effdock-conf-select-v1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

task_id=${SLURM_ARRAY_TASK_ID:-0}
case "$task_id" in
  0)
    variant=control
    change="No loss change; fresh-optimizer continuation control"
    prediction="No systematic gain beyond ordinary fine-tuning variance"
    variant_args=()
    ;;
  1)
    variant=setwise
    change="Add setwise_success_weight=1.0"
    prediction="Validation top-1 <2A improves by at least 1.5 percentage points"
    variant_args=(--setwise-success-weight 1.0)
    ;;
  2)
    variant=pairwise
    change="Add pairwise_success_weight=0.5"
    prediction="Validation top-1 <2A improves by at least 1.5 percentage points"
    variant_args=(--pairwise-success-weight 0.5)
    ;;
  3)
    variant=rmsd_listwise
    change="Add listwise_weight=0.5"
    prediction="Validation top-1 <2A improves by at least 1.5 percentage points"
    variant_args=(--listwise-weight 0.5)
    ;;
  *)
    echo "Unknown sweep task id: $task_id" >&2
    exit 2
    ;;
esac

output_root=${EFFDOCK_CONFIDENCE_OUTPUT_ROOT:-outputs/eff-dock/confidence-selection-v1}
finetune_steps=${EFFDOCK_CONFIDENCE_FINETUNE_STEPS:-1500}
train_limit=${EFFDOCK_CONFIDENCE_TRAIN_LIMIT:-4096}
val_limit=${EFFDOCK_CONFIDENCE_VAL_LIMIT:-512}
eval_complexes=${EFFDOCK_CONFIDENCE_EVAL_COMPLEXES:-512}
eval_every=${EFFDOCK_CONFIDENCE_EVAL_EVERY:-250}
save_every=${EFFDOCK_CONFIDENCE_SAVE_EVERY:-250}
num_workers=${EFFDOCK_CONFIDENCE_NUM_WORKERS:-4}
smoke=${EFFDOCK_CONFIDENCE_SMOKE:-0}

out_dir="${output_root}/${variant}"
metrics="${out_dir}/metrics.json"
mkdir -p "$out_dir"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

train_command=(
  uv run eff-dock confidence train
    --config configs/train_confidence.yaml
    --resume weights/effdock_confidence_extmatch_n80_s25_step42500.pt
    --reset-optimizer
    --finetune-steps "$finetune_steps"
    --eval-on-start
    --out_dir "$out_dir"
    --run_name "effdock-conf-select-v1-${variant}"
    --train_limit "$train_limit"
    --val_limit "$val_limit"
    --eval_complexes "$eval_complexes"
    --eval_every "$eval_every"
    --save_every "$save_every"
    --num_workers "$num_workers"
    --lr 3e-5
    --muon_lr 2e-3
    --seed 43
    --no-use_wandb
    "${variant_args[@]}"
)

if [[ "$smoke" == "1" ]]; then
  "${train_command[@]}"
  exit 0
fi

oms research-runner \
  --question "Can ${variant} improve matched PLINDER pose selection from step 42500?" \
  --hypothesis "The declared pose-set loss improves top-1 selection without changing candidates or architecture." \
  --prediction "$prediction" \
  --baseline "weights/effdock_confidence_extmatch_n80_s25_step42500.pt" \
  --metric "success_selected_lt2 on frozen PLINDER val512" \
  --success "+1.5pp vs initial, above control, frozen composite no worse than -0.5pp" \
  --change "$change" \
  --metrics "$metrics" \
  --no-gate \
  --reason "scripts/check.sh fast and a one-step GPU smoke are required before this array is submitted" \
  -- \
  "${train_command[@]}"
