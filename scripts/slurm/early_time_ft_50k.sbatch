#!/usr/bin/env bash
#SBATCH --job-name=effdock-early-t0p10-50k
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=6000ada
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:4

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

config=configs/train_early_time_ft_50k.yaml
split=data/splits/plinder.json
checkpoint=weights/effdock_geometry_ft_100k_best.pt
output_dir=outputs/eff-dock/early-time-t0p10-50k-v1-20260813
resume_checkpoint=$output_dir/checkpoints/latest.pt
expected_steps=50000
expected_checkpoint_sha256=6932fb3ba6ebac770f714453529656a44b8f33cf15119d23c9e675d2d60b36db
expected_split_sha256=3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b
expected_config_sha256=439d95d7b56e49de6239e113ae4b7e4db94be9e10afe5b8c5f0ece48f076a369
expected_code_sha256=6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc

for required in \
  "$config" \
  "$split" \
  "$checkpoint" \
  .venv/bin/torchrun \
  .venv/bin/eff-dock \
  pyproject.toml \
  uv.lock; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done

actual_checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
actual_split_sha256=$(sha256sum "$split" | awk '{print $1}')
actual_config_sha256=$(sha256sum "$config" | awk '{print $1}')
actual_code_sha256=$(
  {
    find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    sha256sum pyproject.toml uv.lock
  } | sha256sum | awk '{print $1}'
)

[[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]] || {
  echo "source checkpoint hash mismatch" >&2
  exit 2
}
[[ "$actual_split_sha256" == "$expected_split_sha256" ]] || {
  echo "frozen PLINDER split hash mismatch" >&2
  exit 2
}
[[ "$actual_config_sha256" == "$expected_config_sha256" ]] || {
  echo "50k config hash mismatch" >&2
  exit 2
}
[[ "$actual_code_sha256" == "$expected_code_sha256" ]] || {
  echo "training code hash mismatch" >&2
  exit 2
}

mkdir -p "$output_dir"
exec 9>"$output_dir/.run.lock"
if ! flock -n 9; then
  echo "another 50k run already holds $output_dir/.run.lock" >&2
  exit 2
fi

train_mode=(--init-from "$checkpoint")
if [[ -f "$resume_checkpoint" ]]; then
  resume_step=$(
    .venv/bin/python -c \
      "from effdock.checkpoint import load_checkpoint_file; print(int(load_checkpoint_file('$resume_checkpoint')['step']))"
  )
  if (( resume_step >= expected_steps )); then
    echo "run already complete at step $resume_step; refusing duplicate launch" >&2
    exit 2
  fi
  train_mode=(--resume "$resume_checkpoint")
  echo "mode=resume checkpoint=$resume_checkpoint step=$resume_step"
elif compgen -G "$output_dir/checkpoints/*.pt" >/dev/null; then
  echo "checkpoint directory is non-empty but latest.pt is missing; refusing unsafe restart" >&2
  exit 2
else
  echo "mode=init-from checkpoint=$checkpoint source_step=100000"
fi

echo "split=$split split_sha256=$actual_split_sha256 train_declared=47310 val_declared=1076"
echo "filtered_contract_train=47277 filtered_contract_val=1076"
echo "config=$config config_sha256=$actual_config_sha256 total_steps=$expected_steps"
echo "code_sha256=$actual_code_sha256 output_dir=$output_dir"

export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export EFFDOCK_DDP_TIMEOUT_MIN=30

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
.venv/bin/torchrun --standalone --nproc_per_node=4 \
  .venv/bin/eff-dock train \
  --config "$config" \
  "${train_mode[@]}"
