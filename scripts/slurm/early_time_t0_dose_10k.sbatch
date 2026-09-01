#!/usr/bin/env bash
#SBATCH --job-name=effdock-t0-dose-10k
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=6000ada
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --array=0-1%2

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

split=data/splits/plinder.json
checkpoint=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
expected_steps=10000
expected_checkpoint_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
expected_split_sha256=3ac570bf08bced053f1ce040b57efca27c3be616f29a82cd66ef887c08860e6b
expected_code_sha256=6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc
expected_manifest_writer_sha256=7317ee3796b21a32ed0c5a1f628697a42ba32959d682207b203388ff6c4a42ab

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
  0)
    branch=control_t0p10
    config=configs/train_early_time_t0_dose_control_10k.yaml
    expected_config_sha256=13d2f9d8b7d64eb4b0286c6cdf84cba0110364991f9d7f16eba94e5f1be1cbde
    output_dir=outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814
    ;;
  1)
    branch=treatment_t0p15
    config=configs/train_early_time_t0_dose_treatment_10k.yaml
    expected_config_sha256=e24d3068d8c2d768c1643fec581cca41098acb2edb1bd431b29c16ddde0972f4
    output_dir=outputs/eff-dock/early-time-t0-dose-treatment-t0p15-10k-v1-20260814
    ;;
  *)
    echo "invalid dose task: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
    ;;
esac
resume_checkpoint=$output_dir/checkpoints/latest.pt
run_identity=$output_dir/run_identity.json

for required in \
  "$config" \
  "$split" \
  "$checkpoint" \
  .venv/bin/torchrun \
  .venv/bin/eff-dock \
  scripts/write_early_time_t0_dose_run_identity.py \
  pyproject.toml \
  uv.lock; do
  [[ -e "$required" ]] || { echo "missing required path: $required" >&2; exit 2; }
done

producer_array_job_id=${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}
producer_job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}

actual_checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
actual_split_sha256=$(sha256sum "$split" | awk '{print $1}')
actual_config_sha256=$(sha256sum "$config" | awk '{print $1}')
actual_manifest_writer_sha256=$(
  sha256sum scripts/write_early_time_t0_dose_run_identity.py | awk '{print $1}'
)
actual_code_sha256=$(
  {
    find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    sha256sum pyproject.toml uv.lock
  } | sha256sum | awk '{print $1}'
)

[[ "$actual_checkpoint_sha256" == "$expected_checkpoint_sha256" ]] || {
  echo "common EMA initialization hash mismatch" >&2
  exit 2
}
[[ "$actual_split_sha256" == "$expected_split_sha256" ]] || {
  echo "frozen PLINDER split hash mismatch" >&2
  exit 2
}
[[ "$actual_config_sha256" == "$expected_config_sha256" ]] || {
  echo "$branch config hash mismatch" >&2
  exit 2
}
[[ "$actual_code_sha256" == "$expected_code_sha256" ]] || {
  echo "training code hash mismatch" >&2
  exit 2
}
[[ "$actual_manifest_writer_sha256" == "$expected_manifest_writer_sha256" ]] || {
  echo "run identity writer hash mismatch" >&2
  exit 2
}

mkdir -p "$output_dir"
exec 9>"$output_dir/.run.lock"
if ! flock -n 9; then
  echo "another run already holds $output_dir/.run.lock" >&2
  exit 2
fi

train_mode=(--init-from "$checkpoint")
run_mode=fresh
if [[ -f "$resume_checkpoint" ]]; then
  resume_step=$(
    .venv/bin/python -c \
      "from effdock.checkpoint import load_checkpoint_file; print(int(load_checkpoint_file('$resume_checkpoint')['step']))"
  )
  if (( resume_step >= expected_steps )); then
    echo "$branch already complete at step $resume_step; refusing duplicate launch" >&2
    exit 2
  fi
  train_mode=(--resume "$resume_checkpoint")
  run_mode=resume
  echo "branch=$branch mode=resume checkpoint=$resume_checkpoint step=$resume_step"
elif compgen -G "$output_dir/checkpoints/*.pt" >/dev/null; then
  echo "checkpoint directory is non-empty but latest.pt is missing" >&2
  exit 2
else
  echo "branch=$branch mode=init-from checkpoint=$checkpoint source_step=50000 source_weights=ema"
fi

echo "split=$split split_sha256=$actual_split_sha256 train=47277 val=1076"
echo "config=$config config_sha256=$actual_config_sha256 total_steps=$expected_steps"
echo "code_sha256=$actual_code_sha256 output_dir=$output_dir"

.venv/bin/python scripts/write_early_time_t0_dose_run_identity.py start \
  --manifest "$run_identity" \
  --array-job-id "$producer_array_job_id" \
  --job-id "$producer_job_id" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --branch "$branch" \
  --mode "$run_mode" \
  --config "$config" \
  --config-sha256 "$actual_config_sha256" \
  --split "$split" \
  --split-sha256 "$actual_split_sha256" \
  --common-init "$checkpoint" \
  --common-init-sha256 "$actual_checkpoint_sha256" \
  --common-init-step 50000 \
  --code-sha256 "$actual_code_sha256" \
  --output-dir "$output_dir" \
  --expected-steps "$expected_steps"

finalize_failed_identity() {
  exit_code=$?
  trap - EXIT
  if (( exit_code != 0 )); then
    .venv/bin/python scripts/write_early_time_t0_dose_run_identity.py fail \
      --manifest "$run_identity" \
      --array-job-id "$producer_array_job_id" \
      --job-id "$producer_job_id" \
      --task-id "$SLURM_ARRAY_TASK_ID" \
      --exit-code "$exit_code" || true
  fi
  exit "$exit_code"
}
trap finalize_failed_identity EXIT

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

.venv/bin/python scripts/write_early_time_t0_dose_run_identity.py complete \
  --manifest "$run_identity" \
  --array-job-id "$producer_array_job_id" \
  --job-id "$producer_job_id" \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --baseline "$output_dir/checkpoints/rollout_step0000000.pt" \
  --endpoint "$output_dir/checkpoints/rollout_step0010000.pt" \
  --latest "$resume_checkpoint" \
  --expected-steps "$expected_steps"
trap - EXIT
