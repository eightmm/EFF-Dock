#!/usr/bin/env bash
#SBATCH --job-name=effdock-t0-dose-audit
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm

expected_audit_sha256=11df01d88d0dcf30d886c0d961cb8f2b644c74c9b4b55fbc0145dc4e6691041e
expected_parent_audit_sha256=6ee4b5b94f196f20b62fff9d11e3b1a0f6cc3dbdb9c4fb939781c99b8d5208b9
expected_code_sha256=6b41a2f744b9f9678f870f4121160479ed559546306b134ca808fae40664e3cc
actual_audit_sha256=$(sha256sum scripts/audit_early_time_t0_dose_10k.py | awk '{print $1}')
actual_parent_audit_sha256=$(sha256sum scripts/audit_early_time_ft_50k.py | awk '{print $1}')
actual_code_sha256=$(
  {
    find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
    sha256sum pyproject.toml uv.lock
  } | sha256sum | awk '{print $1}'
)
[[ "$actual_audit_sha256" == "$expected_audit_sha256" ]] || {
  echo "paired audit code hash mismatch" >&2
  exit 2
}
[[ "$actual_parent_audit_sha256" == "$expected_parent_audit_sha256" ]] || {
  echo "shared audit helper hash mismatch" >&2
  exit 2
}
[[ "$actual_code_sha256" == "$expected_code_sha256" ]] || {
  echo "checkpoint loader/source hash mismatch" >&2
  exit 2
}

producer_array_job_id=${EFFDOCK_EXPECTED_PRODUCER_ARRAY_JOB_ID:?EFFDOCK_EXPECTED_PRODUCER_ARRAY_JOB_ID is required}
control_dir=outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814
treatment_dir=outputs/eff-dock/early-time-t0-dose-treatment-t0p15-10k-v1-20260814
audit_output=outputs/eff-dock/audits/early-time-t0-dose-paired-10k-v1-20260814-job${producer_array_job_id}.json
.venv/bin/python scripts/audit_early_time_t0_dose_10k.py \
  --control-dir "$control_dir" \
  --treatment-dir "$treatment_dir" \
  --control-config configs/train_early_time_t0_dose_control_10k.yaml \
  --treatment-config configs/train_early_time_t0_dose_treatment_10k.yaml \
  --common-init outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt \
  --expected-producer-array-job-id "$producer_array_job_id" \
  --output "$audit_output"
