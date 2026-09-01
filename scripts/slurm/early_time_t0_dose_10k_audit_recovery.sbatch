#!/usr/bin/env bash
#SBATCH --job-name=effdock-t0-dose-audit-recovery
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
mkdir -p outputs/slurm outputs/eff-dock/audits

expected_recovery_script_sha256=5eddee6b88dcac78c20d4125fc2f0022a59641328eda9ce64887ecaead83e4a8
actual_recovery_script_sha256=$(sha256sum scripts/recover_early_time_t0_dose_audit.py | awk '{print $1}')
[[ "$actual_recovery_script_sha256" == "$expected_recovery_script_sha256" ]] || {
  echo "paired recovery audit script hash mismatch" >&2
  exit 2
}

output=outputs/eff-dock/audits/early-time-t0-dose-paired-10k-v1-20260814-job53926-recovery.json
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing recovery audit output: $output" >&2
  exit 2
fi

.venv/bin/python scripts/recover_early_time_t0_dose_audit.py \
  --control-dir outputs/eff-dock/early-time-t0-dose-control-t0p10-10k-v1-20260814 \
  --treatment-dir outputs/eff-dock/early-time-t0-dose-treatment-t0p15-10k-v1-20260814 \
  --control-config configs/train_early_time_t0_dose_control_10k.yaml \
  --treatment-config configs/train_early_time_t0_dose_treatment_10k.yaml \
  --common-init outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt \
  --expected-producer-array-job-id 53926 \
  --original-audit-job-id 53927 \
  --original-failure-message 'checkpoint loader/source hash mismatch' \
  --output "$output"
