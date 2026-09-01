#!/usr/bin/env bash
#SBATCH --job-name=effdock-pl-k2-coord-audit
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --dependency=afterok:54518
#SBATCH --kill-on-invalid-dep=yes

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/benchmarks/logs

producer_job_id=54518
auditor=scripts/audit_early_time_sampler_plinder_k2_full.py
auditor_sha256=e9f1b490271862178e9226dda4b726c7cb0cb3c5693fd8615e6d5f87fb60c946
run_base=outputs/benchmarks/early_time_sampler_plinder_k2_paired_runs/t0p10-continuation-plinder-k2-v1-pathfix1-20260815
run_root="$run_base/full"
eligibility_manifest="$run_base/eligibility_manifest.json"
eligibility_manifest_sha256=6ebeb2d165e1def6ebf7b5bba301f82d4a9c3ff9d6c5cd43616dcf09edbd38ac
output="$run_base/reports/full_coordinate_audit.json"

verify_sha256() {
  local label=$1
  local path=$2
  local expected=$3
  local actual
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "missing or symlinked $label: $path" >&2
    exit 2
  fi
  read -r actual _ < <(sha256sum "$path")
  if [[ "$actual" != "$expected" ]]; then
    echo "$label SHA-256 mismatch: expected=$expected actual=$actual path=$path" >&2
    exit 2
  fi
}

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo "this audit must run through sbatch with its afterok:$producer_job_id dependency" >&2
  exit 2
fi
if [[ ${SLURM_JOB_PARTITION:-} != "cpu_only" ]]; then
  echo "coordinate audit requires the cpu_only partition" >&2
  exit 2
fi
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 4 ]]; then
  echo "coordinate audit requires exactly four CPUs" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "missing executable .venv/bin/python" >&2
  exit 2
fi
verify_sha256 auditor "$auditor" "$auditor_sha256"
verify_sha256 eligibility-manifest "$eligibility_manifest" "$eligibility_manifest_sha256"

if [[ ! -d "$run_root" || -L "$run_root" ]]; then
  echo "missing or symlinked full-stage root: $run_root" >&2
  exit 2
fi
arms=(s25_ema s50_ema parent50k_plus10k_t0p10_ema)
for shard_index in {0..7}; do
  printf -v shard_name 'shard-%03d-of-008' "$shard_index"
  shard_root="$run_root/$shard_name"
  if [[ ! -f "$shard_root/paired_summary.json" ]]; then
    echo "producer $producer_job_id artifact is incomplete: $shard_root/paired_summary.json" >&2
    exit 2
  fi
  for arm in "${arms[@]}"; do
    arm_root="$shard_root/arms/$arm"
    if [[ ! -f "$arm_root/results.csv" || ! -f "$arm_root/summary.json" || ! -d "$arm_root/poses" ]]; then
      echo "producer $producer_job_id artifact is incomplete: $arm_root" >&2
      exit 2
    fi
  done
done

if [[ -e "$output" || -L "$output" ]]; then
  echo "refusing to overwrite existing coordinate audit output: $output" >&2
  exit 2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

printf 'producer_job_id=%s auditor_sha256=%s eligibility_manifest_sha256=%s\n' \
  "$producer_job_id" "$auditor_sha256" "$eligibility_manifest_sha256"
printf 'run_root=%s output=%s partition=%s cpus=%s\n' \
  "$run_root" "$output" "${SLURM_JOB_PARTITION}" "${SLURM_CPUS_PER_TASK}"

.venv/bin/python "$auditor" \
  --run-root "$run_root" \
  --eligibility-manifest "$eligibility_manifest" \
  --eligibility-manifest-sha256 "$eligibility_manifest_sha256" \
  --output "$output" \
  --fast-valid-mode sampled \
  --fast-valid-sample-count 32 \
  --progress-every 100
