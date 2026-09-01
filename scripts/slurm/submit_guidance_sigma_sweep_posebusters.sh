#!/usr/bin/env bash
# Submit official PoseBusters evaluation for the completed eta=2 sigma sweep.

set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

sweep_root=outputs/benchmarks/guidance_sigma_sweep_eta2_runs/20260809T031535Z
reference_root=outputs/benchmarks/guidance_steric_high_eta_confidence_runs/20260807T045916Z
pb_root="$sweep_root/sigmadock_posebusters"
[[ -f "$sweep_root/audit/full.json" ]] || { echo "missing passed sigma audit" >&2; exit 2; }
[[ -f "$sweep_root/audit/reference.json" ]] || { echo "missing passed reference audit" >&2; exit 2; }
mkdir -p outputs/benchmarks/logs
mkdir "$pb_root" || { echo "refusing to reuse PoseBusters output root: $pb_root" >&2; exit 2; }

submitted=()
committed=0
cleanup() {
  code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
    printf '%s\n' 'status=launcher_failed' "cancelled_jobs=${submitted[*]:-none}" > "$pb_root/.submission.failed"
  fi
  exit "$code"
}
trap cleanup EXIT

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}"
  pyproject.toml uv.lock
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json
  docs/GUIDANCE_SIGMADOCK_COMPATIBLE_POSEBUSTERS_PROTOCOL.md
  scripts/create_execution_capsule.py
  scripts/audit_guidance_sigma_sweep_posebusters_smoke.py
  scripts/slurm/guidance_sigma_sweep_posebusters_array.sbatch
  scripts/slurm/guidance_sigma_sweep_posebusters_smoke_audit.sbatch
  scripts/slurm/guidance_sigma_sweep_posebusters_report.sbatch
  scripts/slurm/submit_guidance_sigma_sweep_posebusters.sh
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
run_id=$(date -u +%Y%m%dT%H%M%SZ)
execution_root=".effdock_execution_capsules/EFFDOCK-SIGMADOCK-COMPATIBLE-PB-SIGMA-ETA2-V1/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" \
  --output "$repo_root/$execution_root" \
  --link-root .venv \
  --link-root outputs \
  --link-root data \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")

git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' \
  'protocol_id=EFFDOCK-SIGMADOCK-COMPATIBLE-PB-SIGMA-ETA2-V1' \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "sweep_root=$sweep_root" \
  "reference_root=$reference_root" \
  "execution_root=$execution_root_abs" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  'posebusters_version=0.6.5' \
  'posebusters_config=redock' \
  'primary_validity=all_27_non_rmsd_checks' \
  'compatibility_validity=sigmadock_listed_26_non_rmsd_checks' \
  'selectors=confidence' \
  'sigmas=0.5,1.0,2.0,3.0,4.0' \
  'datasets=astex:85,posebusters:308' \
  > "$pb_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src"
base_export+=",PYTHONDONTWRITEBYTECODE=1,EFFDOCK_SIGMA_SWEEP_ROOT=$sweep_root"
base_export+=",EFFDOCK_SIGMA_REFERENCE_ROOT=$reference_root"
submit() {
  dependency=$1
  script=$2
  export_spec=$3
  array=${4:-}
  args=(--parsable --export="$export_spec")
  [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || args+=(--array="$array")
  raw=$(sbatch "${args[@]}" "$script")
  job=${raw%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: $raw" >&2; return 1; }
  printf '%s' "$job"
}

smoke_job=$(submit '' "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_posebusters_array.sbatch" "$base_export,EFFDOCK_PB_MODE=smoke" '0-9%10'); submitted+=("$smoke_job")
smoke_audit_job=$(submit "$smoke_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_posebusters_smoke_audit.sbatch" "$base_export"); submitted+=("$smoke_audit_job")
full_job=$(submit "$smoke_audit_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_posebusters_array.sbatch" "$base_export,EFFDOCK_PB_MODE=full" '0-79%24'); submitted+=("$full_job")
report_job=$(submit "$full_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_posebusters_report.sbatch" "$base_export"); submitted+=("$report_job")

printf '%s\n' \
  "smoke_job=$smoke_job" \
  "smoke_audit_job=$smoke_audit_job" \
  "full_job=$full_job" \
  "report_job=$report_job" \
  >> "$pb_root/.submission.pending"
mv "$pb_root/.submission.pending" "$pb_root/.submission"
committed=1
trap - EXIT
printf 'pb_root=%s\nsmoke_job=%s\nsmoke_audit_job=%s\nfull_job=%s\nreport_job=%s\n' \
  "$pb_root" "$smoke_job" "$smoke_audit_job" "$full_job" "$report_job"
