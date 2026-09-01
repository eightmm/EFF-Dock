#!/usr/bin/env bash
# Submit eta=2 sampling for sigma={1,2,3,4}; sigma=0.5 is a frozen reference.

set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -le 1 ]] || { echo "usage: $0 [SAFE_RUN_ID]" >&2; exit 2; }
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }

protocol_id=EFFDOCK-UNIFIED-GUIDANCE-SIGMA-SWEEP-ETA2-V1
output_root="outputs/benchmarks/guidance_sigma_sweep_eta2_runs/$run_id"
reference_root=outputs/benchmarks/guidance_steric_high_eta_confidence_runs/20260807T045916Z
cohort_audit=outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z/audit/combined.json
mkdir -p outputs/benchmarks/guidance_sigma_sweep_eta2_runs outputs/benchmarks/logs
mkdir "$output_root" || { echo "refusing to reuse output root $output_root" >&2; exit 2; }

submitted=()
committed=0
cleanup() {
  code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
  fi
  exit "$code"
}
trap cleanup EXIT

for required in \
  weights/effdock_geometry_ft_100k_best.pt \
  weights/effdock_confidence_extmatch_n80_s25_step42500.pt \
  configs/train.yaml \
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  "$cohort_audit"; do
  [[ -f "$required" ]] || { echo "missing frozen input $required" >&2; exit 2; }
done

.venv/bin/python scripts/audit_guidance_sigma_sweep_eta2.py \
  --output-root "$reference_root" \
  --mode reference \
  --input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --output "$output_root/audit/reference.json" >/dev/null

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}" pyproject.toml uv.lock configs/train.yaml
  docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json docs/GUIDANCE_SIGMA_SWEEP_ETA2_PROTOCOL.md
  scripts/audit_guidance_sigma_sweep_eta2.py scripts/report_guidance_sigma_sweep_eta2.py
  scripts/slurm/guidance_sigma_sweep_eta2_array.sbatch
  scripts/slurm/guidance_sigma_sweep_eta2_audit.sbatch
  scripts/slurm/guidance_sigma_sweep_eta2_report.sbatch
  scripts/slurm/submit_guidance_sigma_sweep_eta2.sh scripts/create_execution_capsule.py
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
execution_root=".effdock_execution_capsules/$protocol_id/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$repo_root/$execution_root" \
  --link-root .venv --link-root data --link-root weights --link-root outputs \
  "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")

git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' "protocol_id=$protocol_id" "run_id=$run_id" \
  "output_root=$output_root" "reference_sigma05_root=$reference_root" \
  'eta=2.0' 'new_sigmas=1.0,2.0,3.0,4.0' 'sampling_budget=N100/S10' \
  "git_commit=$git_commit" "git_diff_sha256=$git_diff_sha256" \
  "execution_root=$execution_root_abs" > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1"
base_export+=",EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_COHORT_MANIFEST=$cohort_audit"
base_export+=",EFFDOCK_SIGMA_REFERENCE_ROOT=$reference_root"
submit() {
  dependency=$1; script=$2; export_spec=$3; array=${4:-}; partition=${5:?partition required}
  args=(--parsable --partition="$partition" --export="$export_spec")
  [[ -z "$dependency" ]] || args+=(--dependency="afterok:$dependency")
  [[ -z "$array" ]] || args+=(--array="$array")
  raw=$(sbatch "${args[@]}" "$script")
  job=${raw%%;*}
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "invalid sbatch response: $raw" >&2; return 1; }
  printf '%s' "$job"
}

smoke_export="$base_export,EFFDOCK_SIGMA_SWEEP_SMOKE=1,EFFDOCK_SIGMA_SWEEP_AUDIT_MODE=smoke"
full_export="$base_export,EFFDOCK_SIGMA_SWEEP_SMOKE=0,EFFDOCK_SIGMA_SWEEP_AUDIT_MODE=full"
smoke_job=$(submit '' "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_eta2_array.sbatch" "$smoke_export" '0-7%4' '6000ada,heavy'); submitted+=("$smoke_job")
smoke_audit_job=$(submit "$smoke_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_eta2_audit.sbatch" "$smoke_export" '' cpu_only); submitted+=("$smoke_audit_job")
sampling_job=$(submit "$smoke_audit_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_eta2_array.sbatch" "$full_export" '0-63%4' '6000ada,heavy'); submitted+=("$sampling_job")
audit_job=$(submit "$sampling_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_eta2_audit.sbatch" "$full_export" '' cpu_only); submitted+=("$audit_job")
report_job=$(submit "$audit_job" "$execution_root_abs/scripts/slurm/guidance_sigma_sweep_eta2_report.sbatch" "$full_export" '' cpu_only); submitted+=("$report_job")

printf '%s\n' \
  "smoke_job=$smoke_job" "smoke_audit_job=$smoke_audit_job" \
  "sampling_job=$sampling_job" "audit_job=$audit_job" "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_job=%s\nsmoke_audit_job=%s\nsampling_job=%s\naudit_job=%s\nreport_job=%s\n' \
  "$output_root" "$smoke_job" "$smoke_audit_job" "$sampling_job" "$audit_job" "$report_job"
