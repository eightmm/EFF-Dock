#!/usr/bin/env bash
# Submit the corrected parallel full/audit/report tail after an existing smoke audit.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -eq 3 ]] || { echo "usage: $0 EXISTING_OUTPUT_ROOT SMOKE_AUDIT_JOB SAFE_CAPSULE_ID" >&2; exit 2; }
output_root=$1
smoke_audit_job=$2
capsule_id=$3
[[ "$output_root" =~ ^outputs/benchmarks/guidance_all_pose_pb_eta_runs/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe output root" >&2; exit 2; }
[[ "$smoke_audit_job" =~ ^[0-9]+$ ]] || { echo "smoke audit job must be numeric" >&2; exit 2; }
[[ "$capsule_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe capsule ID" >&2; exit 2; }
[[ -d "$output_root" ]] || { echo "missing output root" >&2; exit 2; }
[[ ! -e "$output_root/posebusters/full" && ! -e "$output_root/audit/full.json" && ! -e "$output_root/report" ]] || { echo "full outputs already exist" >&2; exit 2; }
protocol_id=EFFDOCK-GUIDANCE-ALL-POSE-PB-ETA-V1
mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=("${package_files[@]}" pyproject.toml uv.lock docs/GUIDANCE_ALL_POSE_PB_ETA_PROTOCOL.md scripts/run_guidance_all_pose_posebusters.py scripts/audit_guidance_all_pose_posebusters.py scripts/report_guidance_all_pose_posebusters.py scripts/slurm/guidance_all_pose_pb_array.sbatch scripts/slurm/guidance_all_pose_pb_audit.sbatch scripts/slurm/guidance_all_pose_pb_report.sbatch scripts/slurm/resume_guidance_all_pose_pb_eta_full.sh scripts/create_execution_capsule.py)
capsule_args=(); for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
execution_root=".effdock_execution_capsules/$protocol_id/$capsule_id"
.venv/bin/python scripts/create_execution_capsule.py --repo-root "$repo_root" --output "$repo_root/$execution_root" --link-root .venv --link-root data --link-root outputs "${capsule_args[@]}" >/dev/null
execution_root_abs=$(readlink -f "$execution_root")
git_commit=$(git rev-parse HEAD); read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
record="$output_root/.submission.full-r2.pending"
printf '%s\n' 'status=submitting' "protocol_id=$protocol_id" "output_root=$output_root" "smoke_audit_job=$smoke_audit_job" 'workers=4' 'posebusters_chunk_size=25' "git_commit=$git_commit" "git_diff_sha256=$git_diff_sha256" "execution_root=$execution_root_abs" > "$record"
base_export="ALL,EFFDOCK_REPO_DIR=$execution_root_abs,PYTHONPATH=$execution_root_abs/src,PYTHONDONTWRITEBYTECODE=1,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_ALL_POSE_PB_MODE=full"
submit(){ dependency=$1; script=$2; array=${3:-}; args=(--parsable --partition=test,cpu_only --export="$base_export" --dependency="afterok:$dependency"); [[ -z "$array" ]] || args+=(--array="$array"); raw=$(sbatch "${args[@]}" "$script"); job=${raw%%;*}; [[ "$job" =~ ^[0-9]+$ ]] || return 1; printf '%s' "$job"; }
submitted=(); committed=0
cleanup(){ code=$?; trap - EXIT; if [[ "$committed" -eq 0 ]]; then for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done; fi; exit "$code"; }
trap cleanup EXIT
full_job=$(submit "$smoke_audit_job" "$execution_root_abs/scripts/slurm/guidance_all_pose_pb_array.sbatch" '0-63%8'); submitted+=("$full_job")
audit_job=$(submit "$full_job" "$execution_root_abs/scripts/slurm/guidance_all_pose_pb_audit.sbatch"); submitted+=("$audit_job")
report_job=$(submit "$audit_job" "$execution_root_abs/scripts/slurm/guidance_all_pose_pb_report.sbatch"); submitted+=("$report_job")
printf '%s\n' "full_job=$full_job" "full_audit_job=$audit_job" "report_job=$report_job" "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$record"
sed -i 's/^status=submitting$/status=submitted/' "$record"; mv "$record" "$output_root/.submission.full-r2"
committed=1; trap - EXIT
printf 'full_job=%s\nfull_audit_job=%s\nreport_job=%s\n' "$full_job" "$audit_job" "$report_job"
