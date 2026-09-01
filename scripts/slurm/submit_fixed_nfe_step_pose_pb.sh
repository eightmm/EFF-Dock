#!/usr/bin/env bash
set -euo pipefail

die() { echo "$*" >&2; exit 2; }
verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" ]] || die "missing frozen input: $path"
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [[ "$actual" == "$expected" ]] || die "SHA-256 mismatch: $path"
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
[[ $# -le 1 ]] || die "usage: $0 [SAFE_RUN_ID]"
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "unsafe run ID"

protocol_id=EFFDOCK-FIXED-NFE-STEP-POSE-PB-V1
output_root="$repo_root/outputs/benchmarks/fixed_nfe_step_pose_pb_runs/$run_id"
source_report="$repo_root/outputs/benchmarks/fixed_nfe_step_pose_runs/20260827T022010Z/report/report.json"
source_report_sha=b66fb23436c4d5b89f3089232c8a40fcf9147010ae9ca67c5f29b6007f2d146f
n100_scores_root="$repo_root/outputs/benchmarks/s50_symmetry_confidence_refined_external_runs/85574152669f7d6a8fa6d60ba2ad7e2e4e9b37a5a5840ac57eeeffe9011100c9/full/u050000"
n40_scores_root="$repo_root/outputs/benchmarks/fixed_nfe_step_pose_runs/20260827T022010Z/full/confidence/u050000"
verify_sha "$source_report" "$source_report_sha"
verify_sha "$repo_root/outputs/benchmarks/fixed_nfe_step_pose_runs/20260827T022010Z/report/complex_metrics.csv" 86ddf0da1f179d2afd702dbabdbb0de18d8ec68b76ea74a0f284c53aac508aaa
[[ -d "$n100_scores_root" && -d "$n40_scores_root" ]] || die "missing frozen U50 score roots"
[[ -x .venv/bin/python ]] || die "missing synchronized .venv"
.venv/bin/python -c 'import posebusters; assert posebusters.__version__ == "0.6.5"'

evaluator_sha=$(sha256sum scripts/evaluate_fixed_nfe_step_pose_pb.py | cut -d' ' -f1)
protocol_sha=$(sha256sum docs/FIXED_NFE_STEP_POSE_PB_PROTOCOL.md | cut -d' ' -f1)
copy_files=(
  pyproject.toml
  uv.lock
  docs/FIXED_NFE_STEP_POSE_PB_PROTOCOL.md
  scripts/evaluate_fixed_nfe_step_pose_pb.py
  scripts/slurm/fixed_nfe_step_pose_pb.sbatch
  scripts/slurm/fixed_nfe_step_pose_pb_report.sbatch
  scripts/slurm/submit_fixed_nfe_step_pose_pb.sh
)
capsule_args=()
for path in "${copy_files[@]}"; do
  [[ -f "$path" ]] || die "missing control file: $path"
  capsule_args+=(--copy-file "$path")
done
control_capsule="$repo_root/.effdock_execution_capsules/$protocol_id/$run_id"
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$control_capsule" \
  --link-root .venv --link-root outputs --link-root data \
  "${capsule_args[@]}" >/dev/null
control_capsule_sha=$(sha256sum "$control_capsule/execution_capsule.json" | cut -d' ' -f1)

mkdir -p outputs/benchmarks/fixed_nfe_step_pose_pb_runs outputs/benchmarks/logs
mkdir "$output_root" || die "refusing to reuse output root: $output_root"
submitted=()
committed=0
cleanup() {
  local code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    local job
    for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
  fi
  exit "$code"
}
trap cleanup EXIT

git_commit=$(git rev-parse HEAD)
git_diff_sha256=$(git diff --no-ext-diff | sha256sum | cut -d' ' -f1)
printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "run_id=$run_id" \
  "output_root=$output_root" \
  'datasets=astex:85,posebusters:308' \
  'conditions=s10_n100:raw+refined,s25_n40:raw+refined' \
  'selector=u050000_argmin_predicted_rmsd' \
  'posebusters_version=0.6.5' \
  'primary_validity=protein_ligand_only_21' \
  'secondary_validity=official_redock_27' \
  'selected_pose_count=1572' \
  'partition=cpu_only' \
  'explicit_cpu_count=none' \
  "source_report=$source_report" \
  "source_report_sha256=$source_report_sha" \
  "control_capsule=$control_capsule" \
  "control_capsule_sha256=$control_capsule_sha" \
  "evaluator_sha256=$evaluator_sha" \
  "protocol_sha256=$protocol_sha" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" > "$output_root/.submission.pending"

base_export="ALL,EFFDOCK_CONTROL_CAPSULE=$control_capsule,EFFDOCK_OUTPUT_ROOT=$output_root"
base_export+=",EFFDOCK_SOURCE_REPORT=$source_report,EFFDOCK_SOURCE_REPORT_SHA256=$source_report_sha"
base_export+=",EFFDOCK_N100_SCORES_ROOT=$n100_scores_root,EFFDOCK_N40_SCORES_ROOT=$n40_scores_root"
base_export+=",EFFDOCK_EVALUATOR_SHA256=$evaluator_sha,EFFDOCK_PROTOCOL_SHA256=$protocol_sha"
smoke_raw=$(sbatch --parsable --partition=cpu_only --array=0-1%2 \
  --export="$base_export,EFFDOCK_RUN_MODE=smoke" \
  "$control_capsule/scripts/slurm/fixed_nfe_step_pose_pb.sbatch")
smoke_job=${smoke_raw%%;*}
[[ "$smoke_job" =~ ^[0-9]+$ ]] || die "invalid smoke sbatch response: $smoke_raw"
submitted+=("$smoke_job")
full_raw=$(sbatch --parsable --partition=cpu_only --dependency="afterok:$smoke_job" \
  --array=0-31%12 --export="$base_export,EFFDOCK_RUN_MODE=full" \
  "$control_capsule/scripts/slurm/fixed_nfe_step_pose_pb.sbatch")
full_job=${full_raw%%;*}
[[ "$full_job" =~ ^[0-9]+$ ]] || die "invalid full sbatch response: $full_raw"
submitted+=("$full_job")
report_raw=$(sbatch --parsable --partition=cpu_only --dependency="afterok:$full_job" \
  --export="$base_export" \
  "$control_capsule/scripts/slurm/fixed_nfe_step_pose_pb_report.sbatch")
report_job=${report_raw%%;*}
[[ "$report_job" =~ ^[0-9]+$ ]] || die "invalid report sbatch response: $report_raw"
submitted+=("$report_job")

printf '%s\n' \
  "smoke_job=$smoke_job" \
  "full_job=$full_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke=%s\nfull=%s\nreport=%s\n' \
  "$output_root" "$smoke_job" "$full_job" "$report_job"
