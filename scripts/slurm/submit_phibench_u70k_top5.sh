#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "${1:-.}" && pwd -P)
cd "$repo_root"

protocol_id=EFFDOCK-PHIBENCH-U70K-TOP5-V1
source_root="$repo_root/outputs/benchmarks/s50_raw_refined_confidence_temporal_external_runs/d97d5eb907acc485dfde4b7fcf88d87b4d5fd8576014d2cfb89dd0518b9c9bb4"
protocol=docs/PHIBENCH_U70K_TOP5_PROTOCOL.md
inputs=docs/PHIBENCH_U70K_TOP5_INPUTS.json
evaluator=scripts/evaluate_phibench_u70k_top5_posebusters_shard.py
helper=scripts/evaluate_external_temporal_posebusters_shard.py
reporter=scripts/report_phibench_u70k_top5.py
pb_job_file=scripts/slurm/phibench_u70k_top5_posebusters.sbatch
report_job_file=scripts/slurm/phibench_u70k_top5_report.sbatch

required=(
  .venv/bin/python "$source_root/report/summary.json" "$protocol" "$inputs"
  "$evaluator" "$helper" "$reporter" "$pb_job_file" "$report_job_file"
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

verify_sha() {
  local path=$1 expected=$2 actual
  [[ -f "$path" && ! -L "$path" ]] || { echo "missing regular file: $path" >&2; exit 2; }
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || { echo "SHA mismatch: $path" >&2; exit 2; }
}
verify_sha "$source_root/report/summary.json" 3365e59753b13464f4911d28f0983e121f3f7be3ea00521f6942a17e167071bf

.venv/bin/python -m py_compile "$evaluator" "$reporter"
.venv/bin/python -m pytest -q tests/test_phibench_u70k_top5.py

protocol_sha=$(sha256sum "$protocol" | cut -d' ' -f1)
inputs_sha=$(sha256sum "$inputs" | cut -d' ' -f1)
evaluator_sha=$(sha256sum "$evaluator" | cut -d' ' -f1)
helper_sha=$(sha256sum "$helper" | cut -d' ' -f1)
reporter_sha=$(sha256sum "$reporter" | cut -d' ' -f1)
pb_job_sha=$(sha256sum "$pb_job_file" | cut -d' ' -f1)
report_job_sha=$(sha256sum "$report_job_file" | cut -d' ' -f1)
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
source_summary_sha=$(sha256sum "$source_root/report/summary.json" | cut -d' ' -f1)
content_id=$(printf '%s\0' "$protocol_id" "$protocol_sha" "$inputs_sha" \
  "$evaluator_sha" "$helper_sha" "$reporter_sha" "$pb_job_sha" \
  "$report_job_sha" "$src_sha" "$source_summary_sha" \
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638 \
  | sha256sum | cut -d' ' -f1)
output_root="$repo_root/outputs/benchmarks/phibench_u70k_top5_runs/$content_id"
execution_root="$repo_root/.effdock_execution_capsules/$protocol_id/$content_id"
[[ ! -e "$output_root" && ! -e "$execution_root" ]] || {
  echo "refusing to reuse output or execution capsule" >&2
  exit 2
}

temporary="$execution_root.tmp.$$"
mkdir -p "$temporary" "$output_root" "$repo_root/outputs/slurm"
trap 'rm -rf "$temporary"' EXIT
cp -a src scripts docs "$temporary/"
ln -s "$repo_root/.venv" "$temporary/.venv"
ln -s "$repo_root/data" "$temporary/data"
ln -s "$repo_root/outputs" "$temporary/outputs"
mv "$temporary" "$execution_root"
trap - EXIT

base_export="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_SOURCE_ROOT=$source_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_PROTOCOL_SHA=$protocol_sha,EFFDOCK_INPUTS_SHA=$inputs_sha,EFFDOCK_EVALUATOR_SHA=$evaluator_sha,EFFDOCK_HELPER_SHA=$helper_sha,EFFDOCK_REPORTER_SHA=$reporter_sha,EFFDOCK_SRC_SHA=$src_sha"
submitted=()
committed=0
cleanup() {
  local code=$?
  trap - EXIT
  if [[ "$committed" -eq 0 ]]; then
    for job in "${submitted[@]}"; do scancel "$job" 2>/dev/null || true; done
  fi
  exit "$code"
}
trap cleanup EXIT

raw=$(sbatch --parsable --hold --array=0 \
  --export="$base_export,EFFDOCK_STAGE=smoke" "$execution_root/$pb_job_file")
smoke_job=${raw%%;*}; [[ "$smoke_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-12%8 \
  --export="$base_export,EFFDOCK_STAGE=full" "$execution_root/$pb_job_file")
full_job=${raw%%;*}; [[ "$full_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_job")
raw=$(sbatch --parsable --dependency="afterok:$full_job" \
  --export="$base_export" "$execution_root/$report_job_file")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

printf '%s\n' \
  'status=submitted' \
  "protocol_id=$protocol_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "source_root=$source_root" \
  'cohort=phibench-derived-203' \
  'ranking=confidence-predicted-rmsd-top5' \
  'posebusters_evaluations=1015' \
  "protocol_sha256=$protocol_sha" \
  "inputs_sha256=$inputs_sha" \
  "evaluator_sha256=$evaluator_sha" \
  "reporter_sha256=$reporter_sha" \
  "src_sha256=$src_sha" \
  "smoke_job=$smoke_job" \
  "full_job=$full_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$output_root/.submission"
scontrol release "$smoke_job"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_job=%s\nfull_job=%s\nreport_job=%s\n' \
  "$output_root" "$smoke_job" "$full_job" "$report_job"
