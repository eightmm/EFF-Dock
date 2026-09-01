#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
[[ $# -ge 2 && $# -le 3 ]] || {
  echo "usage: $0 SOURCE_U50_RUN UPSTREAM_REPORT_JOB [SAFE_RUN_ID]" >&2
  exit 2
}
source_run=$(cd "$1" && pwd -P)
upstream_job=$2
run_id=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$upstream_job" =~ ^[0-9]+$ ]] || { echo "invalid upstream job" >&2; exit 2; }
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }

confidence_sha=fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469
output_root="$repo_root/outputs/benchmarks/openbind_official_top25_u50_runs/$run_id"
metadata=data/external_benchmarks/data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv
protocol=docs/OPENBIND_OFFICIAL_TOP25_PROTOCOL.md
addendum=docs/OPENBIND_OFFICIAL_TOP25_U50_PROTOCOL.md
evaluator=scripts/evaluate_openbind_official_topn.py
pb_job_file=scripts/slurm/openbind_official_top25_posebusters.sbatch
ost_job_file=scripts/slurm/openbind_official_top25_openstructure.sbatch
report_job_file=scripts/slurm/openbind_official_top25_report.sbatch

required=(
  .venv/bin/python .venvs/openstructure/bin/ost "$source_run/.submission"
  "$metadata" "$protocol" "$addendum" "$evaluator" "$pb_job_file" "$ost_job_file"
  "$report_job_file"
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ ! -e "$output_root" ]] || { echo "refusing existing output root: $output_root" >&2; exit 2; }
rg -q "^confidence_sha256=$confidence_sha$" "$source_run/.submission" \
  || { echo "source run is not the frozen U50 selector" >&2; exit 2; }
[[ $(sha256sum "$metadata" | cut -d' ' -f1) == 389a7edca3ac8034d6533da5a3f3235619e7206aef7284441fd52d350bb1c652 ]]
[[ "$(.venvs/openstructure/bin/ost --version)" == "OpenStructure 2.11.1" ]]
.venv/bin/python -c 'import posebusters; assert posebusters.__version__ == "0.6.5"'
.venv/bin/python -m py_compile "$evaluator"
.venv/bin/ruff check "$evaluator" tests/test_openbind_official_topn.py
.venv/bin/python -m pytest -q tests/test_openbind_official_topn.py

protocol_sha256=$(sha256sum "$protocol" | cut -d' ' -f1)
addendum_sha256=$(sha256sum "$addendum" | cut -d' ' -f1)
evaluator_sha256=$(sha256sum "$evaluator" | cut -d' ' -f1)
content_id=$(printf '%s\0' EFFDOCK-OPENBIND-OFFICIAL-TOP25-U50-REPORT-V1 \
  "$protocol_sha256" "$addendum_sha256" "$evaluator_sha256" "$confidence_sha" \
  "$(sha256sum "$pb_job_file" | cut -d' ' -f1)" \
  "$(sha256sum "$ost_job_file" | cut -d' ' -f1)" \
  "$(sha256sum "$report_job_file" | cut -d' ' -f1)" \
  | sha256sum | cut -d' ' -f1)
execution_root="$repo_root/.effdock_execution_capsules/EFFDOCK-OPENBIND-OFFICIAL-TOP25-U50-REPORT-V1/$run_id-${content_id:0:16}"
[[ ! -e "$execution_root" ]] || { echo "refusing existing capsule" >&2; exit 2; }

temporary="$execution_root.tmp.$$"
mkdir -p "$temporary/scripts/slurm" "$temporary/docs" "$output_root" "$repo_root/outputs/slurm"
trap 'rm -rf "$temporary"' EXIT
cp "$evaluator" "$temporary/scripts/"
cp "$pb_job_file" "$ost_job_file" "$report_job_file" "$temporary/scripts/slurm/"
cp "$protocol" "$addendum" "$temporary/docs/"
ln -s "$repo_root/.venv" "$temporary/.venv"
ln -s "$repo_root/.venvs" "$temporary/.venvs"
ln -s "$repo_root/data" "$temporary/data"
ln -s "$repo_root/outputs" "$temporary/outputs"
mv "$temporary" "$execution_root"
trap - EXIT

base_export="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_SOURCE_RUN=$source_run,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_CONFIDENCE_SHA256=$confidence_sha"
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

raw=$(sbatch --parsable --hold --dependency="afterok:$upstream_job" --array=0-0 \
  --export="$base_export,EFFDOCK_STAGE=smoke,EFFDOCK_NUM_SHARDS=1,EFFDOCK_MAX_COMPLEXES=1" \
  "$execution_root/$pb_job_file")
smoke_pb_job=${raw%%;*}; [[ "$smoke_pb_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_pb_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_pb_job" --array=0-0 \
  --export="$base_export,EFFDOCK_STAGE=smoke,EFFDOCK_NUM_SHARDS=1,EFFDOCK_MAX_COMPLEXES=1" \
  "$execution_root/$ost_job_file")
smoke_ost_job=${raw%%;*}; [[ "$smoke_ost_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_ost_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_ost_job" --array=0-63%32 \
  --export="$base_export,EFFDOCK_STAGE=full,EFFDOCK_NUM_SHARDS=64" \
  "$execution_root/$pb_job_file")
full_pb_job=${raw%%;*}; [[ "$full_pb_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_pb_job")
raw=$(sbatch --parsable --dependency="afterok:$full_pb_job" --array=0-63%32 \
  --export="$base_export,EFFDOCK_STAGE=full,EFFDOCK_NUM_SHARDS=64" \
  "$execution_root/$ost_job_file")
full_ost_job=${raw%%;*}; [[ "$full_ost_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_ost_job")
raw=$(sbatch --parsable --dependency="afterok:$full_ost_job" \
  --export="$base_export,EFFDOCK_NUM_SHARDS=64" "$execution_root/$report_job_file")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

printf '%s\n' \
  'status=submitted' \
  'protocol_id=EFFDOCK-OPENBIND-OFFICIAL-TOP25-U50-REPORT-V1' \
  "run_id=$run_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "source_run=$source_run" \
  "upstream_report_job=$upstream_job" \
  "confidence_sha256=$confidence_sha" \
  'denominator=802' \
  'rank=U50_confidence_after_refinement_stable_top25' \
  'metrics=PoseBusters0.6.5+OpenStructure2.11.1_BiSyRMSD_LDDT-PLI' \
  "protocol_sha256=$protocol_sha256" \
  "addendum_sha256=$addendum_sha256" \
  "evaluator_sha256=$evaluator_sha256" \
  "smoke_posebusters_job=$smoke_pb_job" \
  "smoke_openstructure_job=$smoke_ost_job" \
  "full_posebusters_job=$full_pb_job" \
  "full_openstructure_job=$full_ost_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$output_root/.submission"
scontrol release "$smoke_pb_job"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_pb=%s\nsmoke_ost=%s\nfull_pb=%s\nfull_ost=%s\nreport=%s\n' \
  "$output_root" "$smoke_pb_job" "$smoke_ost_job" "$full_pb_job" "$full_ost_job" "$report_job"
