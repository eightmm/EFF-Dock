#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
[[ $# -le 1 ]] || { echo "usage: $0 [SAFE_RUN_ID]" >&2; exit 2; }
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }

protocol_id=EFFDOCK-OPENBIND-OFFICIAL-TOP25-V1
source_run="$repo_root/outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z"
output_root="$repo_root/outputs/benchmarks/openbind_official_top25_runs/$run_id"
metadata=data/external_benchmarks/data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv
protocol=docs/OPENBIND_OFFICIAL_TOP25_PROTOCOL.md
evaluator=scripts/evaluate_openbind_official_topn.py
pb_job_file=scripts/slurm/openbind_official_top25_posebusters.sbatch
ost_job_file=scripts/slurm/openbind_official_top25_openstructure.sbatch
report_job_file=scripts/slurm/openbind_official_top25_report.sbatch

required=(
  .venv/bin/python .venv/bin/ruff .venvs/openstructure/bin/ost
  "$source_run/report/summary.json" "$metadata" "$protocol" "$evaluator"
  "$pb_job_file" "$ost_job_file" "$report_job_file"
  scripts/create_execution_capsule.py tests/test_openbind_official_topn.py
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ ! -e "$output_root" ]] || { echo "refusing existing output root: $output_root" >&2; exit 2; }

read -r metadata_sha256 _ < <(sha256sum "$metadata")
[[ "$metadata_sha256" == 389a7edca3ac8034d6533da5a3f3235619e7206aef7284441fd52d350bb1c652 ]] || {
  echo "OpenBind metadata SHA mismatch" >&2
  exit 2
}
[[ "$(.venvs/openstructure/bin/ost --version)" == "OpenStructure 2.11.1" ]] || {
  echo "OpenStructure version mismatch" >&2
  exit 2
}
.venv/bin/python -c 'import posebusters; assert posebusters.__version__ == "0.6.5"'
.venv/bin/python -m py_compile "$evaluator"
.venv/bin/ruff check "$evaluator" tests/test_openbind_official_topn.py
.venv/bin/python -m pytest -q tests/test_openbind_official_topn.py
.venv/bin/python - <<'PY'
from pathlib import Path
from scripts.evaluate_openbind_official_topn import load_official_cohort

path = Path("data/external_benchmarks/data/OpenBind_EV-A71_2A/EV-A71_2A_metadata.csv")
assert len(load_official_cohort(path)) == 802
PY

protocol_sha256=$(sha256sum "$protocol" | cut -d' ' -f1)
evaluator_sha256=$(sha256sum "$evaluator" | cut -d' ' -f1)
content_id=$(printf '%s\0' \
  "$protocol_id" "$protocol_sha256" "$evaluator_sha256" "$metadata_sha256" \
  "$(sha256sum "$pb_job_file" | cut -d' ' -f1)" \
  "$(sha256sum "$ost_job_file" | cut -d' ' -f1)" \
  "$(sha256sum "$report_job_file" | cut -d' ' -f1)" \
  | sha256sum | cut -d' ' -f1)
execution_root_rel=".effdock_execution_capsules/$protocol_id/${run_id}-${content_id:0:16}"
execution_root="$repo_root/$execution_root_rel"
[[ ! -e "$execution_root" ]] || { echo "refusing existing capsule: $execution_root" >&2; exit 2; }

copy_files=(
  "$protocol" "$evaluator" "$pb_job_file" "$ost_job_file" "$report_job_file"
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$execution_root" \
  --link-root .venv --link-root .venvs --link-root data --link-root outputs \
  "${capsule_args[@]}" >/dev/null

mkdir -p "$output_root" "$repo_root/outputs/slurm"
git_commit=$(git rev-parse HEAD)
read -r git_diff_sha256 _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "run_id=$run_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "source_run=$source_run" \
  'denominator=802' \
  'source_predictions=786' \
  'missing_predictions_count_as_failures=16' \
  'rank=confidence_after_refinement_stable_top25' \
  'metrics=PoseBusters0.6.5+OpenStructure2.11.1_BiSyRMSD_LDDT-PLI' \
  "metadata_sha256=$metadata_sha256" \
  "protocol_sha256=$protocol_sha256" \
  "evaluator_sha256=$evaluator_sha256" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha256" \
  > "$output_root/.submission.pending"

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

base_export="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_SOURCE_RUN=$source_run,EFFDOCK_OUTPUT_ROOT=$output_root"
raw=$(sbatch --parsable --hold --array=0-0 \
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
  --export="$base_export,EFFDOCK_NUM_SHARDS=64" \
  "$execution_root/$report_job_file")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

printf '%s\n' \
  "smoke_posebusters_job=$smoke_pb_job" \
  "smoke_openstructure_job=$smoke_ost_job" \
  "full_posebusters_job=$full_pb_job" \
  "full_openstructure_job=$full_ost_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
scontrol release "$smoke_pb_job"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_posebusters_job=%s\nsmoke_openstructure_job=%s\nfull_posebusters_job=%s\nfull_openstructure_job=%s\nreport_job=%s\n' \
  "$output_root" "$smoke_pb_job" "$smoke_ost_job" "$full_pb_job" "$full_ost_job" "$report_job"
