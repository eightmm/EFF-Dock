#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "${1:-.}" && pwd -P)
cd "$repo_root"

protocol_id=EFFDOCK-FOLDBENCH-POCKET-558-V1
run_id=6ff75eb480e93c2ad3a67382040fb49793002c797e74a070c927cf180975a279
output_root="$repo_root/outputs/benchmarks/foldbench_pocket_558_runs/$run_id"
protocol=docs/FOLDBENCH_FULL_POCKET_PROTOCOL.md
runner=scripts/run_external_temporal_benchmark_shard.py
refiner=scripts/run_guidance_sdf_post_refinement.py
scorer=scripts/score_guidance_sdf_post_refinement_confidence.py
evaluator=scripts/evaluate_external_temporal_posebusters_shard.py
reporter=scripts/report_foldbench_full.py
report_helper=scripts/report_external_temporal_benchmark.py
auditor=scripts/audit_foldbench_full_recovery.py
archiver=scripts/archive_foldbench_failed_sampling.py
submitter=scripts/slurm/submit_foldbench_full_recovery.sh
generation_job_file=scripts/slurm/foldbench_full_pocket.sbatch
audit_job_file=scripts/slurm/foldbench_full_recovery_audit.sbatch
posebusters_job_file=scripts/slurm/foldbench_full_posebusters.sbatch
report_job_file=scripts/slurm/foldbench_full_report.sbatch
manifest=data/external_benchmarks/foldbench_full/manifests/foldbench.json
mapping=data/external_benchmarks/foldbench_full/external/foldbench_smiles.json
centers=data/external_benchmarks/foldbench_full/external/foldbench_reference_pocket_centers.json
docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
confidence=outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/best.pt

required=(
  "$output_root/.submission" .venv/bin/python .venv/bin/eff-dock
  pyproject.toml uv.lock configs/train.yaml "$protocol" "$runner" "$refiner"
  "$scorer" "$evaluator" "$reporter" "$report_helper" "$auditor"
  "$archiver"
  "$submitter"
  "$generation_job_file" "$audit_job_file" "$posebusters_job_file"
  "$report_job_file" "$manifest" "$mapping" "$centers" "$docking" "$confidence"
  scripts/create_execution_capsule.py
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ ! -e "$output_root/full/shards/foldbench.shard-021-of-044.json" ]]
[[ ! -e "$output_root/full/shards/foldbench.shard-022-of-044.json" ]]

verify_sha() {
  local path=$1 expected=$2 actual
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || { echo "SHA mismatch: $path" >&2; exit 2; }
}
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence" ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
verify_sha "$manifest" 3546019734a708265f465c8c74e0fa019accee88242a2c7dcd446706d8fc4957
verify_sha "$mapping" b5f1dd2cd10b163d682fec8ee71a860710cdeb52679839df32b527fed3294d0e
verify_sha "$centers" e9b76ee295195c97024372700fa622312b2698bea2fb6de938af20b5d7774d0c

.venv/bin/python -m py_compile \
  "$runner" "$refiner" "$scorer" "$evaluator" "$reporter" "$auditor" "$archiver"
.venv/bin/ruff check \
  src/effdock/inference/preprocess.py "$runner" "$evaluator" "$reporter" \
  "$auditor" "$archiver"
.venv/bin/python -m pytest -q \
  tests/test_fragment_geometry_audit.py \
  tests/test_external_temporal_benchmark_pipeline.py \
  tests/test_guidance_sdf_post_refinement.py

protocol_sha=$(sha256sum "$protocol" | cut -d' ' -f1)
runner_sha=$(sha256sum "$runner" | cut -d' ' -f1)
refiner_sha=$(sha256sum "$refiner" | cut -d' ' -f1)
scorer_sha=$(sha256sum "$scorer" | cut -d' ' -f1)
evaluator_sha=$(sha256sum "$evaluator" | cut -d' ' -f1)
reporter_sha=$(sha256sum "$reporter" | cut -d' ' -f1)
report_helper_sha=$(sha256sum "$report_helper" | cut -d' ' -f1)
auditor_sha=$(sha256sum "$auditor" | cut -d' ' -f1)
archiver_sha=$(sha256sum "$archiver" | cut -d' ' -f1)
submitter_sha=$(sha256sum "$submitter" | cut -d' ' -f1)
generation_job_sha=$(sha256sum "$generation_job_file" | cut -d' ' -f1)
audit_job_sha=$(sha256sum "$audit_job_file" | cut -d' ' -f1)
posebusters_job_sha=$(sha256sum "$posebusters_job_file" | cut -d' ' -f1)
report_job_sha=$(sha256sum "$report_job_file" | cut -d' ' -f1)
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
content_id=$(printf '%s\0' \
  "$protocol_id" recovery-21-22 "$protocol_sha" "$runner_sha" "$refiner_sha" \
  "$scorer_sha" "$evaluator_sha" "$reporter_sha" "$report_helper_sha" \
  "$auditor_sha" "$archiver_sha" "$submitter_sha" "$generation_job_sha" "$audit_job_sha" "$posebusters_job_sha" \
  "$report_job_sha" "$src_sha" | sha256sum | cut -d' ' -f1)
execution_root="$repo_root/.effdock_execution_capsules/$protocol_id/${run_id}-recovery-${content_id:0:16}"
submission_record="$output_root/.recovery-submission-$content_id"
prior_archive_dir="$output_root/recovery/attempt-1-partial-sampling"
archive_dir="$output_root/recovery/attempt-2-complete-partial-sampling"
[[ ! -e "$execution_root" && ! -e "$submission_record" && ! -e "$archive_dir" ]] || {
  echo "refusing existing recovery state" >&2
  exit 2
}

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}" pyproject.toml uv.lock configs/train.yaml "$protocol"
  scripts/create_execution_capsule.py "$runner" "$refiner" "$scorer" "$evaluator"
  "$reporter" "$report_helper" "$auditor" "$archiver" "$submitter" \
  "$generation_job_file" "$audit_job_file"
  "$posebusters_job_file" "$report_job_file"
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$execution_root" \
  --link-root .venv --link-root data --link-root outputs \
  "${capsule_args[@]}" >/dev/null

mkdir -p "$output_root/ledgers" "$repo_root/outputs/slurm"
export_spec="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_PROTOCOL_SHA=$protocol_sha,EFFDOCK_RUNNER_SHA=$runner_sha,EFFDOCK_REFINER_SHA=$refiner_sha,EFFDOCK_SCORER_SHA=$scorer_sha,EFFDOCK_EVALUATOR_SHA=$evaluator_sha,EFFDOCK_REPORTER_SHA=$reporter_sha,EFFDOCK_AUDITOR_SHA=$auditor_sha,EFFDOCK_SRC_SHA=$src_sha,EFFDOCK_MANIFEST_SHA=3546019734a708265f465c8c74e0fa019accee88242a2c7dcd446706d8fc4957,EFFDOCK_MAPPING_SHA=b5f1dd2cd10b163d682fec8ee71a860710cdeb52679839df32b527fed3294d0e,EFFDOCK_CENTERS_SHA=e9b76ee295195c97024372700fa622312b2698bea2fb6de938af20b5d7774d0c,EFFDOCK_RECOVERY_ARCHIVE=$archive_dir"
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

raw=$(sbatch --parsable --hold --array=0-1%2 \
  --export="$export_spec,EFFDOCK_STAGE=full,EFFDOCK_RECOVERY_SHARDS=21:22" \
  "$execution_root/$generation_job_file")
recovery_job=${raw%%;*}; [[ "$recovery_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$recovery_job")
raw=$(sbatch --parsable --dependency="afterok:$recovery_job" \
  --export="$export_spec" "$execution_root/$audit_job_file")
audit_job=${raw%%;*}; [[ "$audit_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$audit_job")
raw=$(sbatch --parsable --dependency="afterok:$audit_job" --array=0-43%12 \
  --export="$export_spec" "$execution_root/$posebusters_job_file")
posebusters_job=${raw%%;*}; [[ "$posebusters_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$posebusters_job")
raw=$(sbatch --parsable --dependency="afterok:$posebusters_job" \
  --export="$export_spec" "$execution_root/$report_job_file")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

sampling_root="$output_root/full/sampling"
prefix=effdock-foldbench-pocket-558-v1-foldbench-n100-s10-sigma2-unguided
for index in 021 022; do
  for suffix in csv summary.json; do
    source="$prior_archive_dir/$prefix.shard-$index-of-044.$suffix"
    destination="$sampling_root/$prefix.shard-$index-of-044.$suffix"
    [[ -f "$source" ]] || {
      echo "invalid attempt-1 recovery source: $source" >&2
      exit 2
    }
    if [[ -e "$destination" ]]; then
      cmp -s "$source" "$destination" || {
        echo "restored sampling artifact differs from attempt 1: $destination" >&2
        exit 2
      }
    else
      cp --reflink=auto --preserve=mode,timestamps "$source" "$destination"
    fi
  done
done
.venv/bin/python "$archiver" \
  --output-root "$output_root" --archive-dir "$archive_dir" --shards 21 22
scontrol release "$recovery_job"
scancel 63523 63524

printf '%s\n' \
  'status=submitted' \
  "protocol_id=$protocol_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  'recovered_shards=21,22' \
  "archive_dir=$archive_dir" \
  "recovery_job=$recovery_job" \
  "audit_job=$audit_job" \
  "posebusters_job=$posebusters_job" \
  "report_job=$report_job" \
  'superseded_jobs=63523,63524' \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$submission_record"
committed=1
trap - EXIT
printf 'recovery_job=%s\naudit_job=%s\nposebusters_job=%s\nreport_job=%s\n' \
  "$recovery_job" "$audit_job" "$posebusters_job" "$report_job"
