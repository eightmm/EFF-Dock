#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
[[ $# -le 1 ]] || { echo "usage: $0 [SAFE_RUN_ID]" >&2; exit 2; }
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }

protocol_id=EFFDOCK-EXTERNAL-TEMPORAL-U50-REPORT-V1
source_root="$repo_root/outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z"
output_root="$repo_root/outputs/benchmarks/external_temporal_u50_runs/$run_id"
protocol=docs/EXTERNAL_TEMPORAL_U50_REPORT_PROTOCOL.md
runner=scripts/rescore_external_temporal_u50_shard.py
scorer=scripts/score_guidance_sdf_post_refinement_confidence.py
evaluator=scripts/evaluate_external_temporal_posebusters_shard.py
reporter=scripts/report_external_temporal_benchmark.py
rescore_job_file=scripts/slurm/external_temporal_u50_rescore.sbatch
pb_job_file=scripts/slurm/external_temporal_u50_posebusters.sbatch
report_job_file=scripts/slurm/external_temporal_u50_report.sbatch

required=(
  .venv/bin/python "$source_root/report/summary.json" "$protocol" "$runner" "$scorer"
  "$evaluator" "$reporter" "$rescore_job_file" "$pb_job_file" "$report_job_file"
  configs/train.yaml
  outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
  outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ ! -e "$output_root" ]] || { echo "refusing existing output root: $output_root" >&2; exit 2; }

[[ $(find "$source_root/full/refinement" -mindepth 3 -maxdepth 3 -type f -name summary.json | wc -l) -eq 1129 ]] \
  || { echo "source refinement inventory mismatch" >&2; exit 2; }
[[ $(sha256sum configs/train.yaml | cut -d' ' -f1) == 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec ]]
[[ $(sha256sum outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt | cut -d' ' -f1) == 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 ]]
[[ $(sha256sum outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/latest.pt | cut -d' ' -f1) == fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469 ]]

.venv/bin/python -m py_compile "$runner" "$scorer" "$evaluator" "$reporter"
.venv/bin/python -m pytest -q tests/test_external_temporal_benchmark_pipeline.py tests/test_guidance_sdf_post_refinement.py

protocol_sha256=$(sha256sum "$protocol" | cut -d' ' -f1)
runner_sha256=$(sha256sum "$runner" | cut -d' ' -f1)
scorer_sha256=$(sha256sum "$scorer" | cut -d' ' -f1)
evaluator_sha256=$(sha256sum "$evaluator" | cut -d' ' -f1)
reporter_sha256=$(sha256sum "$reporter" | cut -d' ' -f1)
src_sha256=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
content_id=$(printf '%s\0' "$protocol_id" "$protocol_sha256" "$runner_sha256" \
  "$scorer_sha256" "$evaluator_sha256" "$reporter_sha256" "$src_sha256" \
  fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469 \
  | sha256sum | cut -d' ' -f1)
execution_root="$repo_root/.effdock_execution_capsules/$protocol_id/$run_id-${content_id:0:16}"
[[ ! -e "$execution_root" ]] || { echo "refusing existing execution capsule" >&2; exit 2; }

temporary="$execution_root.tmp.$$"
mkdir -p "$temporary" "$output_root/full" "$repo_root/outputs/slurm"
trap 'rm -rf "$temporary"' EXIT
cp -a src scripts configs docs "$temporary/"
ln -s "$repo_root/.venv" "$temporary/.venv"
ln -s "$repo_root/data" "$temporary/data"
ln -s "$repo_root/outputs" "$temporary/outputs"
mv "$temporary" "$execution_root"
trap - EXIT
ln -s "$source_root/full/refinement" "$output_root/full/refinement"

base_export="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_SOURCE_ROOT=$source_root,EFFDOCK_RUNNER_SHA256=$runner_sha256,EFFDOCK_SCORER_SHA256=$scorer_sha256,EFFDOCK_EVALUATOR_SHA256=$evaluator_sha256,EFFDOCK_REPORTER_SHA256=$reporter_sha256,EFFDOCK_SRC_SHA256=$src_sha256"
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

raw=$(sbatch --parsable --hold --array=0-2%3 \
  --export="$base_export,EFFDOCK_STAGE=smoke" "$execution_root/$rescore_job_file")
smoke_job=${raw%%;*}; [[ "$smoke_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-71%16 \
  --export="$base_export,EFFDOCK_STAGE=full" "$execution_root/$rescore_job_file")
full_job=${raw%%;*}; [[ "$full_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_job")
raw=$(sbatch --parsable --dependency="afterok:$full_job" --array=0-71%16 \
  --export="$base_export" "$execution_root/$pb_job_file")
pb_job=${raw%%;*}; [[ "$pb_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$pb_job")
raw=$(sbatch --parsable --dependency="afterok:$pb_job" \
  --export="$base_export" "$execution_root/$report_job_file")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

printf '%s\n' \
  'status=submitted' \
  "protocol_id=$protocol_id" \
  "run_id=$run_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "source_root=$source_root" \
  'selector=U50k_symmetry_confidence_argmin' \
  'confidence_sha256=fd49fa86f67187bf26d6c1bcf2daf925ba3e3b19dfeae733e57535d183280469' \
  'cohorts=phibench203,foldbench66,openbind860' \
  'sampling_and_refinement=reused_immutable' \
  "protocol_sha256=$protocol_sha256" \
  "runner_sha256=$runner_sha256" \
  "scorer_sha256=$scorer_sha256" \
  "evaluator_sha256=$evaluator_sha256" \
  "reporter_sha256=$reporter_sha256" \
  "src_sha256=$src_sha256" \
  "smoke_job=$smoke_job" \
  "full_job=$full_job" \
  "posebusters_job=$pb_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$output_root/.submission"
scontrol release "$smoke_job"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_job=%s\nfull_job=%s\nposebusters_job=%s\nreport_job=%s\n' \
  "$output_root" "$smoke_job" "$full_job" "$pb_job" "$report_job"
