#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "${1:-.}" && pwd -P)
cd "$repo_root"

protocol_id=EFFDOCK-S50-RAW-REFINED-CONFIDENCE-TEMPORAL-EXTERNAL-V1
source_root="$repo_root/outputs/benchmarks/external_temporal_guided_refined_runs/20260825T000806Z"
protocol=docs/S50_RAW_REFINED_CONFIDENCE_TEMPORAL_EXTERNAL_PROTOCOL.md
runner=scripts/run_s50_raw_refined_confidence_temporal_external_shard.py
u50_helper=scripts/rescore_external_temporal_u50_shard.py
scorer=scripts/score_guidance_sdf_post_refinement_confidence.py
evaluator=scripts/evaluate_s50_raw_refined_confidence_temporal_posebusters_shard.py
evaluator_helper=scripts/evaluate_external_temporal_posebusters_shard.py
reporter=scripts/report_s50_raw_refined_confidence_temporal_external.py
rescore_job_file=scripts/slurm/s50_raw_refined_confidence_temporal_external.sbatch
pb_job_file=scripts/slurm/s50_raw_refined_confidence_temporal_posebusters.sbatch
report_job_file=scripts/slurm/s50_raw_refined_confidence_temporal_report.sbatch
checkpoint_root=outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full
docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt

required=(
  .venv/bin/python "$source_root/report/summary.json" "$protocol" "$runner"
  "$u50_helper" "$scorer" "$evaluator" "$evaluator_helper" "$reporter"
  "$rescore_job_file" "$pb_job_file" "$report_job_file" configs/train.yaml
  "$checkpoint_root/best.pt" "$checkpoint_root/latest.pt" "$docking"
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
verify_sha "$checkpoint_root/best.pt" ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638
verify_sha "$checkpoint_root/latest.pt" 2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec

refinement_count=$(find "$source_root/full/refinement" -mindepth 3 -maxdepth 3 -type f -name summary.json | wc -l)
source_shard_count=$(find "$source_root/full/shards" -maxdepth 1 -type f -name '*.json' | wc -l)
[[ "$refinement_count" == 1129 ]] || { echo "expected 1129 refinement summaries" >&2; exit 2; }
[[ "$source_shard_count" == 72 ]] || { echo "expected 72 source shards" >&2; exit 2; }

.venv/bin/python -m py_compile "$runner" "$evaluator" "$reporter"
.venv/bin/python -m pytest -q \
  tests/test_external_temporal_benchmark_pipeline.py \
  tests/test_s50_raw_refined_confidence_temporal_external.py \
  tests/test_guidance_sdf_post_refinement.py

protocol_sha=$(sha256sum "$protocol" | cut -d' ' -f1)
runner_sha=$(sha256sum "$runner" | cut -d' ' -f1)
u50_helper_sha=$(sha256sum "$u50_helper" | cut -d' ' -f1)
scorer_sha=$(sha256sum "$scorer" | cut -d' ' -f1)
evaluator_sha=$(sha256sum "$evaluator" | cut -d' ' -f1)
evaluator_helper_sha=$(sha256sum "$evaluator_helper" | cut -d' ' -f1)
reporter_sha=$(sha256sum "$reporter" | cut -d' ' -f1)
rescore_job_sha=$(sha256sum "$rescore_job_file" | cut -d' ' -f1)
pb_job_sha=$(sha256sum "$pb_job_file" | cut -d' ' -f1)
report_job_sha=$(sha256sum "$report_job_file" | cut -d' ' -f1)
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
source_summary_sha=$(sha256sum "$source_root/report/summary.json" | cut -d' ' -f1)
content_id=$(printf '%s\0' "$protocol_id" "$protocol_sha" "$runner_sha" \
  "$u50_helper_sha" "$scorer_sha" "$evaluator_sha" "$evaluator_helper_sha" \
  "$reporter_sha" "$rescore_job_sha" "$pb_job_sha" "$report_job_sha" "$src_sha" \
  "$source_summary_sha" \
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638 \
  2ea1aca4f1c326cd0841e76c3597e3749231854a523d1ba8bd923c6fb5a9bff8 \
  65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 \
  | sha256sum | cut -d' ' -f1)
output_root="$repo_root/outputs/benchmarks/s50_raw_refined_confidence_temporal_external_runs/$content_id"
execution_root="$repo_root/.effdock_execution_capsules/$protocol_id/$content_id"
[[ ! -e "$output_root" && ! -e "$execution_root" ]] || {
  echo "refusing to reuse output or execution capsule" >&2
  exit 2
}

temporary="$execution_root.tmp.$$"
mkdir -p "$temporary" "$output_root" "$repo_root/outputs/slurm"
trap 'rm -rf "$temporary"' EXIT
cp -a src scripts configs docs "$temporary/"
ln -s "$repo_root/.venv" "$temporary/.venv"
ln -s "$repo_root/data" "$temporary/data"
ln -s "$repo_root/outputs" "$temporary/outputs"
mv "$temporary" "$execution_root"
trap - EXIT

base_export="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_SOURCE_ROOT=$source_root,EFFDOCK_PROTOCOL_SHA=$protocol_sha,EFFDOCK_RUNNER_SHA=$runner_sha,EFFDOCK_U50_HELPER_SHA=$u50_helper_sha,EFFDOCK_SCORER_SHA=$scorer_sha,EFFDOCK_EVALUATOR_SHA=$evaluator_sha,EFFDOCK_EVALUATOR_HELPER_SHA=$evaluator_helper_sha,EFFDOCK_REPORTER_SHA=$reporter_sha,EFFDOCK_SRC_SHA=$src_sha"
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

raw=$(sbatch --parsable --hold --array=0-5%6 \
  --export="$base_export,EFFDOCK_STAGE=smoke" "$execution_root/$rescore_job_file")
smoke_job=${raw%%;*}; [[ "$smoke_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-143%16 \
  --export="$base_export,EFFDOCK_STAGE=full" "$execution_root/$rescore_job_file")
full_job=${raw%%;*}; [[ "$full_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_job")
raw=$(sbatch --parsable --dependency="afterok:$full_job" --array=0-143%24 \
  --export="$base_export" "$execution_root/$pb_job_file")
pb_job=${raw%%;*}; [[ "$pb_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$pb_job")
raw=$(sbatch --parsable --dependency="afterok:$pb_job" \
  --export="$base_export" "$execution_root/$report_job_file")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

printf '%s\n' \
  'status=submitted' \
  "protocol_id=$protocol_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  "source_root=$source_root" \
  'arms=u070000,u100000' \
  'cohorts=phibench203,foldbench66,openbind860' \
  'sampling=N100,S10,sigma2,reused_immutable' \
  'refinement=reused_immutable' \
  "protocol_sha256=$protocol_sha" \
  "runner_sha256=$runner_sha" \
  "evaluator_sha256=$evaluator_sha" \
  "reporter_sha256=$reporter_sha" \
  "src_sha256=$src_sha" \
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
