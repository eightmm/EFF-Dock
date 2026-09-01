#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
[[ $# -le 1 ]] || { echo "usage: $0 [SAFE_RUN_ID]" >&2; exit 2; }
run_id=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "unsafe run ID" >&2; exit 2; }

protocol_id=EFFDOCK-EXTERNAL-TEMPORAL-GUIDED-REFINED-V1
output_root="$repo_root/outputs/benchmarks/external_temporal_guided_refined_runs/$run_id"
[[ ! -e "$output_root" ]] || { echo "refusing existing output root: $output_root" >&2; exit 2; }

docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
confidence=outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/full/best.pt
protocol=docs/EXTERNAL_TEMPORAL_GUIDED_REFINED_PROTOCOL.md
required=(
  .venv/bin/python .venv/bin/eff-dock
  "$docking" "$confidence" configs/train.yaml "$protocol"
  scripts/run_external_temporal_benchmark_shard.py
  scripts/evaluate_external_temporal_posebusters_shard.py
  scripts/report_external_temporal_benchmark.py
  scripts/run_guidance_sdf_post_refinement.py
  scripts/score_guidance_sdf_post_refinement_confidence.py
  scripts/slurm/external_temporal_guided_refined.sbatch
  scripts/slurm/external_temporal_posebusters.sbatch
  scripts/slurm/external_temporal_report.sbatch
)
for path in "${required[@]}"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done

verify_sha() {
  local path=$1 expected=$2 actual
  read -r actual _ < <(sha256sum "$path")
  [[ "$actual" == "$expected" ]] || { echo "SHA mismatch: $path" >&2; exit 2; }
}
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence" 1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha data/external_benchmarks/manifests/phibench.json 2697ecc14a83646a26aac319193f7ad98c202349836fda3bcac4e533f1a10633
verify_sha data/external_benchmarks/manifests/foldbench.json 7f6a77670d28103afc5eb08509a946b35d2b29cf5b17223e7832ff83fd5cb845
verify_sha data/external_benchmarks/manifests/openbind.json f5f8424698fc30970676c52d4e9d4f1b725e8127e540697d25a2d2822982b81d

protocol_sha256=$(sha256sum "$protocol" | cut -d' ' -f1)
runner_sha256=$(sha256sum scripts/run_external_temporal_benchmark_shard.py | cut -d' ' -f1)
pb_sha256=$(sha256sum scripts/evaluate_external_temporal_posebusters_shard.py | cut -d' ' -f1)
report_sha256=$(sha256sum scripts/report_external_temporal_benchmark.py | cut -d' ' -f1)
source_sha256=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
content_id=$(printf '%s\0' \
  "$protocol_id" "$protocol_sha256" "$runner_sha256" "$pb_sha256" "$report_sha256" \
  "$source_sha256" \
  65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 \
  1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030 \
  2697ecc14a83646a26aac319193f7ad98c202349836fda3bcac4e533f1a10633 \
  7f6a77670d28103afc5eb08509a946b35d2b29cf5b17223e7832ff83fd5cb845 \
  f5f8424698fc30970676c52d4e9d4f1b725e8127e540697d25a2d2822982b81d \
  | sha256sum | cut -d' ' -f1)
execution_root_rel=".effdock_execution_capsules/$protocol_id/${run_id}-${content_id:0:16}"
execution_root="$repo_root/$execution_root_rel"
[[ ! -e "$execution_root" ]] || { echo "refusing existing capsule: $execution_root" >&2; exit 2; }

.venv/bin/python -m compileall -q src/effdock \
  scripts/run_external_temporal_benchmark_shard.py \
  scripts/evaluate_external_temporal_posebusters_shard.py \
  scripts/report_external_temporal_benchmark.py \
  scripts/run_guidance_sdf_post_refinement.py \
  scripts/score_guidance_sdf_post_refinement_confidence.py
.venv/bin/ruff check src/effdock \
  scripts/run_external_temporal_benchmark_shard.py \
  scripts/evaluate_external_temporal_posebusters_shard.py \
  scripts/report_external_temporal_benchmark.py \
  scripts/run_guidance_sdf_post_refinement.py \
  scripts/score_guidance_sdf_post_refinement_confidence.py
.venv/bin/python -c 'import effdock; from effdock.cli import main'
.venv/bin/python -m pytest -q \
  tests/test_external_benchmark_data.py \
  tests/test_external_temporal_benchmark_pipeline.py \
  tests/test_guidance_sdf_post_refinement.py

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}"
  pyproject.toml uv.lock configs/train.yaml
  "$protocol"
  scripts/create_execution_capsule.py
  scripts/run_external_temporal_benchmark_shard.py
  scripts/evaluate_external_temporal_posebusters_shard.py
  scripts/report_external_temporal_benchmark.py
  scripts/run_guidance_sdf_post_refinement.py
  scripts/score_guidance_sdf_post_refinement_confidence.py
  scripts/slurm/external_temporal_guided_refined.sbatch
  scripts/slurm/external_temporal_posebusters.sbatch
  scripts/slurm/external_temporal_report.sbatch
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$execution_root" \
  --link-root .venv --link-root data --link-root outputs \
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
  'datasets=phibench:203,foldbench:66,openbind:860' \
  'sampling=N100/S10/sigma2/eta2_normalized_drift' \
  'refinement=max100/adaptive_energy_plateau' \
  'selector=U25k_symmetry_confidence_argmin' \
  'docking_sha256=65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6' \
  'confidence_sha256=1c59034172fb925cc8a70777dcba236be349f1a1de1775d49cc17d492b17c030' \
  "protocol_sha256=$protocol_sha256" \
  "source_tree_sha256=$source_sha256" \
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

export_spec="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_PROTOCOL_SHA256=$protocol_sha256"
raw=$(sbatch --parsable --hold --array=0-2%3 \
  --export="$export_spec,EFFDOCK_STAGE=smoke" \
  "$execution_root/scripts/slurm/external_temporal_guided_refined.sbatch")
smoke_job=${raw%%;*}; [[ "$smoke_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-71%8 \
  --export="$export_spec,EFFDOCK_STAGE=full" \
  "$execution_root/scripts/slurm/external_temporal_guided_refined.sbatch")
full_job=${raw%%;*}; [[ "$full_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_job")
raw=$(sbatch --parsable --dependency="afterok:$full_job" --array=0-71%16 \
  --export="$export_spec" \
  "$execution_root/scripts/slurm/external_temporal_posebusters.sbatch")
posebusters_job=${raw%%;*}; [[ "$posebusters_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$posebusters_job")
raw=$(sbatch --parsable --dependency="afterok:$posebusters_job" \
  --export="$export_spec" \
  "$execution_root/scripts/slurm/external_temporal_report.sbatch")
report_job=${raw%%;*}; [[ "$report_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$report_job")

printf '%s\n' \
  "smoke_job=$smoke_job" \
  "full_job=$full_job" \
  "posebusters_job=$posebusters_job" \
  "report_job=$report_job" \
  "submitted_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "$output_root/.submission.pending"
sed -i 's/^status=submitting$/status=submitted/' "$output_root/.submission.pending"
mv "$output_root/.submission.pending" "$output_root/.submission"
scontrol release "$smoke_job"
committed=1
trap - EXIT
printf 'output_root=%s\nsmoke_job=%s\nfull_job=%s\nposebusters_job=%s\nreport_job=%s\n' \
  "$output_root" "$smoke_job" "$full_job" "$posebusters_job" "$report_job"
