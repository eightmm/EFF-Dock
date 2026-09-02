#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "${1:-.}" && pwd -P)
cd "$repo_root"

protocol_id=EFFDOCK-FOLDBENCH-POCKET-558-V1
protocol=docs/FOLDBENCH_FULL_POCKET_PROTOCOL.md
runner=scripts/run_external_temporal_benchmark_shard.py
refiner=scripts/run_guidance_sdf_post_refinement.py
scorer=scripts/score_guidance_sdf_post_refinement_confidence.py
evaluator=scripts/evaluate_external_temporal_posebusters_shard.py
reporter=scripts/report_foldbench_full.py
report_helper=scripts/report_external_temporal_benchmark.py
generation_job_file=scripts/slurm/foldbench_full_pocket.sbatch
posebusters_job_file=scripts/slurm/foldbench_full_posebusters.sbatch
report_job_file=scripts/slurm/foldbench_full_report.sbatch
manifest=data/external_benchmarks/foldbench_full/manifests/foldbench.json
mapping=data/external_benchmarks/foldbench_full/external/foldbench_smiles.json
centers=data/external_benchmarks/foldbench_full/external/foldbench_reference_pocket_centers.json
docking=outputs/eff-dock/early-time-t0p10-50k-v1-20260813/checkpoints/step50000_ema_common_init.pt
confidence=outputs/eff-dock/s50-raw-refined-confidence-runs/309bb1a7645f8d09a796e45559d018c969d077a937ecb225ea53314a7143c804/training-100k-runs/8641cbe7b5bc99896c6513073ca7d81d4c00db42015a3739685043e5e4fa162f/full/best.pt

required=(
  .venv/bin/python .venv/bin/eff-dock pyproject.toml uv.lock configs/train.yaml
  "$protocol" "$runner" "$refiner" "$scorer" "$evaluator" "$reporter"
  "$report_helper" "$generation_job_file" "$posebusters_job_file"
  "$report_job_file" "$manifest" "$mapping" "$centers" "$docking" "$confidence"
  scripts/create_execution_capsule.py
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
verify_sha configs/train.yaml 39aa62e4a48ed6f3aa4ff59345fb43a81220e2baba22edfd5beb0c4981b307ec
verify_sha "$docking" 65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6
verify_sha "$confidence" ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638

.venv/bin/python -c '
import json
from pathlib import Path
root = Path("data/external_benchmarks/foldbench_full")
manifest = json.loads((root / "manifests/foldbench.json").read_text())
mapping = json.loads((root / "external/foldbench_smiles.json").read_text())
centers = json.loads((root / "external/foldbench_reference_pocket_centers.json").read_text())
directories = list((root / "normalized/foldbench").iterdir())
assert manifest["cohort"] == "full" and manifest["count"] == 558
assert manifest["selected_row_count"] == 558 and manifest["failures"] == []
assert len(manifest["reference_heavy_atom_imputations"]) == 2
assert len(mapping) == len(centers) == len(directories) == 558
assert set(mapping) == set(centers) == {record["id"] for record in manifest["records"]}
'
.venv/bin/python -m py_compile "$runner" "$refiner" "$scorer" "$evaluator" "$reporter"
.venv/bin/ruff check \
  src/effdock/workflows/external_benchmark_data.py "$runner" "$evaluator" "$reporter"
.venv/bin/python -m pytest -q \
  tests/test_external_benchmark_data.py \
  tests/test_external_temporal_benchmark_pipeline.py \
  tests/test_guidance_sdf_post_refinement.py

protocol_sha=$(sha256sum "$protocol" | cut -d' ' -f1)
runner_sha=$(sha256sum "$runner" | cut -d' ' -f1)
refiner_sha=$(sha256sum "$refiner" | cut -d' ' -f1)
scorer_sha=$(sha256sum "$scorer" | cut -d' ' -f1)
evaluator_sha=$(sha256sum "$evaluator" | cut -d' ' -f1)
reporter_sha=$(sha256sum "$reporter" | cut -d' ' -f1)
report_helper_sha=$(sha256sum "$report_helper" | cut -d' ' -f1)
generation_job_sha=$(sha256sum "$generation_job_file" | cut -d' ' -f1)
posebusters_job_sha=$(sha256sum "$posebusters_job_file" | cut -d' ' -f1)
report_job_sha=$(sha256sum "$report_job_file" | cut -d' ' -f1)
manifest_sha=$(sha256sum "$manifest" | cut -d' ' -f1)
mapping_sha=$(sha256sum "$mapping" | cut -d' ' -f1)
centers_sha=$(sha256sum "$centers" | cut -d' ' -f1)
src_sha=$(find src/effdock -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
content_id=$(printf '%s\0' \
  "$protocol_id" "$protocol_sha" "$runner_sha" "$refiner_sha" "$scorer_sha" \
  "$evaluator_sha" "$reporter_sha" "$report_helper_sha" "$generation_job_sha" \
  "$posebusters_job_sha" "$report_job_sha" "$manifest_sha" "$mapping_sha" \
  "$centers_sha" "$src_sha" \
  65be44d7dc8f0867eb9fc5d22214b80f93971ea4702679a527c665046e91e6b6 \
  ce59be42f0ca613871ca079127c3296f5ca9a4ec72e44a9e5cf61878351c2638 \
  | sha256sum | cut -d' ' -f1)
output_root="$repo_root/outputs/benchmarks/foldbench_pocket_558_runs/$content_id"
execution_root="$repo_root/.effdock_execution_capsules/$protocol_id/$content_id"
[[ ! -e "$output_root" && ! -e "$execution_root" ]] || {
  echo "refusing to reuse output or execution capsule" >&2
  exit 2
}

mapfile -t package_files < <(rg --files src/effdock | sort)
copy_files=(
  "${package_files[@]}" pyproject.toml uv.lock configs/train.yaml "$protocol"
  scripts/create_execution_capsule.py "$runner" "$refiner" "$scorer" "$evaluator"
  "$reporter" "$report_helper" "$generation_job_file" "$posebusters_job_file"
  "$report_job_file"
)
capsule_args=()
for path in "${copy_files[@]}"; do capsule_args+=(--copy-file "$path"); done
.venv/bin/python scripts/create_execution_capsule.py \
  --repo-root "$repo_root" --output "$execution_root" \
  --link-root .venv --link-root data --link-root outputs \
  "${capsule_args[@]}" >/dev/null

mkdir -p "$output_root" "$repo_root/outputs/slurm"
git_commit=$(git rev-parse HEAD)
read -r git_diff_sha _ < <(git diff --no-ext-diff | sha256sum)
printf '%s\n' \
  'status=submitting' \
  "protocol_id=$protocol_id" \
  "content_id=$content_id" \
  "execution_root=$execution_root" \
  'dataset=foldbench-pocket-558' \
  'sampling=N100/S10/sigma2/late-power3/unguided' \
  'refinement=all100/max100/adaptive' \
  'selector=U70k_symmetry_confidence_argmin' \
  "manifest_sha256=$manifest_sha" \
  "mapping_sha256=$mapping_sha" \
  "centers_sha256=$centers_sha" \
  "git_commit=$git_commit" \
  "git_diff_sha256=$git_diff_sha" \
  > "$output_root/.submission.pending"

export_spec="ALL,EFFDOCK_EXECUTION_ROOT=$execution_root,EFFDOCK_OUTPUT_ROOT=$output_root,EFFDOCK_PROTOCOL_SHA=$protocol_sha,EFFDOCK_RUNNER_SHA=$runner_sha,EFFDOCK_REFINER_SHA=$refiner_sha,EFFDOCK_SCORER_SHA=$scorer_sha,EFFDOCK_EVALUATOR_SHA=$evaluator_sha,EFFDOCK_REPORTER_SHA=$reporter_sha,EFFDOCK_SRC_SHA=$src_sha,EFFDOCK_MANIFEST_SHA=$manifest_sha,EFFDOCK_MAPPING_SHA=$mapping_sha,EFFDOCK_CENTERS_SHA=$centers_sha"
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

raw=$(sbatch --parsable --hold --array=0-0%1 \
  --export="$export_spec,EFFDOCK_STAGE=smoke" "$execution_root/$generation_job_file")
smoke_job=${raw%%;*}; [[ "$smoke_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$smoke_job")
raw=$(sbatch --parsable --dependency="afterok:$smoke_job" --array=0-43%12 \
  --export="$export_spec,EFFDOCK_STAGE=full" "$execution_root/$generation_job_file")
full_job=${raw%%;*}; [[ "$full_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$full_job")
raw=$(sbatch --parsable --dependency="afterok:$full_job" --array=0-43%12 \
  --export="$export_spec" "$execution_root/$posebusters_job_file")
posebusters_job=${raw%%;*}; [[ "$posebusters_job" =~ ^[0-9]+$ ]] || exit 2; submitted+=("$posebusters_job")
raw=$(sbatch --parsable --dependency="afterok:$posebusters_job" \
  --export="$export_spec" "$execution_root/$report_job_file")
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
