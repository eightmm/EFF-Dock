#!/usr/bin/env bash
#SBATCH --job-name=effdock-eta-v2-parent-gate
#SBATCH --output=outputs/benchmarks/logs/%x-%j.out
#SBATCH --error=outputs/benchmarks/logs/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"

output_root=${EFFDOCK_OUTPUT_ROOT:?EFFDOCK_OUTPUT_ROOT is required}
expected_output_root=outputs/benchmarks/guidance_eta_sweep_v2_runs/20260801T102903Z
if [[ "$output_root" != "$expected_output_root" ]]; then
  echo "parent artifact gate is frozen to $expected_output_root" >&2
  exit 2
fi

source_snapshot="$output_root/recovery/parent_source_d726"
source_manifest="$output_root/recovery/parent_source_d726.manifest.sha256"
cohort_audit="$output_root/audit/combined.json"
aggregate="$output_root/aggregate.json"
posebusters_aggregate="$output_root/posebusters_aggregate.json"
recovery_record="$output_root/recovery/parent-report-47280.json"
source_snapshot=$(readlink -f "$source_snapshot")
source_manifest=$(readlink -f "$source_manifest")

verify_sha256() {
  local path=$1
  local expected=$2
  local observed
  if [[ ! -f "$path" ]]; then
    echo "missing frozen parent artifact: $path" >&2
    exit 2
  fi
  read -r observed _ < <(sha256sum "$path")
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected=$expected observed=$observed" >&2
    exit 2
  fi
}

verify_sha256 "$cohort_audit" \
  dac7903488ccd36552a9bca134e37e633e3f07166d94f0389837012081ff3048
verify_sha256 "$aggregate" \
  167e9f590bc0cc360a5ddbff90d255bb90a06fd85a24a7508e66953a68c9c682
verify_sha256 "$posebusters_aggregate" \
  cad58b05eda850872dd3901c8aa4151e207b9cf173dc26a9bb74331c817c5efa
verify_sha256 "$recovery_record" \
  cd3546a381bf473465bad405a7bb12cec5a9250c86338703363e9aa66c20a302
verify_sha256 "$source_manifest" \
  05941e1999a286dd3c2484ec909c0935d14959b074eb697d602d09218f2d32e1
verify_sha256 scripts/slurm/guidance_eta_sweep_report_recovery.sbatch \
  2deabb3bb213a8a97aec7d725fb0b3e6a909b11f5ac9430238c6b7bf7769a533

if ! (cd "$source_snapshot" && sha256sum --check --quiet "$source_manifest"); then
  echo "historical source snapshot verification failed" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
snapshot_pythonpath="$source_snapshot/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHONPATH="$snapshot_pythonpath" .venv/bin/python -c \
  'import json, sys; from pathlib import Path; import effdock; from effdock.guidance.provenance import guidance_implementation_identity; root=Path(sys.argv[1]).resolve(); imported=Path(effdock.__file__).resolve(); identity=guidance_implementation_identity(); assert imported.is_relative_to(root / "src"); assert identity["sha256"] == sys.argv[2]; recovery=json.loads(Path(sys.argv[3]).read_text()); aggregate=json.loads(Path(sys.argv[4]).read_text()); pb=json.loads(Path(sys.argv[5]).read_text()); assert recovery["status"] == "passed" and recovery["slurm_job_id"] == "47280"; assert recovery["guidance_implementation_sha256"] == sys.argv[2]; assert aggregate["status"] == "complete_strict_full_cohort_paired_descriptive_eta_sweep"; assert aggregate["sampling_inventory"] == {"run_cells":16,"shards_per_cell":8,"total_shard_tasks":128}; assert aggregate["datasets"]["astex"]["coverage"]["count"] == 85; assert aggregate["datasets"]["posebusters"]["coverage"]["count"] == 308; assert pb["status"] == "complete_strict_full_cohort_paired_official_posebusters_eta_sweep"; assert pb["official_inventory"] == {"run_cells":16,"shards_per_cell":8,"total_shard_tasks":128}; assert pb["datasets"]["astex"]["coverage"]["count"] == 85; assert pb["datasets"]["posebusters"]["coverage"]["count"] == 308; print(json.dumps({"status":"passed","historical_implementation_sha256":identity["sha256"],"parent_recovery_job":"47280","sampling_shards":128,"official_posebusters_shards":128},sort_keys=True))' \
  "$source_snapshot" \
  d726ddc4cb89b495f0495aa059faf9efdf33ee76c42c4b14e71356068935c0a5 \
  "$recovery_record" "$aggregate" "$posebusters_aggregate"
