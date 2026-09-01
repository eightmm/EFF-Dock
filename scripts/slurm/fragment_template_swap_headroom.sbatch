#!/usr/bin/env bash
#SBATCH --job-name=effdock-template-headroom
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm outputs/analysis/fragment_template_swap_headroom_v1

expected_identity_sha256=6f195c774e04ba3aa11ff220bd21fbbd3a3b7ce61b95f1a22364e23ded70060e
run_root=outputs/benchmarks/guidance_sigma_sweep_eta2_runs/20260809T031535Z
actual_identity_sha256=$(
  {
    sha256sum \
      scripts/analyze_fragment_template_swap_headroom.py \
      src/effdock/evaluation/fragment_geometry.py \
      src/effdock/evaluation/benchmark.py \
      src/effdock/inference/preprocess.py \
      src/effdock/inference/io.py \
      src/effdock/preprocess/fragments.py \
      src/effdock/workflows/benchmark_inputs.py \
      docs/FRAGMENT_TEMPLATE_SWAP_HEADROOM_PROTOCOL.md \
      docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
      "$run_root/audit/full.json"
    sha256sum "$run_root"/raw/*sigma2000*.csv
  } | sha256sum | awk '{print $1}'
)
if [[ "$actual_identity_sha256" != "$expected_identity_sha256" ]]; then
  echo "fragment-template headroom input/code identity mismatch" >&2
  exit 2
fi

case "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" in
  0)
    dataset=astex
    dataset_root=data/external_benchmarks/data/astex_diverse_set
    expected_count=85
    ;;
  1)
    dataset=posebusters
    dataset_root=data/external_benchmarks/data/posebusters_benchmark_set
    expected_count=308
    ;;
  *)
    echo "invalid array task" >&2
    exit 2
    ;;
esac

prefix=effdock-guidance-sigma-sweep-eta2-v1-${dataset}-n100-s10-sigma2000
all_poses_dir=$run_root/raw/poses/$prefix/$dataset/all_poses
output=outputs/analysis/fragment_template_swap_headroom_v1/${dataset}_sigma2_n100.json
[[ -d "$all_poses_dir" ]] || { echo "missing all-pose directory: $all_poses_dir" >&2; exit 2; }
[[ $(find "$all_poses_dir" -maxdepth 1 -type f -name '*.sdf' | wc -l) -eq "$expected_count" ]] || {
  echo "unexpected all-pose SDF count for $dataset" >&2
  exit 2
}
[[ ! -e "$output" ]] || { echo "refusing to overwrite output: $output" >&2; exit 2; }

summary_args=()
for shard in 000 001 002 003 004 005 006 007; do
  summary_args+=(--summary-csv "$run_root/raw/$prefix.shard-$shard-of-008.csv")
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/analyze_fragment_template_swap_headroom.py \
  --dataset "$dataset" \
  --dataset-root "$dataset_root" \
  --all-poses-dir "$all_poses_dir" \
  --benchmark-input-manifest docs/GUIDANCE_BUDGET1000_FULL_INPUTS.json \
  --expected-poses 100 \
  --expected-sigma 2.0 \
  --max-matches 4096 \
  --require-complete-success \
  --protocol-id EFFDOCK-FRAGMENT-TEMPLATE-SWAP-HEADROOM-V1 \
  --protocol-path docs/FRAGMENT_TEMPLATE_SWAP_HEADROOM_PROTOCOL.md \
  --output "$output" \
  "${summary_args[@]}"
