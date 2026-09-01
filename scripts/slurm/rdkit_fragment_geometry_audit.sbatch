#!/usr/bin/env bash
#SBATCH --job-name=effdock-rdkit-geom
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

repo_dir=${EFFDOCK_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
  repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
fi
cd "$repo_dir"
mkdir -p outputs/slurm outputs/analysis/rdkit_fragment_geometry_v1

expected_identity_sha256=2b62de9acd6c8f037f4c70f4693d404029bb6ba8706050959cb2db5b408d1e73
actual_identity_sha256=$(
  sha256sum \
    scripts/analyze_rdkit_fragment_geometry.py \
    src/effdock/evaluation/fragment_geometry.py \
    src/effdock/inference/preprocess.py \
    src/effdock/preprocess/fragments.py \
    src/effdock/preprocess/ligand.py \
    src/effdock/evaluation/benchmark.py \
    src/effdock/workflows/benchmark_inputs.py \
    docs/RDKIT_FRAGMENT_GEOMETRY_AUDIT_PROTOCOL.md \
    data/splits/plinder.json \
    data/plinder_pool.parquet \
  | sha256sum \
  | awk '{print $1}'
)
if [[ "$actual_identity_sha256" != "$expected_identity_sha256" ]]; then
  echo "RDKit geometry audit input/code identity mismatch" >&2
  exit 2
fi

output=outputs/analysis/rdkit_fragment_geometry_v1/val1076_seed0.json
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing audit output: $output" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python scripts/analyze_rdkit_fragment_geometry.py \
  --split-key val \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --rdkit-seed 0 \
  --output "$output"
