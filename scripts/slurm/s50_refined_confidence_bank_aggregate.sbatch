#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refconf-agg
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --partition=cpu_only
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00

set -euo pipefail
repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
cd "$repo_dir"
source_root="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b"
refined_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/207611a7fed8805b7f52cdf4648da2ea03e7d1e699c7d47106b632215634fdc2/full"
worker="$repo_dir/scripts/materialize_s50_refined_confidence_bank.py"
read -r actual _ < <(sha256sum "$worker")
[[ "$actual" == 07f45445e0d2035d895535dadf1ee73c64a85d1193b36a971c8b4a68a1013768 ]]
export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python "$worker" aggregate \
  --raw-bank-manifest "$source_root/full/bank_manifest.json" \
  --refined-manifest "$refined_root/manifest.json" \
  --input-manifest "$source_root/frozen/input_manifest.json" \
  --output-root "$refined_root/confidence_bank" \
  --train-shards 128 --val-shards 8
