#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-refine-agg
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
content_id=c14df954c757ee092f8299224412c52f4305730348c797416fff4e73a3076050
run_root="$repo_dir/outputs/eff-dock/s50-refined-pose-runs/$content_id"
output_root="$run_root/full"
worker="$repo_dir/scripts/refine_s50_confidence_pose_bank.py"
read -r actual _ < <(sha256sum "$worker")
[[ "$actual" == 2a411292ce29404f9d1c4d5a282db85a346e8f482f52c232d4f007e3e6d34b54 ]]
[[ -f "$run_root/CONTENT_IDENTITY.sha256" ]]
read -r recorded_content_id < "$run_root/CONTENT_IDENTITY.sha256"
[[ "$recorded_content_id" == "$content_id" ]]
export PYTHONPATH="$repo_dir:$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
[[ $(.venv/bin/python -c 'from effdock.guidance.provenance import guidance_implementation_identity; print(guidance_implementation_identity()["sha256"])') == 56e26c3621c84ce4dc0d84e7b627a33941503a4dd646c0ff247ef35e13bf390e ]]
[[ $(.venv/bin/python -c 'from effdock.guidance.parameterization import guidance_parameter_identity; print(guidance_parameter_identity()["sha256"])') == 212a3314a9744729bd98cb4539fed61459619ddaa9b62de4455a2d363a8d4835 ]]
.venv/bin/python "$worker" aggregate \
  --raw-bank-manifest "$source_root/full/bank_manifest.json" \
  --input-manifest "$source_root/frozen/input_manifest.json" \
  --output-root "$output_root" --train-shards 128 --val-shards 8
