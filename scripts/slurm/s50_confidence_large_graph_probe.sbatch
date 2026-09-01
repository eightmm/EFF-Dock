#!/usr/bin/env bash
#SBATCH --job-name=effdock-s50-large-probe
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --partition=heavy
#SBATCH --qos=veryshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

set -euo pipefail

repo_dir=$(cd "${SLURM_SUBMIT_DIR:?}" && pwd -P)
pose_counts=(1 2 4 8)
num_poses=${pose_counts[${SLURM_ARRAY_TASK_ID:?}]}
manifest="$repo_dir/outputs/eff-dock/s50-confidence-matched-runs/14cb2840f040b05127610a1f3dccd8aa5b2573c41156ff446898e2d212f1164b/full/bank_manifest.json"
checkpoint="$repo_dir/outputs/eff-dock/s50-confidence-symmetry-training/88509b2189ab619c91feee843e4d90388b85b36b181532ca0a54ebf271f69b97/recovery/anchors/u001500_job56496_latest.pt"
sample_key='8qfe__1__2.A_3.A_4.A_5.A_6.A__2.B_3.B_4.B_5.B_6.B__2.B'

export PYTHONPATH="$repo_dir/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$repo_dir/.venv/bin/python" "$repo_dir/scripts/probe_s50_confidence_large_graph.py" \
  --bank-manifest "$manifest" \
  --checkpoint "$checkpoint" \
  --sample-key "$sample_key" \
  --num-poses "$num_poses"
