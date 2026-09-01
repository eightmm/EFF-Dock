#!/usr/bin/env bash
#SBATCH --job-name=eff-inter-probe
#SBATCH --partition=cpu_only
#SBATCH --array=0-11
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/guidance/interaction_prior_probe_v2/logs/%A_%a.out
#SBATCH --error=outputs/guidance/interaction_prior_probe_v2/logs/%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

arms=(
  guard_only
  guard_default
  guard_pi
  guard_halogen
  guard_all
  interaction_all_raw
)
priors=(local model)

arm_index=$((SLURM_ARRAY_TASK_ID % ${#arms[@]}))
prior_index=$((SLURM_ARRAY_TASK_ID / ${#arms[@]}))
arm="${arms[$arm_index]}"
prior="${priors[$prior_index]}"
output_dir="outputs/guidance/interaction_prior_probe_v2/1LPZ_CMB/${prior}/${arm}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python -m effdock.workflows.interaction_prior_probe run \
  --protein data/external_benchmarks/data/astex_diverse_set/1LPZ_CMB/1LPZ_CMB_protein.pdb \
  --ligand data/external_benchmarks/data/astex_diverse_set/1LPZ_CMB/1LPZ_CMB_ligand.sdf \
  --pocket-centers-json data/external_test/astex_reference_pocket_centers.json \
  --pocket-center-key 1lpz \
  --sample-id 1LPZ_CMB \
  --output-dir "${output_dir}" \
  --protocol-file docs/INTERACTION_PRIOR_PROBE_PROTOCOL.md \
  --prior "${prior}" \
  --arm "${arm}" \
  --seeds 20260731 20260732 20260733 20260734 20260735 20260736 20260737 20260738 \
  --prior-sigma 0.5 \
  --local-translation-sigma 0.5 \
  --local-rotation-sigma-degrees 15 \
  --protein-shell-cutoff 22 \
  --steps 500 \
  --save-every 5
