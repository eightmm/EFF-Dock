#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
mkdir -p outputs/guidance/interaction_prior_probe_v2/logs

exec sbatch "$@" scripts/slurm/interaction_prior_probe_array.sbatch
