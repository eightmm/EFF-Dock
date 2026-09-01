#!/usr/bin/env bash
# Submit the frozen steric high-eta profile through the shared strict chain.

set -euo pipefail

export EFFDOCK_STANDALONE_PROFILE=steric_high_eta_v1
export EFFDOCK_STANDALONE_LAUNCHER=scripts/slurm/submit_guidance_steric_high_eta_confidence_pb.sh
exec "$(dirname "$0")/submit_guidance_eta_sweep_confidence_standalone_pb.sh" "$@"
