#!/usr/bin/env bash
# Canonical launcher name retained by the eta-sweep pre-registration.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/submit_guidance_eta_sweep_v2.sh" "$@"
