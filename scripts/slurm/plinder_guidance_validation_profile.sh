#!/usr/bin/env bash
# Frozen mechanical constants shared by the PLINDER guidance Slurm stages.

protocol_id=EFFDOCK-PLINDER-GUIDANCE-DEV-V1
plinder_output_prefix=outputs/benchmarks/plinder_guidance_validation_runs
expected_count=1076
full_num_shards=32
eta_values=(0 0.5 1 1.5 2)
eta_tags=(eta0000 eta0500 eta1000 eta1500 eta2000)
eta_count=${#eta_values[@]}
smoke_id='7z9g__1__1.A_1.B_1.C_1.D__1.J__1.J'
smoke_sampling_tasks=$eta_count
full_sampling_tasks=$((full_num_shards * eta_count))
smoke_pb_tasks=$eta_count
full_pb_tasks=$((full_num_shards * eta_count))

verify_frozen_manifest() {
  local label=$1
  local manifest=$2
  local expected_sha256=$3
  local actual_sha256
  if [[ ! -f "$manifest" ]]; then
    echo "missing frozen $label manifest: $manifest" >&2
    exit 2
  fi
  read -r actual_sha256 _ < <(sha256sum "$manifest")
  if [[ "$actual_sha256" != "$expected_sha256" ]] \
    || ! sha256sum --check --quiet "$manifest"; then
    echo "frozen $label manifest verification failed" >&2
    exit 2
  fi
}

require_safe_plinder_output_root() {
  local output_root=$1
  local prefix_slash="$plinder_output_prefix/"
  local component=${output_root#"$prefix_slash"}
  if [[ "$component" == "$output_root" || "$component" == */* ]] \
    || [[ ! "$component" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "unsafe PLINDER guidance output root: $output_root" >&2
    exit 2
  fi
}

verify_plinder_execution_contract() {
  local execution_manifest=${EFFDOCK_PLINDER_EXECUTION_MANIFEST:?missing execution manifest}
  local execution_sha256=${EFFDOCK_PLINDER_EXECUTION_MANIFEST_SHA256:?missing execution manifest SHA-256}
  local frozen_manifest=${EFFDOCK_PLINDER_FROZEN_INPUTS_MANIFEST:?missing frozen-input manifest}
  local frozen_sha256=${EFFDOCK_PLINDER_FROZEN_INPUTS_MANIFEST_SHA256:?missing frozen-input manifest SHA-256}
  verify_frozen_manifest execution "$execution_manifest" "$execution_sha256"
  verify_frozen_manifest inputs "$frozen_manifest" "$frozen_sha256"
}

verify_plinder_raw_manifest() {
  local raw_manifest=${EFFDOCK_PLINDER_RAW_MANIFEST:?missing raw-download manifest}
  .venv/bin/python -c \
    'import sys; from pathlib import Path; sys.path.insert(0,"scripts"); from plinder_guidance_common import load_split_ids,verify_raw_manifest; ids=load_split_ids(Path("data/splits/plinder.json")); verify_raw_manifest(Path(sys.argv[1]),split_ids=ids)' \
    "$raw_manifest"
}

verify_plinder_raw_gate() {
  local raw_manifest=${EFFDOCK_PLINDER_RAW_MANIFEST:?missing raw-download manifest}
  local raw_root=${EFFDOCK_PLINDER_RAW_ROOT:?missing raw root}
  local raw_gate=${EFFDOCK_PLINDER_RAW_GATE:?missing verified raw gate}
  local raw_gate_sidecar=${EFFDOCK_PLINDER_RAW_GATE_SIDECAR:?missing raw-gate sidecar}
  .venv/bin/python -c \
    'import sys; from pathlib import Path; sys.path.insert(0,"scripts"); from plinder_guidance_common import load_split_ids,validate_raw_gate; ids=load_split_ids(Path("data/splits/plinder.json")); validate_raw_gate(Path(sys.argv[1]),Path(sys.argv[2]),raw_manifest=Path(sys.argv[3]),raw_root=Path(sys.argv[4]),split_ids=ids)' \
    "$raw_gate" "$raw_gate_sidecar" "$raw_manifest" "$raw_root"
}
