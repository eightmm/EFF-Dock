#!/usr/bin/env bash
# Frozen shell constants shared by the legacy and steric high-eta workflows.

case "${EFFDOCK_STANDALONE_PROFILE:-legacy_v1}" in
  legacy_v1)
    standalone_profile=legacy_v1
    protocol_id=EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-STANDALONE-PB-V1
    audit_contract=EFFDOCK_CONFIDENCE_STANDALONE_INTEGRITY_V1
    audit_schema_version=effdock.guidance_eta_sweep_confidence_standalone_integrity.v1
    output_prefix=outputs/benchmarks/guidance_eta_sweep_confidence_standalone_runs
    protocol_doc=docs/GUIDANCE_ETA_SWEEP_CONFIDENCE_STANDALONE_PB_PROTOCOL.md
    run_name_prefix=effdock-guidance-direct-drift-eta-sweep-v2
    eta_values=(0.0 0.025 0.05 0.1 0.2 0.3 0.4 0.5)
    eta_tags=(eta0000 eta0025 eta0050 eta0100 eta0200 eta0300 eta0400 eta0500)
    smoke_array_spec=0-15%8
    sampling_array_spec=0-127%8
    posebusters_array_spec=0-255%16
    ;;
  steric_high_eta_v1)
    standalone_profile=steric_high_eta_v1
    protocol_id=EFFDOCK-UNIFIED-GUIDANCE-STERIC-HIGH-ETA-CONFIDENCE-PB-V1
    audit_contract=EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_INTEGRITY_V2
    audit_schema_version=effdock.guidance_steric_high_eta_confidence_integrity.v2
    output_prefix=outputs/benchmarks/guidance_steric_high_eta_confidence_runs
    protocol_doc=docs/GUIDANCE_STERIC_HIGH_ETA_CONFIDENCE_PB_PROTOCOL.md
    run_name_prefix=effdock-guidance-steric-high-eta-v1
    eta_values=(0.0 0.5 1.0 1.5 2.0)
    eta_tags=(eta0000 eta0500 eta1000 eta1500 eta2000)
    smoke_array_spec=0-9%4
    sampling_array_spec=0-79%4
    posebusters_array_spec=0-159%16
    ;;
  *)
    echo "unknown EFFDOCK_STANDALONE_PROFILE: ${EFFDOCK_STANDALONE_PROFILE}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

eta_count=${#eta_values[@]}
dataset_count=2
num_shards=8
selector_count=2
smoke_task_count=$((dataset_count * eta_count))
sampling_task_count=$((dataset_count * eta_count * num_shards))
posebusters_task_count=$((selector_count * dataset_count * eta_count * num_shards))
posebusters_smoke_eta_index=$((eta_count - 1))
