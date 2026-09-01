"""Frozen profiles for parent-free confidence guidance sweeps.

The legacy profile preserves the completed V1 experiment.  New experiments
must use a separate profile so their grid, run names, parameter identities,
and artifact roots cannot be confused with historical outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StandaloneSweepSpec:
    key: str
    protocol_id: str
    audit_contract: str
    audit_schema_version: str
    binding_contract: str
    output_prefix: str
    run_name_prefix: str
    eta_values: tuple[float, ...]
    eta_tags: tuple[str, ...]
    guidance_parameter_sha256: str
    receptor_policy_sha256: str
    physical_parameter_sha256: str
    physical_parameter_version: str
    physical_formula_version: str
    interaction_parameter_sha256: str

    def __post_init__(self) -> None:
        if not self.key or len(self.eta_values) != len(self.eta_tags):
            raise ValueError("standalone sweep profile requires aligned eta values/tags")
        if not self.eta_values or self.eta_values[0] != 0.0:
            raise ValueError("standalone sweep profile requires same-run eta=0 baseline")
        if any(right <= left for left, right in zip(self.eta_values, self.eta_values[1:])):
            raise ValueError("standalone sweep eta values must be strictly increasing")
        if len(set(self.eta_tags)) != len(self.eta_tags):
            raise ValueError("standalone sweep eta tags must be unique")

    def eta_tag(self, eta: float) -> str:
        matches = [
            tag
            for value, tag in zip(self.eta_values, self.eta_tags, strict=True)
            if math.isclose(float(eta), value, rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(f"eta must be one of {self.eta_values}, got {eta!r}")
        return matches[0]

    def expected_run_name(
        self,
        dataset: str,
        eta: float,
        *,
        num_samples: int = 100,
        num_steps: int = 10,
    ) -> str:
        if dataset not in {"astex", "posebusters"}:
            raise ValueError(f"unexpected dataset: {dataset!r}")
        return (
            f"{self.run_name_prefix}-{dataset}-"
            f"n{num_samples}-s{num_steps}-{self.eta_tag(eta)}"
        )


LEGACY_V1 = StandaloneSweepSpec(
    key="legacy_v1",
    protocol_id="EFFDOCK-UNIFIED-GUIDANCE-ETA-SWEEP-CONFIDENCE-STANDALONE-PB-V1",
    audit_contract="EFFDOCK_CONFIDENCE_STANDALONE_INTEGRITY_V1",
    audit_schema_version="effdock.guidance_eta_sweep_confidence_standalone_integrity.v1",
    binding_contract="EFFDOCK_ETA_SWEEP_CONFIDENCE_STANDALONE_OFFICIAL_BINDING_V1",
    output_prefix="outputs/benchmarks/guidance_eta_sweep_confidence_standalone_runs",
    run_name_prefix="effdock-guidance-direct-drift-eta-sweep-v2",
    eta_values=(0.0, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    eta_tags=(
        "eta0000",
        "eta0025",
        "eta0050",
        "eta0100",
        "eta0200",
        "eta0300",
        "eta0400",
        "eta0500",
    ),
    guidance_parameter_sha256="7851dfe3cb2f290d3fce6e3ae2e2fe1d785cd5bc2c730e6d13bbcfb67e2b6012",
    receptor_policy_sha256="7bd75b1ff265b46fb556f7770ed5c393ad349304ae4ceedc0564dde93e26c5fd",
    physical_parameter_sha256="4e65d5c629005c474e1cd107218416beaf035c27e59a817e4baaa7b02b16631e",
    physical_parameter_version="2.0.0",
    physical_formula_version="effff-diagnostic-2",
    interaction_parameter_sha256="b772d431e21bcaecf1648ce4e539b1448e4f9df8ff5bd0db0fdf6407fcd23f16",
)


STERIC_HIGH_ETA_V1 = StandaloneSweepSpec(
    key="steric_high_eta_v1",
    protocol_id="EFFDOCK-UNIFIED-GUIDANCE-STERIC-HIGH-ETA-CONFIDENCE-PB-V1",
    audit_contract="EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_INTEGRITY_V2",
    audit_schema_version="effdock.guidance_steric_high_eta_confidence_integrity.v2",
    binding_contract="EFFDOCK_STERIC_HIGH_ETA_CONFIDENCE_OFFICIAL_BINDING_V2",
    output_prefix="outputs/benchmarks/guidance_steric_high_eta_confidence_runs",
    run_name_prefix="effdock-guidance-steric-high-eta-v1",
    eta_values=(0.0, 0.5, 1.0, 1.5, 2.0),
    eta_tags=("eta0000", "eta0500", "eta1000", "eta1500", "eta2000"),
    guidance_parameter_sha256="6621d17c41aeb6c9685075209155850018c5eb9882489ae209c7c30b8070e89f",
    receptor_policy_sha256="92adb215ccb77aae51ea14d8a2cc33319f70feb8548e9f0b07f500a5bcee1c20",
    physical_parameter_sha256="079940d8b61ed777ea00c3ac9abb101996a618df461deacee3d5ab3189f5d674",
    physical_parameter_version="2.1.0",
    physical_formula_version="effff-diagnostic-2.1",
    interaction_parameter_sha256="b772d431e21bcaecf1648ce4e539b1448e4f9df8ff5bd0db0fdf6407fcd23f16",
)


PROFILES = {profile.key: profile for profile in (LEGACY_V1, STERIC_HIGH_ETA_V1)}


def get_standalone_sweep_spec(profile: str) -> StandaloneSweepSpec:
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"unknown standalone sweep profile {profile!r}; expected one of {tuple(PROFILES)}"
        ) from exc


__all__ = [
    "LEGACY_V1",
    "PROFILES",
    "STERIC_HIGH_ETA_V1",
    "StandaloneSweepSpec",
    "get_standalone_sweep_spec",
]
