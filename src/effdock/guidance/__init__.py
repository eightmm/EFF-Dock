"""Self-contained physical and interaction guidance boundaries for EFF-Dock."""

from .diagnostics import (
    TRACE_SCHEMA_VERSION,
    PoseState,
    make_crystal_perturbations,
    trace_guidance_pose,
    trace_physical_pose,
)
from .errors import UnsupportedPhysicalChemistryError
from .interaction import (
    InteractionEnergyConfig,
    interaction_contact_stats,
    interaction_energy,
    interaction_profile_metadata,
    metal_coordination_v0_contract,
)
from .physical import PhysicalEnergyConfig, physical_energy
from .runtime import (
    GuidanceEnergyConfig,
    PhysicalGuidance,
    PhysicalGuidanceConfig,
    guidance_energy,
)
from .system import (
    InteractionTopology,
    PhysicalSystem,
    build_physical_system,
    type_ligand_interactions,
)
from .topology import PhysicalTopology, build_physical_topology

__all__ = [
    "GuidanceEnergyConfig",
    "InteractionEnergyConfig",
    "InteractionTopology",
    "PhysicalEnergyConfig",
    "PhysicalGuidance",
    "PhysicalGuidanceConfig",
    "PhysicalSystem",
    "PhysicalTopology",
    "PoseState",
    "TRACE_SCHEMA_VERSION",
    "UnsupportedPhysicalChemistryError",
    "build_physical_system",
    "build_physical_topology",
    "guidance_energy",
    "interaction_contact_stats",
    "interaction_energy",
    "interaction_profile_metadata",
    "make_crystal_perturbations",
    "metal_coordination_v0_contract",
    "physical_energy",
    "trace_guidance_pose",
    "trace_physical_pose",
    "type_ligand_interactions",
]
