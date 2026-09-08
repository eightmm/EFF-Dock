"""Self-contained physical, interaction, and chemical guidance for EFF-Dock."""

# Import order follows the module dependency DAG; alphabetical sorting creates
# a cycle because diagnostics imports interaction/runtime during package init.
# ruff: noqa: I001

from .errors import UnsupportedPhysicalChemistryError
from .system import (
    InteractionTopology,
    PhysicalSystem,
    build_physical_system,
    receptor_policy_identity,
    type_ligand_interactions,
)
from .topology import PhysicalTopology, build_physical_topology
from .physical import PhysicalEnergyConfig, physical_energy
from .chemical import (
    ChemicalConstraintEnergyConfig,
    chemical_constraint_diagnostics,
    chemical_constraint_energy,
)
from .feynman_kac import (
    DEFAULT_FK_CONSTRAINT_TERMS,
    SUPPORTED_FK_CONSTRAINT_TERMS,
    FKConstraintConfig,
    FeynmanKacConstraintResampler,
    constraint_potential,
    parse_fk_resample_times,
)
from .interaction import (
    ACTIVE_INTERACTION_TERMS,
    PLANNED_INTERACTION_TERMS,
    InteractionEnergyConfig,
    interaction_contact_stats,
    interaction_energy,
    interaction_profile_metadata,
    metal_coordination_v0_contract,
    metal_coordination_v1_contract,
)
from .runtime import (
    GuidanceEnergyConfig,
    PhysicalGuidance,
    PhysicalGuidanceConfig,
    UnifiedGuidance,
    UnifiedGuidanceConfig,
    guidance_energy,
)
from .diagnostics import (
    TRACE_SCHEMA_VERSION,
    PoseState,
    make_crystal_perturbations,
    trace_guidance_pose,
    trace_physical_pose,
)

__all__ = [
    "ACTIVE_INTERACTION_TERMS",
    "ChemicalConstraintEnergyConfig",
    "DEFAULT_FK_CONSTRAINT_TERMS",
    "FKConstraintConfig",
    "FeynmanKacConstraintResampler",
    "GuidanceEnergyConfig",
    "InteractionEnergyConfig",
    "InteractionTopology",
    "PhysicalEnergyConfig",
    "PhysicalGuidance",
    "PhysicalGuidanceConfig",
    "UnifiedGuidance",
    "UnifiedGuidanceConfig",
    "PhysicalSystem",
    "PhysicalTopology",
    "PLANNED_INTERACTION_TERMS",
    "PoseState",
    "TRACE_SCHEMA_VERSION",
    "SUPPORTED_FK_CONSTRAINT_TERMS",
    "UnsupportedPhysicalChemistryError",
    "build_physical_system",
    "build_physical_topology",
    "chemical_constraint_diagnostics",
    "chemical_constraint_energy",
    "constraint_potential",
    "guidance_energy",
    "interaction_contact_stats",
    "interaction_energy",
    "interaction_profile_metadata",
    "make_crystal_perturbations",
    "metal_coordination_v0_contract",
    "metal_coordination_v1_contract",
    "physical_energy",
    "parse_fk_resample_times",
    "receptor_policy_identity",
    "trace_guidance_pose",
    "trace_physical_pose",
    "type_ligand_interactions",
]
