"""Build static guidance-system tensors without an external force-field engine."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from hashlib import sha256
from pathlib import Path

import torch
from rdkit import Chem
from torch import Tensor

from effdock.preprocess.protein import (
    AA3_TO_IDX,
    NUCLEIC_ACID_RESIDUES,
    WATER_RESIDUES,
    _build_protein_bonds,
    _parse_pdb_lines,
    _patom_pharmacophore,
)

from .errors import UnsupportedPhysicalChemistryError
from .parameterization import (
    element_parameters,
    interaction_parameter_identity,
    load_effff_v2,
    load_interaction_v1,
    parameter_identity,
)
from .topology import PhysicalTopology, build_physical_topology

_SYMBOL_TO_Z = {
    "H": 1,
    "LI": 3,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NA": 11,
    "MG": 12,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "K": 19,
    "CA": 20,
    "V": 23,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "SE": 34,
    "BR": 35,
    "RB": 37,
    "SR": 38,
    "PD": 46,
    "AG": 47,
    "CD": 48,
    "XE": 54,
    "CS": 55,
    "BA": 56,
    "PT": 78,
    "AU": 79,
    "HG": 80,
    "MO": 42,
    "I": 53,
}

_KNOWN_PDB_METAL_ELEMENTS = {
    "LI",
    "NA",
    "K",
    "RB",
    "CS",
    "MG",
    "CA",
    "SR",
    "BA",
    "V",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "AG",
    "CD",
    "HG",
    "PT",
    "PD",
    "AU",
    "MO",
}

_RECEPTOR_POLICIES = frozenset({"fail_closed", "geometry_only"})
_GEOMETRY_ONLY_POLICY_VERSION = "effdock.receptor_geometry_only.v1"
_EFF_FF_SUPPORTED_ATOMIC_NUMBERS = frozenset(
    int(value) for value in load_effff_v2()["elements"]
)
_GENERIC_OBSTACLE_REPULSION_MAX = 10.0
_GENERIC_OBSTACLE_SOFTCORE = 0.5
_GENERIC_OBSTACLE_RADIUS = 1.7
_GENERIC_OBSTACLE_SWITCH_DISTANCE = 2.5
_GENERIC_OBSTACLE_CUTOFF = 3.0


def _is_guidance_metal(atom) -> bool:
    """Recognize guidance-only metals without changing frozen model features."""
    return bool(atom.is_metal) or atom.element.upper() in _KNOWN_PDB_METAL_ELEMENTS


def _receptor_policy_identity(mode: str) -> dict[str, object]:
    """Return an outcome-independent identity for receptor admission/fallback."""
    if mode not in _RECEPTOR_POLICIES:
        raise ValueError(
            f"receptor_policy must be one of {sorted(_RECEPTOR_POLICIES)}, got {mode!r}"
        )
    if mode == "fail_closed":
        payload: dict[str, object] = {
            "schema_version": "effdock.receptor_policy.v1",
            "mode": mode,
            "name": "fail_closed",
            "version": "1.0.0",
            "claim": "Unsupported active-shell receptor chemistry raises a structured error.",
        }
    else:
        payload = {
            "schema_version": "effdock.receptor_policy.v1",
            "mode": mode,
            "name": "geometry_only",
            "version": _GEOMETRY_ONLY_POLICY_VERSION,
            "claim": (
                "Supported nonmetal cofactor atoms are fixed UFF-style repulsion-only "
                "obstacles; unparameterized atoms are bounded generic steric obstacles; "
                "metal sites that fail strict attraction admission are bounded all-ligand "
                "repulsion-only sites. No fallback atom receives an attractive term."
            ),
            "generic_obstacle_formula": (
                "E=E_max*y/(1+y)*S(r), y=(r_rep/sqrt(r^2+alpha^2))^12"
            ),
            "generic_obstacle_constants": {
                "maximum_kcal_mol": _GENERIC_OBSTACLE_REPULSION_MAX,
                "softcore_angstrom": _GENERIC_OBSTACLE_SOFTCORE,
                "radius_angstrom": _GENERIC_OBSTACLE_RADIUS,
                "switch_distance_angstrom": _GENERIC_OBSTACLE_SWITCH_DISTANCE,
                "cutoff_angstrom": _GENERIC_OBSTACLE_CUTOFF,
            },
            "generic_obstacle_parameter_claim": "internal geometry-only diagnostic prior",
            "supported_nonmetal_parameter_set": parameter_identity(),
        }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": sha256(canonical).hexdigest()}


def receptor_policy_identity(mode: str) -> dict[str, object]:
    """Public immutable identity for a receptor admission policy."""
    return _receptor_policy_identity(mode)

_PROTEIN_AROMATIC_RING_TEMPLATES: dict[
    str,
    tuple[tuple[str, tuple[str, ...], bool], ...],
] = {
    "PHE": (("phenyl", ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"), True),),
    "TYR": (("phenyl", ("CG", "CD1", "CE1", "CZ", "CE2", "CD2"), True),),
    "HIS": (("imidazole", ("CG", "ND1", "CE1", "NE2", "CD2"), False),),
    "TRP": (
        ("pyrrole", ("CG", "CD1", "NE1", "CE2", "CD2"), False),
        ("benzene", ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"), True),
    ),
}
_PROTEIN_AROMATIC_RAW_NAMES = {
    "PHE": {"PHE"},
    "TYR": {"TYR"},
    "HIS": {"HIS", "HID", "HIE"},
    "TRP": {"TRP"},
}


def _validate_charge_site_contract(
    *,
    side: str,
    membership: Tensor,
    charge: Tensor,
    labels: tuple[str, ...],
    atom_count: int,
) -> None:
    """Fail closed on malformed static charge-group topology."""
    if membership.ndim != 2:
        raise ValueError(f"{side} charge-site membership must have shape [S,N]")
    if charge.ndim != 1:
        raise ValueError(f"{side} charge-site charge must have shape [S]")
    site_count = int(membership.shape[0])
    if int(charge.numel()) != site_count or len(labels) != site_count:
        raise ValueError(
            f"{side} charge-site membership, charge, and labels must have the same site count"
        )
    sentinel_empty = membership.shape == (0, 0)
    if not sentinel_empty and membership.shape[1] != atom_count:
        raise ValueError(
            f"{side} charge-site membership atom count does not match {side} atom labels"
        )
    if sentinel_empty or site_count == 0:
        return
    if not membership.is_floating_point() or not charge.is_floating_point():
        raise ValueError(f"{side} charge-site membership and charge must be floating point")
    if not bool(torch.isfinite(membership).all()) or not bool(torch.isfinite(charge).all()):
        raise ValueError(f"{side} charge-site membership and charge must be finite")
    if bool((membership < 0).any()):
        raise ValueError(f"{side} charge-site membership weights must be non-negative")
    if not torch.allclose(
        membership.sum(dim=1),
        torch.ones(
            site_count,
            device=membership.device,
            dtype=membership.dtype,
        ),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{side} charge-site membership rows must sum to one")
    if bool((torch.count_nonzero(membership, dim=0) > 1).any()):
        raise ValueError(f"{side} charge-site membership may not overlap")
    if bool((charge == 0).any()) or not torch.allclose(
        charge,
        charge.round(),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{side} charge sites require nonzero integer formal charge")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{side} charge-site labels must be unique")


def _validate_aromatic_ring_contract(
    *,
    side: str,
    membership: Tensor,
    triplet: Tensor,
    system: Tensor,
    reference_area: Tensor,
    is_cation_pi_acceptor: Tensor,
    labels: tuple[str, ...],
    atom_count: int,
) -> None:
    """Fail closed on malformed static aromatic-ring topology."""
    if membership.ndim != 2:
        raise ValueError(f"{side} aromatic-ring membership must have shape [R,N]")
    ring_count = int(membership.shape[0])
    if triplet.shape != (ring_count, 3):
        raise ValueError(f"{side} aromatic-ring triplets must have shape [R,3]")
    for name, value in (
        ("system", system),
        ("reference area", reference_area),
        ("cation-pi acceptor mask", is_cation_pi_acceptor),
    ):
        if value.shape != (ring_count,):
            raise ValueError(f"{side} aromatic-ring {name} must have shape [R]")
    if len(labels) != ring_count:
        raise ValueError(f"{side} aromatic-ring labels must match the ring count")
    sentinel_empty = membership.shape == (0, 0)
    if not sentinel_empty and membership.shape[1] != atom_count:
        raise ValueError(
            f"{side} aromatic-ring membership atom count does not match {side} atom labels"
        )
    if sentinel_empty or ring_count == 0:
        return
    if not membership.is_floating_point() or not reference_area.is_floating_point():
        raise ValueError(
            f"{side} aromatic-ring membership and reference area must be floating point"
        )
    if triplet.dtype != torch.long or system.dtype != torch.long:
        raise ValueError(f"{side} aromatic-ring triplets and systems must be torch.long")
    if is_cation_pi_acceptor.dtype != torch.bool:
        raise ValueError(f"{side} aromatic-ring cation-pi mask must be bool")
    if (
        not bool(torch.isfinite(membership).all())
        or not bool(torch.isfinite(reference_area).all())
        or bool((membership < 0).any())
    ):
        raise ValueError(f"{side} aromatic-ring tensors must be finite and non-negative")
    if not torch.allclose(
        membership.sum(dim=1),
        torch.ones(ring_count, device=membership.device, dtype=membership.dtype),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(f"{side} aromatic-ring membership rows must sum to one")
    member_count = torch.count_nonzero(membership, dim=1)
    if bool(((member_count < 5) | (member_count > 6)).any()):
        raise ValueError(f"{side} aromatic rings must contain five or six atoms")
    if bool((reference_area <= 0).any()) or bool((system < 0).any()):
        raise ValueError(f"{side} aromatic-ring area must be positive and system non-negative")
    system_ids = sorted(set(system.detach().cpu().tolist()))
    if system_ids != list(range(len(system_ids))):
        raise ValueError(f"{side} aromatic-ring system IDs must be contiguous from zero")
    if bool((triplet < 0).any()) or bool((triplet >= atom_count).any()):
        raise ValueError(f"{side} aromatic-ring triplet indices are out of range")
    if bool(
        (
            (triplet[:, 0] == triplet[:, 1])
            | (triplet[:, 0] == triplet[:, 2])
            | (triplet[:, 1] == triplet[:, 2])
        ).any()
    ):
        raise ValueError(f"{side} aromatic-ring triplet atoms must be distinct")
    if not bool(torch.gather(membership, 1, triplet.to(device=membership.device)).gt(0).all()):
        raise ValueError(f"{side} aromatic-ring triplet atoms must belong to their ring")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{side} aromatic-ring labels must be unique")


def _validate_index_vector(
    *,
    name: str,
    value: Tensor,
    upper_bound: int,
) -> None:
    if value.ndim != 1 or value.dtype != torch.long:
        raise ValueError(f"{name} must be a one-dimensional torch.long tensor")
    if bool((value < 0).any()) or bool((value >= upper_bound).any()):
        raise ValueError(f"{name} contains an out-of-range atom index")
    if int(torch.unique(value).numel()) != int(value.numel()):
        raise ValueError(f"{name} atom indices must be unique")


@dataclass(frozen=True)
class InteractionTopology:
    """Static chemical typing and direction references for interaction terms."""

    ligand_neighbor_index: Tensor
    ligand_direction_target_cosine: Tensor
    ligand_direction_geometry_valid: Tensor
    ligand_is_donor: Tensor
    ligand_is_acceptor: Tensor
    ligand_is_hydrophobe: Tensor
    ligand_is_geometry_excluded_hbond_site: Tensor
    protein_is_donor: Tensor
    protein_is_acceptor: Tensor
    protein_is_hydrophobe: Tensor
    protein_outward_direction: Tensor
    protein_direction_target_cosine: Tensor
    protein_direction_quality: Tensor
    protein_direction_valid: Tensor
    protein_is_ambiguous_histidine: Tensor
    protein_is_unsupported_variant: Tensor
    protein_is_geometry_excluded_hbond_site: Tensor
    ligand_atom_labels: tuple[str, ...]
    protein_atom_labels: tuple[str, ...]
    ligand_charge_site_membership: Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float64)
    )
    ligand_charge_site_charge: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    ligand_charge_site_labels: tuple[str, ...] = ()
    ligand_charge_site_exclusion_labels: tuple[str, ...] = ()
    protein_charge_site_membership: Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float64)
    )
    protein_charge_site_charge: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    protein_charge_site_labels: tuple[str, ...] = ()
    protein_charge_site_exclusion_labels: tuple[str, ...] = ()
    ligand_aromatic_ring_membership: Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float64)
    )
    ligand_aromatic_ring_triplet: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.long)
    )
    ligand_aromatic_ring_system: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_aromatic_ring_reference_area: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    ligand_aromatic_ring_is_cation_pi_acceptor: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    ligand_aromatic_ring_labels: tuple[str, ...] = ()
    ligand_aromatic_ring_exclusion_labels: tuple[str, ...] = ()
    protein_aromatic_ring_membership: Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float64)
    )
    protein_aromatic_ring_triplet: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.long)
    )
    protein_aromatic_ring_system: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    protein_aromatic_ring_reference_area: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    protein_aromatic_ring_is_cation_pi_acceptor: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    protein_aromatic_ring_labels: tuple[str, ...] = ()
    protein_aromatic_ring_exclusion_labels: tuple[str, ...] = ()
    ligand_halogen_donor_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_halogen_parent_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    protein_halogen_acceptor_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_halogen_exclusion_labels: tuple[str, ...] = ()
    protein_halogen_exclusion_labels: tuple[str, ...] = ()
    ligand_metal_donor_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_metal_donor_element: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_metal_donor_exclusion_labels: tuple[str, ...] = ()
    metal_coords: Tensor = field(default_factory=lambda: torch.empty((0, 3), dtype=torch.float64))
    metal_atomic_number: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    metal_vacant_direction: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float64)
    )
    metal_fixed_coordination: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    metal_target_coordination: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    metal_ligand_r0: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float64)
    )
    metal_ligand_donor_allowed: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.bool)
    )
    metal_attraction_enabled: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    metal_site_labels: tuple[str, ...] = ()
    metal_profile_labels: tuple[str, ...] = ()
    metal_typing_exclusion_labels: tuple[str, ...] = ()
    # Deprecated Zn-specific aliases.  These remain populated for admitted Zn
    # sites so old diagnostics and serialized fixtures stay readable while the
    # generic metal tensors above are the canonical contract.
    ligand_zinc_donor_index: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_zinc_donor_element: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ligand_zinc_donor_exclusion_labels: tuple[str, ...] = ()
    zinc_coords: Tensor = field(default_factory=lambda: torch.empty((0, 3), dtype=torch.float64))
    zinc_vacant_direction: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float64)
    )
    zinc_receptor_donor_index: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.long)
    )
    zinc_receptor_donor_element: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.long)
    )
    zinc_site_labels: tuple[str, ...] = ()
    zinc_typing_exclusion_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_charge_site_contract(
            side="ligand",
            membership=self.ligand_charge_site_membership,
            charge=self.ligand_charge_site_charge,
            labels=self.ligand_charge_site_labels,
            atom_count=len(self.ligand_atom_labels),
        )
        _validate_charge_site_contract(
            side="protein",
            membership=self.protein_charge_site_membership,
            charge=self.protein_charge_site_charge,
            labels=self.protein_charge_site_labels,
            atom_count=len(self.protein_atom_labels),
        )
        _validate_aromatic_ring_contract(
            side="ligand",
            membership=self.ligand_aromatic_ring_membership,
            triplet=self.ligand_aromatic_ring_triplet,
            system=self.ligand_aromatic_ring_system,
            reference_area=self.ligand_aromatic_ring_reference_area,
            is_cation_pi_acceptor=self.ligand_aromatic_ring_is_cation_pi_acceptor,
            labels=self.ligand_aromatic_ring_labels,
            atom_count=len(self.ligand_atom_labels),
        )
        _validate_aromatic_ring_contract(
            side="protein",
            membership=self.protein_aromatic_ring_membership,
            triplet=self.protein_aromatic_ring_triplet,
            system=self.protein_aromatic_ring_system,
            reference_area=self.protein_aromatic_ring_reference_area,
            is_cation_pi_acceptor=self.protein_aromatic_ring_is_cation_pi_acceptor,
            labels=self.protein_aromatic_ring_labels,
            atom_count=len(self.protein_atom_labels),
        )
        ligand_atom_count = len(self.ligand_atom_labels)
        protein_atom_count = len(self.protein_atom_labels)
        _validate_index_vector(
            name="ligand halogen donor index",
            value=self.ligand_halogen_donor_index,
            upper_bound=ligand_atom_count,
        )
        if self.ligand_halogen_parent_index.shape != self.ligand_halogen_donor_index.shape:
            raise ValueError("ligand halogen donor and parent indices must have the same shape")
        if (
            self.ligand_halogen_parent_index.dtype != torch.long
            or bool((self.ligand_halogen_parent_index < 0).any())
            or bool((self.ligand_halogen_parent_index >= ligand_atom_count).any())
        ):
            raise ValueError("ligand halogen parent indices must be in-range torch.long values")
        _validate_index_vector(
            name="protein halogen acceptor index",
            value=self.protein_halogen_acceptor_index,
            upper_bound=protein_atom_count,
        )
        _validate_index_vector(
            name="ligand metal donor index",
            value=self.ligand_metal_donor_index,
            upper_bound=ligand_atom_count,
        )
        if (
            self.ligand_metal_donor_element.shape != self.ligand_metal_donor_index.shape
            or self.ligand_metal_donor_element.dtype != torch.long
            or bool(
                (
                    ~torch.isin(
                        self.ligand_metal_donor_element,
                        torch.tensor(
                            [7, 8, 16],
                            device=self.ligand_metal_donor_element.device,
                            dtype=torch.long,
                        ),
                    )
                ).any()
            )
        ):
            raise ValueError("ligand metal donor elements must be N, O, or S atomic numbers")
        metal_count = int(self.metal_coords.shape[0])
        generic_metal_shapes = {
            "metal vacant direction": (self.metal_vacant_direction, (metal_count, 3)),
            "metal atomic number": (self.metal_atomic_number, (metal_count,)),
            "metal fixed coordination": (self.metal_fixed_coordination, (metal_count,)),
            "metal target coordination": (self.metal_target_coordination, (metal_count,)),
            "metal ligand equilibrium distance": (self.metal_ligand_r0, (metal_count, 3)),
            "metal ligand donor mask": (
                self.metal_ligand_donor_allowed,
                (metal_count, 3),
            ),
            "metal attraction mask": (self.metal_attraction_enabled, (metal_count,)),
        }
        if self.metal_coords.shape != (metal_count, 3):
            raise ValueError("metal coordinates must have shape [M,3]")
        for name, (value, expected_shape) in generic_metal_shapes.items():
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {list(expected_shape)}")
        for name, value in (
            ("metal coordinates", self.metal_coords),
            ("metal vacant directions", self.metal_vacant_direction),
            ("metal ligand equilibrium distances", self.metal_ligand_r0),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite floating-point values")
        for name, value in (
            ("metal atomic numbers", self.metal_atomic_number),
            ("metal fixed coordination", self.metal_fixed_coordination),
            ("metal target coordination", self.metal_target_coordination),
        ):
            if value.dtype != torch.long:
                raise ValueError(f"{name} must be torch.long")
        if self.metal_ligand_donor_allowed.dtype != torch.bool:
            raise ValueError("metal ligand donor mask must be bool")
        if self.metal_attraction_enabled.dtype != torch.bool:
            raise ValueError("metal attraction mask must be bool")
        if bool((self.metal_atomic_number <= 0).any()):
            raise ValueError("metal atomic numbers must be positive")
        if bool((self.metal_fixed_coordination < 0).any()) or bool(
            (self.metal_target_coordination < self.metal_fixed_coordination).any()
        ):
            raise ValueError(
                "metal coordination counts require 0 <= fixed coordination <= target"
            )
        if bool((self.metal_ligand_r0 < 0).any()):
            raise ValueError("metal ligand equilibrium distances must be non-negative")
        if bool(
            (
                self.metal_ligand_donor_allowed
                != self.metal_ligand_r0.gt(0)
            ).any()
        ):
            raise ValueError(
                "metal ligand donor masks must match positive equilibrium distances"
            )
        if metal_count:
            direction_norm = self.metal_vacant_direction.norm(dim=-1)
            if not torch.allclose(
                direction_norm,
                torch.ones_like(direction_norm),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("metal vacant directions must be unit vectors")
            attractive = self.metal_attraction_enabled
            if bool(
                (
                    attractive
                    & (
                        (self.metal_target_coordination <= 0)
                        | (
                            self.metal_target_coordination
                            - self.metal_fixed_coordination
                            != 1
                        )
                        | ~self.metal_ligand_donor_allowed.any(dim=1)
                    )
                ).any()
            ):
                raise ValueError(
                    "attractive metal profiles require exactly one vacant slot "
                    "and at least one allowed ligand donor"
                )
            if bool(
                (
                    ~attractive
                    & (
                        (self.metal_fixed_coordination != 0)
                        | (self.metal_target_coordination != 0)
                        | self.metal_ligand_donor_allowed.any(dim=1)
                    )
                ).any()
            ):
                raise ValueError(
                    "repulsion-only metal profiles must not declare attractive "
                    "coordination or ligand donors"
                )
        for name, labels in (
            ("metal site labels", self.metal_site_labels),
            ("metal profile labels", self.metal_profile_labels),
        ):
            if len(labels) != metal_count:
                raise ValueError(f"{name} must match the metal site count")
            if any(not str(label).strip() for label in labels):
                raise ValueError(f"{name} must be non-empty")
        if len(set(self.metal_site_labels)) != metal_count:
            raise ValueError("metal site labels must be unique")
        if len(set(self.metal_typing_exclusion_labels)) != len(
            self.metal_typing_exclusion_labels
        ):
            raise ValueError("metal typing exclusion labels must be unique")
        _validate_index_vector(
            name="ligand zinc donor index",
            value=self.ligand_zinc_donor_index,
            upper_bound=ligand_atom_count,
        )
        if (
            self.ligand_zinc_donor_element.shape != self.ligand_zinc_donor_index.shape
            or self.ligand_zinc_donor_element.dtype != torch.long
            or bool(
                (
                    ~torch.isin(
                        self.ligand_zinc_donor_element,
                        torch.tensor(
                            [7, 8, 16],
                            device=self.ligand_zinc_donor_element.device,
                            dtype=torch.long,
                        ),
                    )
                ).any()
            )
        ):
            raise ValueError("ligand zinc donor elements must be N, O, or S atomic numbers")
        zinc_count = int(self.zinc_coords.shape[0])
        if self.zinc_coords.shape != (zinc_count, 3) or self.zinc_vacant_direction.shape != (
            zinc_count,
            3,
        ):
            raise ValueError("zinc coordinates and vacant directions must have shape [Z,3]")
        if self.zinc_receptor_donor_index.shape != (zinc_count, 3):
            raise ValueError("zinc receptor donor indices must have shape [Z,3]")
        if self.zinc_receptor_donor_element.shape != (zinc_count, 3):
            raise ValueError("zinc receptor donor elements must have shape [Z,3]")
        if len(self.zinc_site_labels) != zinc_count:
            raise ValueError("zinc site labels must match the zinc site count")
        if (
            not self.zinc_coords.is_floating_point()
            or not self.zinc_vacant_direction.is_floating_point()
        ):
            raise ValueError("zinc coordinates and vacant directions must be floating point")
        if not bool(torch.isfinite(self.zinc_coords).all()) or not bool(
            torch.isfinite(self.zinc_vacant_direction).all()
        ):
            raise ValueError("zinc coordinates and vacant directions must be finite")
        if self.zinc_receptor_donor_index.dtype != torch.long:
            raise ValueError("zinc receptor donor indices must be torch.long")
        if self.zinc_receptor_donor_element.dtype != torch.long:
            raise ValueError("zinc receptor donor elements must be torch.long")
        if zinc_count and (
            bool((self.zinc_receptor_donor_index < 0).any())
            or bool((self.zinc_receptor_donor_index >= protein_atom_count).any())
            or bool(
                (
                    ~torch.isin(
                        self.zinc_receptor_donor_element,
                        torch.tensor(
                            [7, 8, 16],
                            device=self.zinc_receptor_donor_element.device,
                            dtype=torch.long,
                        ),
                    )
                ).any()
            )
        ):
            raise ValueError("zinc receptor donors require in-range N/O/S atom indices")
        if zinc_count:
            if any(torch.unique(row).numel() != 3 for row in self.zinc_receptor_donor_index):
                raise ValueError("each zinc site requires three distinct receptor donors")
            zinc_direction_norm = self.zinc_vacant_direction.norm(dim=-1)
            if not torch.allclose(
                zinc_direction_norm,
                torch.ones_like(zinc_direction_norm),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("zinc vacant directions must be unit vectors")
            if len(set(self.zinc_site_labels)) != zinc_count:
                raise ValueError("zinc site labels must be unique")

    def to(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> InteractionTopology:
        values: dict[str, Tensor] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, Tensor):
                values[item.name] = value
                continue
            if value.dtype == torch.bool:
                values[item.name] = value.to(device=device, dtype=torch.bool)
            elif not value.is_floating_point():
                values[item.name] = value.to(device=device, dtype=torch.long)
            else:
                values[item.name] = value.to(device=device, dtype=dtype)
        return InteractionTopology(**values)

    def term_counts(self) -> dict[str, int]:
        ligand_donor = int(self.ligand_is_donor.sum().item())
        ligand_acceptor = int(self.ligand_is_acceptor.sum().item())
        ligand_hydrophobe = int(self.ligand_is_hydrophobe.sum().item())
        protein_donor = int(self.protein_is_donor.sum().item())
        protein_acceptor = int(self.protein_is_acceptor.sum().item())
        protein_hydrophobe = int(self.protein_is_hydrophobe.sum().item())
        return {
            "ligand_donors": ligand_donor,
            "ligand_acceptors": ligand_acceptor,
            "ligand_hydrophobes": ligand_hydrophobe,
            "ligand_geometry_excluded_hbond_sites": int(
                self.ligand_is_geometry_excluded_hbond_site.sum().item()
            ),
            "protein_donors": protein_donor,
            "protein_acceptors": protein_acceptor,
            "protein_hydrophobes": protein_hydrophobe,
            "protein_ambiguous_histidine_atoms": int(
                self.protein_is_ambiguous_histidine.sum().item()
            ),
            "protein_unsupported_variant_atoms": int(
                self.protein_is_unsupported_variant.sum().item()
            ),
            "protein_geometry_excluded_hbond_sites": int(
                self.protein_is_geometry_excluded_hbond_site.sum().item()
            ),
            "hydrophobic_candidate_pairs": ligand_hydrophobe * protein_hydrophobe,
            "hydrogen_bond_candidate_pairs": (
                ligand_donor * protein_acceptor + ligand_acceptor * protein_donor
            ),
            "ligand_charge_sites": int(self.ligand_charge_site_charge.numel()),
            "ligand_charge_site_member_atoms": int(
                torch.count_nonzero(self.ligand_charge_site_membership).item()
            ),
            "ligand_charge_site_exclusions": len(self.ligand_charge_site_exclusion_labels),
            "ligand_formal_charge_total_e": int(
                round(float(self.ligand_charge_site_charge.sum().item()))
            ),
            "protein_charge_sites": int(self.protein_charge_site_charge.numel()),
            "protein_charge_site_member_atoms": int(
                torch.count_nonzero(self.protein_charge_site_membership).item()
            ),
            "protein_charge_site_exclusions": len(self.protein_charge_site_exclusion_labels),
            "protein_canonical_charge_total_e": int(
                round(float(self.protein_charge_site_charge.sum().item()))
            ),
            "formal_charge_candidate_pairs": int(
                self.ligand_charge_site_charge.numel() * self.protein_charge_site_charge.numel()
            ),
            "ligand_aromatic_rings": int(self.ligand_aromatic_ring_membership.shape[0]),
            "ligand_aromatic_systems": int(torch.unique(self.ligand_aromatic_ring_system).numel()),
            "protein_aromatic_rings": int(self.protein_aromatic_ring_membership.shape[0]),
            "protein_aromatic_systems": int(
                torch.unique(self.protein_aromatic_ring_system).numel()
            ),
            "pi_stacking_candidate_pairs": int(
                self.ligand_aromatic_ring_membership.shape[0]
                * self.protein_aromatic_ring_membership.shape[0]
            ),
            "cation_pi_candidate_pairs": int(
                (self.ligand_charge_site_charge == 1).sum().item()
                * self.protein_aromatic_ring_is_cation_pi_acceptor.sum().item()
                + (self.protein_charge_site_charge == 1).sum().item()
                * self.ligand_aromatic_ring_is_cation_pi_acceptor.sum().item()
            ),
            "ligand_aromatic_ring_exclusions": len(self.ligand_aromatic_ring_exclusion_labels),
            "protein_aromatic_ring_exclusions": len(self.protein_aromatic_ring_exclusion_labels),
            "ligand_halogen_donors": int(self.ligand_halogen_donor_index.numel()),
            "protein_halogen_acceptors": int(self.protein_halogen_acceptor_index.numel()),
            "halogen_bond_candidate_pairs": int(
                self.ligand_halogen_donor_index.numel()
                * self.protein_halogen_acceptor_index.numel()
            ),
            "ligand_halogen_exclusions": len(self.ligand_halogen_exclusion_labels),
            "protein_halogen_exclusions": len(self.protein_halogen_exclusion_labels),
            "ligand_metal_donors": int(self.ligand_metal_donor_index.numel()),
            "ligand_metal_donor_exclusions": len(
                self.ligand_metal_donor_exclusion_labels
            ),
            "metal_sites": int(self.metal_coords.shape[0]),
            "metal_attractive_sites": int(self.metal_attraction_enabled.sum().item()),
            "metal_repulsion_only_sites": int(
                (~self.metal_attraction_enabled).sum().item()
            ),
            "metal_coordination_candidate_pairs": int(
                self.ligand_metal_donor_index.numel()
                * self.metal_attraction_enabled.sum().item()
            ),
            "metal_repulsion_candidate_pairs": int(
                len(self.ligand_atom_labels) * self.metal_coords.shape[0]
            ),
            "metal_typing_exclusions": len(self.metal_typing_exclusion_labels),
            "ligand_zinc_donors": int(self.ligand_zinc_donor_index.numel()),
            "ligand_zinc_donor_exclusions": len(self.ligand_zinc_donor_exclusion_labels),
            "zinc_sites": int(self.zinc_coords.shape[0]),
            "zinc_candidate_pairs": int(
                self.ligand_zinc_donor_index.numel() * self.zinc_coords.shape[0]
            ),
            "zinc_typing_exclusions": len(self.zinc_typing_exclusion_labels),
        }

    def reference_sha256(self) -> str:
        digest = sha256()
        for item in fields(self):
            field_value = getattr(self, item.name)
            if not isinstance(field_value, Tensor):
                payload = json.dumps(
                    {
                        "name": item.name,
                        "values": list(field_value),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
                digest.update(payload)
                digest.update(b"\0")
                continue
            value = field_value.detach().cpu()
            value = value.to(torch.float64) if value.is_floating_point() else value.to(torch.long)
            payload = json.dumps(
                {
                    "name": item.name,
                    "shape": list(value.shape),
                    "values": value.tolist(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            digest.update(payload)
            digest.update(b"\0")
        return digest.hexdigest()


@dataclass(frozen=True)
class PhysicalSystem:
    topology: PhysicalTopology
    protein_coords: Tensor
    protein_atomic_numbers: Tensor
    protein_uff_x: Tensor
    protein_uff_d: Tensor
    protein_vdw_radius: Tensor
    parameter_set: dict[str, str]
    protein_source_atoms: int
    protein_parameterized_source_atoms: int | None = None
    excluded_nonprotein_atoms: int = 0
    excluded_nonprotein_residues: tuple[str, ...] = ()
    geometry_obstacle_coords: Tensor = field(
        default_factory=lambda: torch.empty((0, 3), dtype=torch.float64)
    )
    geometry_obstacle_atomic_numbers: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    geometry_obstacle_uff_x: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    geometry_obstacle_uff_d: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.float64)
    )
    geometry_obstacle_is_generic: Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.bool)
    )
    geometry_obstacle_labels: tuple[str, ...] = ()
    geometry_obstacle_kinds: tuple[str, ...] = ()
    receptor_policy_mode: str = "fail_closed"
    receptor_policy_identity: dict[str, object] = field(
        default_factory=lambda: _receptor_policy_identity("fail_closed")
    )
    receptor_provenance: dict[str, object] = field(default_factory=dict)
    receptor_policy: str = (
        "records whose normalized residue name maps to a supported amino acid; "
        "ATOM/HETATM record type alone never admits chemistry"
    )
    interaction_topology: InteractionTopology | None = None
    interaction_parameter_set: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.receptor_policy_mode not in _RECEPTOR_POLICIES:
            raise ValueError(
                "receptor_policy_mode must be one of "
                f"{sorted(_RECEPTOR_POLICIES)}, got {self.receptor_policy_mode!r}"
            )
        protein_count = int(self.protein_coords.shape[0])
        if self.protein_coords.shape != (protein_count, 3):
            raise ValueError("protein coordinates must have shape [P,3]")
        for name, value in (
            ("atomic numbers", self.protein_atomic_numbers),
            ("UFF x", self.protein_uff_x),
            ("UFF d", self.protein_uff_d),
            ("vdW radius", self.protein_vdw_radius),
        ):
            if value.shape != (protein_count,):
                raise ValueError(f"protein {name} must have shape [{protein_count}]")
        if self.protein_atomic_numbers.dtype != torch.long:
            raise ValueError("protein atomic numbers must be torch.long")
        for name, value in (
            ("coordinates", self.protein_coords),
            ("UFF x", self.protein_uff_x),
            ("UFF d", self.protein_uff_d),
            ("vdW radius", self.protein_vdw_radius),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"protein {name} must be finite floating point")
        if bool((self.protein_vdw_radius <= 0).any()):
            raise ValueError("protein vdW radii must be positive")
        obstacle_count = int(self.geometry_obstacle_coords.shape[0])
        if self.geometry_obstacle_coords.shape != (obstacle_count, 3):
            raise ValueError("geometry obstacle coordinates must have shape [O,3]")
        for name, value in (
            ("atomic numbers", self.geometry_obstacle_atomic_numbers),
            ("UFF x", self.geometry_obstacle_uff_x),
            ("UFF d", self.geometry_obstacle_uff_d),
            ("generic mask", self.geometry_obstacle_is_generic),
        ):
            if value.shape != (obstacle_count,):
                raise ValueError(
                    f"geometry obstacle {name} must have shape [{obstacle_count}]"
                )
        if self.geometry_obstacle_atomic_numbers.dtype != torch.long:
            raise ValueError("geometry obstacle atomic numbers must be torch.long")
        if self.geometry_obstacle_is_generic.dtype != torch.bool:
            raise ValueError("geometry obstacle generic mask must be bool")
        for name, value in (
            ("coordinates", self.geometry_obstacle_coords),
            ("UFF x", self.geometry_obstacle_uff_x),
            ("UFF d", self.geometry_obstacle_uff_d),
        ):
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"geometry obstacle {name} must be finite floating point")
        if bool((self.geometry_obstacle_uff_x < 0).any()) or bool(
            (self.geometry_obstacle_uff_d < 0).any()
        ):
            raise ValueError("geometry obstacle UFF parameters must be non-negative")
        generic = self.geometry_obstacle_is_generic
        if bool(
            (
                generic
                & (
                    self.geometry_obstacle_uff_x.ne(0)
                    | self.geometry_obstacle_uff_d.ne(0)
                )
            ).any()
        ):
            raise ValueError("generic obstacles must not claim UFF parameters")
        if bool(
            (
                ~generic
                & (
                    self.geometry_obstacle_uff_x.le(0)
                    | self.geometry_obstacle_uff_d.le(0)
                )
            ).any()
        ):
            raise ValueError("UFF-style obstacles require positive UFF parameters")
        if len(self.geometry_obstacle_labels) != obstacle_count or len(
            self.geometry_obstacle_kinds
        ) != obstacle_count:
            raise ValueError("geometry obstacle labels and kinds must match obstacle count")
        if len(set(self.geometry_obstacle_labels)) != obstacle_count:
            raise ValueError("geometry obstacle labels must be unique")
        if not isinstance(self.receptor_policy_identity, dict) or not isinstance(
            self.receptor_provenance, dict
        ):
            raise ValueError("receptor policy identity and provenance must be dictionaries")

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> PhysicalSystem:
        return PhysicalSystem(
            topology=self.topology.to(device, dtype),
            protein_coords=self.protein_coords.to(device=device, dtype=dtype),
            protein_atomic_numbers=self.protein_atomic_numbers.to(device=device),
            protein_uff_x=self.protein_uff_x.to(device=device, dtype=dtype),
            protein_uff_d=self.protein_uff_d.to(device=device, dtype=dtype),
            protein_vdw_radius=self.protein_vdw_radius.to(device=device, dtype=dtype),
            parameter_set=dict(self.parameter_set),
            protein_source_atoms=self.protein_source_atoms,
            protein_parameterized_source_atoms=self.protein_parameterized_source_atoms,
            excluded_nonprotein_atoms=self.excluded_nonprotein_atoms,
            excluded_nonprotein_residues=tuple(self.excluded_nonprotein_residues),
            geometry_obstacle_coords=self.geometry_obstacle_coords.to(
                device=device, dtype=dtype
            ),
            geometry_obstacle_atomic_numbers=self.geometry_obstacle_atomic_numbers.to(
                device=device, dtype=torch.long
            ),
            geometry_obstacle_uff_x=self.geometry_obstacle_uff_x.to(
                device=device, dtype=dtype
            ),
            geometry_obstacle_uff_d=self.geometry_obstacle_uff_d.to(
                device=device, dtype=dtype
            ),
            geometry_obstacle_is_generic=self.geometry_obstacle_is_generic.to(
                device=device, dtype=torch.bool
            ),
            geometry_obstacle_labels=tuple(self.geometry_obstacle_labels),
            geometry_obstacle_kinds=tuple(self.geometry_obstacle_kinds),
            receptor_policy_mode=self.receptor_policy_mode,
            receptor_policy_identity=dict(self.receptor_policy_identity),
            receptor_provenance=dict(self.receptor_provenance),
            receptor_policy=self.receptor_policy,
            interaction_topology=(
                None
                if self.interaction_topology is None
                else self.interaction_topology.to(device, dtype)
            ),
            interaction_parameter_set=(
                None
                if self.interaction_parameter_set is None
                else dict(self.interaction_parameter_set)
            ),
        )


def _smarts_atom_mask(mol: Chem.Mol, patterns: list[str], *, family: str) -> Tensor:
    mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    for pattern in patterns:
        query = Chem.MolFromSmarts(pattern)
        if query is None or query.GetNumAtoms() != 1:
            raise RuntimeError(
                f"interaction-v1 {family} SMARTS must compile to one query atom: {pattern!r}"
            )
        for match in mol.GetSubstructMatches(query, uniquify=True):
            mask[match[0]] = True
    return mask


def _empty_site_membership(n_atoms: int) -> Tensor:
    return torch.empty((0, n_atoms), dtype=torch.float64)


def _canonical_ring_cycle(indices: tuple[int, ...]) -> tuple[int, ...]:
    """Return one deterministic orientation of an already cyclic ring."""
    if not indices:
        return ()
    variants: list[tuple[int, ...]] = []
    for direction in (indices, tuple(reversed(indices))):
        for offset in range(len(direction)):
            variants.append(direction[offset:] + direction[:offset])
    return min(variants)


def _ring_triplet(cycle: tuple[int, ...]) -> tuple[int, int, int]:
    return cycle[0], cycle[len(cycle) // 3], cycle[(2 * len(cycle)) // 3]


def _ligand_ring_reference_area(
    mol: Chem.Mol,
    triplet: tuple[int, int, int],
) -> float | None:
    """Return twice the anchor-triangle area, or a generic graph-only prior."""
    if not mol.GetNumConformers():
        return 1.0
    conformer = mol.GetConformer()
    points = torch.tensor(
        [list(conformer.GetAtomPosition(atom_index)) for atom_index in triplet],
        dtype=torch.float64,
    )
    area = float(torch.linalg.cross(points[1] - points[0], points[2] - points[0]).norm())
    if not math.isfinite(area) or area <= 1e-8:
        return None
    return area


def _compact_fused_system_ids(cycles: list[tuple[int, ...]]) -> list[int]:
    """Group elementary aromatic rings that share an aromatic bond."""
    parent = list(range(len(cycles)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(cycles)):
        left_atoms = set(cycles[left])
        for right in range(left + 1, len(cycles)):
            if len(left_atoms.intersection(cycles[right])) >= 2:
                union(left, right)
    roots = [find(index) for index in range(len(cycles))]
    root_to_system = {root: system_index for system_index, root in enumerate(sorted(set(roots)))}
    return [root_to_system[root] for root in roots]


def _ligand_aromatic_rings(
    mol: Chem.Mol,
) -> dict[str, Tensor | tuple[str, ...]]:
    """Type neutral five- and six-member ligand aromatic rings once with RDKit."""
    n_atoms = mol.GetNumAtoms()
    admitted_cycles: list[tuple[int, ...]] = []
    reference_areas: list[float] = []
    cation_pi_acceptors: list[bool] = []
    labels: list[str] = []
    exclusions: list[str] = []

    cycles = sorted(
        {
            _canonical_ring_cycle(tuple(int(index) for index in ring))
            for ring in Chem.GetSymmSSSR(mol)
        }
    )
    for cycle in cycles:
        ring_label = ",".join(map(str, cycle))
        if len(cycle) not in {5, 6}:
            exclusions.append(f"ligand:ring:{ring_label}:unsupported_size={len(cycle)}")
            continue
        bonds = [
            mol.GetBondBetweenAtoms(cycle[index], cycle[(index + 1) % len(cycle)])
            for index in range(len(cycle))
        ]
        if (
            any(bond is None for bond in bonds)
            or not all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in cycle)
            or not all(bool(bond.GetIsAromatic()) for bond in bonds if bond is not None)
        ):
            exclusions.append(f"ligand:ring:{ring_label}:not_fully_aromatic")
            continue
        charged_atoms = [
            (index, mol.GetAtomWithIdx(index).GetFormalCharge())
            for index in cycle
            if mol.GetAtomWithIdx(index).GetFormalCharge() != 0
        ]
        if charged_atoms:
            exclusions.append(
                f"ligand:ring:{ring_label}:nonzero_formal_charge_atoms="
                + ",".join(f"{index}:{charge:+d}" for index, charge in charged_atoms)
            )
            continue
        triplet = _ring_triplet(cycle)
        reference_area = _ligand_ring_reference_area(mol, triplet)
        if reference_area is None:
            exclusions.append(f"ligand:ring:{ring_label}:degenerate_reference_area")
            continue
        admitted_cycles.append(cycle)
        reference_areas.append(reference_area)
        cation_pi_acceptors.append(
            all(mol.GetAtomWithIdx(index).GetAtomicNum() == 6 for index in cycle)
        )
        labels.append(f"ligand:ring:{ring_label}")

    membership = torch.zeros(
        (len(admitted_cycles), n_atoms),
        dtype=torch.float64,
    )
    for ring_index, cycle in enumerate(admitted_cycles):
        membership[ring_index, list(cycle)] = 1.0 / len(cycle)
    return {
        "membership": membership,
        "triplet": (
            torch.tensor(
                [_ring_triplet(cycle) for cycle in admitted_cycles],
                dtype=torch.long,
            )
            if admitted_cycles
            else torch.empty((0, 3), dtype=torch.long)
        ),
        "system": torch.tensor(
            _compact_fused_system_ids(admitted_cycles),
            dtype=torch.long,
        ),
        "reference_area": torch.tensor(reference_areas, dtype=torch.float64),
        "is_cation_pi_acceptor": torch.tensor(cation_pi_acceptors, dtype=torch.bool),
        "labels": tuple(labels),
        "exclusion_labels": tuple(exclusions),
    }


def _compile_mapped_pair_smarts(
    definition: dict,
    *,
    family: str,
    required_map_fields: tuple[str, ...],
) -> tuple[Chem.Mol, dict[str, int]]:
    """Compile one versioned mapped SMARTS and resolve role maps to query atoms."""
    name = str(definition.get("name", "")).strip()
    pattern = str(definition.get("smarts", "")).strip()
    if not name or not pattern:
        raise RuntimeError(f"interaction-v1 {family} definitions require name and SMARTS")
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise RuntimeError(f"interaction-v1 {family} SMARTS failed to compile: {name!r}")
    map_to_query_index: dict[int, int] = {}
    for atom in query.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if not atom_map:
            continue
        if atom_map in map_to_query_index:
            raise RuntimeError(
                f"interaction-v1 {family} SMARTS has duplicate atom map {atom_map}: {name!r}"
            )
        map_to_query_index[atom_map] = atom.GetIdx()
    resolved: dict[str, int] = {}
    for field_name in required_map_fields:
        atom_map = definition.get(field_name)
        if not isinstance(atom_map, int) or atom_map <= 0:
            raise RuntimeError(
                f"interaction-v1 {family} {field_name} must be a positive integer: {name!r}"
            )
        if atom_map not in map_to_query_index:
            raise RuntimeError(
                f"interaction-v1 {family} {field_name}={atom_map} is absent from SMARTS: {name!r}"
            )
        resolved[field_name] = map_to_query_index[atom_map]
    if len(set(resolved.values())) != len(resolved):
        raise RuntimeError(
            f"interaction-v1 {family} role maps must identify distinct atoms: {name!r}"
        )
    return query, resolved


def _ligand_halogen_donors(
    mol: Chem.Mol,
    definitions: list[dict],
) -> dict[str, Tensor | tuple[str, ...]]:
    if not definitions or not all(isinstance(definition, dict) for definition in definitions):
        raise RuntimeError("interaction-v1 halogen_donor must be a non-empty definition list")
    matched_pairs: set[tuple[int, int]] = set()
    seen_names: set[str] = set()
    for definition in definitions:
        name = str(definition.get("name", "")).strip()
        if name in seen_names:
            raise RuntimeError(f"interaction-v1 duplicate halogen-donor definition: {name!r}")
        seen_names.add(name)
        query, roles = _compile_mapped_pair_smarts(
            definition,
            family="halogen_donor",
            required_map_fields=("parent_atom_map", "donor_atom_map"),
        )
        for match in mol.GetSubstructMatches(query, uniquify=True):
            parent_index = match[roles["parent_atom_map"]]
            donor_index = match[roles["donor_atom_map"]]
            matched_pairs.add((donor_index, parent_index))
    ordered_pairs = sorted(matched_pairs)
    matched_donors = {donor for donor, _ in ordered_pairs}
    exclusions = tuple(
        f"ligand:{atom.GetIdx()}:{atom.GetSymbol()}:"
        "not_matched_by_versioned_neutral_single_C-X_policy"
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in {17, 35, 53} and atom.GetIdx() not in matched_donors
    )
    return {
        "donor_index": torch.tensor(
            [donor for donor, _ in ordered_pairs],
            dtype=torch.long,
        ),
        "parent_index": torch.tensor(
            [parent for _, parent in ordered_pairs],
            dtype=torch.long,
        ),
        "exclusion_labels": exclusions,
    }


def _ligand_zinc_donors(
    mol: Chem.Mol,
    ligand_chemical_acceptor: Tensor,
    sulfur_definitions: list[dict],
) -> dict[str, Tensor | tuple[str, ...]]:
    """Type conservative N/O acceptors and versioned mapped sulfur donors."""
    if not sulfur_definitions or not all(
        isinstance(definition, dict) for definition in sulfur_definitions
    ):
        raise RuntimeError("interaction-v1 metal_sulfur_donor must be a non-empty definition list")
    sulfur_mask = torch.zeros(mol.GetNumAtoms(), dtype=torch.bool)
    seen_names: set[str] = set()
    for definition in sulfur_definitions:
        name = str(definition.get("name", "")).strip()
        if name in seen_names:
            raise RuntimeError(f"interaction-v1 duplicate metal-S-donor definition: {name!r}")
        seen_names.add(name)
        query, roles = _compile_mapped_pair_smarts(
            definition,
            family="metal_sulfur_donor",
            required_map_fields=("donor_atom_map",),
        )
        for match in mol.GetSubstructMatches(query, uniquify=True):
            donor_index = match[roles["donor_atom_map"]]
            if mol.GetAtomWithIdx(donor_index).GetAtomicNum() != 16:
                raise RuntimeError(
                    f"interaction-v1 metal_sulfur_donor SMARTS mapped a non-sulfur atom: {name!r}"
                )
            sulfur_mask[donor_index] = True
    element = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long,
    )
    donor_mask = ligand_chemical_acceptor & torch.isin(
        element,
        torch.tensor([7, 8], dtype=torch.long),
    )
    donor_mask |= sulfur_mask & (element == 16)
    donor_index = donor_mask.nonzero(as_tuple=False).flatten()
    exclusions = tuple(
        f"ligand:{atom.GetIdx()}:{atom.GetSymbol()}:not_matched_by_versioned_zinc_donor_policy"
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in {7, 8, 16} and not bool(donor_mask[atom.GetIdx()])
    )
    return {
        "donor_index": donor_index,
        "donor_element": element[donor_index],
        "exclusion_labels": exclusions,
    }


def _ligand_charge_sites(
    mol: Chem.Mol,
    group_definitions: list[dict],
) -> dict[str, Tensor | tuple[str, ...]]:
    """Build resonance-aware formal-charge sites from the sanitized ligand."""
    n_atoms = mol.GetNumAtoms()
    atom_formal_charge = torch.tensor(
        [atom.GetFormalCharge() for atom in mol.GetAtoms()],
        dtype=torch.float64,
    )
    membership_rows: list[Tensor] = []
    site_charges: list[float] = []
    site_labels: list[str] = []
    exclusion_labels: list[str] = []
    consumed_members: set[int] = set()
    seen_candidates: set[tuple[str, tuple[int, ...]]] = set()

    for definition in group_definitions:
        name = str(definition["name"])
        query = Chem.MolFromSmarts(str(definition["smarts"]))
        if query is None:
            raise RuntimeError(f"interaction-v1 ligand charge SMARTS failed to compile: {name!r}")
        query_map_to_index = {
            atom.GetAtomMapNum(): atom.GetIdx() for atom in query.GetAtoms() if atom.GetAtomMapNum()
        }
        member_maps = tuple(int(value) for value in definition["member_atom_maps"])
        if not member_maps or len(set(member_maps)) != len(member_maps):
            raise RuntimeError(
                f"interaction-v1 ligand charge group {name!r} has invalid member maps"
            )
        missing_maps = sorted(set(member_maps) - set(query_map_to_index))
        if missing_maps:
            raise RuntimeError(
                f"interaction-v1 ligand charge group {name!r} is missing atom maps {missing_maps}"
            )
        expected_charge = float(definition["total_charge_e"])

        for match in mol.GetSubstructMatches(query, uniquify=True):
            members = tuple(sorted(match[query_map_to_index[atom_map]] for atom_map in member_maps))
            candidate_key = (name, members)
            if candidate_key in seen_candidates:
                continue
            seen_candidates.add(candidate_key)
            if consumed_members.intersection(members):
                # A more specific, earlier rule may intentionally subsume this
                # match (for example guanidinium before amidinium).
                continue

            matched_charge = float(atom_formal_charge[torch.tensor(match, dtype=torch.long)].sum())
            if not math.isclose(
                matched_charge,
                expected_charge,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                exclusion_labels.append(
                    f"ligand:{name}:{','.join(map(str, members))}:"
                    f"charge_mismatch={matched_charge:g}"
                )
                continue

            membership = torch.zeros(n_atoms, dtype=torch.float64)
            membership[list(members)] = 1.0 / len(members)
            membership_rows.append(membership)
            site_charges.append(expected_charge)
            site_labels.append(f"ligand:{name}:{','.join(map(str, members))}")
            consumed_members.update(members)

    for atom_index, formal_charge in enumerate(atom_formal_charge.tolist()):
        if formal_charge == 0.0 or atom_index in consumed_members:
            continue
        membership = torch.zeros(n_atoms, dtype=torch.float64)
        membership[atom_index] = 1.0
        membership_rows.append(membership)
        site_charges.append(float(formal_charge))
        atom = mol.GetAtomWithIdx(atom_index)
        site_labels.append(
            f"ligand:atom:{atom_index}:{atom.GetSymbol()}:formal_charge={formal_charge:+g}"
        )

    site_charge = torch.tensor(site_charges, dtype=torch.float64)
    molecular_charge = float(atom_formal_charge.sum())
    assigned_charge = float(site_charge.sum())
    if not math.isclose(
        assigned_charge,
        molecular_charge,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise AssertionError(
            "ligand formal-charge site construction did not conserve charge: "
            f"input={molecular_charge:g}, assigned={assigned_charge:g}"
        )

    return {
        "membership": (
            torch.stack(membership_rows) if membership_rows else _empty_site_membership(n_atoms)
        ),
        "charge": site_charge,
        "labels": tuple(site_labels),
        "exclusion_labels": tuple(exclusion_labels),
    }


def _protein_charge_sites(
    protein_atoms: list,
    keep: Tensor,
    group_definitions: list[dict],
) -> dict[str, Tensor | tuple[str, ...]]:
    """Build complete, shell-safe canonical protein formal-charge sites."""
    if keep.shape != (len(protein_atoms),):
        raise ValueError("protein charge-site keep mask must match protein atoms")
    kept_full_indices = keep.nonzero(as_tuple=False).flatten().tolist()
    full_to_kept = {
        full_index: kept_index for kept_index, full_index in enumerate(kept_full_indices)
    }
    n_kept = len(kept_full_indices)

    rules_by_normalized_residue: dict[str, list[dict]] = {}
    for definition in group_definitions:
        normalized = str(definition["normalized_residue_name"])
        rules_by_normalized_residue.setdefault(
            normalized,
            [],
        ).append(definition)

    residue_atoms: dict[tuple[str, int, str], list[int]] = {}
    for atom_index, atom in enumerate(protein_atoms):
        residue_atoms.setdefault(
            (atom.chain, atom.res_num, atom.icode),
            [],
        ).append(atom_index)

    membership_rows: list[Tensor] = []
    site_charges: list[float] = []
    site_labels: list[str] = []
    exclusion_labels: list[str] = []
    consumed_full_indices: set[int] = set()

    for residue_indices in residue_atoms.values():
        residue = protein_atoms[residue_indices[0]]
        normalized = residue.res_name
        definitions = rules_by_normalized_residue.get(normalized)
        if not definitions:
            continue
        raw_residue = residue.raw_res_name or normalized
        residue_label = f"{residue.chain}:{raw_residue}{residue.res_num}{residue.icode}"
        if any(
            (protein_atoms[index].raw_res_name or protein_atoms[index].res_name) != raw_residue
            or protein_atoms[index].res_name != normalized
            for index in residue_indices
        ):
            exclusion_labels.append(
                f"protein:{residue_label}:{normalized}:inconsistent_residue_identity"
            )
            continue

        matching_definitions = [
            definition
            for definition in definitions
            if raw_residue in {str(value) for value in definition["raw_residue_names"]}
        ]
        if not matching_definitions:
            exclusion_labels.append(
                f"protein:{residue_label}:{normalized}:charge_state_not_admitted"
            )
            continue

        atom_name_to_indices: dict[str, list[int]] = {}
        for atom_index in residue_indices:
            atom_name_to_indices.setdefault(
                protein_atoms[atom_index].atom_name,
                [],
            ).append(atom_index)

        for definition in matching_definitions:
            member_names = tuple(str(value) for value in definition["member_atom_names"])
            if not member_names or len(set(member_names)) != len(member_names):
                raise RuntimeError(
                    "interaction-v1 protein charge group has invalid member "
                    f"atom names for {normalized}"
                )
            missing_or_duplicate = [
                name for name in member_names if len(atom_name_to_indices.get(name, ())) != 1
            ]
            if missing_or_duplicate:
                exclusion_labels.append(
                    f"protein:{residue_label}:{normalized}:"
                    "missing_or_duplicate_members="
                    f"{','.join(missing_or_duplicate)}"
                )
                continue
            full_members = tuple(atom_name_to_indices[name][0] for name in member_names)
            if consumed_full_indices.intersection(full_members):
                exclusion_labels.append(f"protein:{residue_label}:{normalized}:overlapping_members")
                continue
            outside_shell = [
                name
                for name, full_index in zip(
                    member_names,
                    full_members,
                    strict=True,
                )
                if full_index not in full_to_kept
            ]
            if outside_shell:
                exclusion_labels.append(
                    f"protein:{residue_label}:{normalized}:"
                    f"members_outside_shell={','.join(outside_shell)}"
                )
                continue

            membership = torch.zeros(
                n_kept,
                dtype=torch.float64,
            )
            membership[[full_to_kept[index] for index in full_members]] = 1.0 / len(full_members)
            membership_rows.append(membership)
            site_charges.append(float(definition["total_charge_e"]))
            site_labels.append(f"protein:{residue_label}:{normalized}:{'+'.join(member_names)}")
            consumed_full_indices.update(full_members)

    return {
        "membership": (
            torch.stack(membership_rows) if membership_rows else _empty_site_membership(n_kept)
        ),
        "charge": torch.tensor(site_charges, dtype=torch.float64),
        "labels": tuple(site_labels),
        "exclusion_labels": tuple(exclusion_labels),
    }


def _protein_aromatic_rings(
    protein_atoms: list,
    protein_coords: Tensor,
    keep: Tensor,
) -> dict[str, Tensor | tuple[str, ...]]:
    """Build canonical, shell-complete receptor aromatic-ring sites."""
    kept_full_indices = keep.nonzero(as_tuple=False).flatten().tolist()
    full_to_kept = {
        full_index: kept_index for kept_index, full_index in enumerate(kept_full_indices)
    }
    n_kept = len(kept_full_indices)
    residue_atoms: dict[tuple[str, int, str], list[int]] = {}
    for atom_index, atom in enumerate(protein_atoms):
        residue_atoms.setdefault(
            (atom.chain, atom.res_num, atom.icode),
            [],
        ).append(atom_index)

    admitted_members: list[tuple[int, ...]] = []
    admitted_triplets: list[tuple[int, int, int]] = []
    admitted_system_keys: list[tuple[str, int, str]] = []
    reference_areas: list[float] = []
    cation_pi_acceptors: list[bool] = []
    labels: list[str] = []
    exclusions: list[str] = []

    for residue_key, residue_indices in sorted(residue_atoms.items()):
        residue = protein_atoms[residue_indices[0]]
        templates = _PROTEIN_AROMATIC_RING_TEMPLATES.get(residue.res_name)
        if templates is None:
            continue
        raw_residue = residue.raw_res_name or residue.res_name
        residue_label = f"{residue.chain}:{raw_residue}{residue.res_num}{residue.icode}"
        if raw_residue not in _PROTEIN_AROMATIC_RAW_NAMES[residue.res_name]:
            exclusions.append(
                f"protein:{residue_label}:{residue.res_name}:aromatic_state_not_admitted"
            )
            continue
        atom_name_to_indices: dict[str, list[int]] = {}
        for atom_index in residue_indices:
            atom_name_to_indices.setdefault(
                protein_atoms[atom_index].atom_name,
                [],
            ).append(atom_index)
        for ring_name, member_names, is_cation_pi_acceptor in templates:
            missing_or_duplicate = [
                name for name in member_names if len(atom_name_to_indices.get(name, ())) != 1
            ]
            if missing_or_duplicate:
                exclusions.append(
                    f"protein:{residue_label}:{ring_name}:missing_or_duplicate_members="
                    f"{','.join(missing_or_duplicate)}"
                )
                continue
            full_members = tuple(atom_name_to_indices[name][0] for name in member_names)
            outside_shell = [
                name
                for name, full_index in zip(member_names, full_members, strict=True)
                if full_index not in full_to_kept
            ]
            if outside_shell:
                exclusions.append(
                    f"protein:{residue_label}:{ring_name}:members_outside_shell="
                    f"{','.join(outside_shell)}"
                )
                continue
            full_triplet = _ring_triplet(full_members)
            points = protein_coords[torch.tensor(full_triplet, dtype=torch.long)]
            reference_area = float(
                torch.linalg.cross(points[1] - points[0], points[2] - points[0]).norm()
            )
            if not math.isfinite(reference_area) or reference_area <= 1e-8:
                exclusions.append(f"protein:{residue_label}:{ring_name}:degenerate_reference_area")
                continue
            kept_members = tuple(full_to_kept[index] for index in full_members)
            admitted_members.append(kept_members)
            admitted_triplets.append(tuple(full_to_kept[index] for index in full_triplet))
            admitted_system_keys.append(residue_key)
            reference_areas.append(reference_area)
            cation_pi_acceptors.append(is_cation_pi_acceptor)
            labels.append(f"protein:{residue_label}:{ring_name}:{'+'.join(member_names)}")

    membership = torch.zeros(
        (len(admitted_members), n_kept),
        dtype=torch.float64,
    )
    for ring_index, members in enumerate(admitted_members):
        membership[ring_index, list(members)] = 1.0 / len(members)
    unique_system_keys = sorted(set(admitted_system_keys))
    system_by_key = {key: system_index for system_index, key in enumerate(unique_system_keys)}
    return {
        "membership": membership,
        "triplet": (
            torch.tensor(admitted_triplets, dtype=torch.long)
            if admitted_triplets
            else torch.empty((0, 3), dtype=torch.long)
        ),
        "system": torch.tensor(
            [system_by_key[key] for key in admitted_system_keys],
            dtype=torch.long,
        ),
        "reference_area": torch.tensor(reference_areas, dtype=torch.float64),
        "is_cation_pi_acceptor": torch.tensor(cation_pi_acceptors, dtype=torch.bool),
        "labels": tuple(labels),
        "exclusion_labels": tuple(exclusions),
    }


def _protein_halogen_acceptors(
    protein_atoms: list,
    keep: Tensor,
    *,
    chemical_acceptor: Tensor,
    admitted_acceptor: Tensor,
    direction_valid: Tensor,
    geometry_valid: Tensor,
) -> dict[str, Tensor | tuple[str, ...]]:
    """Type strict receptor N/O acceptors plus complete neutral MET SD."""
    kept_full_indices = keep.nonzero(as_tuple=False).flatten().tolist()
    full_to_kept = {
        full_index: kept_index for kept_index, full_index in enumerate(kept_full_indices)
    }
    accepted_full = set(admitted_acceptor.nonzero(as_tuple=False).flatten().tolist())
    exclusions: list[str] = []
    for atom_index in (
        (chemical_acceptor & ~admitted_acceptor).nonzero(as_tuple=False).flatten().tolist()
    ):
        atom = protein_atoms[atom_index]
        if atom.element.upper() not in {"N", "O"}:
            continue
        exclusions.append(
            f"protein:{atom.chain}:{atom.raw_res_name or atom.res_name}"
            f"{atom.res_num}{atom.icode}:{atom.atom_name}:"
            "halogen_acceptor_geometry_not_admitted"
        )

    residue_atoms: dict[tuple[str, int, str], list[int]] = {}
    for atom_index, atom in enumerate(protein_atoms):
        if atom.res_name == "MET":
            residue_atoms.setdefault(
                (atom.chain, atom.res_num, atom.icode),
                [],
            ).append(atom_index)
    for residue_indices in residue_atoms.values():
        residue = protein_atoms[residue_indices[0]]
        raw_residue = residue.raw_res_name or residue.res_name
        residue_label = f"{residue.chain}:{raw_residue}{residue.res_num}{residue.icode}"
        if raw_residue != "MET":
            exclusions.append(f"protein:{residue_label}:MET:variant_not_admitted")
            continue
        by_name: dict[str, list[int]] = {}
        for atom_index in residue_indices:
            by_name.setdefault(protein_atoms[atom_index].atom_name, []).append(atom_index)
        missing_or_duplicate = [
            name for name in ("CG", "SD", "CE") if len(by_name.get(name, ())) != 1
        ]
        if missing_or_duplicate:
            exclusions.append(
                f"protein:{residue_label}:MET:missing_or_duplicate_members="
                f"{','.join(missing_or_duplicate)}"
            )
            continue
        full_indices = {name: by_name[name][0] for name in ("CG", "SD", "CE")}
        outside_shell = [
            name for name, full_index in full_indices.items() if full_index not in full_to_kept
        ]
        if outside_shell:
            exclusions.append(
                f"protein:{residue_label}:MET:members_outside_shell={','.join(outside_shell)}"
            )
            continue
        sulfur_index = full_indices["SD"]
        if not bool(direction_valid[sulfur_index] and geometry_valid[sulfur_index]):
            exclusions.append(f"protein:{residue_label}:MET:SD_direction_geometry_not_admitted")
            continue
        accepted_full.add(sulfur_index)

    accepted_kept = sorted(full_to_kept[index] for index in accepted_full if index in full_to_kept)
    return {
        "acceptor_index": torch.tensor(accepted_kept, dtype=torch.long),
        "exclusion_labels": tuple(exclusions),
    }


def _ligand_neighbor_index(mol: Chem.Mol) -> Tensor:
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        atom_i = bond.GetBeginAtomIdx()
        atom_j = bond.GetEndAtomIdx()
        if (
            mol.GetAtomWithIdx(atom_i).GetAtomicNum() == 1
            or mol.GetAtomWithIdx(atom_j).GetAtomicNum() == 1
        ):
            continue
        edges.extend(((atom_i, atom_j), (atom_j, atom_i)))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).T.contiguous()


def _missing_valence_cone_target(
    heavy_degree: int,
    valence_capacity: int | None,
) -> tuple[float, bool]:
    """Idealized cosine between the outward axis and a missing valence site."""
    if valence_capacity is None or heavy_degree <= 0 or heavy_degree >= valence_capacity:
        return 0.0, False
    # Regular linear/trigonal/tetrahedral valence sites have pairwise
    # direction dot product -1/(m-1).  Projecting any missing site onto the
    # negative sum of k observed bond directions gives this closed form.
    denominator = math.sqrt(heavy_degree * (1.0 - (heavy_degree - 1.0) / (valence_capacity - 1.0)))
    target = heavy_degree / (valence_capacity - 1.0) / denominator
    return float(target), True


def _direction_quality(
    axis_norm: Tensor,
    zero_below: float,
    full_above: float,
) -> Tensor:
    scaled = ((axis_norm - float(zero_below)) / (float(full_above) - float(zero_below))).clamp(
        0.0, 1.0
    )
    result = scaled.pow(3) * (10.0 - 15.0 * scaled + 6.0 * scaled.square())
    return result.clamp(0.0, 1.0)


def _ligand_valence_capacity(atom: Chem.Atom) -> int | None:
    hybridization = atom.GetHybridization()
    if hybridization == Chem.HybridizationType.SP:
        return 2
    if hybridization == Chem.HybridizationType.SP2:
        return 3
    if hybridization == Chem.HybridizationType.SP3:
        return 4
    return None


def type_ligand_interactions(
    mol: Chem.Mol,
) -> dict[str, Tensor | tuple[str, ...]]:
    """Strictly type ligand interaction sites from versioned local SMARTS."""
    raw = load_interaction_v1()
    patterns = raw["ligand_smarts"]
    ligand_donor = _smarts_atom_mask(
        mol,
        patterns["donor"],
        family="donor",
    )
    ligand_acceptor = _smarts_atom_mask(
        mol,
        patterns["acceptor"],
        family="acceptor",
    )
    ligand_hydrophobe = _smarts_atom_mask(
        mol,
        patterns["hydrophobe"],
        family="hydrophobe",
    )
    ligand_elements = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long,
    )
    supported_hbond_elements = torch.tensor([7, 8], dtype=torch.long)
    ligand_supported_hbond = torch.isin(
        ligand_elements,
        supported_hbond_elements,
    )
    ligand_donor &= ligand_supported_hbond
    ligand_acceptor &= ligand_supported_hbond
    ligand_chemical_acceptor = ligand_acceptor.clone()
    ligand_hydrophobe &= ligand_elements == 6
    ligand_neighbors = _ligand_neighbor_index(mol)
    ligand_degree = torch.zeros(mol.GetNumAtoms(), dtype=torch.long)
    if ligand_neighbors.numel():
        ligand_degree.index_add_(
            0,
            ligand_neighbors[0],
            torch.ones(ligand_neighbors.shape[1], dtype=torch.long),
        )
    target_and_valid = [
        _missing_valence_cone_target(
            int(ligand_degree[index]),
            _ligand_valence_capacity(atom),
        )
        for index, atom in enumerate(mol.GetAtoms())
    ]
    direction_target_cosine = torch.tensor(
        [item[0] for item in target_and_valid],
        dtype=torch.float64,
    )
    direction_geometry_valid = torch.tensor(
        [item[1] for item in target_and_valid],
        dtype=torch.bool,
    )
    geometry_excluded_hbond_site = (ligand_donor | ligand_acceptor) & ~direction_geometry_valid
    ligand_donor &= direction_geometry_valid
    ligand_acceptor &= direction_geometry_valid
    charge_sites = _ligand_charge_sites(
        mol,
        raw["ligand_delocalized_charge_groups"],
    )
    aromatic_rings = _ligand_aromatic_rings(mol)
    pair_smarts = raw["ligand_pair_smarts"]
    halogen_donors = _ligand_halogen_donors(
        mol,
        pair_smarts["halogen_donor"],
    )
    zinc_donors = _ligand_zinc_donors(
        mol,
        ligand_chemical_acceptor,
        pair_smarts["metal_sulfur_donor"],
    )
    return {
        "neighbor_index": ligand_neighbors,
        "direction_target_cosine": direction_target_cosine,
        "direction_geometry_valid": direction_geometry_valid,
        "is_donor": ligand_donor,
        "is_acceptor": ligand_acceptor,
        "is_hydrophobe": ligand_hydrophobe,
        "is_geometry_excluded_hbond_site": (geometry_excluded_hbond_site),
        "atom_labels": tuple(f"{atom.GetIdx()}:{atom.GetSymbol()}" for atom in mol.GetAtoms()),
        "charge_site_membership": charge_sites["membership"],
        "charge_site_charge": charge_sites["charge"],
        "charge_site_labels": charge_sites["labels"],
        "charge_site_exclusion_labels": charge_sites["exclusion_labels"],
        "aromatic_ring_membership": aromatic_rings["membership"],
        "aromatic_ring_triplet": aromatic_rings["triplet"],
        "aromatic_ring_system": aromatic_rings["system"],
        "aromatic_ring_reference_area": aromatic_rings["reference_area"],
        "aromatic_ring_is_cation_pi_acceptor": aromatic_rings["is_cation_pi_acceptor"],
        "aromatic_ring_labels": aromatic_rings["labels"],
        "aromatic_ring_exclusion_labels": aromatic_rings["exclusion_labels"],
        "halogen_donor_index": halogen_donors["donor_index"],
        "halogen_parent_index": halogen_donors["parent_index"],
        "halogen_exclusion_labels": halogen_donors["exclusion_labels"],
        "zinc_donor_index": zinc_donors["donor_index"],
        "zinc_donor_element": zinc_donors["donor_element"],
        "zinc_donor_exclusion_labels": zinc_donors["exclusion_labels"],
    }


def _protein_valence_geometry(
    res_name: str,
    atom_name: str,
    element: str,
) -> tuple[int, int] | None:
    """Return ``(valence capacity, expected heavy degree)`` for admitted sites."""
    element = element.upper()
    if element == "N":
        if atom_name == "N":
            # Only an internal peptide N is admitted. A free terminus or chain
            # gap has a different/unknown protonation geometry and fails closed.
            return 3, 2
        return {
            ("ARG", "NE"): (3, 2),
            ("ARG", "NH1"): (3, 1),
            ("ARG", "NH2"): (3, 1),
            ("ASN", "ND2"): (3, 1),
            ("GLN", "NE2"): (3, 1),
            ("HIS", "ND1"): (3, 2),
            ("HIS", "NE2"): (3, 2),
            ("LYS", "NZ"): (4, 1),
            ("TRP", "NE1"): (3, 2),
        }.get((res_name, atom_name))
    if element == "O":
        if atom_name == "O":
            return 3, 1
        if (res_name, atom_name) in {
            ("SER", "OG"),
            ("THR", "OG1"),
        }:
            return 4, 1
        if (res_name, atom_name) in {
            ("ASN", "OD1"),
            ("ASP", "OD1"),
            ("ASP", "OD2"),
            ("GLN", "OE1"),
            ("GLU", "OE1"),
            ("GLU", "OE2"),
            ("TYR", "OH"),
        }:
            return 3, 1
    if element == "S" and (res_name, atom_name) == ("MET", "SD"):
        return 4, 2
    return None


def _build_interaction_topology(
    mol: Chem.Mol,
    protein_atoms: list,
    protein_coords: Tensor,
    keep: Tensor,
) -> InteractionTopology:
    raw = load_interaction_v1()
    ligand_typing = type_ligand_interactions(mol)
    protein_charge_sites = _protein_charge_sites(
        protein_atoms,
        keep,
        raw["protein_charge_groups"],
    )

    atom_lookup = {
        ((atom.chain, atom.res_num, atom.icode), atom.atom_name): index
        for index, atom in enumerate(protein_atoms)
    }
    protein_bond_src, protein_bond_dst = _build_protein_bonds(
        protein_atoms,
        atom_lookup,
    )
    direction_sum = torch.zeros_like(protein_coords)
    site_bond_quality = torch.ones(
        len(protein_atoms),
        dtype=protein_coords.dtype,
    )
    if protein_bond_src:
        source = torch.tensor(protein_bond_src, dtype=torch.long)
        target = torch.tensor(protein_bond_dst, dtype=torch.long)
        bond_direction = protein_coords[source] - protein_coords[target]
        bond_length = bond_direction.norm(
            dim=-1,
            keepdim=True,
        )
        bond_quality = _direction_quality(
            bond_length.squeeze(-1),
            float(raw["defaults"]["bond_direction_quality_zero_below_angstrom"]),
            float(raw["defaults"]["bond_direction_quality_full_above_angstrom"]),
        )
        bond_direction = bond_direction / bond_length.clamp_min(
            float(raw["defaults"]["direction_epsilon_angstrom"])
        )
        bond_direction = bond_direction * bond_quality.unsqueeze(-1)
        direction_sum.index_add_(
            0,
            source,
            bond_direction,
        )
        site_bond_quality = site_bond_quality.scatter_reduce(
            0,
            source,
            bond_quality,
            reduce="prod",
            include_self=True,
        )
    direction_norm = direction_sum.norm(dim=-1)
    direction_epsilon = float(raw["defaults"]["direction_epsilon_angstrom"])
    quality_zero = float(raw["defaults"]["direction_quality_zero_below"])
    quality_full = float(raw["defaults"]["direction_quality_full_above"])
    direction_quality = (
        _direction_quality(
            direction_norm,
            quality_zero,
            quality_full,
        )
        * site_bond_quality
    )
    direction_valid = (direction_norm > quality_zero) & (site_bond_quality > 0)
    outward_direction = direction_sum / direction_norm.clamp_min(direction_epsilon).unsqueeze(-1)
    protein_degree = torch.zeros(len(protein_atoms), dtype=torch.long)
    if protein_bond_src:
        protein_degree.index_add_(
            0,
            torch.tensor(protein_bond_src, dtype=torch.long),
            torch.ones(len(protein_bond_src), dtype=torch.long),
        )
    protein_target_and_valid: list[tuple[float, bool]] = []
    for index, atom in enumerate(protein_atoms):
        geometry = _protein_valence_geometry(
            atom.res_name,
            atom.atom_name,
            atom.element,
        )
        if geometry is None:
            protein_target_and_valid.append((0.0, False))
            continue
        capacity, expected_degree = geometry
        actual_degree = int(protein_degree[index])
        if actual_degree != expected_degree:
            protein_target_and_valid.append((0.0, False))
            continue
        protein_target_and_valid.append(_missing_valence_cone_target(actual_degree, capacity))
    protein_direction_target_cosine = torch.tensor(
        [item[0] for item in protein_target_and_valid],
        dtype=torch.float64,
    )
    protein_geometry_valid = torch.tensor(
        [item[1] for item in protein_target_and_valid],
        dtype=torch.bool,
    )

    protein_flags = [_patom_pharmacophore(atom.res_name, atom.atom_name) for atom in protein_atoms]
    protein_donor = torch.tensor(
        [flag[0] for flag in protein_flags],
        dtype=torch.bool,
    )
    protein_acceptor = torch.tensor(
        [flag[1] for flag in protein_flags],
        dtype=torch.bool,
    )
    protein_hydrophobe = torch.tensor(
        [flag[4] for flag in protein_flags],
        dtype=torch.bool,
    )
    protein_hydrophobe &= torch.tensor(
        [atom.element.upper() == "C" for atom in protein_atoms],
        dtype=torch.bool,
    )
    supported_histidine_variants = {"HID", "HIE", "HIP"}
    unsupported_variant = torch.tensor(
        [
            (atom.raw_res_name or atom.res_name) != atom.res_name
            and (atom.raw_res_name or atom.res_name) not in supported_histidine_variants
            for atom in protein_atoms
        ],
        dtype=torch.bool,
    )
    protein_donor &= ~unsupported_variant
    protein_acceptor &= ~unsupported_variant
    protein_hydrophobe &= ~unsupported_variant
    ambiguous_histidine = torch.tensor(
        [
            atom.res_name == "HIS"
            and (atom.raw_res_name or atom.res_name) == "HIS"
            and atom.atom_name in {"ND1", "NE2"}
            for atom in protein_atoms
        ],
        dtype=torch.bool,
    )
    for index, atom in enumerate(protein_atoms):
        if atom.res_name != "HIS" or atom.atom_name not in {"ND1", "NE2"}:
            continue
        raw_residue = atom.raw_res_name or atom.res_name
        protein_donor[index] = (
            (raw_residue == "HID" and atom.atom_name == "ND1")
            or (raw_residue == "HIE" and atom.atom_name == "NE2")
            or raw_residue == "HIP"
        )
        protein_acceptor[index] = (raw_residue == "HID" and atom.atom_name == "NE2") or (
            raw_residue == "HIE" and atom.atom_name == "ND1"
        )
    protein_chemical_acceptor = protein_acceptor.clone()
    protein_supported_hbond = torch.tensor(
        [atom.element.upper() in {"N", "O"} for atom in protein_atoms],
        dtype=torch.bool,
    )
    hbond_geometry_valid = protein_supported_hbond & direction_valid & protein_geometry_valid
    geometry_excluded_hbond_site = (
        (protein_donor | protein_acceptor) & protein_supported_hbond & ~hbond_geometry_valid
    )
    protein_donor &= hbond_geometry_valid
    protein_acceptor &= hbond_geometry_valid
    protein_aromatic_rings = _protein_aromatic_rings(
        protein_atoms,
        protein_coords,
        keep,
    )
    protein_halogen_acceptors = _protein_halogen_acceptors(
        protein_atoms,
        keep,
        chemical_acceptor=protein_chemical_acceptor,
        admitted_acceptor=protein_acceptor,
        direction_valid=direction_valid,
        geometry_valid=protein_geometry_valid,
    )

    return InteractionTopology(
        ligand_neighbor_index=ligand_typing["neighbor_index"],
        ligand_direction_target_cosine=ligand_typing["direction_target_cosine"],
        ligand_direction_geometry_valid=ligand_typing["direction_geometry_valid"],
        ligand_is_donor=ligand_typing["is_donor"],
        ligand_is_acceptor=ligand_typing["is_acceptor"],
        ligand_is_hydrophobe=ligand_typing["is_hydrophobe"],
        ligand_is_geometry_excluded_hbond_site=ligand_typing["is_geometry_excluded_hbond_site"],
        protein_is_donor=protein_donor[keep],
        protein_is_acceptor=protein_acceptor[keep],
        protein_is_hydrophobe=protein_hydrophobe[keep],
        protein_outward_direction=outward_direction[keep],
        protein_direction_target_cosine=protein_direction_target_cosine[keep],
        protein_direction_quality=direction_quality[keep],
        protein_direction_valid=direction_valid[keep],
        protein_is_ambiguous_histidine=ambiguous_histidine[keep],
        protein_is_unsupported_variant=unsupported_variant[keep],
        protein_is_geometry_excluded_hbond_site=(geometry_excluded_hbond_site[keep]),
        ligand_atom_labels=ligand_typing["atom_labels"],
        protein_atom_labels=tuple(
            (
                f"{atom.chain}:{atom.raw_res_name or atom.res_name}"
                f"{atom.res_num}{atom.icode}:{atom.atom_name}"
            )
            for atom, selected in zip(
                protein_atoms,
                keep.tolist(),
                strict=True,
            )
            if selected
        ),
        ligand_charge_site_membership=ligand_typing["charge_site_membership"],
        ligand_charge_site_charge=ligand_typing["charge_site_charge"],
        ligand_charge_site_labels=ligand_typing["charge_site_labels"],
        ligand_charge_site_exclusion_labels=ligand_typing["charge_site_exclusion_labels"],
        protein_charge_site_membership=protein_charge_sites["membership"],
        protein_charge_site_charge=protein_charge_sites["charge"],
        protein_charge_site_labels=protein_charge_sites["labels"],
        protein_charge_site_exclusion_labels=protein_charge_sites["exclusion_labels"],
        ligand_aromatic_ring_membership=ligand_typing["aromatic_ring_membership"],
        ligand_aromatic_ring_triplet=ligand_typing["aromatic_ring_triplet"],
        ligand_aromatic_ring_system=ligand_typing["aromatic_ring_system"],
        ligand_aromatic_ring_reference_area=ligand_typing["aromatic_ring_reference_area"],
        ligand_aromatic_ring_is_cation_pi_acceptor=ligand_typing[
            "aromatic_ring_is_cation_pi_acceptor"
        ],
        ligand_aromatic_ring_labels=ligand_typing["aromatic_ring_labels"],
        ligand_aromatic_ring_exclusion_labels=ligand_typing["aromatic_ring_exclusion_labels"],
        protein_aromatic_ring_membership=protein_aromatic_rings["membership"],
        protein_aromatic_ring_triplet=protein_aromatic_rings["triplet"],
        protein_aromatic_ring_system=protein_aromatic_rings["system"],
        protein_aromatic_ring_reference_area=protein_aromatic_rings["reference_area"],
        protein_aromatic_ring_is_cation_pi_acceptor=protein_aromatic_rings["is_cation_pi_acceptor"],
        protein_aromatic_ring_labels=protein_aromatic_rings["labels"],
        protein_aromatic_ring_exclusion_labels=protein_aromatic_rings["exclusion_labels"],
        ligand_halogen_donor_index=ligand_typing["halogen_donor_index"],
        ligand_halogen_parent_index=ligand_typing["halogen_parent_index"],
        protein_halogen_acceptor_index=protein_halogen_acceptors["acceptor_index"],
        ligand_halogen_exclusion_labels=ligand_typing["halogen_exclusion_labels"],
        protein_halogen_exclusion_labels=protein_halogen_acceptors["exclusion_labels"],
        ligand_metal_donor_index=ligand_typing["zinc_donor_index"],
        ligand_metal_donor_element=ligand_typing["zinc_donor_element"],
        ligand_metal_donor_exclusion_labels=ligand_typing[
            "zinc_donor_exclusion_labels"
        ],
        metal_coords=torch.empty((0, 3), dtype=torch.float64),
        metal_atomic_number=torch.empty(0, dtype=torch.long),
        metal_vacant_direction=torch.empty((0, 3), dtype=torch.float64),
        metal_fixed_coordination=torch.empty(0, dtype=torch.long),
        metal_target_coordination=torch.empty(0, dtype=torch.long),
        metal_ligand_r0=torch.empty((0, 3), dtype=torch.float64),
        metal_ligand_donor_allowed=torch.empty((0, 3), dtype=torch.bool),
        metal_attraction_enabled=torch.empty(0, dtype=torch.bool),
        metal_site_labels=(),
        metal_profile_labels=(),
        metal_typing_exclusion_labels=(),
        ligand_zinc_donor_index=ligand_typing["zinc_donor_index"],
        ligand_zinc_donor_element=ligand_typing["zinc_donor_element"],
        ligand_zinc_donor_exclusion_labels=ligand_typing["zinc_donor_exclusion_labels"],
        zinc_coords=torch.empty((0, 3), dtype=torch.float64),
        zinc_vacant_direction=torch.empty((0, 3), dtype=torch.float64),
        zinc_receptor_donor_index=torch.empty((0, 3), dtype=torch.long),
        zinc_receptor_donor_element=torch.empty((0, 3), dtype=torch.long),
        zinc_site_labels=(),
        zinc_typing_exclusion_labels=(),
    )


def _within_shell(coords: Tensor, near: Tensor, cutoff: float, chunk_size: int = 1024) -> Tensor:
    minimum = torch.full((coords.shape[0],), torch.inf, dtype=coords.dtype)
    for start in range(0, near.shape[0], chunk_size):
        stop = min(start + chunk_size, near.shape[0])
        minimum = torch.minimum(
            minimum,
            torch.cdist(coords, near[start:stop]).amin(dim=1),
        )
    return minimum <= float(cutoff)


def _water_oxygen_coords(pdb_path: Path) -> Tensor:
    """Read only water oxygen coordinates for strict Zn-site admission.

    The shared protein parser intentionally discards water.  Zinc V0 must
    nevertheless fail closed when an unresolved coordination water occupies
    the fourth site, so this narrow read happens before that information is
    discarded.
    """
    coordinates: list[tuple[float, float, float]] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[16:17] not in {"", " ", "A"}:
            continue
        if line[17:20].strip().upper() not in WATER_RESIDUES:
            continue
        atom_name = line[12:16].strip().upper()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if element and element != "O":
            continue
        if not element and not atom_name.startswith("O"):
            continue
        try:
            coordinates.append(
                (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            )
        except ValueError:
            continue
    if not coordinates:
        return torch.empty((0, 3), dtype=torch.float64)
    return torch.tensor(coordinates, dtype=torch.float64)


def _nonprimary_altloc_contacts_near_zinc(
    pdb_path: Path,
    zinc_coords: Tensor,
) -> tuple[str, ...]:
    """Find raw B/C/... donor or water records hidden by the shared parser."""
    contacts: list[str] = []
    for line in pdb_path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        altloc = line[16:17].strip().upper()
        if altloc in {"", "A"}:
            continue
        residue = line[17:20].strip().upper()
        atom_name = line[12:16].strip().upper()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            element = atom_name.lstrip("0123456789")[:1]
        is_water_oxygen = residue in WATER_RESIDUES and (
            element == "O" or atom_name.startswith("O")
        )
        if is_water_oxygen:
            cutoff = 2.5
        elif element in {"N", "O"}:
            cutoff = 2.5
        elif element == "S":
            cutoff = 2.8
        else:
            continue
        try:
            coords = torch.tensor(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=torch.float64,
            )
        except ValueError:
            continue
        distance = float((coords - zinc_coords).norm())
        if distance > cutoff:
            continue
        chain = line[21:22].strip()
        residue_number = line[22:26].strip()
        insertion_code = line[26:27].strip()
        contacts.append(
            f"{line[:6].strip()}:{chain}:{residue}{residue_number}"
            f"{insertion_code}:{atom_name}:altloc={altloc}:distance={distance:.3f}"
        )
    return tuple(sorted(contacts))


def _raw_pdb_records(pdb_path: Path) -> list[dict[str, object]]:
    """Retain occupancy/altloc/water information discarded by the shared parser."""
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(pdb_path.read_text().splitlines(), start=1):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip().upper()
        residue = line[17:20].strip().upper()
        record_type = line[:6].strip()
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        if not element:
            stripped = atom_name.lstrip("0123456789")
            if record_type == "ATOM":
                # Protein atom name ``CA`` is alpha carbon, not calcium.
                element = stripped[:1]
            elif residue in _KNOWN_PDB_METAL_ELEMENTS:
                element = residue
            elif len(stripped) >= 2 and stripped[:2] in _SYMBOL_TO_Z:
                element = stripped[:2]
            else:
                element = stripped[:1]
        try:
            coords = torch.tensor(
                (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ),
                dtype=torch.float64,
            )
        except ValueError:
            continue
        occupancy_text = line[54:60].strip() if len(line) >= 60 else ""
        try:
            occupancy = float(occupancy_text) if occupancy_text else 1.0
        except ValueError:
            occupancy = float("nan")
        residue_number_text = line[22:26].strip()
        try:
            residue_number = int(residue_number_text)
        except ValueError:
            residue_number = 0
        records.append(
            {
                "line_number": line_number,
                "record_type": record_type,
                "atom_name": atom_name,
                "residue": residue,
                "chain": line[21:22],
                "residue_number": residue_number,
                "icode": line[26:27].strip(),
                "altloc": line[16:17].strip().upper(),
                "element": element,
                "coords": coords,
                "occupancy": occupancy,
                "is_water": residue in WATER_RESIDUES,
            }
        )
    return records


def _raw_record_site_key(record: dict[str, object]) -> tuple[str, str, str, int, str, str]:
    return (
        str(record["element"]).upper(),
        str(record["residue"]).upper(),
        str(record["chain"]),
        int(record["residue_number"]),
        str(record["icode"]),
        str(record["atom_name"]).upper(),
    )


def _raw_record_provenance(record: dict[str, object]) -> dict[str, object]:
    return {
        "line_number": int(record["line_number"]),
        "record_type": str(record["record_type"]),
        "element": str(record["element"]).upper(),
        "residue": str(record["residue"]),
        "chain": str(record["chain"]),
        "residue_number": int(record["residue_number"]),
        "icode": str(record["icode"]),
        "atom_name": str(record["atom_name"]),
        "altloc": str(record["altloc"]),
        "occupancy": float(record["occupancy"]),
    }


def _select_altloc_representative(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Select one unresolved site coordinate without averaging conformers."""
    if not records:
        raise ValueError("alternate-location group is empty")

    def priority(record: dict[str, object]) -> tuple[float, str, int]:
        occupancy = float(record["occupancy"])
        occupancy_priority = -occupancy if math.isfinite(occupancy) else math.inf
        return (
            occupancy_priority,
            str(record["altloc"]).upper(),
            int(record["line_number"]),
        )

    return min(records, key=priority)


def _reject_hidden_active_metal_altlocs(
    pdb_path: Path,
    near: Tensor,
    protein_cutoff: float,
) -> None:
    """Reject active metal records that the normalized parser would discard.

    The shared protein parser intentionally keeps only blank/A conformers.
    Guidance must inspect raw metal records first so a B/C-only pocket metal
    cannot disappear and silently bypass the metal interaction boundary.
    """
    raw_metals = [
        record
        for record in _raw_pdb_records(pdb_path)
        if str(record["element"]).upper() in _KNOWN_PDB_METAL_ELEMENTS
    ]
    if not raw_metals:
        return
    raw_coords = torch.stack([record["coords"] for record in raw_metals])
    active = _within_shell(raw_coords, near, protein_cutoff)
    hidden = [
        record
        for record, in_shell in zip(raw_metals, active.tolist(), strict=True)
        if in_shell and str(record["altloc"]).upper() not in {"", "A"}
    ]
    if not hidden:
        return
    elements = sorted({str(record["element"]).upper() for record in hidden})
    details = {
        "elements": elements,
        "records": [
            {
                "line_number": int(record["line_number"]),
                "residue": str(record["residue"]),
                "atom_name": str(record["atom_name"]),
                "altloc": str(record["altloc"]),
                "occupancy": float(record["occupancy"]),
            }
            for record in hidden
        ],
    }
    if elements == ["ZN"]:
        _unsupported_zinc_site(
            "active Zn record has a non-primary alternate location hidden by normalization",
            **details,
        )
    _unsupported_metal_profile(
        "active metal record has a non-primary alternate location hidden by normalization",
        **details,
    )


def _metal_site_label(atom) -> str:
    return (
        f"{atom.chain}:{atom.raw_res_name or atom.res_name}"
        f"{atom.res_num}{atom.icode}:{atom.atom_name}"
    )


def _unsupported_metal_profile(message: str, **details: object) -> None:
    raise UnsupportedPhysicalChemistryError(
        "unsupported_metal_profile",
        message,
        details=details,
    )


def _metal_coordination_profiles() -> dict[str, dict[str, object]]:
    raw_profiles = load_interaction_v1().get("metal_coordination_profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise RuntimeError(
            "interaction-v1 metal_coordination_profiles must be a non-empty object"
        )
    profiles: dict[str, dict[str, object]] = {}
    for raw_element, raw_profile in raw_profiles.items():
        element = str(raw_element).strip().upper()
        if (
            not element
            or not isinstance(raw_profile, dict)
            or element in profiles
        ):
            raise RuntimeError(
                "interaction-v1 metal_coordination_profiles contains an invalid "
                f"or duplicate entry: {raw_element!r}"
            )
        profile = dict(raw_profile)
        atomic_number = profile.get("atomic_number")
        identity_residue = str(profile.get("identity_residue", "")).strip().upper()
        auto_attraction = profile.get("auto_attraction")
        target_coordination = profile.get("target_coordination")
        geometries = profile.get("geometries")
        allowed_ligand = profile.get("allowed_ligand_elements")
        allowed_receptor = profile.get("allowed_receptor_elements")
        r0 = profile.get("r0_angstrom")
        cutoffs = profile.get("receptor_detection_cutoff_angstrom")
        water_policy = profile.get("water_policy")
        if (
            not isinstance(atomic_number, int)
            or atomic_number <= 0
            or _SYMBOL_TO_Z.get(element) != atomic_number
            or identity_residue != element
            or not isinstance(auto_attraction, bool)
            or not isinstance(target_coordination, int)
            or target_coordination < 0
            or not isinstance(geometries, list)
            or not isinstance(allowed_ligand, list)
            or not isinstance(allowed_receptor, list)
            or not isinstance(r0, dict)
            or not isinstance(cutoffs, dict)
            or water_policy not in {"reject", "admit_fixed_oxygen", "trace_only"}
        ):
            raise RuntimeError(
                f"interaction-v1 malformed metal coordination profile: {element}"
            )
        for name, values in (
            ("allowed_ligand_elements", allowed_ligand),
            ("allowed_receptor_elements", allowed_receptor),
        ):
            if (
                any(str(value).upper() not in {"N", "O", "S"} for value in values)
                or len({str(value).upper() for value in values}) != len(values)
            ):
                raise RuntimeError(
                    f"interaction-v1 {element} {name} must contain unique N/O/S values"
                )
        for symbol in ("N", "O", "S"):
            try:
                r0_value = float(r0[symbol])
                cutoff_value = float(cutoffs[symbol])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"interaction-v1 {element} requires numeric N/O/S metal distances"
                ) from exc
            if (
                not math.isfinite(r0_value)
                or r0_value < 0
                or not math.isfinite(cutoff_value)
                or cutoff_value < 0
            ):
                raise RuntimeError(
                    f"interaction-v1 {element} metal distances must be finite and non-negative"
                )
        if auto_attraction:
            if (
                target_coordination < 2
                or not geometries
                or not allowed_ligand
                or not allowed_receptor
                or water_policy == "trace_only"
            ):
                raise RuntimeError(
                    f"interaction-v1 attractive {element} profile is incomplete"
                )
        elif target_coordination or geometries or allowed_ligand:
            raise RuntimeError(
                f"interaction-v1 repulsion-only {element} profile must not declare "
                "an attractive geometry or ligand donor set"
            )
        nearby_cutoff = profile.get("nearby_metal_cutoff_angstrom")
        if (
            not isinstance(nearby_cutoff, (int, float))
            or not math.isfinite(float(nearby_cutoff))
            or float(nearby_cutoff) <= 0
        ):
            raise RuntimeError(
                f"interaction-v1 {element} nearby metal cutoff must be positive"
            )
        profiles[element] = profile
    return profiles


def _profile_label(element: str, profile: dict[str, object]) -> str:
    oxidation = str(profile.get("assumed_oxidation_state", "unresolved"))
    status = str(profile.get("attraction_status", "unspecified"))
    return f"{element}:{oxidation}:{status}"


def _validate_raw_metal_site_records(
    atom,
    profile: dict[str, object],
    raw_records: list[dict[str, object]],
) -> None:
    """Fail closed on alternate/partial/duplicated metal and donor records."""
    element = atom.element.upper()
    site_coords = torch.tensor(atom.coords, dtype=torch.float64)
    site_label = _metal_site_label(atom)
    fail = _unsupported_zinc_site if element == "ZN" else _unsupported_metal_profile
    matching_metals = [
        record
        for record in raw_records
        if record["element"] == element
        and record["residue"] == (atom.raw_res_name or atom.res_name).upper()
        and record["chain"] == atom.chain
        and record["residue_number"] == atom.res_num
        and record["icode"] == atom.icode
        and record["atom_name"] == atom.atom_name.upper()
    ]
    if len(matching_metals) != 1:
        fail(
            "metal identity is duplicated, alternate, or absent in the raw PDB records",
            metal_site=site_label,
            matching_record_count=len(matching_metals),
        )
    metal_record = matching_metals[0]
    occupancy = float(metal_record["occupancy"])
    if (
        metal_record["altloc"]
        or not math.isfinite(occupancy)
        or not math.isclose(occupancy, 1.0, rel_tol=0.0, abs_tol=1e-3)
    ):
        fail(
            "metal attraction/repulsion requires one full-occupancy primary conformation",
            metal_site=site_label,
            altloc=metal_record["altloc"],
            occupancy=occupancy,
        )

    cutoffs = profile["receptor_detection_cutoff_angstrom"]
    ambiguous_contacts: list[str] = []
    contact_identity_counts: dict[tuple[object, ...], int] = {}
    for record in raw_records:
        symbol = str(record["element"])
        cutoff = float(cutoffs.get(symbol, 0.0)) if isinstance(cutoffs, dict) else 0.0
        if record is metal_record or cutoff <= 0:
            continue
        if symbol not in {"N", "O", "S"}:
            continue
        distance = float((record["coords"] - site_coords).norm())
        if distance > cutoff:
            continue
        contact_identity = (
            record["chain"],
            record["residue"],
            record["residue_number"],
            record["icode"],
            record["atom_name"],
        )
        contact_identity_counts[contact_identity] = (
            contact_identity_counts.get(contact_identity, 0) + 1
        )
        contact_occupancy = float(record["occupancy"])
        if (
            record["altloc"]
            or not math.isfinite(contact_occupancy)
            or not math.isclose(
                contact_occupancy,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-3,
            )
        ):
            ambiguous_contacts.append(
                f"line={record['line_number']}:{record['chain']}:"
                f"{record['residue']}{record['residue_number']}{record['icode']}:"
                f"{record['atom_name']}:altloc={record['altloc'] or '-'}:"
                f"occupancy={contact_occupancy:g}:distance={distance:.3f}"
            )
    duplicate_contacts = [
        list(identity)
        for identity, count in contact_identity_counts.items()
        if count != 1
    ]
    if duplicate_contacts:
        fail(
            "metal site has duplicated donor/water identities",
            metal_site=site_label,
            duplicate_records=duplicate_contacts,
        )
    if ambiguous_contacts:
        fail(
            "metal site has alternate-location or partial-occupancy donor/water records",
            metal_site=site_label,
            ambiguous_records=sorted(ambiguous_contacts),
        )


def _unsupported_zinc_site(message: str, **details: object) -> None:
    raise UnsupportedPhysicalChemistryError(
        "unsupported_zinc_site",
        message,
        details=details,
    )


def _build_zinc_site(
    zinc_atom,
    *,
    all_parsed_atoms: list,
    protein_atoms: list,
    protein_coords: Tensor,
    keep: Tensor,
    protein_pdb: Path,
    coordinate_origin: Tensor | None,
) -> dict[str, Tensor | tuple[str, ...]]:
    """Admit one strict three-receptor-donor tetrahedral Zn(II) site."""
    zinc_absolute = torch.tensor(zinc_atom.coords, dtype=torch.float64)
    zinc_label = (
        f"{zinc_atom.chain}:{zinc_atom.raw_res_name or zinc_atom.res_name}"
        f"{zinc_atom.res_num}{zinc_atom.icode}:{zinc_atom.atom_name}"
    )
    alternate_contacts = _nonprimary_altloc_contacts_near_zinc(
        protein_pdb,
        zinc_absolute,
    )
    if alternate_contacts:
        _unsupported_zinc_site(
            "Zn(II) V0 does not admit non-primary alternate-location "
            "receptor donor or coordination-water records",
            zinc_site=zinc_label,
            alternate_records=list(alternate_contacts),
        )

    other_metals = [
        atom
        for atom in all_parsed_atoms
        if atom is not zinc_atom
        and _is_guidance_metal(atom)
        and float((torch.tensor(atom.coords, dtype=torch.float64) - zinc_absolute).norm()) <= 5.0
    ]
    if other_metals:
        _unsupported_zinc_site(
            "Zn(II) V0 does not admit binuclear or nearby multi-metal sites",
            zinc_site=zinc_label,
            nearby_metals=sorted(
                {str(atom.raw_res_name or atom.res_name) for atom in other_metals}
            ),
        )

    waters = _water_oxygen_coords(protein_pdb)
    if waters.numel() and bool(((waters - zinc_absolute).norm(dim=-1) <= 2.5).any()):
        _unsupported_zinc_site(
            "Zn(II) V0 does not admit a resolved coordination water",
            zinc_site=zinc_label,
            water_cutoff_angstrom=2.5,
        )

    displacement = protein_coords - zinc_absolute
    distance = displacement.norm(dim=-1)
    element = [atom.element.upper() for atom in protein_atoms]
    contact = torch.tensor(
        [
            (
                (symbol in {"N", "O"} and float(distance[index]) <= 2.5)
                or (symbol == "S" and float(distance[index]) <= 2.8)
            )
            for index, symbol in enumerate(element)
        ],
        dtype=torch.bool,
    )
    contact_indices = contact.nonzero(as_tuple=False).flatten().tolist()

    residue_contacts: dict[tuple[str, int, str], list[int]] = {}
    for atom_index in contact_indices:
        atom = protein_atoms[atom_index]
        residue_contacts.setdefault(
            (atom.chain, atom.res_num, atom.icode),
            [],
        ).append(atom_index)

    admitted: list[int] = []
    unsupported_contacts: list[str] = []
    for residue_indices in residue_contacts.values():
        residue = protein_atoms[residue_indices[0]]
        raw_residue = (residue.raw_res_name or residue.res_name).upper()
        by_name: dict[str, list[int]] = {}
        for atom_index in residue_indices:
            by_name.setdefault(protein_atoms[atom_index].atom_name, []).append(atom_index)
        duplicate_names = sorted(name for name, indices in by_name.items() if len(indices) != 1)
        if duplicate_names:
            _unsupported_zinc_site(
                "Zn(II) V0 donor identity is duplicated or alternate",
                zinc_site=zinc_label,
                duplicate_atom_names=duplicate_names,
            )

        if residue.res_name == "HIS":
            histidine_indices = [
                index for name in ("ND1", "NE2") for index in by_name.get(name, ())
            ]
            if raw_residue == "HID":
                admitted.extend(by_name.get("NE2", ()))
                unsupported_contacts.extend(
                    f"{raw_residue}:{protein_atoms[index].atom_name}"
                    for index in histidine_indices
                    if protein_atoms[index].atom_name != "NE2"
                )
            elif raw_residue == "HIE":
                admitted.extend(by_name.get("ND1", ()))
                unsupported_contacts.extend(
                    f"{raw_residue}:{protein_atoms[index].atom_name}"
                    for index in histidine_indices
                    if protein_atoms[index].atom_name != "ND1"
                )
            elif raw_residue == "HIS":
                if len(histidine_indices) != 1:
                    _unsupported_zinc_site(
                        "plain HIS is admitted only when exactly one ring nitrogen "
                        "geometrically resolves the Zn donor",
                        zinc_site=zinc_label,
                        histidine_contact_count=len(histidine_indices),
                    )
                admitted.extend(histidine_indices)
            else:
                unsupported_contacts.extend(
                    f"{raw_residue}:{protein_atoms[index].atom_name}" for index in histidine_indices
                )
            continue

        if residue.res_name in {"ASP", "GLU"} and raw_residue == residue.res_name:
            names = ("OD1", "OD2") if residue.res_name == "ASP" else ("OE1", "OE2")
            carboxylate_indices = [index for name in names for index in by_name.get(name, ())]
            if len(carboxylate_indices) > 1:
                _unsupported_zinc_site(
                    "Zn(II) V0 does not admit bidentate carboxylate coordination",
                    zinc_site=zinc_label,
                    residue=raw_residue,
                    atom_names=[protein_atoms[index].atom_name for index in carboxylate_indices],
                )
            admitted.extend(carboxylate_indices)
            unsupported_contacts.extend(
                f"{raw_residue}:{protein_atoms[index].atom_name}"
                for index in residue_indices
                if index not in carboxylate_indices
            )
            continue

        if residue.res_name == "CYS" and raw_residue == "CYM":
            admitted.extend(by_name.get("SG", ()))
            unsupported_contacts.extend(
                f"{raw_residue}:{protein_atoms[index].atom_name}"
                for index in residue_indices
                if protein_atoms[index].atom_name != "SG"
            )
            continue

        unsupported_contacts.extend(
            f"{raw_residue}:{protein_atoms[index].atom_name}" for index in residue_indices
        )

    if unsupported_contacts:
        _unsupported_zinc_site(
            "Zn(II) V0 found unsupported N/O/S coordination contacts",
            zinc_site=zinc_label,
            contacts=sorted(unsupported_contacts),
        )
    admitted = sorted(set(admitted))
    if len(admitted) != 3:
        _unsupported_zinc_site(
            "Zn(II) V0 requires exactly three fixed receptor donors",
            zinc_site=zinc_label,
            receptor_donor_count=len(admitted),
            donor_labels=[
                (
                    f"{protein_atoms[index].chain}:"
                    f"{protein_atoms[index].raw_res_name or protein_atoms[index].res_name}"
                    f"{protein_atoms[index].res_num}{protein_atoms[index].icode}:"
                    f"{protein_atoms[index].atom_name}"
                )
                for index in admitted
            ],
        )

    kept_full_indices = keep.nonzero(as_tuple=False).flatten().tolist()
    full_to_kept = {
        full_index: kept_index for kept_index, full_index in enumerate(kept_full_indices)
    }
    outside_shell = [index for index in admitted if index not in full_to_kept]
    if outside_shell:
        _unsupported_zinc_site(
            "Zn(II) receptor donors must all be present in the active shell",
            zinc_site=zinc_label,
            outside_shell_indices=outside_shell,
        )

    donor_vectors = displacement[torch.tensor(admitted, dtype=torch.long)]
    donor_directions = donor_vectors / donor_vectors.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    pair_cosine = donor_directions @ donor_directions.T
    upper = pair_cosine[torch.triu_indices(3, 3, offset=1).unbind()].clamp(-1.0, 1.0)
    pair_angles = torch.rad2deg(torch.acos(upper))
    tetrahedral_angle = math.degrees(math.acos(-1.0 / 3.0))
    if bool((torch.abs(pair_angles - tetrahedral_angle) > 25.0).any()):
        _unsupported_zinc_site(
            "Zn(II) receptor-donor angles do not match the tetrahedral V0 contract",
            zinc_site=zinc_label,
            donor_angles_degrees=pair_angles.tolist(),
            target_degrees=tetrahedral_angle,
            tolerance_degrees=25.0,
        )
    vacant_raw = -donor_directions.sum(dim=0)
    vacant_norm = vacant_raw.norm()
    if not bool(torch.isfinite(vacant_norm)) or float(vacant_norm) <= 1e-8:
        _unsupported_zinc_site(
            "Zn(II) V0 vacant coordination direction is degenerate",
            zinc_site=zinc_label,
        )
    vacant = vacant_raw / vacant_norm
    zinc_coordinate = zinc_absolute
    if coordinate_origin is not None:
        zinc_coordinate = zinc_coordinate - coordinate_origin.detach().cpu().to(torch.float64).view(
            3
        )
    return {
        "coords": zinc_coordinate.view(1, 3),
        "vacant_direction": vacant.view(1, 3),
        "receptor_donor_index": torch.tensor(
            [[full_to_kept[index] for index in admitted]],
            dtype=torch.long,
        ),
        "receptor_donor_element": torch.tensor(
            [[_SYMBOL_TO_Z[protein_atoms[index].element.upper()] for index in admitted]],
            dtype=torch.long,
        ),
        "labels": (zinc_label,),
        "exclusion_labels": (),
    }


def _ideal_coordination_vectors(geometry: str) -> Tensor:
    if geometry == "tetrahedral":
        return torch.tensor(
            (
                (1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
            ),
            dtype=torch.float64,
        ) / math.sqrt(3.0)
    if geometry == "square_planar":
        return torch.tensor(
            ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            dtype=torch.float64,
        )
    if geometry == "octahedral":
        return torch.tensor(
            (
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ),
            dtype=torch.float64,
        )
    if geometry == "pentagonal_bipyramidal":
        equatorial = [
            (
                math.cos(2.0 * math.pi * index / 5.0),
                math.sin(2.0 * math.pi * index / 5.0),
                0.0,
            )
            for index in range(5)
        ]
        return torch.tensor(
            (*equatorial, (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            dtype=torch.float64,
        )
    raise RuntimeError(f"interaction-v1 unsupported coordination geometry: {geometry!r}")


def _pair_angle_signature(directions: Tensor) -> Tensor:
    pair_index = torch.triu_indices(
        directions.shape[0],
        directions.shape[0],
        offset=1,
    )
    cosine = (directions @ directions.T)[pair_index.unbind()].clamp(-1.0, 1.0)
    return torch.sort(torch.rad2deg(torch.acos(cosine))).values


def _fit_one_vacancy_geometry(
    donor_directions: Tensor,
    geometries: list[object],
    *,
    tolerance_degrees: float,
    ambiguity_margin_degrees: float,
    metal_site: str,
) -> tuple[str, Tensor, float]:
    """Fit an unordered fixed-donor shell to a regular one-vacancy template."""
    observed = _pair_angle_signature(donor_directions)
    geometry_scores: list[tuple[float, str]] = []
    for raw_geometry in geometries:
        geometry = str(raw_geometry).strip().lower()
        ideal = _ideal_coordination_vectors(geometry)
        if ideal.shape[0] != donor_directions.shape[0] + 1:
            continue
        candidate_scores: list[float] = []
        for vacancy in range(ideal.shape[0]):
            fixed = torch.cat((ideal[:vacancy], ideal[vacancy + 1 :]), dim=0)
            expected = _pair_angle_signature(fixed)
            candidate_scores.append(
                float(torch.sqrt(torch.mean((observed - expected).square())))
            )
        geometry_scores.append((min(candidate_scores), geometry))
    if not geometry_scores:
        _unsupported_metal_profile(
            "metal profile has no geometry matching its target coordination",
            metal_site=metal_site,
            fixed_donor_count=int(donor_directions.shape[0]),
            geometries=[str(value) for value in geometries],
        )
    geometry_scores.sort()
    best_rms, best_geometry = geometry_scores[0]
    if best_rms > tolerance_degrees:
        _unsupported_metal_profile(
            "fixed receptor-donor angles do not satisfy the metal profile",
            metal_site=metal_site,
            geometry=best_geometry,
            angle_rms_degrees=best_rms,
            tolerance_degrees=tolerance_degrees,
            observed_pair_angles_degrees=observed.tolist(),
        )
    if (
        len(geometry_scores) > 1
        and geometry_scores[1][0] - best_rms < ambiguity_margin_degrees
    ):
        _unsupported_metal_profile(
            "metal coordination geometry is ambiguous between profile templates",
            metal_site=metal_site,
            best_geometry=best_geometry,
            best_rms_degrees=best_rms,
            second_geometry=geometry_scores[1][1],
            second_rms_degrees=geometry_scores[1][0],
            required_margin_degrees=ambiguity_margin_degrees,
        )
    vacant_raw = -donor_directions.sum(dim=0)
    vacant_norm = vacant_raw.norm()
    if not bool(torch.isfinite(vacant_norm)) or float(vacant_norm) <= 1e-8:
        _unsupported_metal_profile(
            "one-vacancy metal direction is degenerate",
            metal_site=metal_site,
            geometry=best_geometry,
        )
    return best_geometry, vacant_raw / vacant_norm, best_rms


def _canonical_receptor_metal_donor(atom) -> bool:
    """Conservative canonical protein donor identity for generic metal sites."""
    element = atom.element.upper()
    raw_residue = (atom.raw_res_name or atom.res_name).upper()
    atom_name = atom.atom_name.upper()
    if element == "O":
        # Only unmodified canonical amino-acid oxygens are admitted.  This
        # includes backbone carbonyl/OXT and canonical side-chain oxygens.
        return raw_residue == atom.res_name and atom.res_name in AA3_TO_IDX
    if element == "N":
        if atom.res_name != "HIS" or atom_name not in {"ND1", "NE2"}:
            return False
        if raw_residue == "HID":
            return atom_name == "NE2"
        if raw_residue == "HIE":
            return atom_name == "ND1"
        return raw_residue == "HIS"
    if element == "S":
        return (
            atom.res_name == "CYS"
            and raw_residue in {"CYS", "CYM"}
            and atom_name == "SG"
        ) or (
            atom.res_name == "MET"
            and raw_residue == "MET"
            and atom_name == "SD"
        )
    return False


def _build_generic_attractive_metal_site(
    metal_atom,
    profile: dict[str, object],
    *,
    protein_atoms: list,
    protein_coords: Tensor,
    keep: Tensor,
    raw_records: list[dict[str, object]],
    coordinate_origin: Tensor | None,
) -> dict[str, object]:
    """Build one strict, regular, exactly-one-vacancy non-Zn metal site."""
    element = metal_atom.element.upper()
    site_label = _metal_site_label(metal_atom)
    metal_absolute = torch.tensor(metal_atom.coords, dtype=torch.float64)
    target_coordination = int(profile["target_coordination"])
    fixed_required = target_coordination - 1
    allowed_receptor = {
        str(value).upper() for value in profile["allowed_receptor_elements"]
    }
    cutoffs = profile["receptor_detection_cutoff_angstrom"]

    displacement = protein_coords - metal_absolute
    distance = displacement.norm(dim=-1)
    all_contact_indices = [
        index
        for index, atom in enumerate(protein_atoms)
        if atom.element.upper() in {"N", "O", "S"}
        and float(cutoffs[atom.element.upper()]) > 0
        and float(distance[index]) <= float(cutoffs[atom.element.upper()])
    ]
    unsupported_element_contacts = [
        (
            f"{protein_atoms[index].chain}:"
            f"{protein_atoms[index].raw_res_name or protein_atoms[index].res_name}"
            f"{protein_atoms[index].res_num}{protein_atoms[index].icode}:"
            f"{protein_atoms[index].atom_name}"
        )
        for index in all_contact_indices
        if protein_atoms[index].element.upper() not in allowed_receptor
    ]
    if unsupported_element_contacts:
        _unsupported_metal_profile(
            "metal site has close donor elements outside the attractive profile",
            metal_site=site_label,
            contacts=sorted(unsupported_element_contacts),
            allowed_receptor_elements=sorted(allowed_receptor),
        )
    contact_indices = [
        index
        for index in all_contact_indices
        if protein_atoms[index].element.upper() in allowed_receptor
    ]
    identity_keys: dict[tuple[str, int, str, str], list[int]] = {}
    for index in contact_indices:
        atom = protein_atoms[index]
        identity_keys.setdefault(
            (atom.chain, atom.res_num, atom.icode, atom.atom_name),
            [],
        ).append(index)
    duplicate_donors = [
        key for key, indices in identity_keys.items() if len(indices) != 1
    ]
    if duplicate_donors:
        _unsupported_metal_profile(
            "metal receptor donor identity is duplicated",
            metal_site=site_label,
            duplicate_donors=[list(key) for key in duplicate_donors],
        )
    unsupported_contacts = [
        (
            f"{protein_atoms[index].chain}:"
            f"{protein_atoms[index].raw_res_name or protein_atoms[index].res_name}"
            f"{protein_atoms[index].res_num}{protein_atoms[index].icode}:"
            f"{protein_atoms[index].atom_name}"
        )
        for index in contact_indices
        if not _canonical_receptor_metal_donor(protein_atoms[index])
    ]
    if unsupported_contacts:
        _unsupported_metal_profile(
            "metal site has non-canonical receptor donor contacts",
            metal_site=site_label,
            contacts=sorted(unsupported_contacts),
        )
    protein_donor_indices = sorted(
        index
        for index in contact_indices
        if _canonical_receptor_metal_donor(protein_atoms[index])
    )

    water_cutoff = float(cutoffs.get("O", 0.0))
    water_records = [
        record
        for record in raw_records
        if bool(record["is_water"])
        and record["element"] == "O"
        and water_cutoff > 0
        and float((record["coords"] - metal_absolute).norm()) <= water_cutoff
    ]
    water_policy = str(profile["water_policy"])
    if water_records and water_policy == "reject":
        _unsupported_metal_profile(
            "metal profile rejects resolved coordination water",
            metal_site=site_label,
            water_count=len(water_records),
            water_cutoff_angstrom=water_cutoff,
        )
    if water_policy == "admit_fixed_oxygen" and element == "MG" and not water_records:
        _unsupported_metal_profile(
            "strict Mg(II) attraction requires at least one retained coordination water",
            metal_site=site_label,
            water_cutoff_angstrom=water_cutoff,
        )
    admitted_water = water_records if water_policy == "admit_fixed_oxygen" else []
    fixed_count = len(protein_donor_indices) + len(admitted_water)
    if fixed_count != fixed_required:
        _unsupported_metal_profile(
            "attractive metal profile requires exactly one vacant coordination slot",
            metal_site=site_label,
            fixed_donor_count=fixed_count,
            required_fixed_donor_count=fixed_required,
            protein_donor_count=len(protein_donor_indices),
            retained_water_count=len(admitted_water),
            target_coordination=target_coordination,
        )

    kept_full_indices = keep.nonzero(as_tuple=False).flatten().tolist()
    full_to_kept = {
        full_index: kept_index for kept_index, full_index in enumerate(kept_full_indices)
    }
    outside_shell = [
        index for index in protein_donor_indices if index not in full_to_kept
    ]
    if outside_shell:
        _unsupported_metal_profile(
            "metal receptor donors must all be present in the active shell",
            metal_site=site_label,
            outside_shell_indices=outside_shell,
        )

    donor_vectors = [
        displacement[index] for index in protein_donor_indices
    ] + [
        record["coords"] - metal_absolute for record in admitted_water
    ]
    donor_tensor = torch.stack(donor_vectors)
    donor_directions = donor_tensor / donor_tensor.norm(
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)
    geometry, vacant, angle_rms = _fit_one_vacancy_geometry(
        donor_directions,
        profile["geometries"],
        tolerance_degrees=float(profile["geometry_rms_tolerance_degrees"]),
        ambiguity_margin_degrees=float(
            profile["geometry_ambiguity_margin_degrees"]
        ),
        metal_site=site_label,
    )
    metal_coordinate = metal_absolute
    if coordinate_origin is not None:
        metal_coordinate = metal_coordinate - coordinate_origin.detach().cpu().to(
            torch.float64
        ).view(3)
    donor_keys = tuple(
        [f"protein:{index}" for index in protein_donor_indices]
        + [
            (
                f"water:{record['chain']}:{record['residue']}"
                f"{record['residue_number']}{record['icode']}:{record['atom_name']}"
            )
            for record in admitted_water
        ]
    )
    return {
        "coords": metal_coordinate,
        "vacant_direction": vacant,
        "fixed_coordination": fixed_count,
        "target_coordination": target_coordination,
        "donor_keys": donor_keys,
        "geometry": geometry,
        "angle_rms_degrees": angle_rms,
    }


def _metal_r0_and_allowed(profile: dict[str, object]) -> tuple[Tensor, Tensor]:
    ordered_elements = ("N", "O", "S")
    allowed_symbols = {
        str(value).upper() for value in profile["allowed_ligand_elements"]
    }
    allowed = torch.tensor(
        [symbol in allowed_symbols for symbol in ordered_elements],
        dtype=torch.bool,
    )
    raw_r0 = profile["r0_angstrom"]
    r0 = torch.tensor(
        [
            float(raw_r0[symbol]) if symbol in allowed_symbols else 0.0
            for symbol in ordered_elements
        ],
        dtype=torch.float64,
    )
    if bool((allowed != r0.gt(0)).any()):
        raise RuntimeError(
            "interaction-v1 allowed metal ligand donors require positive N/O/S r0"
        )
    return r0, allowed


def _build_metal_topology_payload(
    active_metals: list,
    *,
    profiles: dict[str, dict[str, object]],
    all_parsed_atoms: list,
    protein_atoms: list,
    protein_coords: Tensor,
    keep: Tensor,
    protein_pdb: Path,
    coordinate_origin: Tensor | None,
) -> dict[str, object]:
    """Dispatch active standalone ions to strict attraction or repulsion-only rows."""
    raw_records = _raw_pdb_records(protein_pdb)
    metal_coords: list[Tensor] = []
    metal_atomic_numbers: list[int] = []
    vacant_directions: list[Tensor] = []
    fixed_coordination: list[int] = []
    target_coordination: list[int] = []
    ligand_r0: list[Tensor] = []
    ligand_allowed: list[Tensor] = []
    attraction_enabled: list[bool] = []
    site_labels: list[str] = []
    profile_labels: list[str] = []
    typing_exclusions: list[str] = []
    used_donor_keys: dict[str, str] = {}
    zinc_coords: list[Tensor] = []
    zinc_vacant: list[Tensor] = []
    zinc_receptor_index: list[Tensor] = []
    zinc_receptor_element: list[Tensor] = []
    zinc_labels: list[str] = []
    zinc_exclusions: list[str] = []

    for metal_atom in active_metals:
        element = metal_atom.element.upper()
        profile = profiles[element]
        site_label = _metal_site_label(metal_atom)
        _validate_raw_metal_site_records(metal_atom, profile, raw_records)
        nearby_cutoff = float(profile["nearby_metal_cutoff_angstrom"])
        nearby_metals = [
            atom
            for atom in all_parsed_atoms
            if atom is not metal_atom
            and _is_guidance_metal(atom)
            and float(
                (
                    torch.tensor(atom.coords, dtype=torch.float64)
                    - torch.tensor(metal_atom.coords, dtype=torch.float64)
                ).norm()
            )
            <= nearby_cutoff
        ]
        if nearby_metals:
            fail = _unsupported_zinc_site if element == "ZN" else _unsupported_metal_profile
            fail(
                "nearby-metal, binuclear, and polynuclear sites are not admitted",
                metal_site=site_label,
                nearby_metal_cutoff_angstrom=nearby_cutoff,
                nearby_metals=[
                    _metal_site_label(atom) for atom in nearby_metals
                ],
            )

        enabled = bool(profile["auto_attraction"])
        if enabled and element == "ZN":
            site = _build_zinc_site(
                metal_atom,
                all_parsed_atoms=all_parsed_atoms,
                protein_atoms=protein_atoms,
                protein_coords=protein_coords,
                keep=keep,
                protein_pdb=protein_pdb,
                coordinate_origin=coordinate_origin,
            )
            site_coord = site["coords"][0]
            vacant = site["vacant_direction"][0]
            fixed = 3
            target = int(profile["target_coordination"])
            kept_full_indices = keep.nonzero(as_tuple=False).flatten().tolist()
            donor_keys = tuple(
                f"protein:{kept_full_indices[int(index)]}"
                for index in site["receptor_donor_index"][0].tolist()
            )
            zinc_coords.append(site["coords"])
            zinc_vacant.append(site["vacant_direction"])
            zinc_receptor_index.append(site["receptor_donor_index"])
            zinc_receptor_element.append(site["receptor_donor_element"])
            zinc_labels.extend(site["labels"])
            zinc_exclusions.extend(site["exclusion_labels"])
        elif enabled:
            site = _build_generic_attractive_metal_site(
                metal_atom,
                profile,
                protein_atoms=protein_atoms,
                protein_coords=protein_coords,
                keep=keep,
                raw_records=raw_records,
                coordinate_origin=coordinate_origin,
            )
            site_coord = site["coords"]
            vacant = site["vacant_direction"]
            fixed = int(site["fixed_coordination"])
            target = int(site["target_coordination"])
            donor_keys = tuple(site["donor_keys"])
        else:
            site_coord = torch.tensor(metal_atom.coords, dtype=torch.float64)
            if coordinate_origin is not None:
                site_coord = site_coord - coordinate_origin.detach().cpu().to(
                    torch.float64
                ).view(3)
            vacant = torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64)
            fixed = 0
            target = 0
            donor_keys = ()
            status = str(profile.get("attraction_status", "disabled"))
            typing_exclusions.append(
                f"{site_label}:attraction_disabled:{status}"
            )

        for donor_key in donor_keys:
            previous_site = used_donor_keys.get(donor_key)
            if previous_site is not None:
                _unsupported_metal_profile(
                    "a receptor donor may not be shared by admitted metal sites",
                    donor=donor_key,
                    first_metal_site=previous_site,
                    second_metal_site=site_label,
                )
            used_donor_keys[donor_key] = site_label

        if enabled:
            r0, allowed = _metal_r0_and_allowed(profile)
        else:
            r0 = torch.zeros(3, dtype=torch.float64)
            allowed = torch.zeros(3, dtype=torch.bool)
        metal_coords.append(site_coord)
        metal_atomic_numbers.append(int(profile["atomic_number"]))
        vacant_directions.append(vacant)
        fixed_coordination.append(fixed)
        target_coordination.append(target)
        ligand_r0.append(r0)
        ligand_allowed.append(allowed)
        attraction_enabled.append(enabled)
        site_labels.append(site_label)
        profile_labels.append(_profile_label(element, profile))

    return {
        "metal_coords": (
            torch.stack(metal_coords)
            if metal_coords
            else torch.empty((0, 3), dtype=torch.float64)
        ),
        "metal_atomic_number": torch.tensor(metal_atomic_numbers, dtype=torch.long),
        "metal_vacant_direction": (
            torch.stack(vacant_directions)
            if vacant_directions
            else torch.empty((0, 3), dtype=torch.float64)
        ),
        "metal_fixed_coordination": torch.tensor(
            fixed_coordination,
            dtype=torch.long,
        ),
        "metal_target_coordination": torch.tensor(
            target_coordination,
            dtype=torch.long,
        ),
        "metal_ligand_r0": (
            torch.stack(ligand_r0)
            if ligand_r0
            else torch.empty((0, 3), dtype=torch.float64)
        ),
        "metal_ligand_donor_allowed": (
            torch.stack(ligand_allowed)
            if ligand_allowed
            else torch.empty((0, 3), dtype=torch.bool)
        ),
        "metal_attraction_enabled": torch.tensor(
            attraction_enabled,
            dtype=torch.bool,
        ),
        "metal_site_labels": tuple(site_labels),
        "metal_profile_labels": tuple(profile_labels),
        "metal_typing_exclusion_labels": tuple(typing_exclusions),
        "zinc_coords": (
            torch.cat(zinc_coords, dim=0)
            if zinc_coords
            else torch.empty((0, 3), dtype=torch.float64)
        ),
        "zinc_vacant_direction": (
            torch.cat(zinc_vacant, dim=0)
            if zinc_vacant
            else torch.empty((0, 3), dtype=torch.float64)
        ),
        "zinc_receptor_donor_index": (
            torch.cat(zinc_receptor_index, dim=0)
            if zinc_receptor_index
            else torch.empty((0, 3), dtype=torch.long)
        ),
        "zinc_receptor_donor_element": (
            torch.cat(zinc_receptor_element, dim=0)
            if zinc_receptor_element
            else torch.empty((0, 3), dtype=torch.long)
        ),
        "zinc_site_labels": tuple(zinc_labels),
        "zinc_typing_exclusion_labels": tuple(zinc_exclusions),
    }


_METAL_PAYLOAD_TENSOR_KEYS = (
    "metal_coords",
    "metal_atomic_number",
    "metal_vacant_direction",
    "metal_fixed_coordination",
    "metal_target_coordination",
    "metal_ligand_r0",
    "metal_ligand_donor_allowed",
    "metal_attraction_enabled",
    "zinc_coords",
    "zinc_vacant_direction",
    "zinc_receptor_donor_index",
    "zinc_receptor_donor_element",
)
_METAL_PAYLOAD_LABEL_KEYS = (
    "metal_site_labels",
    "metal_profile_labels",
    "metal_typing_exclusion_labels",
    "zinc_site_labels",
    "zinc_typing_exclusion_labels",
)


def _merge_metal_topology_payloads(
    payloads: list[dict[str, object]],
) -> dict[str, object]:
    """Concatenate independently admitted/fallback metal sites deterministically."""
    if not payloads:
        return {
            "metal_coords": torch.empty((0, 3), dtype=torch.float64),
            "metal_atomic_number": torch.empty(0, dtype=torch.long),
            "metal_vacant_direction": torch.empty((0, 3), dtype=torch.float64),
            "metal_fixed_coordination": torch.empty(0, dtype=torch.long),
            "metal_target_coordination": torch.empty(0, dtype=torch.long),
            "metal_ligand_r0": torch.empty((0, 3), dtype=torch.float64),
            "metal_ligand_donor_allowed": torch.empty((0, 3), dtype=torch.bool),
            "metal_attraction_enabled": torch.empty(0, dtype=torch.bool),
            "metal_site_labels": (),
            "metal_profile_labels": (),
            "metal_typing_exclusion_labels": (),
            "zinc_coords": torch.empty((0, 3), dtype=torch.float64),
            "zinc_vacant_direction": torch.empty((0, 3), dtype=torch.float64),
            "zinc_receptor_donor_index": torch.empty((0, 3), dtype=torch.long),
            "zinc_receptor_donor_element": torch.empty((0, 3), dtype=torch.long),
            "zinc_site_labels": (),
            "zinc_typing_exclusion_labels": (),
        }
    merged: dict[str, object] = {}
    for key in _METAL_PAYLOAD_TENSOR_KEYS:
        values = [payload[key] for payload in payloads]
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError(f"metal topology payload {key} must contain tensors")
        merged[key] = torch.cat(values, dim=0)
    for key in _METAL_PAYLOAD_LABEL_KEYS:
        merged[key] = tuple(
            label
            for payload in payloads
            for label in tuple(payload[key])
        )
    return merged


def _repulsion_only_metal_payload(
    *,
    coords: Tensor,
    atomic_number: int,
    site_label: str,
    element: str,
    failure: UnsupportedPhysicalChemistryError,
    coordinate_origin: Tensor | None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Represent an unresolved site only by the existing bounded metal repulsion."""
    site_coords = coords.detach().cpu().to(torch.float64).view(3)
    if coordinate_origin is not None:
        site_coords = site_coords - coordinate_origin.detach().cpu().to(torch.float64).view(3)
    failure_dict = failure.as_dict()
    provenance = {
        "metal_site": site_label,
        "element": element,
        "action": "bounded_all_ligand_repulsion_only",
        "vacant_direction_semantics": (
            "inactive_unit_placeholder_required_by_interaction_v1_tensor_contract"
        ),
        "code": failure_dict["code"],
        "message": failure_dict["message"],
        "details": failure_dict["details"],
    }
    exclusion = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    return (
        {
            "metal_coords": site_coords.view(1, 3),
            "metal_atomic_number": torch.tensor([atomic_number], dtype=torch.long),
            "metal_vacant_direction": torch.tensor(
                [[1.0, 0.0, 0.0]], dtype=torch.float64
            ),
            "metal_fixed_coordination": torch.zeros(1, dtype=torch.long),
            "metal_target_coordination": torch.zeros(1, dtype=torch.long),
            "metal_ligand_r0": torch.zeros((1, 3), dtype=torch.float64),
            "metal_ligand_donor_allowed": torch.zeros((1, 3), dtype=torch.bool),
            "metal_attraction_enabled": torch.zeros(1, dtype=torch.bool),
            "metal_site_labels": (site_label,),
            "metal_profile_labels": (
                f"{element}:unresolved:geometry_only_repulsion_v1",
            ),
            "metal_typing_exclusion_labels": (exclusion,),
            "zinc_coords": torch.empty((0, 3), dtype=torch.float64),
            "zinc_vacant_direction": torch.empty((0, 3), dtype=torch.float64),
            "zinc_receptor_donor_index": torch.empty((0, 3), dtype=torch.long),
            "zinc_receptor_donor_element": torch.empty((0, 3), dtype=torch.long),
            "zinc_site_labels": (),
            "zinc_typing_exclusion_labels": (),
        },
        provenance,
    )


def _geometry_obstacle_payload(
    atoms: list,
    *,
    coordinate_origin: Tensor | None,
) -> dict[str, object]:
    """Type fixed nonmetal cofactor atoms as repulsion-only geometry obstacles."""
    if not atoms:
        return {
            "coords": torch.empty((0, 3), dtype=torch.float64),
            "atomic_numbers": torch.empty(0, dtype=torch.long),
            "uff_x": torch.empty(0, dtype=torch.float64),
            "uff_d": torch.empty(0, dtype=torch.float64),
            "is_generic": torch.empty(0, dtype=torch.bool),
            "labels": (),
            "kinds": (),
        }
    coords = torch.tensor([atom.coords for atom in atoms], dtype=torch.float64)
    if coordinate_origin is not None:
        coords = coords - coordinate_origin.detach().cpu().to(torch.float64).view(1, 3)
    atomic_numbers = torch.tensor(
        [_SYMBOL_TO_Z.get(atom.element.upper(), 0) for atom in atoms],
        dtype=torch.long,
    )
    is_generic = torch.tensor(
        [int(number) not in _EFF_FF_SUPPORTED_ATOMIC_NUMBERS for number in atomic_numbers],
        dtype=torch.bool,
    )
    uff_x = torch.zeros(len(atoms), dtype=torch.float64)
    uff_d = torch.zeros(len(atoms), dtype=torch.float64)
    supported_index = (~is_generic).nonzero(as_tuple=False).flatten()
    if supported_index.numel():
        params = element_parameters(
            atomic_numbers[supported_index],
            dtype=torch.float64,
        )
        uff_x[supported_index] = params.uff_x
        uff_d[supported_index] = params.uff_d
    labels = tuple(
        (
            f"{atom.chain}:{atom.raw_res_name or atom.res_name}"
            f"{atom.res_num}{atom.icode}:{atom.atom_name}:{atom.element.upper()}"
        )
        for atom in atoms
    )
    kinds = tuple(
        "generic_bounded_steric_v1" if generic else "effff_v2_repulsion_only"
        for generic in is_generic.tolist()
    )
    return {
        "coords": coords,
        "atomic_numbers": atomic_numbers,
        "uff_x": uff_x,
        "uff_d": uff_d,
        "is_generic": is_generic,
        "labels": labels,
        "kinds": kinds,
    }


def build_physical_system(
    mol: Chem.Mol,
    protein_pdb: str | Path,
    *,
    fragment_id: Tensor,
    near_coords: Tensor,
    protein_cutoff: float = 10.0,
    coordinate_origin: Tensor | None = None,
    receptor_policy: str = "fail_closed",
) -> PhysicalSystem:
    """Parameterize a heavy-atom ligand and a local rigid receptor shell."""
    if protein_cutoff <= 0:
        raise ValueError("protein_cutoff must be positive")
    if receptor_policy not in _RECEPTOR_POLICIES:
        raise ValueError(
            f"receptor_policy must be one of {sorted(_RECEPTOR_POLICIES)}, "
            f"got {receptor_policy!r}"
        )
    topology = build_physical_topology(mol, fragment_id)
    protein_path = Path(protein_pdb)
    raw_records = _raw_pdb_records(protein_path)
    parsed_atoms = [
        atom for atom in _parse_pdb_lines(protein_path) if atom.element.upper() != "H"
    ]

    # Keep parsed coordinates in double precision for the diagnostic path.
    # Callers that need float32 explicitly downcast the completed system in
    # ``PhysicalSystem.to``.
    near = near_coords.detach().cpu().to(torch.float64).view(-1, 3)
    if not near.shape[0] or not bool(torch.isfinite(near).all()):
        raise ValueError("physical system shell reference coordinates are empty or non-finite")
    nonprimary_nonmetal_altloc_records = [
        record
        for record in raw_records
        if str(record["altloc"]).upper() not in {"", "A"}
        and str(record["element"]).upper() != "H"
        and str(record["element"]).upper() not in _KNOWN_PDB_METAL_ELEMENTS
        and not bool(record["is_water"])
        and str(record["residue"]).upper() not in NUCLEIC_ACID_RESIDUES
    ]
    if nonprimary_nonmetal_altloc_records:
        nonprimary_coords = torch.stack(
            [record["coords"] for record in nonprimary_nonmetal_altloc_records]
        )
        nonprimary_active = _within_shell(nonprimary_coords, near, protein_cutoff)
        nonprimary_nonmetal_altloc_records = sorted(
            (
                record
                for record, active in zip(
                    nonprimary_nonmetal_altloc_records,
                    nonprimary_active.tolist(),
                    strict=True,
                )
                if active
            ),
            key=lambda record: int(record["line_number"]),
        )
    if receptor_policy == "fail_closed":
        _reject_hidden_active_metal_altlocs(
            protein_path,
            near,
            protein_cutoff,
        )
    if not parsed_atoms:
        raise ValueError("physical system receptor contains no heavy atoms")
    protein_atoms = [atom for atom in parsed_atoms if atom.res_name in AA3_TO_IDX]
    excluded_nonprotein = [atom for atom in parsed_atoms if atom.res_name not in AA3_TO_IDX]
    all_metal_atoms = [atom for atom in excluded_nonprotein if _is_guidance_metal(atom)]
    active_metals: list = []
    if all_metal_atoms:
        all_metal_coords = torch.tensor(
            [atom.coords for atom in all_metal_atoms],
            dtype=torch.float64,
        )
        active_mask = _within_shell(all_metal_coords, near, protein_cutoff)
        active_metals = [
            atom
            for atom, active in zip(
                all_metal_atoms,
                active_mask.tolist(),
                strict=True,
            )
            if active
        ]
    primary_metal_keys = {
        (
            atom.element.upper(),
            (atom.raw_res_name or atom.res_name).upper(),
            atom.chain,
            atom.res_num,
            atom.icode,
            atom.atom_name.upper(),
        )
        for atom in all_metal_atoms
    }
    hidden_active_metal_groups: list[dict[str, object]] = []
    if receptor_policy == "geometry_only":
        raw_metals = [
            record
            for record in raw_records
            if str(record["element"]).upper() in _KNOWN_PDB_METAL_ELEMENTS
            and str(record["altloc"]).upper() not in {"", "A"}
            and _raw_record_site_key(record) not in primary_metal_keys
        ]
        grouped_raw_metals: dict[
            tuple[str, str, str, int, str, str],
            list[dict[str, object]],
        ] = {}
        for record in raw_metals:
            grouped_raw_metals.setdefault(_raw_record_site_key(record), []).append(record)
        for site_key in sorted(grouped_raw_metals):
            records = sorted(
                grouped_raw_metals[site_key],
                key=lambda record: int(record["line_number"]),
            )
            group_coords = torch.stack([record["coords"] for record in records])
            if not bool(_within_shell(group_coords, near, protein_cutoff).any()):
                continue
            hidden_active_metal_groups.append(
                {
                    "site_key": site_key,
                    "representative": _select_altloc_representative(records),
                    "records": records,
                }
            )
    profiles: dict[str, dict[str, object]] = {}
    if active_metals:
        profiles = _metal_coordination_profiles()
        if receptor_policy == "fail_closed":
            for atom in active_metals:
                element = atom.element.upper()
                raw_residue = (atom.raw_res_name or atom.res_name).upper()
                profile = profiles.get(element)
                if profile is None:
                    _unsupported_metal_profile(
                        "active standalone metal has no versioned guidance profile",
                        metal_site=_metal_site_label(atom),
                        element=element,
                        registered_elements=sorted(profiles),
                    )
                if raw_residue != str(profile["identity_residue"]).upper():
                    fail = (
                        _unsupported_zinc_site
                        if element == "ZN"
                        else _unsupported_metal_profile
                    )
                    fail(
                        "standalone metal requires matching PDB residue and element identity",
                        metal_site=_metal_site_label(atom),
                        element=element,
                        residue_name=raw_residue,
                        required_residue_name=profile["identity_residue"],
                    )
    elif hidden_active_metal_groups:
        profiles = _metal_coordination_profiles()

    active_metal_ids = {id(atom) for atom in active_metals}
    policy_excluded_nonprotein = [
        atom for atom in excluded_nonprotein if id(atom) not in active_metal_ids
    ]
    active_nonprotein_atoms: list = []
    if policy_excluded_nonprotein:
        excluded_coords = torch.tensor(
            [atom.coords for atom in policy_excluded_nonprotein],
            dtype=torch.float64,
        )
        active_excluded = _within_shell(excluded_coords, near, protein_cutoff)
        active_nonprotein_atoms = [
            atom
            for atom, active in zip(
                policy_excluded_nonprotein,
                active_excluded.tolist(),
                strict=True,
            )
            if active
        ]
        if active_nonprotein_atoms and receptor_policy == "fail_closed":
            active_residues = sorted(
                {atom.res_name for atom in active_nonprotein_atoms}
            )
            raise UnsupportedPhysicalChemistryError(
                "active_nonprotein_residue",
                "EFF-FF-v2 does not parameterize non-protein residues in "
                f"the active shell: {active_residues}",
                details={
                    "residue_names": active_residues,
                    "record_types": sorted(
                        {atom.record_type for atom in active_nonprotein_atoms}
                    ),
                },
            )
    if not protein_atoms:
        raise ValueError("physical system receptor contains no parameterizable protein atoms")

    coords = torch.tensor([atom.coords for atom in protein_atoms], dtype=torch.float64)
    keep = _within_shell(coords, near, protein_cutoff)
    if not bool(keep.any()):
        raise ValueError("physical system receptor shell is empty")
    kept_atoms = [
        atom for atom, selected in zip(protein_atoms, keep.tolist(), strict=True) if selected
    ]
    unsupported = sorted(
        {atom.element.upper() for atom in kept_atoms if atom.element.upper() not in _SYMBOL_TO_Z}
    )
    if unsupported:
        raise UnsupportedPhysicalChemistryError(
            "unsupported_receptor_element",
            f"EFF-FF-v2 unsupported receptor elements in the active shell: {unsupported}",
            details={"elements": unsupported},
        )
    atomic_numbers = torch.tensor(
        [_SYMBOL_TO_Z[atom.element.upper()] for atom in kept_atoms],
        dtype=torch.long,
    )
    params = element_parameters(atomic_numbers, dtype=torch.float64)
    kept_coords = coords[keep]
    if coordinate_origin is not None:
        kept_coords = kept_coords - coordinate_origin.detach().cpu().to(torch.float64).view(1, 3)
    interaction_topology = _build_interaction_topology(
        mol,
        protein_atoms,
        coords,
        keep,
    )
    metal_fallbacks: list[dict[str, object]] = []
    metal_payloads: list[dict[str, object]] = []
    if active_metals:
        if receptor_policy == "fail_closed":
            metal_payloads.append(
                _build_metal_topology_payload(
                    active_metals,
                    profiles=profiles,
                    all_parsed_atoms=parsed_atoms,
                    protein_atoms=protein_atoms,
                    protein_coords=coords,
                    keep=keep,
                    protein_pdb=protein_path,
                    coordinate_origin=coordinate_origin,
                )
            )
        else:
            for atom in active_metals:
                element = atom.element.upper()
                raw_residue = (atom.raw_res_name or atom.res_name).upper()
                profile = profiles.get(element)
                try:
                    if profile is None:
                        _unsupported_metal_profile(
                            "active standalone metal has no versioned guidance profile",
                            metal_site=_metal_site_label(atom),
                            element=element,
                            registered_elements=sorted(profiles),
                        )
                    if raw_residue != str(profile["identity_residue"]).upper():
                        fail = (
                            _unsupported_zinc_site
                            if element == "ZN"
                            else _unsupported_metal_profile
                        )
                        fail(
                            "standalone metal requires matching PDB residue and element identity",
                            metal_site=_metal_site_label(atom),
                            element=element,
                            residue_name=raw_residue,
                            required_residue_name=profile["identity_residue"],
                        )
                    metal_payloads.append(
                        _build_metal_topology_payload(
                            [atom],
                            profiles=profiles,
                            all_parsed_atoms=parsed_atoms,
                            protein_atoms=protein_atoms,
                            protein_coords=coords,
                            keep=keep,
                            protein_pdb=protein_path,
                            coordinate_origin=coordinate_origin,
                        )
                    )
                except UnsupportedPhysicalChemistryError as error:
                    fallback_payload, provenance = _repulsion_only_metal_payload(
                        coords=torch.tensor(atom.coords, dtype=torch.float64),
                        atomic_number=_SYMBOL_TO_Z[element],
                        site_label=_metal_site_label(atom),
                        element=element,
                        failure=error,
                        coordinate_origin=coordinate_origin,
                    )
                    metal_payloads.append(fallback_payload)
                    metal_fallbacks.append(provenance)
    if hidden_active_metal_groups:
        for group in hidden_active_metal_groups:
            record = group["representative"]
            records = group["records"]
            element = str(record["element"]).upper()
            altlocs = ",".join(
                sorted({str(item["altloc"]).upper() for item in records})
            )
            site_label = (
                f"{record['chain']}:{record['residue']}"
                f"{record['residue_number']}{record['icode']}:{record['atom_name']}"
                f":altloc_group={altlocs}"
            )
            code = "unsupported_zinc_site" if element == "ZN" else "unsupported_metal_profile"
            failure = UnsupportedPhysicalChemistryError(
                code,
                "active metal record has a non-primary alternate location hidden by normalization",
                details={
                    "elements": [element],
                    "records": [_raw_record_provenance(item) for item in records],
                    "coordinate_selection": (
                        "highest_finite_occupancy_then_altloc_then_line_number"
                    ),
                    "selected_line_number": int(record["line_number"]),
                },
            )
            fallback_payload, provenance = _repulsion_only_metal_payload(
                coords=record["coords"],
                atomic_number=_SYMBOL_TO_Z[element],
                site_label=site_label,
                element=element,
                failure=failure,
                coordinate_origin=coordinate_origin,
            )
            metal_payloads.append(fallback_payload)
            metal_fallbacks.append(provenance)
    if metal_payloads:
        metal_payload = _merge_metal_topology_payloads(metal_payloads)
        interaction_topology = replace(
            interaction_topology,
            metal_coords=metal_payload["metal_coords"],
            metal_atomic_number=metal_payload["metal_atomic_number"],
            metal_vacant_direction=metal_payload["metal_vacant_direction"],
            metal_fixed_coordination=metal_payload["metal_fixed_coordination"],
            metal_target_coordination=metal_payload["metal_target_coordination"],
            metal_ligand_r0=metal_payload["metal_ligand_r0"],
            metal_ligand_donor_allowed=metal_payload[
                "metal_ligand_donor_allowed"
            ],
            metal_attraction_enabled=metal_payload["metal_attraction_enabled"],
            metal_site_labels=metal_payload["metal_site_labels"],
            metal_profile_labels=metal_payload["metal_profile_labels"],
            metal_typing_exclusion_labels=metal_payload[
                "metal_typing_exclusion_labels"
            ],
            zinc_coords=metal_payload["zinc_coords"],
            zinc_vacant_direction=metal_payload["zinc_vacant_direction"],
            zinc_receptor_donor_index=metal_payload["zinc_receptor_donor_index"],
            zinc_receptor_donor_element=metal_payload[
                "zinc_receptor_donor_element"
            ],
            zinc_site_labels=metal_payload["zinc_site_labels"],
            zinc_typing_exclusion_labels=metal_payload[
                "zinc_typing_exclusion_labels"
            ],
        )
    obstacle_payload = _geometry_obstacle_payload(
        active_nonprotein_atoms if receptor_policy == "geometry_only" else [],
        coordinate_origin=coordinate_origin,
    )
    filtered_water_atoms = sum(
        1
        for record in raw_records
        if bool(record["is_water"]) and str(record["element"]).upper() != "H"
    )
    filtered_nucleic_acid_atoms = sum(
        1
        for record in raw_records
        if str(record["residue"]).upper() in NUCLEIC_ACID_RESIDUES
        and str(record["element"]).upper() != "H"
    )
    policy_identity = _receptor_policy_identity(receptor_policy)
    receptor_provenance = {
        "schema_version": "effdock.receptor_provenance.v1",
        "mode": receptor_policy,
        "policy_mode": receptor_policy,
        "policy_sha256": policy_identity["sha256"],
        "active_shell_cutoff_angstrom": float(protein_cutoff),
        "standard_protein_source_heavy_atoms": len(protein_atoms),
        "active_nonprotein_geometry_obstacle_atoms": len(active_nonprotein_atoms)
        if receptor_policy == "geometry_only"
        else 0,
        "active_nonprotein_geometry_obstacle_residues": sorted(
            {atom.res_name for atom in active_nonprotein_atoms}
        )
        if receptor_policy == "geometry_only"
        else [],
        "geometry_obstacle_uff_style_atoms": int(
            (~obstacle_payload["is_generic"]).sum().item()
        ),
        "geometry_obstacle_generic_atoms": int(
            obstacle_payload["is_generic"].sum().item()
        ),
        "obstacle_count": int(obstacle_payload["coords"].shape[0]),
        "metal_strict_attractive_sites": int(
            interaction_topology.metal_attraction_enabled.sum().item()
        ),
        "metal_repulsion_only_sites": int(
            (~interaction_topology.metal_attraction_enabled).sum().item()
        ),
        "metal_fallbacks": metal_fallbacks,
        "metal_fallback_count": len(metal_fallbacks),
        "metal_fallback_reasons": {
            code: sum(1 for fallback in metal_fallbacks if fallback["code"] == code)
            for code in sorted({fallback["code"] for fallback in metal_fallbacks})
        },
        "filtered_records": {
            "water_heavy_atoms": filtered_water_atoms,
            "nucleic_acid_heavy_atoms": filtered_nucleic_acid_atoms,
            "nonprimary_nonmetal_altloc_heavy_atoms": len(
                nonprimary_nonmetal_altloc_records
            ),
            "nonprimary_nonmetal_altloc_records": [
                _raw_record_provenance(record)
                for record in nonprimary_nonmetal_altloc_records
            ],
            "policy": (
                "shared parser filters water, nucleic-acid, and non-primary "
                "alternate-location records"
            ),
        },
    }
    return PhysicalSystem(
        topology=topology,
        protein_coords=kept_coords,
        protein_atomic_numbers=atomic_numbers,
        protein_uff_x=params.uff_x,
        protein_uff_d=params.uff_d,
        protein_vdw_radius=params.vdw_radius,
        parameter_set=parameter_identity(),
        protein_source_atoms=len(parsed_atoms),
        protein_parameterized_source_atoms=len(protein_atoms),
        excluded_nonprotein_atoms=len(policy_excluded_nonprotein),
        excluded_nonprotein_residues=tuple(
            sorted({atom.res_name for atom in policy_excluded_nonprotein})
        ),
        geometry_obstacle_coords=obstacle_payload["coords"],
        geometry_obstacle_atomic_numbers=obstacle_payload["atomic_numbers"],
        geometry_obstacle_uff_x=obstacle_payload["uff_x"],
        geometry_obstacle_uff_d=obstacle_payload["uff_d"],
        geometry_obstacle_is_generic=obstacle_payload["is_generic"],
        geometry_obstacle_labels=obstacle_payload["labels"],
        geometry_obstacle_kinds=obstacle_payload["kinds"],
        receptor_policy_mode=receptor_policy,
        receptor_policy_identity=policy_identity,
        receptor_provenance=receptor_provenance,
        receptor_policy=(
            (
                "normalized standard amino acids plus versioned standalone-metal "
                "profiles: strict one-vacancy Zn(II)/hydrated Mg(II) attraction and "
                "registered common-metal repulsion-only rows; ambiguous identity, "
                "partial occupancy, clusters, and all other active-shell "
                "nonprotein chemistry fail closed"
            )
            if receptor_policy == "fail_closed"
            else (
                "normalized standard amino acids retain existing physical/interaction "
                "typing; active nonprotein nonmetals are fixed repulsion-only geometry "
                "obstacles; unresolved metals are bounded all-ligand repulsion-only; "
                "water and nucleic-acid records remain explicitly filtered"
            )
        ),
        interaction_topology=interaction_topology,
        interaction_parameter_set=interaction_parameter_identity(),
    )


__all__ = [
    "InteractionTopology",
    "PhysicalSystem",
    "build_physical_system",
    "receptor_policy_identity",
    "type_ligand_interactions",
]
