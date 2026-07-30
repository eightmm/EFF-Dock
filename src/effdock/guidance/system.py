"""Build static guidance-system tensors without an external force-field engine."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from hashlib import sha256
from pathlib import Path

import torch
from rdkit import Chem
from torch import Tensor

from effdock.preprocess.protein import (
    AA3_TO_IDX,
    _build_protein_bonds,
    _parse_pdb_lines,
    _patom_pharmacophore,
)

from .errors import UnsupportedPhysicalChemistryError
from .parameterization import (
    element_parameters,
    interaction_parameter_identity,
    load_interaction_v1,
    parameter_identity,
)
from .topology import PhysicalTopology, build_physical_topology

_SYMBOL_TO_Z = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "SE": 34,
    "BR": 35,
    "I": 53,
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
            if item.name == "ligand_neighbor_index":
                values[item.name] = value.to(device=device, dtype=torch.long)
            elif value.dtype == torch.bool:
                values[item.name] = value.to(device=device, dtype=torch.bool)
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
    parameter_set: dict[str, str]
    protein_source_atoms: int
    protein_parameterized_source_atoms: int | None = None
    excluded_nonprotein_atoms: int = 0
    excluded_nonprotein_residues: tuple[str, ...] = ()
    receptor_policy: str = (
        "records whose normalized residue name maps to a supported amino acid; "
        "ATOM/HETATM record type alone never admits chemistry"
    )
    interaction_topology: InteractionTopology | None = None
    interaction_parameter_set: dict[str, str] | None = None

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> PhysicalSystem:
        return PhysicalSystem(
            topology=self.topology.to(device, dtype),
            protein_coords=self.protein_coords.to(device=device, dtype=dtype),
            protein_atomic_numbers=self.protein_atomic_numbers.to(device=device),
            protein_uff_x=self.protein_uff_x.to(device=device, dtype=dtype),
            protein_uff_d=self.protein_uff_d.to(device=device, dtype=dtype),
            parameter_set=dict(self.parameter_set),
            protein_source_atoms=self.protein_source_atoms,
            protein_parameterized_source_atoms=self.protein_parameterized_source_atoms,
            excluded_nonprotein_atoms=self.excluded_nonprotein_atoms,
            excluded_nonprotein_residues=tuple(self.excluded_nonprotein_residues),
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


def build_physical_system(
    mol: Chem.Mol,
    protein_pdb: str | Path,
    *,
    fragment_id: Tensor,
    near_coords: Tensor,
    protein_cutoff: float = 10.0,
    coordinate_origin: Tensor | None = None,
) -> PhysicalSystem:
    """Parameterize a heavy-atom ligand and a local rigid receptor shell."""
    if protein_cutoff <= 0:
        raise ValueError("protein_cutoff must be positive")
    topology = build_physical_topology(mol, fragment_id)
    parsed_atoms = [
        atom for atom in _parse_pdb_lines(Path(protein_pdb)) if atom.element.upper() != "H"
    ]
    if not parsed_atoms:
        raise ValueError("physical system receptor contains no heavy atoms")

    # Keep parsed coordinates in double precision for the diagnostic path.
    # Callers that need float32 explicitly downcast the completed system in
    # ``PhysicalSystem.to``.
    near = near_coords.detach().cpu().to(torch.float64).view(-1, 3)
    if not near.shape[0] or not bool(torch.isfinite(near).all()):
        raise ValueError("physical system shell reference coordinates are empty or non-finite")
    protein_atoms = [atom for atom in parsed_atoms if atom.res_name in AA3_TO_IDX]
    excluded_nonprotein = [atom for atom in parsed_atoms if atom.res_name not in AA3_TO_IDX]
    if excluded_nonprotein:
        excluded_coords = torch.tensor(
            [atom.coords for atom in excluded_nonprotein],
            dtype=torch.float64,
        )
        active_excluded = _within_shell(excluded_coords, near, protein_cutoff)
        if bool(active_excluded.any()):
            active_residues = sorted(
                {
                    atom.res_name
                    for atom, active in zip(
                        excluded_nonprotein,
                        active_excluded.tolist(),
                        strict=True,
                    )
                    if active
                }
            )
            raise UnsupportedPhysicalChemistryError(
                "active_nonprotein_residue",
                "EFF-FF-v2 does not parameterize non-protein residues in "
                f"the active shell: {active_residues}",
                details={
                    "residue_names": active_residues,
                    "record_types": sorted(
                        {
                            atom.record_type
                            for atom, active in zip(
                                excluded_nonprotein,
                                active_excluded.tolist(),
                                strict=True,
                            )
                            if active
                        }
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
    return PhysicalSystem(
        topology=topology,
        protein_coords=kept_coords,
        protein_atomic_numbers=atomic_numbers,
        protein_uff_x=params.uff_x,
        protein_uff_d=params.uff_d,
        parameter_set=parameter_identity(),
        protein_source_atoms=len(parsed_atoms),
        protein_parameterized_source_atoms=len(protein_atoms),
        excluded_nonprotein_atoms=len(excluded_nonprotein),
        excluded_nonprotein_residues=tuple(sorted({atom.res_name for atom in excluded_nonprotein})),
        interaction_topology=interaction_topology,
        interaction_parameter_set=interaction_parameter_identity(),
    )


__all__ = [
    "InteractionTopology",
    "PhysicalSystem",
    "build_physical_system",
    "type_ligand_interactions",
]
