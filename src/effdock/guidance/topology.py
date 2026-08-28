"""Deterministic ligand topology construction for guidance energy terms."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, fields
from hashlib import sha256

import torch
from rdkit import Chem
from torch import Tensor

from .errors import UnsupportedPhysicalChemistryError
from .parameterization import element_parameters, load_effff_v2


@dataclass(frozen=True)
class PhysicalTopology:
    atomic_numbers: Tensor
    fragment_id: Tensor
    mass: Tensor
    uff_x: Tensor
    uff_d: Tensor
    vdw_radius: Tensor
    bond_index: Tensor
    bond_r0: Tensor
    bond_k: Tensor
    angle_index: Tensor
    angle_theta0: Tensor
    angle_k: Tensor
    proper_index: Tensor
    proper_periodicity: Tensor
    proper_phase: Tensor
    proper_k: Tensor
    proper_weight: Tensor
    proper_cut_bond_id: Tensor
    improper_index: Tensor
    improper_phi0: Tensor
    improper_k: Tensor
    improper_planar: Tensor
    ligand_pair_index: Tensor
    ligand_pair_scale: Tensor

    @property
    def num_atoms(self) -> int:
        return int(self.atomic_numbers.numel())

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> PhysicalTopology:
        values: dict[str, Tensor] = {}
        integer_names = {
            "atomic_numbers",
            "fragment_id",
            "vdw_radius",
            "bond_index",
            "angle_index",
            "proper_index",
            "proper_periodicity",
            "proper_cut_bond_id",
            "improper_index",
            "improper_planar",
            "ligand_pair_index",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(
                device=device,
                dtype=value.dtype if item.name in integer_names else dtype,
            )
        return PhysicalTopology(**values)

    def term_counts(self) -> dict[str, int]:
        return {
            "atoms": self.num_atoms,
            "cut_bonds": int(self.bond_index.shape[1]),
            "cross_fragment_angles": int(self.angle_index.shape[1]),
            "torsion_cut_bonds": int(torch.unique(self.proper_cut_bond_id).numel()),
            "torsion_quads": int(self.proper_index.shape[1]),
            "cross_fragment_impropers": int(self.improper_index.shape[1]),
            "interfragment_nonbonded_pairs": int(self.ligand_pair_index.shape[1]),
        }

    def reference_sha256(self) -> str:
        """Hash fragment topology and input-reference geometric targets."""
        names = (
            "atomic_numbers",
            "fragment_id",
            "bond_index",
            "bond_r0",
            "angle_index",
            "angle_theta0",
            "proper_index",
            "proper_periodicity",
            "proper_phase",
            "proper_weight",
            "proper_cut_bond_id",
            "improper_index",
            "improper_phi0",
            "improper_planar",
            "ligand_pair_index",
            "ligand_pair_scale",
        )
        digest = sha256()
        for name in names:
            value = getattr(self, name).detach().cpu()
            value = value.to(torch.float64) if value.is_floating_point() else value.to(torch.long)
            payload = json.dumps(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "values": value.tolist(),
                },
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            digest.update(payload)
            digest.update(b"\0")
        return digest.hexdigest()


def _index_tensor(rows: list[tuple[int, ...]], width: int) -> Tensor:
    if not rows:
        return torch.empty(width, 0, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long).T.contiguous()


def _reference_dihedral(coords: Tensor, indices: tuple[int, int, int, int]) -> float:
    p0, p1, p2, p3 = (coords[index].to(torch.float64) for index in indices)
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_hat = b1 / b1.norm().clamp_min(1e-12)
    v = b0 - (b0 * b1_hat).sum() * b1_hat
    w = b2 - (b2 * b1_hat).sum() * b1_hat
    if float(v.norm()) < 1e-12 or float(w.norm()) < 1e-12:
        return 0.0
    x = (v * w).sum()
    y = (torch.linalg.cross(b1_hat, v, dim=-1) * w).sum()
    return float(torch.atan2(y, x))


def _reference_angle(coords: Tensor, indices: tuple[int, int, int]) -> float:
    left = coords[indices[0]] - coords[indices[1]]
    right = coords[indices[2]] - coords[indices[1]]
    if float(left.norm()) < 1e-8 or float(right.norm()) < 1e-8:
        raise UnsupportedPhysicalChemistryError(
            "degenerate_ligand_reference_geometry",
            "ligand input conformer contains a degenerate covalent angle",
            details={"atom_indices": list(indices)},
        )
    value = torch.atan2(
        torch.linalg.cross(left, right, dim=-1).norm(),
        (left * right).sum(),
    )
    if not bool(torch.isfinite(value)):
        raise UnsupportedPhysicalChemistryError(
            "nonfinite_ligand_reference_geometry",
            "ligand input conformer produced a non-finite covalent angle",
            details={"atom_indices": list(indices)},
        )
    return float(value)


def _graph_distances(n_atoms: int, bonds: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    result = [[n_atoms + 1] * n_atoms for _ in range(n_atoms)]
    for source in range(n_atoms):
        result[source][source] = 0
        queue = [source]
        for node in queue:
            for neighbor in adjacency[node]:
                if result[source][neighbor] <= result[source][node] + 1:
                    continue
                result[source][neighbor] = result[source][node] + 1
                queue.append(neighbor)
    return result


def build_physical_topology(mol: Chem.Mol, fragment_id: Tensor) -> PhysicalTopology:
    """Build only coordinate-varying terms for the fragment representation."""
    n_atoms = mol.GetNumAtoms()
    fragment_id = fragment_id.detach().cpu().to(torch.long).view(-1)
    if fragment_id.numel() != n_atoms:
        raise ValueError("fragment_id length must match ligand atoms")
    if mol.GetNumConformers() == 0:
        raise ValueError("physical topology requires a ligand conformer")

    raw = load_effff_v2()
    defaults = raw["defaults"]
    atomic_numbers = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atom_params = element_parameters(atomic_numbers, dtype=torch.float64)
    coords = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float64)
    if not bool(torch.isfinite(coords).all()):
        raise UnsupportedPhysicalChemistryError(
            "nonfinite_ligand_reference_geometry",
            "ligand input conformer contains non-finite coordinates",
        )

    all_bonds: list[tuple[int, int]] = []
    bond_rows: list[tuple[int, int]] = []
    bond_r0: list[float] = []
    bond_k: list[float] = []
    cross_bonds: list[Chem.Bond] = []
    for bond in mol.GetBonds():
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        all_bonds.append((i, j))
        if int(fragment_id[i]) == int(fragment_id[j]):
            continue
        reference_distance = float((coords[i] - coords[j]).norm())
        if not math.isfinite(reference_distance) or reference_distance < 1e-8:
            raise UnsupportedPhysicalChemistryError(
                "degenerate_ligand_reference_geometry",
                "ligand input conformer contains a degenerate cut bond",
                details={"atom_indices": [i, j]},
            )
        bond_rows.append((i, j))
        bond_r0.append(reference_distance)
        order = max(1.0, float(bond.GetBondTypeAsDouble()))
        bond_k.append(float(defaults["bond_k"]) * order)
        cross_bonds.append(bond)

    angle_rows: list[tuple[int, int, int]] = []
    angle_theta0: list[float] = []
    angle_k: list[float] = []
    for center in range(n_atoms):
        neighbors = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(center).GetNeighbors())
        for left, right in itertools.combinations(neighbors, 2):
            fragments = {
                int(fragment_id[left]),
                int(fragment_id[center]),
                int(fragment_id[right]),
            }
            if len(fragments) == 1:
                continue
            row = (left, center, right)
            angle_rows.append(row)
            angle_theta0.append(_reference_angle(coords, row))
            angle_k.append(float(defaults["angle_k"]))

    proper_seen: set[tuple[int, int, int, int]] = set()
    proper_rows: list[tuple[int, int, int, int]] = []
    proper_periodicity: list[int] = []
    proper_phase: list[float] = []
    proper_k: list[float] = []
    proper_weight: list[float] = []
    proper_cut_bond_id: list[int] = []
    for cut_bond_id, bond in enumerate(cross_bonds):
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(j).GetNeighbors() if atom.GetIdx() != k)
        right = sorted(atom.GetIdx() for atom in mol.GetAtomWithIdx(k).GetNeighbors() if atom.GetIdx() != j)
        bond_rows_local: list[tuple[int, int, int, int]] = []
        for i, l in itertools.product(left, right):
            row = (i, j, k, l)
            reverse = tuple(reversed(row))
            key = min(row, reverse)
            if key in proper_seen:
                continue
            proper_seen.add(key)
            bond_rows_local.append(row)
        if not bond_rows_local:
            continue
        weight = 1.0 / len(bond_rows_local)
        for row in bond_rows_local:
            proper_rows.append(row)
            planar = (
                bond.GetIsConjugated()
                or mol.GetAtomWithIdx(j).GetIsAromatic()
                or mol.GetAtomWithIdx(k).GetIsAromatic()
            )
            proper_periodicity.append(2 if planar else 3)
            proper_phase.append(math.pi if planar else 0.0)
            proper_k.append(float(defaults["proper_k"]))
            proper_weight.append(weight)
            proper_cut_bond_id.append(cut_bond_id)

    improper_rows: list[tuple[int, int, int, int]] = []
    improper_phi0: list[float] = []
    improper_k: list[float] = []
    improper_planar: list[bool] = []
    for center in range(n_atoms):
        atom = mol.GetAtomWithIdx(center)
        neighbors = sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
        if len(neighbors) < 3:
            continue
        is_chiral = atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
        is_planar = atom.GetIsAromatic() or atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2
        if not (is_chiral or is_planar):
            continue
        for chosen in itertools.combinations(neighbors, 3):
            row = (chosen[0], center, chosen[1], chosen[2])
            if len({int(fragment_id[index]) for index in row}) == 1:
                continue
            improper_rows.append(row)
            improper_phi0.append(_reference_dihedral(coords, row) if is_chiral else 0.0)
            planar = bool(is_planar and not is_chiral)
            improper_k.append(
                float(
                    defaults[
                        "planar_improper_k" if planar else "chiral_improper_k"
                    ]
                )
            )
            improper_planar.append(planar)

    distances = _graph_distances(n_atoms, all_bonds)
    pair_rows: list[tuple[int, int]] = []
    pair_scale: list[float] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if int(fragment_id[i]) == int(fragment_id[j]):
                continue
            separation = distances[i][j]
            if separation <= 2:
                continue
            pair_rows.append((i, j))
            pair_scale.append(
                float(defaults["one_four_lj_scale"]) if separation == 3 else 1.0
            )

    return PhysicalTopology(
        atomic_numbers=atomic_numbers,
        fragment_id=fragment_id,
        mass=atom_params.mass,
        uff_x=atom_params.uff_x,
        uff_d=atom_params.uff_d,
        vdw_radius=atom_params.vdw_radius,
        bond_index=_index_tensor(bond_rows, 2),
        bond_r0=torch.tensor(bond_r0, dtype=torch.float64),
        bond_k=torch.tensor(bond_k, dtype=torch.float64),
        angle_index=_index_tensor(angle_rows, 3),
        angle_theta0=torch.tensor(angle_theta0, dtype=torch.float64),
        angle_k=torch.tensor(angle_k, dtype=torch.float64),
        proper_index=_index_tensor(proper_rows, 4),
        proper_periodicity=torch.tensor(proper_periodicity, dtype=torch.long),
        proper_phase=torch.tensor(proper_phase, dtype=torch.float64),
        proper_k=torch.tensor(proper_k, dtype=torch.float64),
        proper_weight=torch.tensor(proper_weight, dtype=torch.float64),
        proper_cut_bond_id=torch.tensor(proper_cut_bond_id, dtype=torch.long),
        improper_index=_index_tensor(improper_rows, 4),
        improper_phi0=torch.tensor(improper_phi0, dtype=torch.float64),
        improper_k=torch.tensor(improper_k, dtype=torch.float64),
        improper_planar=torch.tensor(improper_planar, dtype=torch.bool),
        ligand_pair_index=_index_tensor(pair_rows, 2),
        ligand_pair_scale=torch.tensor(pair_scale, dtype=torch.float64),
    )


__all__ = ["PhysicalTopology", "build_physical_topology"]
