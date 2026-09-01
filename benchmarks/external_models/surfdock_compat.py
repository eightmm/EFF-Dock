"""Runtime-only compatibility fixes for the frozen SurfDock checkout."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rdkit import Chem
from rdkit.Chem import AllChem


def assign_missing_stereochemistry_from_3d(mol: Chem.Mol) -> int:
    """Perceive missing double-bond stereo before SurfDock discards SDF coordinates."""

    before = sum(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    if mol.GetNumConformers():
        Chem.AssignStereochemistryFrom3D(mol)
    after = sum(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    )
    return max(0, after - before)


def preserve_biopython_pdb_on_rdkit_failure(
    mol: Chem.Mol | None,
    filename: str,
    writer: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Keep the already-written PDBIO file when SurfDock's optional rewrite fails."""

    if mol is not None:
        return writer(mol, filename, *args, **kwargs)
    path = Path(filename)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"RDKit returned no receptor and no PDBIO fallback exists: {path}")
    print(
        "[SurfDock compatibility] retained the valid PDBIO receptor because "
        f"RDKit could not perform its optional rewrite: {path}",
        flush=True,
    )
    return None


def install_surfdock_compat() -> None:
    """Patch imported SurfDock aliases without modifying the frozen upstream tree."""

    import datasets.process_mols as process_mols

    if getattr(process_mols, "_effdock_compat_installed", False):
        return

    original_reader = process_mols.read_molecule
    original_generate = process_mols.generate_conformer
    original_writer = process_mols.Chem.MolToPDBFile

    def compatible_reader(*args: Any, **kwargs: Any) -> Chem.Mol | None:
        mol = original_reader(*args, **kwargs)
        if mol is not None:
            assigned = assign_missing_stereochemistry_from_3d(mol)
            if assigned:
                print(
                    "[SurfDock compatibility] restored "
                    f"{assigned} missing double-bond stereo assignment(s) from 3D.",
                    flush=True,
                )
        return mol

    def compatible_generate_conformer(
        mol: Chem.Mol,
        useRandomCoords: bool = True,
    ) -> None:
        error: Exception | None = None
        try:
            original_generate(mol, useRandomCoords=useRandomCoords)
        except Exception as caught:
            error = caught
        if mol.GetNumConformers():
            return

        # The released helper ignores the final EmbedMolecule status and then
        # calls MMFF with conformer 0. Retry with bounded ETKDG settings and
        # report a real failure rather than surfacing the opaque Bad Conformer Id.
        for enforce_chirality in (True, False):
            mol.RemoveAllConformers()
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxAttempts = 2000
            params.enforceChirality = enforce_chirality
            params.randomSeed = 0xEFFD0C
            if AllChem.EmbedMolecule(mol, params) == 0 and mol.GetNumConformers():
                print(
                    "[SurfDock compatibility] recovered a failed independent "
                    f"conformer (enforceChirality={enforce_chirality}).",
                    flush=True,
                )
                return
        detail = f": {error}" if error is not None else ""
        raise RuntimeError(f"SurfDock conformer generation failed after bounded retries{detail}")

    def compatible_writer(
        mol: Chem.Mol | None,
        filename: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return preserve_biopython_pdb_on_rdkit_failure(
            mol, filename, original_writer, *args, **kwargs
        )

    process_mols.read_molecule = compatible_reader
    process_mols.generate_conformer = compatible_generate_conformer
    process_mols.Chem.MolToPDBFile = compatible_writer
    process_mols._effdock_compat_installed = True
