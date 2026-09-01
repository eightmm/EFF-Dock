#!/usr/bin/env python3
"""Repair DiffBindFR's non-standard two-character PDB chain identifiers.

DiffBindFR renames chains beyond ``Z`` to ``AA``, ``BA``, ... when exporting a
flexible receptor.  Those identifiers insert one byte into the fixed-width PDB
record and shift residue/coordinate columns, so ProDy cannot read the structure
during native MDN scoring.  This adapter preserves every raw export, writes a
fixed-width sibling PDB, and points ``results.csv`` at that sibling.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

PDB_RECORDS = ("ATOM  ", "HETATM", "TER   ")
MIN_CHAIN_SIGNATURE_OVERLAP = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, required=True)
    return parser.parse_args()


def chain_token(line: str) -> str:
    """Return a standard or DiffBindFR-expanded chain token from a PDB line."""

    if len(line) <= 21 or not line.startswith(PDB_RECORDS):
        return ""
    if len(line) > 22 and line[21].isalpha() and line[22].isalpha():
        return line[21:23]
    return line[21]


def ordered_atom_chains(path: Path) -> tuple[list[str], Counter[str]]:
    ordered: list[str] = []
    counts: Counter[str] = Counter()
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            token = chain_token(line)
            if not token:
                raise ValueError(f"Missing chain identifier in {path}: {line.rstrip()}")
            if token not in counts:
                ordered.append(token)
            counts[token] += 1
    if not ordered:
        raise ValueError(f"No protein atom records found in {path}")
    return ordered, counts


def normalize_chain_line(line: str, token: str, mapped: str) -> str:
    """Return a fixed-width PDB line with ``token`` replaced by ``mapped``."""

    if len(mapped) != 1:
        raise ValueError(f"Source chain is not a one-character PDB ID: {mapped!r}")
    if len(token) == 2:
        return line[:21] + mapped + line[23:]
    return line[:21] + mapped + line[22:]


def atom_signatures(path: Path, chain_map: dict[str, str] | None = None) -> dict[str, Counter[tuple[str, ...]]]:
    """Collect coordinate-independent PDB atom identities by normalized chain."""

    signatures: dict[str, Counter[tuple[str, ...]]] = {}
    with path.open(errors="replace") as handle:
        for raw_line in handle:
            if not raw_line.startswith(("ATOM  ", "HETATM")):
                continue
            token = chain_token(raw_line)
            mapped = chain_map.get(token, token) if chain_map is not None else token
            line = normalize_chain_line(raw_line, token, mapped)
            signature = (
                line[:6],
                line[12:16].strip(),
                line[16],
                line[17:20].strip(),
                line[22:26].strip(),
                line[26],
                line[76:78].strip(),
            )
            signatures.setdefault(mapped, Counter())[signature] += 1
    return signatures


def repair_pdb(source: Path, exported: Path) -> tuple[Path, dict[str, object]]:
    source_chains, source_counts = ordered_atom_chains(source)
    exported_chains, exported_counts = ordered_atom_chains(exported)
    malformed = [token for token in exported_chains if len(token) == 2]
    if not malformed:
        return exported, {"repaired": False, "chain_map": {}}
    if len(source_chains) != len(exported_chains):
        raise ValueError(
            f"Chain-count mismatch for {exported}: source={source_chains}, "
            f"exported={exported_chains}"
        )

    # DiffBindFR's frozen environment uses Python 3.9, where ``zip(strict=...)``
    # does not exist.  The explicit cardinality check above provides the same
    # fail-closed guarantee without widening the model environment.
    chain_map = dict(zip(exported_chains, source_chains))
    source_signatures = atom_signatures(source)
    exported_signatures = atom_signatures(exported, chain_map)
    validation: dict[str, dict[str, object]] = {}
    for exported_chain, source_chain in chain_map.items():
        source_signature = source_signatures[source_chain]
        exported_signature = exported_signatures[source_chain]
        shared = sum((source_signature & exported_signature).values())
        source_total = sum(source_signature.values())
        exported_total = sum(exported_signature.values())
        # Flexible-receptor export legitimately drops crystallographic waters,
        # cofactors, and unresolved side-chain atoms from the source PDB.  The
        # safety property needed for chain relabeling is that nearly every
        # *exported* atom belongs to the mapped source chain, not that every
        # source record survives model preparation.
        overlap = shared / exported_total if exported_total else 0.0
        validation[f"{exported_chain}->{source_chain}"] = {
            "source_atoms": source_counts[source_chain],
            "exported_atoms": exported_counts[exported_chain],
            "shared_atom_signatures": shared,
            "exported_signature_overlap": overlap,
            "source_signature_coverage": shared / source_total if source_total else 0.0,
        }
        if overlap < MIN_CHAIN_SIGNATURE_OVERLAP:
            raise ValueError(
                f"Atom-identity overlap too low for {exported_chain}->{source_chain} "
                f"in {exported}: {overlap:.6f} < {MIN_CHAIN_SIGNATURE_OVERLAP:.6f}"
            )

    repaired = exported.with_suffix(".chain_compat.pdb")
    temporary = repaired.with_suffix(repaired.suffix + ".tmp")
    with exported.open(errors="replace") as source_handle, temporary.open("w") as output:
        for line in source_handle:
            token = chain_token(line)
            if token in chain_map:
                line = normalize_chain_line(line, token, chain_map[token])
            output.write(line)
    os.replace(temporary, repaired)

    repaired_chains, repaired_counts = ordered_atom_chains(repaired)
    expected_repaired_counts = Counter(
        {chain_map[token]: count for token, count in exported_counts.items()}
    )
    if repaired_chains != source_chains or repaired_counts != expected_repaired_counts:
        raise ValueError(f"Post-repair chain validation failed for {repaired}")
    with repaired.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            float(line[30:38])
            float(line[38:46])
            float(line[46:54])

    return repaired, {
        "repaired": True,
        "chain_map": chain_map,
        "source": str(source.resolve()),
        "raw_export": str(exported.resolve()),
        "compatible_export": str(repaired.resolve()),
        "minimum_exported_signature_overlap": MIN_CHAIN_SIGNATURE_OVERLAP,
        "chain_validation": validation,
    }


def main() -> None:
    args = parse_args()
    results_csv = args.results_csv.resolve()
    if not results_csv.is_file():
        raise FileNotFoundError(results_csv)
    with results_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {results_csv}")
        required = {"protein", "protein_pdb"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing columns {sorted(required)} in {results_csv}")
        fieldnames = reader.fieldnames
        rows = list(reader)

    audit_by_pair: dict[tuple[Path, Path], dict[str, object]] = {}
    for row in rows:
        source = Path(row["protein"]).resolve()
        exported = Path(row["protein_pdb"]).resolve()
        if exported.name.endswith(".chain_compat.pdb"):
            continue
        pair = (source, exported)
        if pair not in audit_by_pair:
            compatible, audit = repair_pdb(source, exported)
            audit_by_pair[pair] = {**audit, "compatible": str(compatible)}
        row["protein_pdb"] = str(audit_by_pair[pair]["compatible"])

    changed = sum(bool(item["repaired"]) for item in audit_by_pair.values())
    if changed:
        backup = results_csv.with_name("results.pre_chain_compat.csv")
        if not backup.exists():
            backup.write_bytes(results_csv.read_bytes())
        temporary = results_csv.with_suffix(results_csv.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, results_csv)

    audit_path = results_csv.with_name("chain_compat_audit.json")
    audit_payload = {
        "schema_version": 1,
        "results_csv": str(results_csv),
        "rows": len(rows),
        "unique_exports_checked": len(audit_by_pair),
        "unique_exports_repaired": changed,
        "repairs": [item for item in audit_by_pair.values() if item["repaired"]],
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n")
    print(
        f"DiffBindFR chain compatibility: repaired={changed}/"
        f"{len(audit_by_pair)} exports"
    )


if __name__ == "__main__":
    main()
