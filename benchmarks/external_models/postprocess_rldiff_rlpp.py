#!/usr/bin/env python3
"""Run RLDiff RL++ minimization/reranking on an existing native pose bank."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

SMINA_UNSUPPORTED_RECEPTOR_ELEMENTS = frozenset({"B", "V", "Mo", "Xe"})


def _pdb_element(line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element[0].upper() + element[1:].lower()
    atom_name = line[12:16].strip()
    match = re.match(r"[0-9]*([A-Za-z]{1,2})", atom_name)
    if match is None:
        return ""
    raw = match.group(1)
    return raw[0].upper() + raw[1:].lower()


def prepare_smina_receptor(
    source: Path,
    destination: Path,
    *,
    unsupported_elements: frozenset[str] = SMINA_UNSUPPORTED_RECEPTOR_ELEMENTS,
) -> dict[str, object]:
    """Remove complete non-protein residues containing smina-unsupported atoms.

    Smina aborts before scoring when a receptor contains an unsupported AutoDock
    atom type.  Removing only the offending atom would leave a chemically broken
    cofactor, so this compatibility policy removes the complete HETATM residue
    and records every exclusion.  Unsupported ATOM records fail closed.
    """
    lines = source.read_text().splitlines(keepends=True)
    excluded_residues: set[tuple[str, str, str, str]] = set()
    excluded_elements: set[str] = set()
    for line in lines:
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            continue
        element = _pdb_element(line)
        if element not in unsupported_elements:
            continue
        residue_key = (line[17:20].strip(), line[21:22], line[22:26], line[26:27])
        if record != "HETATM":
            raise ValueError(
                f"Unsupported smina receptor element {element!r} occurs in an "
                f"ATOM record in {source}: {line.rstrip()}"
            )
        excluded_residues.add(residue_key)
        excluded_elements.add(element)

    destination.parent.mkdir(parents=True, exist_ok=True)
    removed_atom_count = 0
    output_lines: list[str] = []
    for line in lines:
        record = line[:6].strip()
        residue_key = (line[17:20].strip(), line[21:22], line[22:26], line[26:27])
        if record == "HETATM" and residue_key in excluded_residues:
            removed_atom_count += 1
            continue
        output_lines.append(line)
    destination.write_text("".join(output_lines))
    return {
        "source": str(source.resolve()),
        "prepared": str(destination.resolve()),
        "policy": "strip_unsupported_hetero_residues",
        "unsupported_elements": sorted(excluded_elements),
        "excluded_residues": [
            {
                "resname": residue[0],
                "chain": residue[1].strip(),
                "resseq": residue[2].strip(),
                "icode": residue[3].strip(),
            }
            for residue in sorted(excluded_residues)
        ],
        "removed_atom_count": removed_atom_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rldiff-root", type=Path, required=True)
    parser.add_argument("--smina-path", type=Path, required=True)
    parser.add_argument("--gnina-path", type=Path, required=True)
    parser.add_argument("--samples-per-complex", type=int, required=True)
    parser.add_argument(
        "--receptor-compat-policy",
        choices=("fail", "strip_unsupported_hetero_residues"),
        default="fail",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rldiff_root = Path(os.path.abspath(args.rldiff_root))
    sys.path.insert(0, str(rldiff_root))
    from utils.minimize_utils import minimize_and_rerank

    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_ids = [row["complex_name"] for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protein_path_map: dict[str, str] = {}
    receptor_audit: dict[str, dict[str, object]] = {}
    for row in rows:
        target_id = row["complex_name"].replace("/", "-")
        receptor = Path(row["experimental_protein"])
        if args.receptor_compat_policy == "strip_unsupported_hetero_residues":
            prepared = args.output_dir / "smina_receptors" / f"{target_id}.pdb"
            audit = prepare_smina_receptor(receptor, prepared)
            protein_path_map[target_id] = str(prepared)
            receptor_audit[target_id] = audit
        else:
            protein_path_map[target_id] = str(receptor)
            receptor_audit[target_id] = {
                "source": str(receptor.resolve()),
                "prepared": str(receptor.resolve()),
                "policy": "fail",
                "unsupported_elements": [],
                "excluded_residues": [],
                "removed_atom_count": 0,
            }
    (args.output_dir / "receptor_compatibility.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": args.receptor_compat_policy,
                "targets": receptor_audit,
            },
            indent=2,
        )
        + "\n"
    )

    raw_coverage: dict[str, int] = {}
    for target_id in target_ids:
        candidates = list(args.raw_output_dir.glob(f"index*___{target_id}"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one raw output directory for {target_id}; found {len(candidates)}"
            )
        raw_dir = candidates[0]
        raw_poses = [
            path for path in raw_dir.glob("rank*.sdf") if re.fullmatch(r"rank\d+\.sdf", path.name)
        ]
        raw_coverage[target_id] = len(raw_poses)
        if len(raw_poses) != args.samples_per_complex:
            raise RuntimeError(
                f"Raw pose count for {target_id} is {len(raw_poses)}; "
                f"expected {args.samples_per_complex}"
            )
        link = args.output_dir / raw_dir.name
        if link.is_symlink():
            if link.resolve() != raw_dir.resolve():
                raise RuntimeError(f"Existing raw-pose link points elsewhere: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to overwrite non-symlink path: {link}")
        else:
            link.symlink_to(raw_dir.resolve(), target_is_directory=True)

    minimize_and_rerank(
        out_dir=args.output_dir,
        protein_path_map=protein_path_map,
        smina_path=str(args.smina_path.resolve()),
        gnina_path=str(args.gnina_path.resolve()),
        n_workers=1,
    )

    coverage = {}
    for target_id in target_ids:
        poses = sorted((args.output_dir / "minimized_poses" / target_id).glob("rank*_gnina*.sdf"))
        coverage[target_id] = {
            "raw_pose_count": raw_coverage[target_id],
            "selected_pose_count": len(poses),
        }
    summary = {
        "schema_version": 1,
        "receptor_compat_policy": args.receptor_compat_policy,
        "expected_targets": len(target_ids),
        "targets_with_any_pose": sum(
            value["selected_pose_count"] > 0 for value in coverage.values()
        ),
        "targets_with_expected_pose_count": sum(
            value["selected_pose_count"] == args.samples_per_complex for value in coverage.values()
        ),
        "coverage": coverage,
    }
    (args.output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["targets_with_expected_pose_count"] != summary["expected_targets"]:
        raise SystemExit(
            "Incomplete RL++ postprocessing: "
            f"{summary['targets_with_expected_pose_count']}/{summary['expected_targets']} "
            "targets have the expected pose count"
        )


if __name__ == "__main__":
    main()
