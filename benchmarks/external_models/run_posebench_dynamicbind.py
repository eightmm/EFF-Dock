#!/usr/bin/env python3
"""Run pinned DynamicBind target inputs with isolated, auditable outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

_SAFE_RANK_RENAME = '''def rename_files_by_confidence(directory_path, molecule_type: Literal["ligand", "receptor"] = "ligand"):
    """Rank files without overwriting a source file during an in-place permutation."""
    files = [file for file in os.listdir(directory_path) if f"{molecule_type}_lddt" in file]
    files.sort(key=lambda filename: (-float(os.path.splitext(filename)[0].split("_lddt")[-1].split("_affinity")[0]), -float(os.path.splitext(filename)[0].split("_affinity")[-1])))
    rename_plan = []
    for rank, filename in enumerate(files, start=1):
        confidence = os.path.splitext(filename)[0].split("_lddt")[-1].split("_affinity")[0]
        affinity = os.path.splitext(filename)[0].split("_affinity")[-1]
        extension = os.path.splitext(filename)[-1]
        temporary = f".effdock_rank_tmp_{uuid.uuid4().hex}{extension}"
        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, temporary))
        rename_plan.append((temporary, f"rank{rank}_{molecule_type}_lddt{confidence}_affinity{affinity}{extension}"))
    for temporary, destination in rename_plan:
        os.rename(os.path.join(directory_path, temporary), os.path.join(directory_path, destination))
'''


def prepare_compatible_runner(dynamicbind_root: Path, output_dir: Path) -> Path:
    """Create an isolated runner with collision-safe native confidence reranking."""
    source_path = dynamicbind_root / "run_single_protein_inference.py"
    source = source_path.read_text()
    function_start = source.index("def rename_files_by_confidence(")
    function_end = source.index("\ndef swap_dir_names", function_start)
    patched = source[:function_start] + _SAFE_RANK_RENAME + source[function_end:]
    script_folder_line = "script_folder = os.path.dirname(file_path)"
    if script_folder_line not in patched:
        raise ValueError("DynamicBind runner script_folder marker is missing")
    patched = patched.replace(
        script_folder_line,
        f"script_folder = {str(dynamicbind_root)!r}",
        1,
    )
    runner = output_dir / "workdirs" / "run_single_protein_inference_compat.py"
    runner.write_text(patched)
    metadata = {
        "schema_version": 1,
        "source": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "compatibility_fix": "two_phase_collision_safe_native_confidence_reranking",
    }
    (output_dir / "dynamicbind_compatibility.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamicbind-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--protein-dir", type=Path, required=True)
    parser.add_argument("--ligand-csv-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--samples-per-complex", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--target-id", action="append", default=[])
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dynamicbind_root = args.dynamicbind_root.resolve()
    python = args.python.resolve()
    protein_dir = args.protein_dir.resolve()
    ligand_csv_dir = args.ligand_csv_dir.resolve()
    output_dir = args.output_dir.resolve()
    result_root = dynamicbind_root / "inference" / "outputs" / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    (output_dir / "poses").mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)
    (output_dir / "workdirs").mkdir(parents=True, exist_ok=True)
    compatible_runner = prepare_compatible_runner(dynamicbind_root, output_dir)

    ligand_inputs = sorted(ligand_csv_dir.glob("*.csv"))
    if args.target_id:
        requested = set(args.target_id)
        ligand_inputs = [path for path in ligand_inputs if path.stem in requested]
        missing = requested.difference(path.stem for path in ligand_inputs)
        if missing:
            raise ValueError(f"DynamicBind target IDs not found: {sorted(missing)}")
    if not ligand_inputs:
        raise ValueError(f"No DynamicBind ligand CSVs found in {ligand_csv_dir}")

    coverage: dict[str, dict[str, object]] = {}
    for ligand_csv in ligand_inputs:
        target_id = ligand_csv.stem
        protein_matches = sorted(protein_dir.glob(f"{target_id}_*.pdb"))
        if len(protein_matches) != 1:
            coverage[target_id] = {
                "pose_count": 0,
                "return_code": None,
                "error": f"expected one receptor, found {len(protein_matches)}",
            }
            continue
        protein = protein_matches[0]
        header = f"{args.run_tag}_{target_id}"
        target_result = result_root / header / "index0_idx_0"
        pose_pattern = re.compile(r"rank\d+_ligand_.*\.sdf")

        def poses() -> list[Path]:
            if not target_result.is_dir():
                return []
            return sorted(
                path
                for path in target_result.glob("rank*_ligand*.sdf")
                if pose_pattern.fullmatch(path.name)
            )

        return_code = 0
        error = None
        if len(poses()) != args.samples_per_complex:
            target_workdir = output_dir / "workdirs" / target_id
            target_workdir.mkdir(parents=True, exist_ok=True)
            command = [
                str(python),
                str(compatible_runner),
                str(protein),
                str(ligand_csv),
                "--samples_per_complex",
                str(args.samples_per_complex),
                "--savings_per_complex",
                "1",
                "--inference_steps",
                str(args.inference_steps),
                "--batch_size",
                str(args.batch_size),
                "--cache_path",
                str(output_dir / "cache" / target_id),
                "--header",
                header,
                "--device",
                "0",
                "--python",
                str(python),
                "--relax_python",
                str(python),
                "--results",
                str(result_root),
                "--no_relax",
                "--paper",
                "--seed",
                str(args.seed),
            ]
            completed = subprocess.run(command, cwd=target_workdir, check=False)
            return_code = completed.returncode
            if return_code != 0:
                error = f"upstream exit code {return_code}"

        found_poses = poses()
        pose_link = output_dir / "poses" / target_id
        if found_poses and not pose_link.exists():
            pose_link.symlink_to(target_result)
        coverage[target_id] = {
            "pose_count": len(found_poses),
            "return_code": return_code,
            "error": error,
            "result_dir": str(target_result),
        }

    expected = len(ligand_inputs)
    complete = sum(
        row["pose_count"] == args.samples_per_complex for row in coverage.values()
    )
    summary = {
        "schema_version": 1,
        "expected_targets": expected,
        "targets_with_any_pose": sum(row["pose_count"] > 0 for row in coverage.values()),
        "targets_with_expected_pose_count": complete,
        "expected_poses_per_target": args.samples_per_complex,
        "seed": args.seed,
        "inference_steps": args.inference_steps,
        "target_filter": args.target_id,
        "coverage": coverage,
    }
    (output_dir / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.fail_on_incomplete and complete != expected:
        raise SystemExit(f"Incomplete DynamicBind coverage: {complete}/{expected}")


if __name__ == "__main__":
    main()
