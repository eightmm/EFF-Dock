#!/usr/bin/env python3
"""Run the pinned PoseBench FABind pipeline with an explicit repeat seed."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--fabind-dir", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--protein-dir", type=Path, required=True)
    parser.add_argument("--save-mols-dir", type=Path, required=True)
    parser.add_argument("--save-pt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=128)
    parser.add_argument("--cuda-device-index", type=int, default=0)
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    python = str(args.python.resolve())
    fabind_dir = args.fabind_dir.resolve()
    args.save_mols_dir.mkdir(parents=True, exist_ok=True)
    args.save_pt_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            python,
            str(fabind_dir / "inference_preprocess_mol_confs.py"),
            "--index_csv",
            str(args.input_csv.resolve()),
            "--save_mols_dir",
            str(args.save_mols_dir.resolve()),
            "--num_threads",
            "1",
        ]
    )
    run(
        [
            python,
            str(fabind_dir / "inference_preprocess_protein.py"),
            "--pdb_file_dir",
            str(args.protein_dir.resolve()),
            "--save_pt_dir",
            str(args.save_pt_dir.resolve()),
            "--cuda_device_index",
            str(args.cuda_device_index),
        ]
    )
    run(
        [
            python,
            str(fabind_dir / "fabind_inference.py"),
            "--ckpt",
            str(args.checkpoint.resolve()),
            "--batch_size",
            "4",
            "--seed",
            str(args.seed),
            "--test-gumbel-soft",
            "--redocking",
            "--post-optim",
            "--write-mol-to-file",
            "--sdf-output-path-post-optim",
            str(args.output_dir.resolve()),
            "--index-csv",
            str(args.input_csv.resolve()),
            "--preprocess-dir",
            str(args.save_pt_dir.resolve()),
            "--cuda_device_index",
            str(args.cuda_device_index),
        ]
    )


if __name__ == "__main__":
    main()
