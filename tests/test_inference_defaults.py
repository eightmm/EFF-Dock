from pathlib import Path

import pytest

from effdock.inference.defaults import (
    DEFAULT_CONFIDENCE_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_DOCKING_CHECKPOINT,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_NUM_STEPS,
    DEFAULT_POCKET_CUTOFF,
    DEFAULT_SIGMA,
)
from effdock.inference.docking import build_arg_parser as build_dock_parser
from effdock.workflows.evaluate import build_arg_parser as build_evaluate_parser


def test_promoted_sampling_budget_is_n100_s10() -> None:
    assert DEFAULT_DOCKING_CHECKPOINT == Path(
        "weights/effdock_docking_early_time_t0p10_50k.pt"
    )
    assert DEFAULT_CONFIDENCE_CHECKPOINT == Path(
        "weights/effdock_confidence_s50_raw_refined_u70k.pt"
    )
    assert DEFAULT_NUM_SAMPLES == 100
    assert DEFAULT_NUM_STEPS == 10
    assert DEFAULT_NUM_SAMPLES * DEFAULT_NUM_STEPS == 1_000
    assert DEFAULT_SIGMA == 2.0


def test_dock_uses_promoted_inference_stack_by_default() -> None:
    args = build_dock_parser().parse_args(
        [
            "--protein",
            "receptor.pdb",
            "--ligand",
            "ligand.sdf",
            "--pocket-center",
            "0,0,0",
        ]
    )

    assert args.checkpoint == DEFAULT_DOCKING_CHECKPOINT
    assert args.confidence_checkpoint == DEFAULT_CONFIDENCE_CHECKPOINT
    assert args.config == DEFAULT_CONFIG
    assert args.num_samples == DEFAULT_NUM_SAMPLES
    assert args.num_steps == DEFAULT_NUM_STEPS
    assert args.sigma == DEFAULT_SIGMA
    assert args.pocket_cutoff == DEFAULT_POCKET_CUTOFF


def test_evaluate_uses_same_promoted_inference_stack_by_default() -> None:
    args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            "astex",
            "--data-dir",
            "data/astex",
            "--pocket-centers",
            "centers.json",
        ]
    )

    assert args.checkpoint == DEFAULT_DOCKING_CHECKPOINT
    assert args.confidence_checkpoint == DEFAULT_CONFIDENCE_CHECKPOINT
    assert args.config == DEFAULT_CONFIG
    assert args.num_samples == DEFAULT_NUM_SAMPLES
    assert args.num_steps == DEFAULT_NUM_STEPS
    assert args.sigma == DEFAULT_SIGMA
    assert args.pocket_cutoff == DEFAULT_POCKET_CUTOFF


def test_dock_always_annotates_confidence_and_evaluate_can_opt_out() -> None:
    args = ["--protein", "receptor.pdb", "--ligand", "ligand.sdf", "--pocket-center", "0,0,0"]
    dock_args = build_dock_parser().parse_args([*args, "--no-score"])
    assert dock_args.confidence_checkpoint == DEFAULT_CONFIDENCE_CHECKPOINT
    assert dock_args.score is False
    with pytest.raises(SystemExit):
        build_dock_parser().parse_args([*args, "--no-confidence"])
    evaluate_args = build_evaluate_parser().parse_args(
        [
            "--dataset",
            "astex",
            "--data-dir",
            "data/astex",
            "--pocket-centers",
            "centers.json",
            "--no-confidence",
        ]
    )

    assert evaluate_args.confidence_checkpoint is None
    assert isinstance(DEFAULT_DOCKING_CHECKPOINT, Path)
