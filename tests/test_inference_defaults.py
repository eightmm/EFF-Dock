from pathlib import Path

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
from effdock.inference.sampler import sample_unified, sample_unified_multi_sigma
from effdock.workflows.evaluate import build_arg_parser as build_evaluate_parser


def test_public_sampling_budget_is_n100_s10() -> None:
    assert DEFAULT_NUM_SAMPLES == 100
    assert DEFAULT_NUM_STEPS == 10
    assert DEFAULT_SIGMA == 2.0
    assert sample_unified.__kwdefaults__["num_steps"] == DEFAULT_NUM_STEPS
    assert sample_unified_multi_sigma.__kwdefaults__["num_steps"] == DEFAULT_NUM_STEPS


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
    parser = build_evaluate_parser()
    dataset_action = next(action for action in parser._actions if action.dest == "dataset")
    assert dataset_action.choices == ("astex", "posebusters")
    args = parser.parse_args(
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


def test_confidence_can_be_disabled_explicitly() -> None:
    dock_args = build_dock_parser().parse_args(
        [
            "--protein",
            "receptor.pdb",
            "--ligand",
            "ligand.sdf",
            "--pocket-center",
            "0,0,0",
            "--no-confidence",
        ]
    )
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

    assert dock_args.confidence_checkpoint is None
    assert evaluate_args.confidence_checkpoint is None
    assert isinstance(DEFAULT_DOCKING_CHECKPOINT, Path)
