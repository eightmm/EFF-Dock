from __future__ import annotations

from effdock.guidance.provenance import guidance_implementation_identity


def test_guidance_implementation_covers_sampling_dynamics_source_closure() -> None:
    first = guidance_implementation_identity()
    second = guidance_implementation_identity()
    assert first == second
    assert len(first["sha256"]) == 64

    files = set(first["files"])
    required = {
        "checkpoint.py",
        "data/dataset.py",
        "evaluation/benchmark.py",
        "evaluation/pose_scoring.py",
        "evaluation/pose_validity.py",
        "geometry/flow_matching.py",
        "geometry/se3.py",
        "inference/defaults.py",
        "inference/docking.py",
        "inference/preprocess.py",
        "inference/sampler.py",
        "models/effdock.py",
        "models/equivariant.py",
        "models/nn_utils.py",
        "preprocess/fragments.py",
        "preprocess/graph.py",
        "preprocess/graph_types.py",
        "preprocess/ligand.py",
        "preprocess/protein.py",
        "workflows/benchmark_inputs.py",
        "workflows/evaluate.py",
        "workflows/guidance_coverage_audit.py",
    }
    assert required <= files
