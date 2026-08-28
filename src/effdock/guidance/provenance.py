"""Deterministic implementation identity shared by audit and sampling."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from rdkit import rdBase


def guidance_implementation_identity() -> dict[str, object]:
    """Hash every in-repository source/input that defines guided evaluation."""
    effdock_root = Path(__file__).resolve().parents[1]
    project_root = effdock_root.parents[1]
    source_paths = [
        effdock_root / "checkpoint.py",
        effdock_root / "workflows" / "benchmark_inputs.py",
        effdock_root / "workflows" / "evaluate.py",
        effdock_root / "workflows" / "guidance_coverage_audit.py",
        *sorted((effdock_root / "data").glob("*.py")),
        *sorted((effdock_root / "evaluation").glob("*.py")),
        *sorted((effdock_root / "geometry").glob("*.py")),
        *sorted((effdock_root / "guidance").glob("*.py")),
        *sorted((effdock_root / "guidance" / "parameters").glob("*")),
        *sorted((effdock_root / "inference").glob("*.py")),
        *sorted((effdock_root / "models").glob("*.py")),
        *sorted((effdock_root / "preprocess").glob("*.py")),
    ]
    source_paths = sorted(
        {path.resolve() for path in source_paths if path.is_file()},
        key=lambda path: path.relative_to(effdock_root).as_posix(),
    )
    runtime_versions = {
        name: importlib.metadata.version(name)
        for name in (
            "cuequivariance",
            "e3nn",
            "numpy",
            "rdkit",
            "torch",
            "torch-cluster",
            "torch-geometric",
            "torch-scatter",
            "torch-sparse",
        )
    }
    runtime_versions["rdkit_runtime"] = rdBase.rdkitVersion
    runtime_versions["torch_runtime"] = torch.__version__
    project_inputs = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (project_root / "pyproject.toml", project_root / "uv.lock")
        if path.is_file()
    }
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_GUIDANCE_IMPLEMENTATION_V1\0")
    relative_paths: list[str] = []
    for path in source_paths:
        relative = path.relative_to(effdock_root).as_posix()
        relative_paths.append(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(
        json.dumps(runtime_versions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(
        json.dumps(project_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return {
        "schema_version": "effdock.guidance_implementation.v1",
        "sha256": digest.hexdigest(),
        "files": relative_paths,
        "runtime_versions": runtime_versions,
        "project_inputs": project_inputs,
    }


def physical_system_reference_sha256(system) -> str:
    """Hash every fixed tensor/parameter input consumed by GuidanceEnergy."""
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_PHYSICAL_SYSTEM_REFERENCE_V1\0")
    digest.update(system.topology.reference_sha256().encode("ascii"))
    digest.update(b"\0")
    interaction_hash = (
        system.interaction_topology.reference_sha256()
        if system.interaction_topology is not None
        else ""
    )
    digest.update(interaction_hash.encode("ascii"))
    digest.update(b"\0")
    for name in (
        "protein_coords",
        "protein_atomic_numbers",
        "protein_uff_x",
        "protein_uff_d",
        "protein_vdw_radius",
        "geometry_obstacle_coords",
        "geometry_obstacle_atomic_numbers",
        "geometry_obstacle_uff_x",
        "geometry_obstacle_uff_d",
        "geometry_obstacle_is_generic",
    ):
        value = getattr(system, name, None)
        if value is None:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\0")
    for name in ("parameter_set", "interaction_parameter_set"):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                getattr(system, name),
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["guidance_implementation_identity", "physical_system_reference_sha256"]
