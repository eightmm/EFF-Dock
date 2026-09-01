#!/usr/bin/env python3
"""Apply the frozen adaptive rigid-fragment refiner to the sealed S50 bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem

from effdock.guidance import InteractionEnergyConfig, build_physical_system
from effdock.guidance.parameterization import (
    guidance_parameter_identity,
    load_effff_v2,
)
from effdock.guidance.provenance import guidance_implementation_identity
from effdock.preprocess.protein import (
    METAL_ATOM_TOKENS,
    METAL_OTHER_TOKEN,
    RES_ATOM_TOKEN,
    UNK_ATOM_TOKEN,
)
from effdock.workflows.benchmark_inputs import load_benchmark_ligand
from effdock.workflows.evaluate import file_sha256
from effdock.workflows.relax_guidance import (
    RigidRelaxationConfig,
    relax_rigid_fragments_batch,
)

PROTOCOL_ID = "EFFDOCK-S50-REFINED-POSE-BANK-V2"
SCHEMA_VERSION = "effdock.s50_refined_pose_bank.v2"
MANIFEST_SCHEMA_VERSION = "effdock.s50_refined_pose_bank.manifest.v2"
EXPECTED_SAMPLES = 100
REFINEMENT_STEPS = 100
REFINEMENT_BATCH_SIZE = 20
PROTEIN_SHELL = 18.0
RAW_BANK_PROTOCOL = "EFFDOCK-S50-CONFIDENCE-TRAINING-BANK-V1"
RAW_BANK_SCHEMA = "effdock.s50_confidence_bank.manifest.v1"


class RefinedBankError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    attempt = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(attempt, path)
    finally:
        attempt.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    attempt = Path(raw)
    try:
        torch.save(payload, attempt)
        with attempt.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(attempt, path)
    finally:
        attempt.unlink(missing_ok=True)


def _refinement_implementation() -> dict[str, Any]:
    """Snapshot the implementation once for a worker process.

    The guidance provenance helper hashes files on disk.  A long-running shard
    must not let unrelated edits to repository metadata change the identity of
    later records after the executable modules have already been imported.
    """
    return {
        "guidance": guidance_implementation_identity(),
        "parameters": guidance_parameter_identity(),
        "torch": torch.__version__,
    }


def _semantic_implementation_contract(identity: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime-relevant portion of a recorded implementation.

    Historical V2 shards called ``guidance_implementation_identity`` once per
    complex.  That helper re-read ``pyproject.toml`` even though the worker had
    already imported its executable modules.  Repository packaging edits could
    therefore change only that file hash and the derived guidance digest inside
    otherwise identical records.  Keep every executable/runtime/parameter
    field exact while excluding only those two observational metadata fields.
    """
    if not isinstance(identity, dict):
        raise RefinedBankError("invalid refinement implementation identity")
    guidance = identity.get("guidance")
    project_inputs = guidance.get("project_inputs") if isinstance(guidance, dict) else None
    if not isinstance(guidance, dict) or not isinstance(project_inputs, dict):
        raise RefinedBankError("invalid guidance implementation identity")
    for label, value in (
        ("guidance sha256", guidance.get("sha256")),
        ("pyproject.toml sha256", project_inputs.get("pyproject.toml")),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise RefinedBankError(f"invalid {label}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise RefinedBankError(f"invalid {label}") from exc
    normalized_guidance = {
        key: value
        for key, value in guidance.items()
        if key not in {"sha256", "project_inputs"}
    }
    normalized_guidance["project_inputs"] = {
        key: value for key, value in project_inputs.items() if key != "pyproject.toml"
    }
    return {
        "guidance": normalized_guidance,
        "parameters": identity.get("parameters"),
        "torch": identity.get("torch"),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().to(torch.float32).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(b"EFFDOCK_S50_REFINED_POSES_V1\0")
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_ids_sha256(ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RefinedBankError(f"expected JSON object: {path}")
    return value


def _load_source_contract(
    raw_bank_manifest: Path, input_manifest: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    bank = _load_json(raw_bank_manifest)
    if (
        bank.get("schema_version") != RAW_BANK_SCHEMA
        or bank.get("protocol_id") != RAW_BANK_PROTOCOL
        or bank.get("status") != "complete"
        or bank.get("claim_eligible") is not True
    ):
        raise RefinedBankError("raw S50 bank is not a complete claim-eligible bank")
    records = bank.get("records")
    if not isinstance(records, list) or len(records) != 44127:
        raise RefinedBankError("raw S50 bank record inventory mismatch")
    frozen = _load_json(input_manifest)
    frozen_records = frozen.get("records")
    if not isinstance(frozen_records, list):
        raise RefinedBankError("frozen input manifest has no records")
    by_id = {
        str(row["sample_key"]): row for row in frozen_records if row.get("status") == "eligible"
    }
    bank_ids = {str(row["sample_key"]) for row in records}
    if bank_ids != set(by_id):
        raise RefinedBankError("raw bank and frozen eligible IDs differ")
    return bank, by_id


_TOKEN_TO_RES_ATOM = {token: pair for pair, token in RES_ATOM_TOKEN.items()}
_TOKEN_TO_METAL = {token: element for element, token in METAL_ATOM_TOKENS.items()}
_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _generic_obstacle_atom_name(serial: int) -> str:
    """Give each lossy catch-all receptor atom a stable PDB identifier."""
    if serial <= 0 or serial >= 36**3:
        raise RefinedBankError(
            "processed protein has too many atoms for a unique generic obstacle label"
        )
    value = serial
    digits: list[str] = []
    for _ in range(3):
        value, remainder = divmod(value, 36)
        digits.append(_BASE36[remainder])
    return "X" + "".join(reversed(digits))


def _protein_pdb_text(protein: dict[str, torch.Tensor]) -> str:
    required = {"patom_coords", "patom_token", "patom_residue_id"}
    if not required.issubset(protein):
        raise RefinedBankError("processed protein lacks atom coordinates/tokens/residues")
    coords = torch.as_tensor(protein["patom_coords"], dtype=torch.float32)
    tokens = torch.as_tensor(protein["patom_token"], dtype=torch.long)
    residue_ids = torch.as_tensor(protein["patom_residue_id"], dtype=torch.long)
    if (
        coords.ndim != 2
        or coords.shape[1] != 3
        or tokens.shape != (coords.shape[0],)
        or residue_ids.shape != (coords.shape[0],)
        or not bool(torch.isfinite(coords).all())
    ):
        raise RefinedBankError("processed protein atom tensors are malformed")
    residue_names: dict[int, str] = {}
    for residue_id, token in zip(residue_ids.tolist(), tokens.tolist(), strict=True):
        pair = _TOKEN_TO_RES_ATOM.get(token)
        if pair is not None and pair[0] not in {"ANY", "UNK"}:
            previous = residue_names.setdefault(residue_id, pair[0])
            if previous != pair[0]:
                raise RefinedBankError("one processed residue has conflicting residue tokens")
    lines: list[str] = []
    for serial, (residue_id, token, xyz) in enumerate(
        zip(residue_ids.tolist(), tokens.tolist(), coords.tolist(), strict=True), start=1
    ):
        if token in _TOKEN_TO_RES_ATOM:
            residue_name, atom_name = _TOKEN_TO_RES_ATOM[token]
            if residue_name in {"ANY", "UNK"}:
                # The frozen protein tensor intentionally collapses modified or
                # otherwise unknown residues onto backbone/terminal fallback
                # tokens.  When another atom identifies the residue, retain it;
                # otherwise keep the atom as an explicit UNK geometry obstacle
                # instead of inventing a canonical amino-acid identity.
                residue_name = residue_names.get(residue_id, "UNK")
            if atom_name.startswith("C"):
                element = "C"
            elif atom_name.startswith("N"):
                element = "N"
            elif atom_name.startswith("O"):
                element = "O"
            elif atom_name.startswith("S"):
                element = "S"
            else:
                raise RefinedBankError(f"cannot infer element for atom {atom_name}")
            record_type = "ATOM"
        elif token in _TOKEN_TO_METAL:
            element = _TOKEN_TO_METAL[token]
            residue_name = element
            atom_name = element
            record_type = "HETATM"
        elif token in {UNK_ATOM_TOKEN, METAL_OTHER_TOKEN}:
            # The lossy processed representation does not retain the element
            # behind either catch-all token.  Preserve its exact coordinate as
            # an untyped geometry-only obstacle.  The synthetic X element is
            # deliberately absent from EFF-FF, so receptor_policy=geometry_only
            # dispatches it to the bounded generic repulsion and never assigns
            # an attractive or element-specific term.
            element = "X"
            residue_name = "GEO"
            # Multiple lossy catch-all atoms may belong to the same processed
            # residue.  Their coordinates are distinct physical obstacles, so
            # retain all of them while assigning a stable unique diagnostic
            # atom identifier.  The explicit element column remains X and the
            # guidance path therefore stays generic repulsion-only.
            atom_name = _generic_obstacle_atom_name(serial)
            record_type = "HETATM"
        else:
            raise RefinedBankError(f"unknown processed protein token {token}")
        x, y, z = xyz
        lines.append(
            f"{record_type:<6}{serial:5d} {atom_name:>4s} {residue_name:>3s} "
            f"A{residue_id + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00 20.00          {element:>2s}"
        )
    return "\n".join(lines) + "\nEND\n"


def _refinement_config() -> RigidRelaxationConfig:
    return RigidRelaxationConfig(
        initialization_mode="model_prior",
        prior_sigma_angstrom=2.0,
        max_steps=REFINEMENT_STEPS,
        save_every=REFINEMENT_STEPS,
        base_step_size=1.0,
        max_translation_step_angstrom=0.10,
        max_rotation_step_degrees=5.0,
        max_atom_step_angstrom=0.10,
        max_backtracks=12,
        convergence_displacement_angstrom=0.01,
        convergence_patience=5,
        convergence_energy_absolute_kcal_mol=None,
        convergence_energy_relative=None,
        convergence_energy_patience=5,
        convergence_energy_min_steps=20,
        physical_cutoff_angstrom=8.0,
        protein_shell_cutoff_angstrom=PROTEIN_SHELL,
    )


def _refine_one(
    raw_record: dict[str, Any],
    frozen_record: dict[str, Any],
    *,
    device: torch.device,
    implementation: dict[str, Any],
) -> dict[str, Any]:
    raw_path = Path(str(raw_record["pt_path"])).resolve(strict=True)
    if file_sha256(raw_path) != raw_record["pt_sha256"]:
        raise RefinedBankError(f"changed raw pose payload: {raw_path}")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    sample_id = str(raw_record["sample_key"])
    if raw.get("pid") != sample_id or raw.get("num_samples") != EXPECTED_SAMPLES:
        raise RefinedBankError(f"{sample_id}: raw payload identity/count mismatch")
    initial = torch.as_tensor(raw["pose_atom_coords"], dtype=torch.float32)
    crystal = torch.as_tensor(raw["lig_atom_coords_crystal_centered"], dtype=torch.float32)
    fragment_id = torch.as_tensor(raw["fragment_id"], dtype=torch.long)
    pocket_center = torch.as_tensor(raw["pocket_center_used"], dtype=torch.float32)
    if (
        initial.shape[0] != EXPECTED_SAMPLES
        or initial.shape[1:] != crystal.shape
        or pocket_center.shape != (3,)
        or not bool(torch.isfinite(initial).all())
    ):
        raise RefinedBankError(f"{sample_id}: malformed raw pose tensors")
    processed_protein_path = Path(str(frozen_record["processed_protein"]["path"])).resolve(
        strict=True
    )
    if file_sha256(processed_protein_path) != frozen_record["processed_protein"]["sha256"]:
        raise RefinedBankError(f"{sample_id}: processed protein changed")
    protein = torch.load(processed_protein_path, map_location="cpu", weights_only=False)
    molecule, _ = load_benchmark_ligand(str(frozen_record["canonical_smiles"]), random_seed=0)
    pdb_text = _protein_pdb_text(protein)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as handle:
        handle.write(pdb_text)
        receptor_path = Path(handle.name)
    try:
        system = build_physical_system(
            molecule,
            receptor_path,
            fragment_id=fragment_id,
            near_coords=pocket_center.view(1, 3),
            protein_cutoff=PROTEIN_SHELL,
            coordinate_origin=pocket_center,
            receptor_policy="geometry_only",
        ).to(device=device, dtype=torch.float32)
    finally:
        receptor_path.unlink(missing_ok=True)
    config = _refinement_config()
    interaction = InteractionEnergyConfig()
    final_parts: list[torch.Tensor] = []
    statuses: list[str] = []
    terminal_steps: list[int] = []
    backtracks: list[int] = []
    zero_center = torch.zeros(3, dtype=torch.float32, device=device)
    crystal_device = crystal.to(device)
    for start in range(0, EXPECTED_SAMPLES, REFINEMENT_BATCH_SIZE):
        stop = start + REFINEMENT_BATCH_SIZE
        run = relax_rigid_fragments_batch(
            crystal_device,
            initial[start:stop].to(device),
            system,
            config=config,
            mode="unified",
            pocket_center=zero_center,
            interaction_config=interaction,
            collect_every_step_metrics=False,
            collect_contact_stats=False,
        )
        final_parts.append(run.frames[-1].detach().cpu().to(torch.float32))
        statuses.extend(run.statuses)
        terminal_steps.extend(int(value) for value in run.terminal_steps)
        backtracks.extend(int(value) for value in run.total_backtracks)
    refined = torch.cat(final_parts, dim=0)
    if refined.shape != initial.shape or not bool(torch.isfinite(refined).all()):
        raise RefinedBankError(f"{sample_id}: refinement returned invalid coordinates")
    accepted_statuses = {
        "max_steps",
        "converged_displacement",
        "converged_energy_plateau",
        "line_search_failed",
    }
    if any(status not in accepted_statuses for status in statuses):
        raise RefinedBankError(f"{sample_id}: unusable refinement status {statuses}")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "sample_key": sample_id,
        "system_id": raw_record["system_id"],
        "split": raw_record["split"],
        "split_index": int(raw_record["split_index"]),
        "source_pt_path": str(raw_path),
        "source_pt_sha256": raw_record["pt_sha256"],
        "source_pose_ensemble_sha256": raw_record["pose_ensemble_sha256"],
        "num_samples": EXPECTED_SAMPLES,
        "refinement_steps": REFINEMENT_STEPS,
        "refinement_batch_size": REFINEMENT_BATCH_SIZE,
        "refinement_mode": "unified",
        "receptor_policy": "geometry_only",
        "processed_receptor_reconstruction": "patom_token_to_synthetic_pdb_v2",
        "solver_config": asdict(config),
        "implementation": implementation,
        "pose_atom_coords_refined": refined,
        "refined_pose_ensemble_sha256": _tensor_sha256(refined),
        "terminal_statuses": statuses,
        "terminal_steps": torch.tensor(terminal_steps, dtype=torch.long),
        "total_backtracks": torch.tensor(backtracks, dtype=torch.long),
    }


def _selected_records(
    bank: dict[str, Any], *, split: str, shard_index: int, num_shards: int
) -> list[dict[str, Any]]:
    records = sorted(
        (row for row in bank["records"] if row.get("split") == split),
        key=lambda row: int(row["split_index"]),
    )
    return [row for row in records if (int(row["split_index"]) - 1) % num_shards == shard_index]


def preflight(args: argparse.Namespace) -> None:
    """Verify label-free ligand-element and processed-receptor coverage."""
    bank, frozen_by_id = _load_source_contract(args.raw_bank_manifest, args.input_manifest)
    supported = {int(value) for value in load_effff_v2()["elements"]}
    element_complex_counts: Counter[int] = Counter()
    receptor_token_counts: Counter[int] = Counter()
    generic_receptor_records = 0
    ordered_ids: list[str] = []
    records = sorted(bank["records"], key=lambda row: (row["split"], int(row["split_index"])))
    for index, raw_record in enumerate(records, start=1):
        sample_id = str(raw_record["sample_key"])
        frozen_record = frozen_by_id[sample_id]
        molecule = Chem.MolFromSmiles(str(frozen_record["canonical_smiles"]))
        if molecule is None:
            raise RefinedBankError(f"{sample_id}: canonical SMILES cannot be parsed")
        element_complex_counts.update({atom.GetAtomicNum() for atom in molecule.GetAtoms()})
        protein_path = Path(str(frozen_record["processed_protein"]["path"])).resolve(strict=True)
        if file_sha256(protein_path) != frozen_record["processed_protein"]["sha256"]:
            raise RefinedBankError(f"{sample_id}: processed protein changed")
        protein = torch.load(protein_path, map_location="cpu", weights_only=False)
        _protein_pdb_text(protein)
        tokens = torch.as_tensor(protein["patom_token"], dtype=torch.long)
        receptor_token_counts.update(int(value) for value in tokens.tolist())
        if bool(((tokens == UNK_ATOM_TOKEN) | (tokens == METAL_OTHER_TOKEN)).any()):
            generic_receptor_records += 1
        ordered_ids.append(sample_id)
        if index % 1000 == 0:
            print(f"[preflight] {index}/{len(records)}", flush=True)
    observed = set(element_complex_counts)
    if observed != supported:
        raise RefinedBankError(
            "EFF-FF element table differs from the frozen ligand cohort: "
            f"missing={sorted(observed - supported)} extra={sorted(supported - observed)}"
        )
    summary = {
        "schema_version": "effdock.s50_refined_pose_bank.preflight.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_label_free",
        "record_count": len(records),
        "record_ids_sha256": _ordered_ids_sha256(ordered_ids),
        "source_raw_bank_manifest": str(args.raw_bank_manifest.resolve()),
        "source_raw_bank_manifest_sha256": file_sha256(args.raw_bank_manifest),
        "source_input_manifest": str(args.input_manifest.resolve()),
        "source_input_manifest_sha256": file_sha256(args.input_manifest),
        "supported_and_observed_atomic_numbers": sorted(observed),
        "element_complex_counts": {
            str(key): element_complex_counts[key] for key in sorted(element_complex_counts)
        },
        "receptor_token_counts": {
            str(key): receptor_token_counts[key] for key in sorted(receptor_token_counts)
        },
        "generic_receptor_record_count": generic_receptor_records,
        "implementation": {
            **_refinement_implementation(),
            "worker_sha256": file_sha256(Path(__file__)),
        },
    }
    _atomic_write(args.output, _canonical_bytes(summary))
    _atomic_write(
        args.output.with_suffix(args.output.suffix + ".sha256"),
        f"{file_sha256(args.output)}  {args.output.resolve()}\n".encode(),
    )


def generate_shard(args: argparse.Namespace) -> None:
    bank, frozen_by_id = _load_source_contract(args.raw_bank_manifest, args.input_manifest)
    records = _selected_records(
        bank,
        split=args.split,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.only_id:
        requested = list(dict.fromkeys(args.only_id))
        split_records = {
            str(row["sample_key"]): row for row in bank["records"] if row.get("split") == args.split
        }
        missing = [sample_id for sample_id in requested if sample_id not in split_records]
        if missing:
            raise RefinedBankError(f"requested IDs are absent from split {args.split}: {missing}")
        records = [split_records[sample_id] for sample_id in requested]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RefinedBankError("selected refinement shard is empty")
    shard_root = (
        args.output_root.resolve()
        / "shards"
        / args.split
        / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}"
    )
    summary_path = shard_root / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to reuse completed shard: {summary_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA refinement requested but CUDA is unavailable")
    implementation = _refinement_implementation()
    output_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(records, start=1):
        sample_id = str(raw_record["sample_key"])
        path = shard_root / sample_id / "refined_step100.pt"
        if path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if (
                payload.get("protocol_id") != PROTOCOL_ID
                or payload.get("sample_key") != sample_id
                or payload.get("source_pt_sha256") != raw_record["pt_sha256"]
            ):
                raise RefinedBankError(f"{sample_id}: existing refined payload mismatch")
        else:
            payload = _refine_one(
                raw_record,
                frozen_by_id[sample_id],
                device=device,
                implementation=implementation,
            )
            _atomic_torch_save(path, payload)
        output_records.append(
            {
                "sample_key": sample_id,
                "system_id": raw_record["system_id"],
                "split": args.split,
                "split_index": int(raw_record["split_index"]),
                "pt_path": str(path.resolve()),
                "pt_sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "refined_pose_ensemble_sha256": payload["refined_pose_ensemble_sha256"],
            }
        )
        print(
            f"[{args.split} {args.shard_index}/{args.num_shards}] "
            f"{index}/{len(records)} {sample_id}",
            flush=True,
        )
    summary = {
        "schema_version": "effdock.s50_refined_pose_bank.shard.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "split": args.split,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_raw_bank_manifest": str(args.raw_bank_manifest.resolve()),
        "source_raw_bank_manifest_sha256": file_sha256(args.raw_bank_manifest),
        "source_input_manifest": str(args.input_manifest.resolve()),
        "source_input_manifest_sha256": file_sha256(args.input_manifest),
        "record_count": len(output_records),
        "record_ids_sha256": _ordered_ids_sha256([row["sample_key"] for row in output_records]),
        "records": output_records,
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "device": str(device),
        },
    }
    _atomic_write(summary_path, _canonical_bytes(summary))


def aggregate(args: argparse.Namespace) -> None:
    bank, _ = _load_source_contract(args.raw_bank_manifest, args.input_manifest)
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")
    preflight_path = output_root.parent / "preflight.json"
    preflight = _load_json(preflight_path)
    if (
        preflight.get("protocol_id") != PROTOCOL_ID
        or preflight.get("status") != "complete_label_free"
        or not isinstance(preflight.get("implementation"), dict)
    ):
        raise RefinedBankError(f"invalid refinement preflight: {preflight_path}")
    preflight_implementation = preflight["implementation"]
    expected_implementation = {
        "guidance": preflight_implementation.get("guidance"),
        "parameters": preflight_implementation.get("parameters"),
        "torch": preflight_implementation.get("torch")
        or preflight_implementation.get("guidance", {})
        .get("runtime_versions", {})
        .get("torch_runtime"),
    }
    expected_semantic_contract = _semantic_implementation_contract(
        expected_implementation
    )
    all_records: list[dict[str, Any]] = []
    implementation_identities: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    for split, num_shards in (("train", args.train_shards), ("val", args.val_shards)):
        for shard_index in range(num_shards):
            path = (
                output_root
                / "shards"
                / split
                / f"shard-{shard_index:03d}-of-{num_shards:03d}"
                / "summary.json"
            )
            summary = _load_json(path)
            if (
                summary.get("protocol_id") != PROTOCOL_ID
                or summary.get("status") != "complete"
                or summary.get("split") != split
                or int(summary.get("shard_index", -1)) != shard_index
                or int(summary.get("num_shards", -1)) != num_shards
            ):
                raise RefinedBankError(f"invalid shard summary: {path}")
            if summary.get("source_raw_bank_manifest_sha256") != file_sha256(
                args.raw_bank_manifest
            ):
                raise RefinedBankError(f"source bank drift in {path}")
            for record in summary.get("records", []):
                pt_path = Path(str(record["pt_path"])).resolve(strict=True)
                if file_sha256(pt_path) != record["pt_sha256"]:
                    raise RefinedBankError(f"changed refined payload: {pt_path}")
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
                coords = payload.get("pose_atom_coords_refined")
                if (
                    payload.get("protocol_id") != PROTOCOL_ID
                    or payload.get("sample_key") != record["sample_key"]
                    or not torch.is_tensor(coords)
                    or coords.shape[0] != EXPECTED_SAMPLES
                    or not bool(torch.isfinite(coords).all())
                    or _tensor_sha256(coords) != record["refined_pose_ensemble_sha256"]
                ):
                    raise RefinedBankError(f"invalid refined payload: {pt_path}")
                implementation_identities.append(payload.get("implementation"))
                all_records.append(record)
            shard_summaries.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    source_records = sorted(
        bank["records"], key=lambda row: (row["split"], int(row["split_index"]))
    )
    observed_records = sorted(all_records, key=lambda row: (row["split"], int(row["split_index"])))
    if [row["sample_key"] for row in observed_records] != [
        row["sample_key"] for row in source_records
    ]:
        raise RefinedBankError("refined manifest does not cover the exact raw bank")
    identity_counts: Counter[str] = Counter()
    guidance_sha_counts: Counter[str] = Counter()
    pyproject_sha_counts: Counter[str] = Counter()
    for identity in implementation_identities:
        if _semantic_implementation_contract(identity) != expected_semantic_contract:
            raise RefinedBankError("refinement runtime implementation changed across records")
        identity_counts[hashlib.sha256(_canonical_bytes(identity)).hexdigest()] += 1
        guidance = identity["guidance"]
        guidance_sha_counts[str(guidance["sha256"])] += 1
        pyproject_sha_counts[str(guidance["project_inputs"]["pyproject.toml"])] += 1
    inventory: dict[str, Any] = {}
    for split in ("train", "val"):
        ids = [row["sample_key"] for row in observed_records if row["split"] == split]
        inventory[split] = {
            "record_count": len(ids),
            "record_ids_sha256": _ordered_ids_sha256(ids),
        }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "claim_eligible": True,
        "source_raw_bank_manifest": str(args.raw_bank_manifest.resolve()),
        "source_raw_bank_manifest_sha256": file_sha256(args.raw_bank_manifest),
        "source_input_manifest": str(args.input_manifest.resolve()),
        "source_input_manifest_sha256": file_sha256(args.input_manifest),
        "settings": {
            "num_samples": EXPECTED_SAMPLES,
            "steps": REFINEMENT_STEPS,
            "batch_size": REFINEMENT_BATCH_SIZE,
            "mode": "unified",
            "receptor_policy": "geometry_only",
            "protein_shell_angstrom": PROTEIN_SHELL,
            "solver": asdict(_refinement_config()),
        },
        "implementation": expected_implementation,
        "implementation_observation": {
            "policy": "runtime_exact_pyproject_metadata_normalized_to_preflight_v1",
            "record_count": len(implementation_identities),
            "full_identity_sha256_counts": dict(sorted(identity_counts.items())),
            "guidance_sha256_counts": dict(sorted(guidance_sha_counts.items())),
            "pyproject_sha256_counts": dict(sorted(pyproject_sha_counts.items())),
        },
        "inventory": inventory,
        "records": observed_records,
        "shard_summaries": shard_summaries,
    }
    _atomic_write(manifest_path, _canonical_bytes(manifest))
    _atomic_write(
        manifest_path.with_suffix(".json.sha256"),
        f"{file_sha256(manifest_path)}  {manifest_path}\n".encode(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--raw-bank-manifest", type=Path, required=True)
    preflight_parser.add_argument("--input-manifest", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)
    generate = sub.add_parser("generate-shard")
    generate.add_argument("--raw-bank-manifest", type=Path, required=True)
    generate.add_argument("--input-manifest", type=Path, required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--split", choices=("train", "val"), required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    generate.add_argument("--limit", type=int)
    generate.add_argument("--only-id", action="append")
    generate.add_argument("--device", default="cuda")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--raw-bank-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--input-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--output-root", type=Path, required=True)
    aggregate_parser.add_argument("--train-shards", type=int, default=128)
    aggregate_parser.add_argument("--val-shards", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "preflight":
        preflight(args)
    elif args.command == "generate-shard":
        if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("invalid shard index/count")
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        generate_shard(args)
    elif args.command == "aggregate":
        aggregate(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
