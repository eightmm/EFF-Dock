#!/usr/bin/env python3
"""Compare first, Vina-like, and RMSD-oracle selectors on the eta sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from effdock.workflows.guidance_budget_full_report import load_full_cohort_audits
from effdock.workflows.guidance_budget_posebusters_report import _aggregate_cell
from effdock.workflows.guidance_budget_report import (
    DATASETS,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_EXPECTED_SHARDS,
)
from effdock.workflows.guidance_eta_sweep_posebusters_report import (
    _require_input_hash_verification as _require_oracle_bindings,
)
from effdock.workflows.guidance_eta_sweep_report import (
    CONDITION,
    ETA_TAGS,
    ETA_VALUES,
    NUM_SAMPLES,
    NUM_STEPS,
    expected_run_name,
)
from effdock.workflows.guidance_eta_sweep_report import (
    PROTOCOL_ID as PARENT_PROTOCOL_ID,
)
from effdock.workflows.guidance_eta_sweep_selector_binding import PROTOCOL_ID
from effdock.workflows.guidance_eta_sweep_selector_report import (
    _join_sampling_outcomes,
    aggregate_selector_cell,
    paired_outcomes,
    summarize_outcomes,
)
from effdock.workflows.guidance_eta_sweep_selector_report import (
    _require_bindings as _require_selector_bindings,
)
from effdock.workflows.posebusters_report import file_sha256

SELECTORS = ("first", "vina", "oracle")
SELECTOR_PAIRS = (
    ("first", "vina"),
    ("first", "oracle"),
    ("vina", "oracle"),
)


def _load_frozen_report(path: Path, *, selector: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing selector report: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    expected_protocol = PARENT_PROTOCOL_ID if selector == "oracle" else PROTOCOL_ID
    if payload.get("protocol_id") != expected_protocol:
        raise ValueError(f"{path}: protocol mismatch for selector {selector}")
    observed_selector = (
        payload.get("posebusters", {}).get("selector")
        if selector == "oracle"
        else payload.get("selector")
    )
    if observed_selector != selector:
        raise ValueError(f"{path}: selector mismatch")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError(f"{path}: dataset coverage mismatch")
    for dataset in DATASETS:
        cells = datasets[dataset].get("cells", {})
        if set(cells) != set(ETA_TAGS):
            raise ValueError(f"{path}: eta coverage mismatch for {dataset}")
    return payload


def _load_parent_sampling_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing parent sampling report: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError(f"{path}: parent sampling protocol mismatch")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise ValueError(f"{path}: parent sampling dataset coverage mismatch")
    for dataset in DATASETS:
        if set(datasets[dataset].get("cells", {})) != set(ETA_TAGS):
            raise ValueError(f"{path}: parent sampling eta coverage mismatch for {dataset}")
    return payload


def build_report(
    *,
    oracle_input_dir: Path,
    selector_input_root: Path,
    sampling_dir: Path,
    cohort_audit: Path,
    oracle_report: Path,
    first_report: Path,
    vina_report: Path,
    parent_sampling_report: Path,
    expected_shards: int = DEFAULT_EXPECTED_SHARDS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if expected_shards < 1 or bootstrap_resamples < 1:
        raise ValueError("expected_shards and bootstrap_resamples must be >= 1")
    report_paths = {
        "first": first_report,
        "vina": vina_report,
        "oracle": oracle_report,
    }
    frozen_reports = {
        selector: _load_frozen_report(path, selector=selector)
        for selector, path in report_paths.items()
    }
    _load_parent_sampling_report(parent_sampling_report)
    audits = load_full_cohort_audits(cohort_audit)
    expected_runs = {
        expected_run_name(dataset, eta) for dataset in DATASETS for eta in ETA_VALUES
    }
    _require_oracle_bindings(
        oracle_input_dir,
        sampling_dir,
        expected_runs,
        expected_shards,
    )
    for selector in ("first", "vina"):
        _require_selector_bindings(
            input_dir=selector_input_root / selector,
            sampling_dir=sampling_dir,
            selector=selector,
            expected_shards=expected_shards,
        )

    result: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "parent_sampling_protocol_id": PARENT_PROTOCOL_ID,
        "status": "complete_posthoc_paired_selector_comparison",
        "claim_boundary": (
            "all selector and eta cells reported; no automatic eta/selector choice or "
            "production admission"
        ),
        "estimand": "one selected top-1 pose per complex; never pooled across selectors",
        "selectors": {
            "first": "candidate index zero; no-ranking baseline",
            "vina": "argmin of the frozen repository Torch Vina-like plus DG-strain score",
            "oracle": "minimum symmetry-aware crystal RMSD; non-deployable upper bound",
        },
        "condition": {
            "name": CONDITION,
            "num_samples": NUM_SAMPLES,
            "num_steps": NUM_STEPS,
            "model_pose_step_budget": NUM_SAMPLES * NUM_STEPS,
        },
        "bootstrap": {
            "method": "paired complex-ID bootstrap, percentile 95% CI",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
        },
        "sources": {
            selector: {
                "report": str(report_paths[selector]),
                "report_sha256": file_sha256(report_paths[selector]),
                "official_input_dir": str(
                    oracle_input_dir
                    if selector == "oracle"
                    else selector_input_root / selector
                ),
            }
            for selector in SELECTORS
        },
        "datasets": {},
    }
    result["sources"]["parent_sampling"] = {
        "report": str(parent_sampling_report),
        "report_sha256": file_sha256(parent_sampling_report),
        "sampling_dir": str(sampling_dir),
    }

    for dataset in DATASETS:
        audit = audits[dataset]
        ids = audit["ids"]
        cells: dict[str, Any] = {}
        for eta, eta_tag in zip(ETA_VALUES, ETA_TAGS, strict=True):
            run_name = expected_run_name(dataset, eta)
            outcomes_by_selector: dict[str, dict[str, dict[str, Any]]] = {}
            selector_summaries: dict[str, Any] = {}
            for selector in SELECTORS:
                if selector == "oracle":
                    official_rows, aggregate, _ = _aggregate_cell(
                        oracle_input_dir / run_name,
                        ids,
                        run_name=run_name,
                        dataset=dataset,
                        condition=CONDITION,
                        arm=eta_tag,
                        num_samples=NUM_SAMPLES,
                        num_steps=NUM_STEPS,
                        expected_shards=expected_shards,
                    )
                else:
                    official_rows, aggregate, _ = aggregate_selector_cell(
                        selector_input_root / selector / run_name,
                        ids,
                        run_name=run_name,
                        dataset=dataset,
                        eta_tag=eta_tag,
                        selector=selector,
                        expected_shards=expected_shards,
                    )
                outcomes = _join_sampling_outcomes(
                    sampling_dir=sampling_dir,
                    run_name=run_name,
                    selector=selector,
                    ids=ids,
                    official_rows=official_rows,
                )
                outcomes_by_selector[selector] = outcomes
                selector_summaries[selector] = {
                    "posebusters_valid_count": aggregate["posebusters_valid_count"],
                    "posebusters_valid_pct": aggregate["posebusters_valid_pct"],
                    "check_pass_pct": aggregate["check_pass_pct"],
                    "module_pass_pct": aggregate["module_pass_pct"],
                    "selected_pose_metrics": summarize_outcomes(outcomes),
                }
                frozen_cell = frozen_reports[selector]["datasets"][dataset]["cells"][eta_tag]
                if float(frozen_cell["posebusters_valid_pct"]) != float(
                    aggregate["posebusters_valid_pct"]
                ):
                    raise ValueError(
                        f"{dataset}/{eta_tag}/{selector}: frozen report differs from raw official data"
                    )

            pairwise = {
                f"{comparison}_minus_{baseline}": paired_outcomes(
                    outcomes_by_selector[baseline],
                    outcomes_by_selector[comparison],
                    ids,
                    baseline_label=baseline,
                    comparison_label=comparison,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                )
                for baseline, comparison in SELECTOR_PAIRS
            }
            cells[eta_tag] = {
                "eta": eta,
                "eta_tag": eta_tag,
                "run_name": run_name,
                "selectors": selector_summaries,
                "paired_selector_comparisons": pairwise,
            }
        result["datasets"][dataset] = {
            "coverage": {
                "count": len(ids),
                "ids_sha256": audit["ids_sha256"],
                "audit_path": audit["source_path"],
                "audit_sha256": audit["source_sha256"],
            },
            "cells": cells,
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-input-dir", type=Path, required=True)
    parser.add_argument("--selector-input-root", type=Path, required=True)
    parser.add_argument("--sampling-dir", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--oracle-report", type=Path, required=True)
    parser.add_argument("--first-report", type=Path, required=True)
    parser.add_argument("--vina-report", type=Path, required=True)
    parser.add_argument("--parent-sampling-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=DEFAULT_EXPECTED_SHARDS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    report = build_report(
        oracle_input_dir=args.oracle_input_dir,
        selector_input_root=args.selector_input_root,
        sampling_dir=args.sampling_dir,
        cohort_audit=args.cohort_audit,
        oracle_report=args.oracle_report,
        first_report=args.first_report,
        vina_report=args.vina_report,
        parent_sampling_report=args.parent_sampling_report,
        expected_shards=args.expected_shards,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
