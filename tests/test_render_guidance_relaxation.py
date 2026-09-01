from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

RENDERER = Path(__file__).parents[1] / "scripts" / "figures" / "render_guidance_relaxation.py"


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    crystal = torch.tensor(
        [
            [-0.7, 0.0, 0.0],
            [0.7, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    frames = torch.stack((crystal + 0.5, crystal + 0.1))
    run_metrics = [
        {
            "step": step,
            "energy_groups": {
                "combined": 3.0 - step,
                "physical": 2.0 - step,
                "interaction": 1.0,
            },
            "energies": {
                "interaction_hydrophobic": -0.5,
                "interaction_hydrogen_bond": -0.4 + 0.4 * step,
                "interaction_screened_formal_charge": -0.25,
                "interaction_pi_stacking": -0.2,
                "interaction_cation_pi": -0.1,
                "interaction_halogen_bond": -0.05,
                "interaction_metal_coordination": -0.025,
            },
            "raw_rmsd_angstrom": 0.5 - 0.4 * step,
        }
        for step in range(2)
    ]
    run_keys = (
        "unified",
        "physical_only",
        "model-prior/seed-7/unified",
        "pocket-control-seed-11",
    )
    summary = {
        "sample_id": "inline-ensemble",
        "initialization": {"kind": "model_prior", "sigma": 0.5},
        "run_labels": {
            "pocket-control-seed-11": "Pocket Gaussian · seed 11 · Physical only",
        },
        "runs": {key: {"metrics": run_metrics} for key in run_keys},
    }
    bundle_runs = {
        "unified": {"frames": frames, "saved_steps": torch.tensor([0, 1])},
        "physical_only": {"frames": frames, "saved_steps": torch.tensor([0, 1])},
        "model-prior/seed-7/unified": {
            "frames": frames,
            "saved_steps": torch.tensor([0, 1]),
            "prior": "model_prior",
            "seed": 7,
            "mode": "unified",
        },
        "pocket-control-seed-11": {
            "frames": frames,
            "saved_steps": torch.tensor([0, 1]),
        },
    }
    summary_path = tmp_path / "summary.json"
    trajectory_path = tmp_path / "trajectory.pt"
    summary_path.write_text(json.dumps(summary))
    torch.save(
        {
            "crystal_coords": crystal,
            "protein_coords": torch.empty(0, 3),
            "pocket_center": torch.zeros(3),
            "fragment_id": torch.tensor([0, 1]),
            "bonds": torch.tensor([[0, 1]]),
            "runs": bundle_runs,
        },
        trajectory_path,
    )
    return summary_path, trajectory_path


@pytest.mark.parametrize("explicit_inline_only", [False, True])
def test_inline_renderer_supports_arbitrary_runs_without_static_artifacts(
    tmp_path: Path,
    explicit_inline_only: bool,
) -> None:
    summary_path, trajectory_path = _write_fixture(tmp_path)
    inline_path = tmp_path / "ensemble-inline.html"
    command = [
        sys.executable,
        str(RENDERER),
        str(summary_path),
        str(trajectory_path),
        "--inline-html",
        str(inline_path),
        "--max-gif-frames",
        "1",
    ]
    if explicit_inline_only:
        command.append("--inline-only")
    subprocess.run(command, check=True, capture_output=True, text=True)

    assert inline_path.is_file()
    assert not (tmp_path / "relaxation.gif").exists()
    assert not (tmp_path / "dashboard.png").exists()

    fragment = inline_path.read_text()
    match = re.search(
        r'<script type="application/json" id="[^"]+-data">(.*?)</script>',
        fragment,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert list(payload["runs"]) == [
        "unified",
        "physical_only",
        "model-prior/seed-7/unified",
        "pocket-control-seed-11",
    ]
    assert payload["initializationLabel"] == "Model fragment prior · σ=0.5 Å"
    assert payload["runs"]["unified"]["label"] == "Unified guidance"
    assert payload["runs"]["physical_only"]["label"] == "Physical only"
    assert (
        payload["runs"]["model-prior/seed-7/unified"]["label"]
        == "Model prior · seed 7 · Unified guidance"
    )
    assert (
        payload["runs"]["pocket-control-seed-11"]["label"]
        == "Pocket Gaussian · seed 11 · Physical only"
    )
    interaction_terms = payload["runs"]["unified"]["interactionTerms"]
    assert [term["key"] for term in interaction_terms] == [
        "interaction_hydrophobic",
        "interaction_hydrogen_bond",
        "interaction_screened_formal_charge",
        "interaction_pi_stacking",
        "interaction_cation_pi",
        "interaction_halogen_bond",
        "interaction_metal_coordination",
    ]
    assert [term["label"] for term in interaction_terms] == [
        "hydrophobic",
        "H-bond",
        "charge",
        "π-stack",
        "cation–π",
        "halogen",
        "metal",
    ]
    assert interaction_terms[1]["values"] == [-0.4, 0.0]
    assert "Object.entries(data.runs)" in fragment
    assert "Math.abs(value) > 1e-12" in fragment
    assert len(fragment.encode()) < 2_000_000


def test_render_static_remains_available_by_explicit_request(tmp_path: Path) -> None:
    summary_path, trajectory_path = _write_fixture(tmp_path)
    inline_path = tmp_path / "ensemble-inline.html"
    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            str(summary_path),
            str(trajectory_path),
            "--inline-html",
            str(inline_path),
            "--render-static",
            "--max-gif-frames",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert inline_path.is_file()
    assert (tmp_path / "relaxation.gif").is_file()
    assert (tmp_path / "dashboard.png").is_file()
