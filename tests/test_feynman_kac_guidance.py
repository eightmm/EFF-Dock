"""Tests for experimental constraint-only Feynman--Kac resampling."""

from __future__ import annotations

import pytest
import torch
from rdkit import Chem

import effdock.guidance.feynman_kac as fk
from effdock.guidance.parameterization import element_parameters
from effdock.guidance.system import PhysicalSystem
from effdock.guidance.topology import build_physical_topology
from effdock.workflows.evaluate import (
    build_arg_parser as build_evaluate_parser,
)
from effdock.workflows.evaluate import (
    fk_post_resampling_mutation_classification,
)


def test_parse_fk_resample_times_is_strict() -> None:
    assert fk.parse_fk_resample_times("0.25, 0.5,0.75") == (0.25, 0.5, 0.75)
    assert fk.parse_fk_resample_times("") == ()

    for invalid in ("0", "1", "0.5,0.5", "0.7,0.2", "abc"):
        with pytest.raises(ValueError):
            fk.parse_fk_resample_times(invalid)


def test_constraint_potential_uses_only_the_explicit_nonnegative_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_physical_energy(coords, system, config):
        del system, config
        batch_size = coords.shape[0]
        zero = coords.new_zeros(batch_size)
        components = {name: zero.clone() for name in fk.DEFAULT_FK_CONSTRAINT_TERMS}
        components["ligand_intra_bond"] = torch.tensor([1.0, 2.0], device=coords.device)
        components["protein_ligand_steric_barrier"] = torch.tensor([3.0, 4.0], device=coords.device)
        components["ligand_intra_lj_attractive"] = torch.full_like(zero, 1.0e6)
        components["protein_ligand_lj_attractive"] = torch.full_like(zero, 1.0e6)
        components["total"] = sum(components.values(), start=zero)
        return components

    monkeypatch.setattr(fk, "physical_energy", fake_physical_energy)
    potential, selected = fk.constraint_potential(
        torch.zeros(2, 1, 3),
        object(),
        fk.FKConstraintConfig(beta=1.0),
    )

    torch.testing.assert_close(potential, torch.tensor([4.0, 6.0]))
    assert set(selected) == set(fk.DEFAULT_FK_CONSTRAINT_TERMS)
    assert "ligand_intra_lj_attractive" not in selected
    assert "protein_ligand_lj_attractive" not in selected


@pytest.mark.parametrize("bad_value", [-0.1, float("nan"), float("inf")])
def test_constraint_potential_rejects_invalid_constraint_terms(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: float,
) -> None:
    def fake_physical_energy(coords, system, config):
        del system, config
        zero = coords.new_zeros(coords.shape[0])
        components = {name: zero.clone() for name in fk.DEFAULT_FK_CONSTRAINT_TERMS}
        components["ligand_intra_bond"] = torch.full_like(zero, bad_value)
        components["total"] = zero
        return components

    monkeypatch.setattr(fk, "physical_energy", fake_physical_energy)
    with pytest.raises(FloatingPointError):
        fk.constraint_potential(
            torch.zeros(2, 1, 3),
            object(),
            fk.FKConstraintConfig(beta=1.0),
        )


def test_zero_beta_is_an_exact_noop_without_energy_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("zero-beta FK must not evaluate the potential")

    monkeypatch.setattr(fk, "constraint_potential", must_not_run)
    resampler = fk.FeynmanKacConstraintResampler(
        object(),
        fk.FKConstraintConfig(beta=0.0),
    )
    source = resampler.resample(
        torch.zeros(4, 1, 3),
        prior_sigma=torch.ones(4),
        requested_time=0.5,
        actual_time=0.5,
    )

    torch.testing.assert_close(source, torch.arange(4))
    assert resampler.diagnostics()["num_resampling_events"] == 0


@pytest.mark.parametrize("bad_beta", [-1.0, float("nan"), float("inf")])
def test_fk_beta_must_be_finite_and_nonnegative(bad_beta: float) -> None:
    with pytest.raises(ValueError, match="beta"):
        fk.FKConstraintConfig(beta=bad_beta)


def test_fk_diagnostics_record_translation_sde_contract() -> None:
    config = fk.FKConstraintConfig(
        beta=0.0,
        dynamics="translation_score_corrected_sde_deterministic_so3",
        translation_sde_base_sigma=0.3,
    )
    diagnostics = fk.FeynmanKacConstraintResampler(object(), config).diagnostics()

    assert diagnostics["schema_version"] == "effdock.fk_constraint_resampling.v2"
    assert diagnostics["dynamics"] == "translation_score_corrected_sde_deterministic_so3"
    assert diagnostics["translation_sde_base_sigma"] == 0.3

    with pytest.raises(ValueError, match="must agree"):
        fk.FKConstraintConfig(
            beta=1.0,
            dynamics="translation_score_corrected_sde_deterministic_so3",
            translation_sde_base_sigma=0.0,
        )


def test_high_temperature_selects_low_potential_and_records_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potential = torch.tensor([0.0, 100.0, 100.0, 100.0])

    def fake_constraint_potential(coords, system, config):
        del coords, system, config
        return potential, {"ligand_intra_bond": potential}

    monkeypatch.setattr(fk, "constraint_potential", fake_constraint_potential)
    resampler = fk.FeynmanKacConstraintResampler(
        object(),
        fk.FKConstraintConfig(beta=1.0, seed=7),
    )
    source = resampler.resample(
        torch.zeros(4, 1, 3),
        prior_sigma=torch.ones(4),
        requested_time=0.5,
        actual_time=0.5,
    )

    torch.testing.assert_close(source, torch.zeros(4, dtype=torch.long))
    diagnostics = resampler.diagnostics()
    assert diagnostics["num_resampling_events"] == 1
    assert diagnostics["final_unique_initial_ancestors"] == 1
    assert diagnostics["events"][0]["unique_parent_count"] == 1
    assert diagnostics["events"][0]["ess_fraction"] == pytest.approx(0.25)


def test_resampling_preserves_prior_sigma_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potential = torch.tensor([0.0, 100.0, 100.0, 0.0])

    def fake_constraint_potential(coords, system, config):
        del coords, system, config
        return potential, {"ligand_intra_bond": potential}

    monkeypatch.setattr(fk, "constraint_potential", fake_constraint_potential)
    resampler = fk.FeynmanKacConstraintResampler(
        object(),
        fk.FKConstraintConfig(beta=1.0, seed=9),
    )
    source = resampler.resample(
        torch.zeros(4, 1, 3),
        prior_sigma=torch.tensor([0.5, 0.5, 1.0, 1.0]),
        requested_time=0.5,
        actual_time=0.5,
    )

    torch.testing.assert_close(source, torch.tensor([0, 0, 3, 3]))
    assert resampler.diagnostics()["events"][0]["group_sizes"] == [2, 2]


def test_difference_schedule_uses_the_previous_endpoint_potential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    potentials = iter(
        (
            torch.ones(4),
            torch.tensor([2.0, 3.0, 4.0, 5.0]),
        )
    )

    def fake_constraint_potential(coords, system, config):
        del coords, system, config
        potential = next(potentials)
        return potential, {"ligand_intra_bond": potential}

    monkeypatch.setattr(fk, "constraint_potential", fake_constraint_potential)
    resampler = fk.FeynmanKacConstraintResampler(
        object(),
        fk.FKConstraintConfig(beta=1.0, seed=13),
    )
    for time in (0.25, 0.5):
        resampler.resample(
            torch.zeros(4, 1, 3),
            prior_sigma=torch.ones(4),
            requested_time=time,
            actual_time=time,
        )

    second_event = resampler.diagnostics()["events"][1]
    assert second_event["delta_min"] == 1.0
    assert second_event["delta_median"] == 2.0
    assert second_event["delta_max"] == 4.0


def test_real_physical_system_potential_penalizes_a_close_protein_contact() -> None:
    mol = Chem.MolFromSmiles("C")
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    mol.AddConformer(conformer)
    topology = build_physical_topology(
        mol,
        torch.zeros(1, dtype=torch.long),
    ).to(torch.device("cpu"), torch.float64)
    protein_atomic_numbers = torch.tensor([8], dtype=torch.long)
    protein_parameters = element_parameters(
        protein_atomic_numbers,
        dtype=torch.float64,
    )
    system = PhysicalSystem(
        topology=topology,
        protein_coords=torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float64),
        protein_atomic_numbers=protein_atomic_numbers,
        protein_uff_x=protein_parameters.uff_x,
        protein_uff_d=protein_parameters.uff_d,
        protein_vdw_radius=protein_parameters.vdw_radius,
        parameter_set={"name": "test", "version": "test"},
        protein_source_atoms=1,
    )
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0]], [[8.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )

    potential, selected = fk.constraint_potential(
        coords,
        system,
        fk.FKConstraintConfig(beta=1.0),
    )

    assert float(potential[0]) > float(potential[1])
    assert bool((potential >= 0.0).all())
    assert float(selected["protein_ligand_steric_barrier"][0]) > 0.0
    assert float(selected["receptor_geometry_obstacle_uff_repulsive"].sum()) == 0.0
    assert float(selected["receptor_geometry_obstacle_generic_repulsive"].sum()) == 0.0

    resampler = fk.FeynmanKacConstraintResampler(
        system,
        fk.FKConstraintConfig(beta=1.0, seed=11),
    )
    source = resampler.resample(
        coords,
        prior_sigma=torch.ones(2, dtype=torch.float64),
        requested_time=0.5,
        actual_time=0.5,
    )
    torch.testing.assert_close(source, torch.tensor([1, 1]))


def test_evaluate_cli_parses_fk_contract_without_changing_defaults() -> None:
    required = [
        "--dataset",
        "astex",
        "--data-dir",
        "data/astex",
        "--pocket-centers",
        "centers.json",
    ]
    baseline = build_evaluate_parser().parse_args(required)
    assert baseline.fk_constraint_beta == 0.0
    assert baseline.fk_resample_times == ()
    assert baseline.fk_resample_method == "systematic"
    assert baseline.fk_resample_translation_jitter == 0.0
    assert baseline.fk_resample_rotation_jitter == 0.0
    assert baseline.translation_sde_base_sigma == 0.0

    configured = build_evaluate_parser().parse_args(
        required
        + [
            "--fk-constraint-beta",
            "0.2",
            "--fk-resample-times",
            "0.25,0.5,0.75",
            "--fk-resample-method",
            "multinomial",
            "--translation-sde-base-sigma",
            "0.3",
        ]
    )
    assert configured.fk_constraint_beta == 0.2
    assert configured.fk_resample_times == (0.25, 0.5, 0.75)
    assert configured.fk_resample_method == "multinomial"
    assert configured.translation_sde_base_sigma == 0.3


@pytest.mark.parametrize(
    ("sde", "translation_jitter", "rotation_jitter", "expected"),
    (
        (False, 0.0, 0.0, "none"),
        (False, 0.01, 0.0, "heuristic_not_marginal_preserving_sde"),
        (False, 0.0, 0.01, "heuristic_not_marginal_preserving_sde"),
        (True, 0.0, 0.0, "none_sde_is_continuous_translation_dynamics"),
    ),
)
def test_fk_post_resampling_mutation_provenance_is_exact(
    sde: bool,
    translation_jitter: float,
    rotation_jitter: float,
    expected: str,
) -> None:
    assert (
        fk_post_resampling_mutation_classification(
            translation_sde_enabled=sde,
            translation_jitter=translation_jitter,
            rotation_jitter=rotation_jitter,
        )
        == expected
    )
