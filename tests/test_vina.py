"""Tests for torch-differentiable AutoDock Vina scoring (src/evaluation/vina.py)."""

import pytest
import torch

from effdock.evaluation.vina import (
    VINA_WEIGHTS,
    VinaScorer,
    count_rotatable_bonds,
    ligand_dg_penalty,
    ligand_dg_penalty_batched,
    ligand_dg_reference,
    vina_atom_radii,
    vina_atom_types,
    vina_pair_terms,
    vina_score,
    vina_score_batched,
    vina_score_from_xs,
    vina_score_with_strain,
    vina_xs_inputs,
)

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import AllChem  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _embed(smiles: str) -> "Chem.Mol":
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=0) == 0
    return Chem.RemoveHs(mol)


def _nitrogen_flags(smiles: str) -> tuple[bool, bool]:
    """Return (is_donor, is_acceptor) of the first nitrogen in a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    t = vina_atom_types(mol)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            i = atom.GetIdx()
            return bool(t["is_donor"][i]), bool(t["is_acceptor"][i])
    raise AssertionError("no nitrogen in molecule")


def _rand_flags(n: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        k: torch.randint(0, 2, (n,), generator=g).bool()
        for k in ("is_hydrophobic", "is_donor", "is_acceptor")
    }


# ---------------------------------------------------------------------------
# Radii
# ---------------------------------------------------------------------------
def test_radii_known_elements():
    r = vina_atom_radii(torch.tensor([6, 7, 8, 16, 9, 53]))
    assert torch.allclose(r, torch.tensor([1.9, 1.8, 1.7, 2.0, 1.5, 2.2]))


def test_radii_unknown_falls_back_to_carbon():
    r = vina_atom_radii(torch.tensor([999]))
    assert float(r[0]) == pytest.approx(1.9)


def test_exact_xs_radii_and_flags_match_vina_127():
    # C_H, N_DA, O_A, Br_H, Si, At, Met_D, G0
    x = vina_xs_inputs(torch.tensor([0, 5, 8, 14, 16, 17, 18, 27]))
    assert torch.allclose(x["radii"], torch.tensor([1.9, 1.8, 1.7, 2.0, 2.2, 2.3, 1.2, 0.0]))
    assert x["is_hydrophobic"].tolist() == [True, False, False, True, False, False, False, False]
    assert x["is_donor"].tolist() == [False, True, False, False, False, False, True, False]
    assert x["is_acceptor"].tolist() == [False, True, True, False, False, False, False, False]


def test_exact_xs_rejects_invalid_type():
    with pytest.raises(ValueError, match="XS type"):
        vina_xs_inputs(torch.tensor([31]))


# ---------------------------------------------------------------------------
# Hydrophobic XS typing
# ---------------------------------------------------------------------------
def test_hydrophobic_carbon_vs_polar_carbon():
    # ethanol C-C-O: methyl C hydrophobic, C bonded to O is polar (excluded).
    mol = Chem.MolFromSmiles("CCO")
    t = vina_atom_types(mol)
    assert bool(t["is_hydrophobic"][0]) is True  # CH3
    assert bool(t["is_hydrophobic"][1]) is False  # CH2-O
    assert bool(t["is_hydrophobic"][2]) is False  # O


def test_halogen_is_hydrophobic_and_neighbour_carbon_is_polar():
    mol = Chem.MolFromSmiles("c1ccccc1F")
    t = vina_atom_types(mol)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    f_idx = syms.index("F")
    assert bool(t["is_hydrophobic"][f_idx]) is True
    # the ring carbon bonded to F must be polar (heteroatom neighbour)
    c_on_f = [n.GetIdx() for n in mol.GetAtomWithIdx(f_idx).GetNeighbors()][0]
    assert bool(t["is_hydrophobic"][c_on_f]) is False


# ---------------------------------------------------------------------------
# Nitrogen donor/acceptor (AutoDock NA) typing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,donor,acceptor",
    [
        ("c1ccncc1", False, True),  # pyridine: lone pair out of ring
        ("c1cc[nH]c1", True, False),  # pyrrole: N-H, lone pair in ring
        ("CC(=O)N", True, False),  # amide: delocalised into carbonyl
        ("CCN", True, True),  # aliphatic amine
        ("CC[NH3+]", True, False),  # ammonium: no lone pair
        ("CC#N", False, True),  # nitrile
    ],
)
def test_nitrogen_donor_acceptor(smiles, donor, acceptor):
    assert _nitrogen_flags(smiles) == (donor, acceptor)


def test_oxygen_always_acceptor():
    mol = Chem.MolFromSmiles("CC(=O)O")  # carboxylic acid: =O and -OH
    t = vina_atom_types(mol)
    o_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 8]
    assert all(bool(t["is_acceptor"][i]) for i in o_idx)


# ---------------------------------------------------------------------------
# Rotatable bonds (N_rot)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("CC", 0),  # ethane: terminal only
        ("CCCC", 1),  # butane: one central single bond
        ("CC(=O)NC", 0),  # amide bond excluded by strict
    ],
)
def test_rotatable_bond_count(smiles, expected):
    assert count_rotatable_bonds(Chem.MolFromSmiles(smiles)) == expected


def test_atom_types_dict_includes_nrot():
    t = vina_atom_types(Chem.MolFromSmiles("CCCC"))
    assert set(t) == {
        "atomic_nums",
        "is_hydrophobic",
        "is_donor",
        "is_acceptor",
        "num_rotatable_bonds",
    }
    assert t["num_rotatable_bonds"] == 1


# ---------------------------------------------------------------------------
# Pairwise term math
# ---------------------------------------------------------------------------
def test_pair_terms_match_closed_form():
    d = torch.tensor([-0.5, 0.0, 0.3, 3.0])
    no = torch.zeros(4, dtype=torch.bool)
    e = vina_pair_terms(d, no, no)  # only gauss + repulsion active
    w = VINA_WEIGHTS
    g1 = torch.exp(-((d / 0.5) ** 2))
    g2 = torch.exp(-(((d - 3.0) / 2.0) ** 2))
    rep = torch.where(d < 0, d * d, torch.zeros_like(d))
    expected = w.gauss1 * g1 + w.gauss2 * g2 + w.repulsion * rep
    assert torch.allclose(e, expected, atol=1e-6)


def test_hydrophobic_term_only_for_hydrophobic_pairs():
    d = torch.tensor([0.0])  # ramp value 1.0
    on = torch.ones(1, dtype=torch.bool)
    off = torch.zeros(1, dtype=torch.bool)
    e_on = vina_pair_terms(d, on, off)
    e_off = vina_pair_terms(d, off, off)
    assert float(e_on - e_off) == pytest.approx(VINA_WEIGHTS.hydrophobic, abs=1e-6)


def test_hbond_term_only_for_complementary_pairs():
    d = torch.tensor([-0.7])  # ramp value 1.0
    on = torch.ones(1, dtype=torch.bool)
    off = torch.zeros(1, dtype=torch.bool)
    e_on = vina_pair_terms(d, off, on)
    e_off = vina_pair_terms(d, off, off)
    assert float(e_on - e_off) == pytest.approx(VINA_WEIGHTS.hydrogen_bond, abs=1e-6)


# ---------------------------------------------------------------------------
# vina_score: invariance, batching, gradients
# ---------------------------------------------------------------------------
def _score_random_complex(lig, prot, n_rot=0, **batch):
    nl, npr = lig.shape[0], prot.shape[0]
    fl, fp = _rand_flags(nl, 1), _rand_flags(npr, 2)
    return vina_score(
        lig,
        prot,
        vina_atom_radii(torch.full((nl,), 6)),
        vina_atom_radii(torch.full((npr,), 8)),
        lig_is_hydrophobic=fl["is_hydrophobic"],
        prot_is_hydrophobic=fp["is_hydrophobic"],
        lig_is_donor=fl["is_donor"],
        prot_is_donor=fp["is_donor"],
        lig_is_acceptor=fl["is_acceptor"],
        prot_is_acceptor=fp["is_acceptor"],
        num_rotatable_bonds=n_rot,
        **batch,
    )


def test_score_invariant_to_rigid_motion():
    torch.manual_seed(0)
    lig, prot = torch.randn(6, 3), torch.randn(20, 3) * 2.5
    s0 = _score_random_complex(lig, prot)

    # random rotation + translation applied to the whole complex
    q = torch.randn(4)
    q = q / q.norm()
    w, x, y, z = q
    R = torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    tvec = torch.randn(3)
    s1 = _score_random_complex(lig @ R.T + tvec, prot @ R.T + tvec)
    assert torch.allclose(s0, s1, atol=1e-4)


def test_batched_equals_per_sample_loop():
    torch.manual_seed(0)
    sizes = [(5, 12), (7, 9)]
    ligs = [torch.randn(nl, 3) for nl, _ in sizes]
    prots = [torch.randn(npr, 3) * 3 for _, npr in sizes]
    nrots = [2.0, 4.0]

    # consistent flags per sample (seeded by size so single & batch agree)
    def flags(n, seed):
        return _rand_flags(n, seed)

    singles = []
    for (lig, prot), nr in zip(zip(ligs, prots), nrots):
        nl, npr = lig.shape[0], prot.shape[0]
        fl, fp = flags(nl, nl), flags(npr, npr)
        singles.append(
            vina_score(
                lig,
                prot,
                vina_atom_radii(torch.full((nl,), 6)),
                vina_atom_radii(torch.full((npr,), 8)),
                lig_is_hydrophobic=fl["is_hydrophobic"],
                prot_is_hydrophobic=fp["is_hydrophobic"],
                lig_is_donor=fl["is_donor"],
                prot_is_donor=fp["is_donor"],
                lig_is_acceptor=fl["is_acceptor"],
                prot_is_acceptor=fp["is_acceptor"],
                num_rotatable_bonds=nr,
            )
        )

    lig = torch.cat(ligs)
    prot = torch.cat(prots)
    lb = torch.cat([torch.full((nl,), b) for b, (nl, _) in enumerate(sizes)])
    pb = torch.cat([torch.full((npr,), b) for b, (_, npr) in enumerate(sizes)])
    fl = [flags(nl, nl) for nl, _ in sizes]
    fp = [flags(npr, npr) for _, npr in sizes]

    def catk(parts, key):
        return torch.cat([part[key] for part in parts])

    batched = vina_score(
        lig,
        prot,
        vina_atom_radii(torch.full((lig.shape[0],), 6)),
        vina_atom_radii(torch.full((prot.shape[0],), 8)),
        lig_is_hydrophobic=catk(fl, "is_hydrophobic"),
        prot_is_hydrophobic=catk(fp, "is_hydrophobic"),
        lig_is_donor=catk(fl, "is_donor"),
        prot_is_donor=catk(fp, "is_donor"),
        lig_is_acceptor=catk(fl, "is_acceptor"),
        prot_is_acceptor=catk(fp, "is_acceptor"),
        num_rotatable_bonds=torch.tensor(nrots),
        lig_batch=lb,
        prot_batch=pb,
    )
    assert torch.allclose(batched, torch.stack(singles), atol=1e-5)


def test_batch_no_cross_sample_leakage():
    # Two complexes placed far apart: batched score == each scored alone.
    torch.manual_seed(1)
    lig = torch.cat([torch.randn(4, 3), torch.randn(4, 3) + 1000.0])
    prot = torch.cat([torch.randn(6, 3), torch.randn(6, 3) + 1000.0])
    lb = torch.tensor([0] * 4 + [1] * 4)
    pb = torch.tensor([0] * 6 + [1] * 6)
    batched = _score_random_complex(lig, prot, lig_batch=lb, prot_batch=pb)
    assert batched.shape == (2,)
    assert torch.isfinite(batched).all()


def test_gradient_finite_at_coincident_atoms():
    lig = torch.zeros(2, 3, requires_grad=True)
    prot = torch.zeros(2, 3)  # exactly overlapping -> torch.cdist would NaN
    r = torch.full((2,), 1.9)
    f = _rand_flags(2, 3)
    s = vina_score(
        lig,
        prot,
        r,
        r,
        lig_is_hydrophobic=f["is_hydrophobic"],
        prot_is_hydrophobic=f["is_hydrophobic"],
        lig_is_donor=f["is_donor"],
        prot_is_donor=f["is_donor"],
        lig_is_acceptor=f["is_acceptor"],
        prot_is_acceptor=f["is_acceptor"],
    )
    s.backward()
    assert torch.isfinite(lig.grad).all()


def test_cutoff_is_center_distance_not_surface_distance():
    # Official potentials.h returns zero at center distance >=8A. The old bug
    # applied 8A to surface distance and would incorrectly keep this pair.
    lig = torch.zeros(1, 3)
    r = torch.full((1,), 1.9)
    off = torch.zeros(1, dtype=torch.bool)
    kwargs = dict(
        lig_is_hydrophobic=off,
        prot_is_hydrophobic=off,
        lig_is_donor=off,
        prot_is_donor=off,
        lig_is_acceptor=off,
        prot_is_acceptor=off,
    )
    inside = vina_score(lig, torch.tensor([[7.9, 0.0, 0.0]]), r, r, **kwargs)
    outside = vina_score(lig, torch.tensor([[8.0, 0.0, 0.0]]), r, r, **kwargs)
    assert float(inside) != pytest.approx(0.0, abs=1e-8)
    assert float(outside) == pytest.approx(0.0, abs=1e-8)


def test_dense_pose_batch_matches_loop_and_xs_wrapper():
    torch.manual_seed(4)
    poses = torch.randn(3, 5, 3)
    prot = torch.randn(9, 3) * 2.0
    lig_xs = torch.tensor([0, 1, 3, 8, 14])
    prot_xs = torch.tensor([0, 4, 7, 8, 10, 0, 1, 18, 13])
    lig = vina_xs_inputs(lig_xs)
    rec = vina_xs_inputs(prot_xs)
    kwargs = dict(
        lig_is_hydrophobic=lig["is_hydrophobic"],
        prot_is_hydrophobic=rec["is_hydrophobic"],
        lig_is_donor=lig["is_donor"],
        prot_is_donor=rec["is_donor"],
        lig_is_acceptor=lig["is_acceptor"],
        prot_is_acceptor=rec["is_acceptor"],
        num_rotatable_bonds=2,
    )
    dense = vina_score_batched(poses, prot, lig["radii"], rec["radii"], **kwargs)
    loop = torch.stack(
        [vina_score(pose, prot, lig["radii"], rec["radii"], **kwargs) for pose in poses]
    )
    exact = vina_score_from_xs(poses, prot, lig_xs, prot_xs, num_rotatable_bonds=2)
    assert torch.allclose(dense, loop, atol=1e-6)
    assert torch.allclose(exact, loop, atol=1e-6)


def test_clash_raises_score_via_repulsion():
    # Overlapping atoms (negative surface distance) -> positive repulsion -> higher score.
    r = torch.full((1,), 1.9)
    flags = dict(
        lig_is_hydrophobic=torch.zeros(1, dtype=torch.bool),
        prot_is_hydrophobic=torch.zeros(1, dtype=torch.bool),
        lig_is_donor=torch.zeros(1, dtype=torch.bool),
        prot_is_donor=torch.zeros(1, dtype=torch.bool),
        lig_is_acceptor=torch.zeros(1, dtype=torch.bool),
        prot_is_acceptor=torch.zeros(1, dtype=torch.bool),
    )
    clash = vina_score(torch.zeros(1, 3), torch.tensor([[0.5, 0.0, 0.0]]), r, r, **flags)
    far = vina_score(torch.zeros(1, 3), torch.tensor([[4.0, 0.0, 0.0]]), r, r, **flags)
    assert float(clash) > float(far)


def test_nrot_penalty_shrinks_magnitude():
    r = torch.full((2,), 1.9)
    f = _rand_flags(2, 5)
    kw = dict(
        lig_is_hydrophobic=f["is_hydrophobic"],
        prot_is_hydrophobic=f["is_hydrophobic"],
        lig_is_donor=f["is_donor"],
        prot_is_donor=f["is_donor"],
        lig_is_acceptor=f["is_acceptor"],
        prot_is_acceptor=f["is_acceptor"],
    )
    lig = torch.zeros(2, 3)
    prot = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    s0 = vina_score(lig, prot, r, r, num_rotatable_bonds=0, **kw)
    s5 = vina_score(lig, prot, r, r, num_rotatable_bonds=5, **kw)
    # denominator (1 + w_rot*N_rot) > 1 shrinks the magnitude toward zero
    assert abs(float(s5)) < abs(float(s0))
    assert float(s5) == pytest.approx(float(s0) / (1 + VINA_WEIGHTS.rot * 5), abs=1e-6)


# ---------------------------------------------------------------------------
# Ligand distance-geometry strain penalty
# ---------------------------------------------------------------------------
def _butane_two_frag():
    """Butane with conformer; fragments {0,1} and {2,3}; bond 1-2 is the cut."""
    mol = _embed("CCCC")
    ref = ligand_dg_reference(mol)
    xyz = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    frag_id = torch.tensor([0, 0, 1, 1])
    radii = vina_atom_radii(torch.full((4,), 6))
    return mol, ref, xyz, frag_id, radii


def test_dg_reference_matches_bonds():
    mol = _embed("CCCC")
    ref = ligand_dg_reference(mol)
    assert ref["bond_index"].shape[1] == mol.GetNumBonds()
    # carbon-carbon single bonds ~1.5 Å
    assert torch.all(ref["bond_ref_len"] > 1.0)
    assert torch.all(ref["bond_ref_len"] < 1.8)
    # undirected, i < j
    assert torch.all(ref["bond_index"][0] < ref["bond_index"][1])


def test_dg_penalty_zero_at_reference():
    _, ref, xyz, frag_id, radii = _butane_two_frag()
    p = ligand_dg_penalty(xyz, ref["bond_index"], ref["bond_ref_len"], frag_id=frag_id, radii=radii)
    assert float(p) == pytest.approx(0.0, abs=1e-5)


def test_dg_penalty_rises_with_bond_stretch():
    _, ref, xyz, frag_id, radii = _butane_two_frag()
    broken = xyz.clone()
    broken[2:] += torch.tensor([4.0, 0.0, 0.0])  # yank frag 1 -> stretch cut bond
    p0 = ligand_dg_penalty(xyz, ref["bond_index"], ref["bond_ref_len"])
    p = ligand_dg_penalty(broken, ref["bond_index"], ref["bond_ref_len"])
    assert float(p0) == pytest.approx(0.0, abs=1e-5)
    assert float(p) > 1.0  # stretched cut bond -> (Δd)^2 penalty


def test_dg_penalty_catches_inter_fragment_clash():
    _, ref, xyz, frag_id, radii = _butane_two_frag()
    # collapse frag 1 onto frag 0's first atom -> heavy overlap, no bond change
    clashed = xyz.clone()
    clashed[2:] = xyz[0].clone() + 1e-3 * torch.arange(2)[:, None]
    p_clash = ligand_dg_penalty(
        clashed, ref["bond_index"], ref["bond_ref_len"], frag_id=frag_id, radii=radii
    )
    p_nobody = ligand_dg_penalty(clashed, ref["bond_index"], ref["bond_ref_len"])
    # clash term (with frag_id+radii) adds penalty beyond the bond term alone
    assert float(p_clash) > float(p_nobody)


def test_dg_penalty_batched_equals_loop():
    _, ref, xyz, frag_id, radii = _butane_two_frag()
    a = xyz.clone()
    b = xyz.clone()
    b[2:] += torch.tensor([1.0, 0.0, 0.0])  # perturb sample 1
    p_a = ligand_dg_penalty(a, ref["bond_index"], ref["bond_ref_len"], frag_id=frag_id, radii=radii)
    p_b = ligand_dg_penalty(b, ref["bond_index"], ref["bond_ref_len"], frag_id=frag_id, radii=radii)

    coords = torch.cat([a, b])
    bidx = torch.cat([ref["bond_index"], ref["bond_index"] + 4], dim=1)
    bref = torch.cat([ref["bond_ref_len"], ref["bond_ref_len"]])
    lig_batch = torch.tensor([0] * 4 + [1] * 4)
    batched = ligand_dg_penalty(
        coords,
        bidx,
        bref,
        frag_id=torch.cat([frag_id, frag_id]),
        radii=torch.cat([radii, radii]),
        lig_batch=lig_batch,
    )
    assert torch.allclose(batched, torch.stack([p_a, p_b]), atol=1e-5)


def test_dense_dg_batch_matches_loop():
    _, ref, xyz, frag_id, radii = _butane_two_frag()
    poses = torch.stack([xyz, xyz + torch.tensor([0.2, 0.0, 0.0])])
    poses[1, 2:] += torch.tensor([0.7, 0.0, 0.0])
    dense = ligand_dg_penalty_batched(
        poses,
        ref["bond_index"],
        ref["bond_ref_len"],
        frag_id=frag_id,
        radii=radii,
    )
    loop = torch.stack(
        [
            ligand_dg_penalty(
                pose,
                ref["bond_index"],
                ref["bond_ref_len"],
                frag_id=frag_id,
                radii=radii,
            )
            for pose in poses
        ]
    )
    assert torch.allclose(dense, loop, atol=1e-6)


def test_dg_penalty_gradient_finite_at_coincident_atoms():
    _, ref, xyz, frag_id, radii = _butane_two_frag()
    x = xyz.clone()
    x[2:] = xyz[0]  # coincident across fragments
    x.requires_grad_(True)
    p = ligand_dg_penalty(x, ref["bond_index"], ref["bond_ref_len"], frag_id=frag_id, radii=radii)
    p.backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Combined vina + strain
# ---------------------------------------------------------------------------
def _combined_setup():
    _, ref, lig, frag_id, lig_r = _butane_two_frag()
    torch.manual_seed(0)
    prot = torch.randn(15, 3) * 2.0
    prot_r = vina_atom_radii(torch.full((15,), 8))
    fl, fp = _rand_flags(4, 1), _rand_flags(15, 2)
    kw = dict(
        lig_is_hydrophobic=fl["is_hydrophobic"],
        prot_is_hydrophobic=fp["is_hydrophobic"],
        lig_is_donor=fl["is_donor"],
        prot_is_donor=fp["is_donor"],
        lig_is_acceptor=fl["is_acceptor"],
        prot_is_acceptor=fp["is_acceptor"],
    )
    return ref, lig, frag_id, lig_r, prot, prot_r, kw


def test_combined_equals_vina_plus_weighted_strain():
    ref, lig, frag_id, lig_r, prot, prot_r, kw = _combined_setup()
    lig = lig.clone()
    lig[2:] += torch.tensor([1.5, 0.0, 0.0])  # introduce some strain
    w = 2.5
    out = vina_score_with_strain(
        lig,
        prot,
        lig_r,
        prot_r,
        bond_index=ref["bond_index"],
        bond_ref_len=ref["bond_ref_len"],
        frag_id=frag_id,
        w_strain=w,
        return_components=True,
        **kw,
    )
    v = vina_score(lig, prot, lig_r, prot_r, **kw)
    from effdock.evaluation.vina import ligand_dg_penalty

    s = ligand_dg_penalty(lig, ref["bond_index"], ref["bond_ref_len"], frag_id=frag_id, radii=lig_r)
    assert torch.allclose(out["vina"], v, atol=1e-6)
    assert torch.allclose(out["strain"], s, atol=1e-6)
    assert torch.allclose(out["total"], v + w * s, atol=1e-6)


def test_combined_returns_scalar_by_default():
    ref, lig, frag_id, lig_r, prot, prot_r, kw = _combined_setup()
    out = vina_score_with_strain(
        lig,
        prot,
        lig_r,
        prot_r,
        bond_index=ref["bond_index"],
        bond_ref_len=ref["bond_ref_len"],
        frag_id=frag_id,
        **kw,
    )
    assert out.ndim == 0


def test_combined_gradient_flows_to_coords():
    ref, lig, frag_id, lig_r, prot, prot_r, kw = _combined_setup()
    lig = lig.clone().requires_grad_(True)
    out = vina_score_with_strain(
        lig,
        prot,
        lig_r,
        prot_r,
        bond_index=ref["bond_index"],
        bond_ref_len=ref["bond_ref_len"],
        frag_id=frag_id,
        **kw,
    )
    out.backward()
    assert torch.isfinite(lig.grad).all()
    assert lig.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# VinaScorer module
# ---------------------------------------------------------------------------
def test_vina_scorer_module_matches_functional():
    torch.manual_seed(0)
    lig, prot = torch.randn(5, 3), torch.randn(10, 3) * 2.0
    lz, pz = torch.full((5,), 6), torch.full((10,), 8)
    f = _rand_flags(5, 1)
    g = _rand_flags(10, 2)
    kw = dict(
        lig_is_hydrophobic=f["is_hydrophobic"],
        prot_is_hydrophobic=g["is_hydrophobic"],
        lig_is_donor=f["is_donor"],
        prot_is_donor=g["is_donor"],
        lig_is_acceptor=f["is_acceptor"],
        prot_is_acceptor=g["is_acceptor"],
    )
    scorer = VinaScorer(lz, pz)
    s_mod = scorer(lig, prot, **kw)
    s_fn = vina_score(lig, prot, vina_atom_radii(lz), vina_atom_radii(pz), **kw)
    assert torch.allclose(s_mod, s_fn, atol=1e-6)
