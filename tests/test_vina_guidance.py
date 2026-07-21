import torch

from effdock.evaluation.vina import vina_score_batched, vina_xs_inputs
from effdock.evaluation.vina_guidance import VinaGuidance, VinaGuidanceConfig


def _guidance(lig_coords: torch.Tensor, prot_coords: torch.Tensor, frag_id=None) -> VinaGuidance:
    if frag_id is None:
        frag_id = torch.zeros(lig_coords.shape[0], dtype=torch.long)
    lig = vina_xs_inputs(torch.zeros(lig_coords.shape[0], dtype=torch.long))
    prot = vina_xs_inputs(torch.zeros(prot_coords.shape[0], dtype=torch.long))
    return VinaGuidance(
        prot_coords=prot_coords,
        prot_radii=prot["radii"],
        prot_is_hydrophobic=prot["is_hydrophobic"],
        prot_is_donor=prot["is_donor"],
        prot_is_acceptor=prot["is_acceptor"],
        lig_radii=lig["radii"],
        lig_is_hydrophobic=lig["is_hydrophobic"],
        lig_is_donor=lig["is_donor"],
        lig_is_acceptor=lig["is_acceptor"],
        num_rotatable_bonds=0,
        bond_index=torch.empty(2, 0, dtype=torch.long),
        bond_ref_len=torch.empty(0),
        frag_id=frag_id,
        config=VinaGuidanceConfig(start_t=0.5, max_atom_force=100.0),
    )


def test_guidance_is_zero_before_start_and_finite_after():
    coords = torch.tensor([[3.5, 0.0, 0.0]])
    callback = _guidance(coords, torch.zeros(1, 3))
    v0, w0 = callback(coords, torch.tensor([0]), coords, 0.49)
    v1, w1 = callback(coords, torch.tensor([0]), coords, 1.0)
    assert torch.equal(v0, torch.zeros_like(v0))
    assert torch.equal(w0, torch.zeros_like(w0))
    assert torch.isfinite(v1).all() and v1.abs().sum() > 0
    assert torch.equal(w1, torch.zeros_like(w1))


def test_translation_guidance_is_descent_direction():
    coords = torch.tensor([[4.2, 0.0, 0.0]])
    prot_coords = torch.zeros(1, 3)
    callback = _guidance(coords, prot_coords)
    v, _ = callback(coords, torch.tensor([0]), coords, 1.0)
    lig = vina_xs_inputs(torch.tensor([0]))
    prot = vina_xs_inputs(torch.tensor([0]))
    kwargs = dict(
        lig_is_hydrophobic=lig["is_hydrophobic"],
        prot_is_hydrophobic=prot["is_hydrophobic"],
        lig_is_donor=lig["is_donor"],
        prot_is_donor=prot["is_donor"],
        lig_is_acceptor=lig["is_acceptor"],
        prot_is_acceptor=prot["is_acceptor"],
    )
    e0 = vina_score_batched(coords[None], prot_coords, lig["radii"], prot["radii"], **kwargs)
    e1 = vina_score_batched(
        (coords + 1e-2 * v)[None], prot_coords, lig["radii"], prot["radii"], **kwargs
    )
    assert float(e1) < float(e0)


def test_guidance_translation_and_rotation_are_equivariant():
    coords = torch.tensor([[3.0, 0.7, 0.0], [3.0, -0.7, 0.0]])
    center = coords.mean(dim=0, keepdim=True)
    prot = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.3]])
    callback = _guidance(coords, prot)
    v, w = callback(coords, torch.tensor([0, 0]), center, 1.0)

    q = torch.tensor([0.3, -0.4, 0.5, 0.7])
    q = q / q.norm()
    qw, qx, qy, qz = q
    rot = torch.tensor(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    callback_rot = _guidance(coords @ rot.T, prot @ rot.T)
    v_rot, w_rot = callback_rot(
        coords @ rot.T, torch.tensor([0, 0]), center @ rot.T, 1.0
    )
    assert torch.allclose(v_rot, v @ rot.T, atol=2e-5)
    assert torch.allclose(w_rot, w @ rot.T, atol=2e-5)
