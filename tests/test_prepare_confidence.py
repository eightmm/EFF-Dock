from __future__ import annotations

import torch

from effdock.workflows.prepare_confidence import compute_pose_labels


def test_compute_pose_labels_uses_pocket_centered_frame() -> None:
    ligand = {"atom_coords": torch.tensor([[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]])}
    center = torch.tensor([10.0, 0.0, 0.0])
    poses = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]
    )

    labels = compute_pose_labels(ligand, center, poses)

    assert torch.equal(labels["atom_disp"], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert torch.equal(labels["pose_rmsd"], torch.tensor([0.0, 1.0]))
