import cuequivariance as cue
import cuequivariance_torch as cuet
import pytest
import torch

from effdock.device import move_module_to_device, resolve_device
from effdock.geometry.se3 import quaternion_to_matrix
from effdock.models.equivariant import _linear_method, linear_method_for_device
from effdock.training.trainer import _training_autocast, setup_ddp


def test_auto_device_prefers_mps_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device() == torch.device("mps")


def test_explicit_device_is_preserved() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_non_cuda_models_use_native_equivariant_kernels() -> None:
    with linear_method_for_device(torch.device("mps")):
        assert _linear_method() == "naive"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_equivariant_linear_and_tensor_product_backward() -> None:
    device = torch.device("mps")
    irreps = cue.Irreps("O3", "2x0e + 1x1o")
    harmonics = cue.Irreps("O3", "1x0e + 1x1o")
    with linear_method_for_device(device):
        linear = cuet.Linear(irreps, irreps, layout=cue.mul_ir, method=_linear_method())
        product = cuet.FullyConnectedTensorProduct(
            irreps,
            harmonics,
            irreps,
            layout_in1=cue.mul_ir,
            layout_in2=cue.mul_ir,
            layout_out=cue.mul_ir,
            internal_weights=True,
            shared_weights=True,
            method=_linear_method(),
        )
    linear = move_module_to_device(linear, device)
    product = move_module_to_device(product, device)
    features = torch.randn(4, irreps.dim, device=device, requires_grad=True)
    directions = torch.randn(4, harmonics.dim, device=device)
    loss = linear(features).square().mean() + product(features, directions).square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_newton_euler_mps_float64_work_falls_back_to_cpu() -> None:
    from effdock.models.effdock import newton_euler_aggregate

    device = torch.device("mps")
    f_atom = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    atom_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
    centers = torch.tensor([[0.5, 0.0, 0.0]], device=device)
    frag_ids = torch.tensor([0, 0], device=device)
    sizes = torch.tensor([2], device=device)
    outputs = newton_euler_aggregate(f_atom, atom_pos, centers, frag_ids, 1, sizes)

    assert all(value.device.type == "mps" for value in outputs)
    assert all(torch.isfinite(value).all() for value in outputs)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_contextual_confidence_frame_recovery_uses_cpu_float64_svd() -> None:
    from effdock.confidence.features import recover_frag_state

    device = torch.device("mps")
    template = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [-0.2, 0.8, 0.1], [0.1, -0.3, 1.0]],
        device=device,
    )
    frag_ids = torch.zeros(4, dtype=torch.long, device=device)
    local = template - template.mean(0)
    rotation = quaternion_to_matrix(
        torch.tensor([0.7, 0.2, -0.4, 0.5], dtype=torch.float32, device=device)
    )
    posed = template @ rotation.T + torch.tensor([2.3, -1.7, 0.5], device=device)
    center, quaternion = recover_frag_state(
        posed,
        local,
        frag_ids,
        1,
        frame_policy="contextual_v1",
        template_atom_pos=template,
    )
    recovered = quaternion_to_matrix(quaternion)

    assert center.device.type == "mps"
    assert torch.allclose(recovered, rotation.unsqueeze(0), atol=1e-4, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_muon_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.randn(4, 4, device="mps"))
    optimizer = torch.optim.Muon([parameter])
    parameter.square().sum().backward()
    optimizer.step()

    assert torch.isfinite(parameter).all()
    with linear_method_for_device(torch.device("cpu")):
        assert _linear_method() == "naive"


def test_cpu_training_autocast_is_disabled() -> None:
    with _training_autocast(torch.device("cpu"), torch.bfloat16, enabled=True):
        value = torch.ones(2) * 2
    assert value.dtype == torch.float32


def test_multi_process_training_fails_clearly_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="currently requires CUDA"):
        setup_ddp()
