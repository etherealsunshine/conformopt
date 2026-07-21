import torch

from probe4_core import (
    LearnedEnergy,
    first_order_refine,
    residue_onehot,
    rotate_points,
    torsion_to_coords,
)


def test_rodrigues_rotation_is_differentiable():
    angle = torch.tensor(torch.pi / 2, requires_grad=True)
    point = torch.tensor([[1.0, 0.0, 0.0]])
    rotated = rotate_points(point, torch.zeros(3), torch.tensor([0.0, 0.0, 1.0]), angle)
    torch.testing.assert_close(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
    rotated[:, 1].sum().backward()
    assert angle.grad is not None


def test_torsion_to_coords_preserves_bond_geometry():
    template = torch.tensor([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    result = torsion_to_coords(
        template,
        ["CB", "CG"],
        torch.tensor([1.2]),
        [("CA", "CB", ("CG",))],
        {"CA": torch.zeros(3)},
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(result[1] - result[0]),
        torch.linalg.vector_norm(template[1] - template[0]),
    )


def test_first_order_endpoint_retains_parameter_gradient():
    torch.manual_seed(1)
    model = LearnedEnergy(density_feat_dim=8, hidden=16, n_layers=2)
    final, _ = first_order_refine(
        model,
        torch.randn(8),
        residue_onehot("ARG", device="cpu"),
        torch.randn(4),
        n_chi=4,
        steps=3,
        alpha=0.1,
        training=True,
    )
    (final.square().sum()).backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
