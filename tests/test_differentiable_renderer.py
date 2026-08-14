import math

import torch

from density_denoiser.differentiable_renderer import (
    coefficients_for_elements,
    nerf_chain,
    nerf_place,
    render_cctbx_density,
)


def test_nerf_chain_preserves_requested_bonds_and_is_differentiable():
    initial = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [2.0, 1.2, 0.0]]],
        dtype=torch.float64,
    )
    lengths = torch.tensor([[1.33, 1.52]], dtype=torch.float64)
    angles = torch.tensor([[2.1, 1.9]], dtype=torch.float64)
    torsions = torch.tensor([[0.4, -1.2]], dtype=torch.float64, requires_grad=True)

    coordinates = nerf_chain(initial, lengths, angles, torsions)
    actual_lengths = (coordinates[..., 1:, :] - coordinates[..., :-1, :]).norm(dim=-1)
    assert torch.allclose(actual_lengths[..., 2:], lengths, atol=1e-12, rtol=0.0)
    loss = coordinates.square().sum()
    loss.backward()
    assert torch.isfinite(torsions.grad).all()
    assert torch.linalg.vector_norm(torsions.grad) > 0


def test_cctbx_density_supports_batches_and_masks():
    grid = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64
    )
    atoms = torch.tensor([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]], dtype=torch.float64)
    b_factors = torch.tensor([[12.0], [18.0]], dtype=torch.float64)
    coeffs = coefficients_for_elements(["C"])
    density = render_cctbx_density(
        grid,
        atoms,
        b_factors,
        coeffs,
        u_base=0.05,
        grid_mask=torch.tensor([True, True, False]),
    )
    assert density.shape == (2, 3)
    assert torch.all(density[:, 2] == 0)
    assert torch.all(density[:, :2] > 0)
    assert density[0, 0] > density[0, 1]


def test_cctbx_exponent_table_has_a_straight_through_gradient():
    grid = torch.tensor([[0.2, 0.1, 0.0]], dtype=torch.float64)
    atoms = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    b_factors = torch.tensor([15.0], dtype=torch.float64)
    coeffs = coefficients_for_elements(["C"])
    value = render_cctbx_density(grid, atoms, b_factors, coeffs, u_base=0.05).sum()
    value.backward()
    assert torch.isfinite(atoms.grad).all()
    assert torch.linalg.vector_norm(atoms.grad) > 0


def test_nerf_place_zero_torsion_is_finite():
    point = nerf_place(
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64),
        1.5,
        math.pi / 2,
        0.0,
    )
    assert torch.isfinite(point).all()
