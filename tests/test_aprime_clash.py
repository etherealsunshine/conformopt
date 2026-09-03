import numpy as np
import torch

from scripts.aprime_clash import ClashResidualContext


def test_clash_residual_and_jacobian_match_finite_difference():
    context = ClashResidualContext(
        dynamic_pair_i=torch.tensor([0], dtype=torch.long),
        dynamic_pair_j=torch.tensor([1], dtype=torch.long),
        dynamic_pair_threshold=torch.tensor([2.0], dtype=torch.float64),
        environment_pair_i=torch.tensor([2], dtype=torch.long),
        environment_xyz=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        environment_threshold=torch.tensor([2.5], dtype=torch.float64),
        pair_cutoff_A=4.5,
        threshold_scale=0.75,
        dynamic_atom_count=3,
        source_pdb="synthetic",
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    residual = context.residual(coordinates, weight=4.0)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: context.residual(value, weight=4.0), coordinates,
    ).detach().numpy()
    step = 1.0e-6
    finite_difference = np.zeros_like(jacobian)
    for slot in range(2):
        for atom in range(3):
            for axis in range(3):
                plus = coordinates.detach().numpy().copy()
                minus = coordinates.detach().numpy().copy()
                plus[slot, atom, axis] += step
                minus[slot, atom, axis] -= step
                finite_difference[:, slot, atom, axis] = (
                    context.residual(torch.from_numpy(plus), 4.0).detach().numpy()
                    - context.residual(torch.from_numpy(minus), 4.0).detach().numpy()
                ) / (2.0 * step)
    assert residual.shape == (4,)
    assert np.allclose(jacobian, finite_difference, atol=1.0e-6, rtol=1.0e-6)


def test_non_overlapping_contact_has_zero_residual():
    context = ClashResidualContext(
        dynamic_pair_i=torch.tensor([0], dtype=torch.long),
        dynamic_pair_j=torch.tensor([1], dtype=torch.long),
        dynamic_pair_threshold=torch.tensor([2.0], dtype=torch.float64),
        environment_pair_i=torch.empty(0, dtype=torch.long),
        environment_xyz=torch.empty((0, 3), dtype=torch.float64),
        environment_threshold=torch.empty(0, dtype=torch.float64),
        pair_cutoff_A=4.5,
        threshold_scale=0.75,
        dynamic_atom_count=2,
        source_pdb="synthetic",
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    assert torch.equal(context.residual(coordinates, weight=25.0), torch.zeros(2, dtype=torch.float64))


def test_pair_count_normalization_uses_both_slots():
    context = ClashResidualContext(
        dynamic_pair_i=torch.tensor([0], dtype=torch.long),
        dynamic_pair_j=torch.tensor([1], dtype=torch.long),
        dynamic_pair_threshold=torch.tensor([2.0], dtype=torch.float64),
        environment_pair_i=torch.empty(0, dtype=torch.long),
        environment_xyz=torch.empty((0, 3), dtype=torch.float64),
        environment_threshold=torch.empty(0, dtype=torch.float64),
        pair_cutoff_A=4.5,
        threshold_scale=0.75,
        dynamic_atom_count=2,
        source_pdb="synthetic",
    )
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    residual = context.residual(
        coordinates, weight=4.0, normalize_by_pair_count=True,
    )
    assert context.pair_count == 1
    assert context.residual_pair_count == 2
    assert np.allclose(residual.numpy(), [np.sqrt(2.0), np.sqrt(0.5)])
