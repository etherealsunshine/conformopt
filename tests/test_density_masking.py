import numpy as np
import pytest
import torch

from density_denoiser.five_site_optimizer import (
    _density_mse,
    _normalized_variance_weights,
    _reachable_volume_mask,
)


def test_reachable_volume_mask_is_union_of_padded_atom_positions():
    grid = np.stack(np.meshgrid(
        np.arange(3, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        indexing="ij",
    ), axis=-1)
    mask = _reachable_volume_mask(
        grid,
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        padding=0.6,
    )
    assert mask.reshape(-1).tolist() == [True, False, True]


def test_reachable_variance_weights_have_unit_mean_and_weight_density_mse():
    states = torch.tensor([
        [0.0, 1.0, 2.0],
        [0.0, 2.0, 4.0],
        [0.0, 3.0, 6.0],
    ])
    weights = _normalized_variance_weights(states)
    assert float(weights.mean()) == pytest.approx(1.0)
    assert weights.tolist() == pytest.approx([0.0, 0.6, 2.4])

    rendered = torch.tensor([1.0, 1.0, 1.0])
    target = torch.zeros(3)
    assert float(_density_mse(rendered, target, weights)) == pytest.approx(1.0)


def test_reachable_variance_requires_actual_state_variation():
    with pytest.raises(ValueError, match="variance must be positive"):
        _normalized_variance_weights(torch.ones((2, 4)))
