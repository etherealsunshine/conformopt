import math

import torch

from density_denoiser.five_site_optimizer import (
    _apply_stage1_chi_noise_,
    _linear_chi_noise_sd_degrees,
)


def test_linear_chi_noise_schedule_starts_configured_and_ends_zero():
    values = [
        _linear_chi_noise_sd_degrees(30.0, step, 500)
        for step in range(1, 501)
    ]
    assert values[0] == 30.0
    assert values[-1] == 0.0
    assert all(left >= right for left, right in zip(values, values[1:]))
    assert math.isclose(values[249], 30.0 * (1.0 - 249.0 / 499.0))


def test_zero_chi_noise_is_bitwise_noop_and_does_not_consume_rng():
    chi = torch.tensor([[0.4, -0.8], [1.2, -2.4]])
    before = chi.clone()
    generator = torch.Generator().manual_seed(41)
    expected = torch.randn((3,), generator=generator)

    generator = torch.Generator().manual_seed(41)
    applied = _apply_stage1_chi_noise_(
        chi,
        initial_sd_degrees=0.0,
        step=1,
        total_steps=500,
        generator=generator,
    )
    observed = torch.randn((3,), generator=generator)

    assert applied == 0.0
    assert torch.equal(chi, before)
    assert torch.equal(observed, expected)


def test_nonzero_chi_noise_is_seeded_and_chi_only():
    left = torch.zeros((4, 3))
    right = torch.zeros((4, 3))
    left_generator = torch.Generator().manual_seed(44)
    right_generator = torch.Generator().manual_seed(44)

    left_sd = _apply_stage1_chi_noise_(
        left,
        initial_sd_degrees=10.0,
        step=1,
        total_steps=500,
        generator=left_generator,
    )
    right_sd = _apply_stage1_chi_noise_(
        right,
        initial_sd_degrees=10.0,
        step=1,
        total_steps=500,
        generator=right_generator,
    )

    assert left_sd == right_sd == 10.0
    assert torch.equal(left, right)
    assert torch.count_nonzero(left) > 0
