import math

import torch

from density_denoiser.five_site_optimizer import (
    _initialize_chi_offsets,
    _unique_canonical_centers_radians,
)
from experiments.probe4.core import wrap_angles


def _mapping(base: torch.Tensor, direction: torch.Tensor):
    def delta_from_physical(desired: torch.Tensor) -> torch.Tensor:
        return direction * wrap_angles(desired - base)

    def physical_from_delta(delta: torch.Tensor) -> torch.Tensor:
        return wrap_angles(base + direction * delta)

    return delta_from_physical, physical_from_delta


def _nearest_center_error(
    values: torch.Tensor, centers: tuple[float, ...]
) -> torch.Tensor:
    center_tensor = torch.tensor(centers, dtype=values.dtype)
    delta = torch.atan2(
        torch.sin(values[:, None] - center_tensor[None, :]),
        torch.cos(values[:, None] - center_tensor[None, :]),
    )
    return delta.abs().min(dim=1).values


def test_control_initialization_is_bitwise_historical_randn():
    base = torch.tensor([0.3, -1.1])
    direction = torch.tensor([1.0, -1.0])
    delta_from_physical, _ = _mapping(base, direction)
    expected_generator = torch.Generator().manual_seed(407)
    actual_generator = torch.Generator().manual_seed(407)
    expected = torch.randn((4, 2), generator=expected_generator)
    actual = _initialize_chi_offsets(
        mode="deposited_a_cloud_60",
        resname="HIS",
        n_chi=2,
        K=4,
        base_physical_chi=base,
        delta_from_physical_chi=delta_from_physical,
        generator=actual_generator,
        device=torch.device("cpu"),
        jitter_degrees=12.0,
    )
    assert torch.equal(actual, expected)


def test_wide_cloud_is_exactly_twice_control_draw():
    base = torch.tensor([0.3, -1.1])
    direction = torch.tensor([1.0, -1.0])
    delta_from_physical, _ = _mapping(base, direction)
    expected_generator = torch.Generator().manual_seed(91)
    actual_generator = torch.Generator().manual_seed(91)
    expected = 2.0 * torch.randn((4, 2), generator=expected_generator)
    actual = _initialize_chi_offsets(
        mode="deposited_a_cloud_120",
        resname="HIS",
        n_chi=2,
        K=4,
        base_physical_chi=base,
        delta_from_physical_chi=delta_from_physical,
        generator=actual_generator,
        device=torch.device("cpu"),
        jitter_degrees=12.0,
    )
    assert torch.equal(actual, expected)


def test_canonical_free_is_deposited_a_independent_and_balanced():
    direction = torch.tensor([1.0, -1.0])
    physical_results = []
    for base in (torch.tensor([0.2, -0.7]), torch.tensor([-2.4, 1.9])):
        delta_from_physical, physical_from_delta = _mapping(base, direction)
        offsets = _initialize_chi_offsets(
            mode="canonical_stratified_free",
            resname="HIS",
            n_chi=2,
            K=4,
            base_physical_chi=base,
            delta_from_physical_chi=delta_from_physical,
            generator=torch.Generator().manual_seed(12),
            device=torch.device("cpu"),
            jitter_degrees=0.0,
        )
        physical_results.append(physical_from_delta(offsets))

    assert torch.allclose(physical_results[0], physical_results[1], atol=1e-6)
    physical = physical_results[0]
    for chi_index in range(2):
        centers = _unique_canonical_centers_radians("HIS", chi_index)
        assert torch.all(
            _nearest_center_error(physical[:, chi_index], centers) < 1e-6
        )
        assigned = [
            min(
                range(len(centers)),
                key=lambda index: abs(math.atan2(
                    math.sin(float(value) - centers[index]),
                    math.cos(float(value) - centers[index]),
                )),
            )
            for value in physical[:, chi_index]
        ]
        counts = [assigned.count(index) for index in range(len(centers))]
        assert max(counts) - min(counts) <= 1


def test_canonical_anchor_preserves_slot_zero_and_stratifies_others():
    base = torch.tensor([0.37, -1.24, 2.61])
    direction = torch.tensor([1.0, -1.0, 1.0])
    delta_from_physical, physical_from_delta = _mapping(base, direction)
    offsets = _initialize_chi_offsets(
        mode="canonical_stratified_a_anchor",
        resname="MET",
        n_chi=3,
        K=4,
        base_physical_chi=base,
        delta_from_physical_chi=delta_from_physical,
        generator=torch.Generator().manual_seed(77),
        device=torch.device("cpu"),
        jitter_degrees=0.0,
    )
    physical = physical_from_delta(offsets)
    assert torch.allclose(physical[0], base, atol=1e-6)
    assert len({tuple(row.tolist()) for row in physical[1:]}) == 3
    for chi_index in range(3):
        centers = _unique_canonical_centers_radians("MET", chi_index)
        assert torch.all(
            _nearest_center_error(physical[1:, chi_index], centers) < 1e-6
        )


def test_terminal_pi_and_minus_pi_are_one_physical_center():
    centers = _unique_canonical_centers_radians("ARG", 3)
    assert len(centers) == 3
