import torch

from density_denoiser.five_site_optimizer import (
    _inverse_kinematics_to_peak,
    _merge_occupancies_for_respawn,
    _reset_adam_parameter_slice,
    _respawn_due,
)


def test_zero_respawn_cadence_is_literal_noop():
    assert not any(_respawn_due(0, step, 500) for step in range(501))
    assert _respawn_due(100, 100, 500)
    assert not _respawn_due(100, 500, 500)


def test_merge_preserves_total_and_pair_occupancy():
    logits = torch.log(
        torch.tensor([0.35, 0.15, 0.30, 0.20], dtype=torch.float64)
    ).requires_grad_(True)
    before = torch.softmax(logits.detach(), dim=0)
    after = _merge_occupancies_for_respawn(logits, keeper=0, freed=1)
    torch.testing.assert_close(after.sum(), torch.tensor(1.0, dtype=after.dtype))
    torch.testing.assert_close(
        after[0] + after[1], before[0] + before[1]
    )
    assert 0.0 < float(after[1]) <= 1e-6


def test_reset_adam_parameter_slice_only_clears_selected_slot_moments():
    parameter = torch.ones((3, 2), requires_grad=True)
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    parameter.sum().backward()
    optimizer.step()
    before = optimizer.state[parameter]["exp_avg"].clone()
    _reset_adam_parameter_slice(optimizer, parameter, 1)
    state = optimizer.state[parameter]
    torch.testing.assert_close(state["exp_avg"][1], torch.zeros(2))
    torch.testing.assert_close(state["exp_avg_sq"][1], torch.zeros(2))
    torch.testing.assert_close(state["exp_avg"][0], before[0])
    torch.testing.assert_close(state["exp_avg"][2], before[2])


def test_local_inverse_kinematics_reaches_peak_with_one_heavy_atom():
    initial = torch.tensor([0.0])
    peak = torch.tensor([1.5, 0.0, 0.0])

    def coordinates_from_chi(chi):
        return torch.stack(
            (
                torch.stack((chi[0], chi[0] * 0.0, chi[0] * 0.0)),
                torch.stack((chi[0] * 0.0, chi[0], chi[0] * 0.0)),
            )
        )

    fitted, atom_index, distance = _inverse_kinematics_to_peak(
        initial,
        peak,
        coordinates_from_chi,
        steps=100,
        learning_rate=0.1,
    )
    assert atom_index == 0
    assert distance < 0.02
    assert abs(float(fitted[0]) - 1.5) < 0.02
