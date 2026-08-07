import numpy as np

from scripts.analyze_langevin_trajectories import (
    loss_at_step,
    nearest_well_indices,
    trajectory_slot_metrics,
)


def test_loss_at_step_aligns_losses_after_initial_state():
    steps = np.asarray([0, 1, 2, 3])
    phases = np.asarray([1, 1, 1, 2])
    losses = np.asarray([4.0, 3.0, 2.0])
    assert loss_at_step(steps, phases, losses, 2) == 3.0
    assert np.isnan(loss_at_step(steps, phases, losses, 3))


def test_well_crossing_and_circular_distance():
    chi = np.deg2rad(np.asarray([
        [[-70.0]],
        [[-50.0]],
        [[50.0]],
        [[70.0]],
    ]))
    metrics = trajectory_slot_metrics(
        chi,
        base_physical_chi=np.asarray([0.0]),
        delta_direction=np.asarray([1.0]),
        residue_type="SER",
    )[0]
    assert metrics["rotamer_well_boundary_crossings"] == 1
    assert metrics["steps_with_any_well_boundary_crossing"] == 1
    assert np.isclose(metrics["chi_space_distance_travelled_degrees"], 140.0)


def test_nearest_well_treats_minus_pi_and_pi_as_same_center():
    values = np.deg2rad(np.asarray([[[-179.0]], [[179.0]]]))
    wells = nearest_well_indices(values, "SER")
    assert wells[0, 0, 0] == wells[1, 0, 0]
