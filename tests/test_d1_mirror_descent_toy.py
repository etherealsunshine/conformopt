import numpy as np

from scripts.run_d1_mirror_descent_toy import (
    START_Q, START_W, TARGET, TRUE_Q, TRUE_W, models, normalized_mirror_gradient, run,
)


def test_exact_qp_collapses_second_slot_on_first_evaluation():
    result = run("qp", eta=0.0)
    first = result["trajectory"][0]
    assert first["w"][1] == 0.0
    np.testing.assert_allclose(result["initial_q"], START_Q)


def test_mirror_descent_keeps_both_slots_and_recovers_toy_truth():
    result = run("mirror", eta=0.01)
    assert result["strictly_positive_final_weights"]
    assert result["slot_separation"] > 0.8 * result["truth_separation"]
    assert result["q_error"] < 0.25
    assert result["w_error"] < 0.10
    np.testing.assert_allclose(result["true_q"], TRUE_Q)
    np.testing.assert_allclose(result["true_w"], TRUE_W)
    assert TARGET.shape == models(TRUE_Q).shape[1:]


def test_mirror_gradient_direction_is_invariant_to_residual_scale():
    gradient = np.asarray([2.0, -5.0])
    np.testing.assert_allclose(
        normalized_mirror_gradient(gradient),
        normalized_mirror_gradient(100.0 * gradient),
    )


def test_toy_mirror_update_is_invariant_to_residual_scale_at_fixed_geometry():
    from scripts.run_d1_mirror_descent_toy import loss_and_gradients, mirror_update

    _, _, gradient, _ = loss_and_gradients(START_Q, START_W)
    baseline = mirror_update(START_W, gradient, eta=0.01)
    for scale in (10.0, 100.0):
        _, _, scaled_gradient, _ = loss_and_gradients(
            START_Q, START_W, occupancy_residual_scale=scale,
        )
        np.testing.assert_allclose(
            baseline, mirror_update(START_W, scaled_gradient, eta=0.01),
            atol=1e-12, rtol=0.0,
        )
