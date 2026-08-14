import math

import numpy as np
import pytest

from scripts.occupancy_selection import (
    diagnose_cardinality_caps,
    evaluate_qfit_coupled_thresholds,
    legacy_cull,
    qfit_bic,
    select_decoupled_miqp,
    solve_affine_qp,
)


def test_legacy_cull_is_only_a_final_zeroing_step():
    weights = np.array([0.89, 0.11, 0.02])
    np.testing.assert_allclose(legacy_cull(weights), [0.89, 0.11, 0.0])
    np.testing.assert_allclose(weights, [0.89, 0.11, 0.02])


def test_qfit_bic_counts_selected_conformers_not_cardinality_cap():
    bic, k = qfit_bic(3.0, 500, 6, 2)
    expected_k = 3.0 * 6 * 2 * 0.8
    expected_bic = 500 * math.log(3.0 / 500) + expected_k * math.log(500)
    assert math.isclose(k, expected_k)
    assert math.isclose(bic, expected_bic)


def test_decoupled_selection_keeps_an_eleven_percent_state():
    target = np.array([0.89, 0.11])
    models = np.eye(2)

    def fake_solver(target, models, *, cardinality, threshold):
        assert threshold == 0.02
        assert cardinality == 2
        return np.array([0.89, 0.11]), 0.0

    result = select_decoupled_miqp(
        target,
        models,
        cardinality_cap=4,
        t_min=0.02,
        n_atoms=1,
        solve_miqp=fake_solver,
    )
    assert result["cardinality_cap"] == 4
    assert result["effective_cardinality_cap"] == 2
    assert result["selected_slots"] == [0, 1]
    np.testing.assert_allclose(result["weights"], [0.89, 0.11])
    assert len(result["bic_candidates"]) == 1


def test_coupled_thresholds_use_threshold_as_both_floor_and_implicit_cap():
    target = np.array([1.0, 0.0])
    models = np.eye(2)
    calls = []

    def fake_solver(target, models, *, cardinality, threshold):
        calls.append((cardinality, threshold))
        return np.array([1.0, 0.0]), 0.0

    rows = evaluate_qfit_coupled_thresholds(
        target,
        models,
        n_atoms=1,
        thresholds=(1.0, 0.5),
        solve_miqp=fake_solver,
    )
    assert [row["t_min"] for row in rows] == [1.0, 0.5]
    assert calls == [(None, 1.0), (None, 0.5)]


def test_invalid_selection_floor_is_rejected():
    with pytest.raises(ValueError, match="t_min"):
        select_decoupled_miqp(
            np.ones(2),
            np.eye(2),
            cardinality_cap=4,
            t_min=0.0,
            n_atoms=1,
            solve_miqp=lambda *args, **kwargs: (np.ones(2), 0.0),
        )


def test_affine_qp_recovers_occupancy_and_intercept_without_a_floor():
    model = np.array([[1.0, 0.0, 0.5]])
    target = 0.35 + 0.7 * model[0]
    weights, intercept, rss = solve_affine_qp(target, model)
    np.testing.assert_allclose(weights, [0.7], atol=1e-5)
    assert math.isclose(intercept, 0.35, abs_tol=1e-5)
    assert rss < 1e-10


def test_cap_diagnostic_keeps_requested_caps_and_reports_effective_cap():
    target = np.array([1.0, 0.0])
    models = np.eye(2)

    def fake_solver(target, models, *, cardinality, threshold):
        if cardinality == 1:
            return np.array([1.0, 0.0]), 1.0
        return np.array([0.8, 0.2]), 0.1

    rows = diagnose_cardinality_caps(
        target,
        models,
        cardinality_caps=(1, 2, 3, 4),
        t_min=0.02,
        n_atoms=1,
        solve_miqp=fake_solver,
    )
    assert [row["cardinality_cap"] for row in rows] == [1, 2, 3, 4]
    assert [row["effective_cardinality_cap"] for row in rows] == [1, 2, 2, 2]
    assert rows[0]["n_selected_conformers"] == 1
    assert rows[1]["n_selected_conformers"] == 2
