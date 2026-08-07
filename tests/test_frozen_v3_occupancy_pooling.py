import numpy as np

from scripts.diagnose_frozen_v3_occupancy_pooling import (
    deposited_midpoint,
    identify_single_recovery,
    lawson_hanson_nnls,
    solve_gram_nnls,
)
from scripts.diagnose_unmatched_target_density import safe_ratio


def test_identify_single_recovery_preserves_raw_142_45_definition():
    row = {
        "found_A": "False",
        "found_B": "True",
        "target_A_occupancy": "0.3",
        "target_B_occupancy": "0.7",
    }
    result = identify_single_recovery(row)
    assert result is not None
    assert result["recovery_rank"] == "major_only"
    assert result["recovered_state"] == "B"
    assert result["missed_state"] == "A"


def test_deposited_midpoint_uses_equivalent_terminal_labels():
    names = ["CB", "CG", "OD1", "OD2"]
    deposited_a = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, -1.0, 0.0],
    ])
    deposited_b = deposited_a[[0, 1, 3, 2]]
    midpoint, permutation = deposited_midpoint(
        deposited_a, deposited_b, names, "ASP"
    )
    assert permutation == [0, 1, 3, 2]
    np.testing.assert_allclose(midpoint, deposited_a)


def test_gram_nnls_recovers_nonnegative_weights():
    columns = np.asarray([
        [1.0, 0.2],
        [0.1, 1.0],
        [0.7, 0.3],
    ])
    expected = np.asarray([0.35, 0.65])
    target = columns @ expected
    fitted, _gram, _right, diagnostics = solve_gram_nnls(columns, target)
    np.testing.assert_allclose(fitted, expected, atol=1e-7)
    assert diagnostics["density_residual_relative_l2"] < 1e-8


def test_lawson_hanson_nnls_clamps_negative_unconstrained_weight():
    matrix = np.eye(2)
    fitted, _iterations, optimality = lawson_hanson_nnls(
        matrix, np.asarray([0.4, -0.2])
    )
    np.testing.assert_allclose(fitted, [0.4, 0.0], atol=1e-10)
    assert optimality < 1e-10


def test_unmatched_density_safe_ratio():
    assert safe_ratio(0.25, 0.5) == 0.5
    assert np.isnan(safe_ratio(1.0, 0.0))
