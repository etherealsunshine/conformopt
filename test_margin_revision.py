import numpy as np

from scripts.summarize_frozen_v3_margin_revision import (
    average_ranks,
    build_rows,
    spearman_correlation,
)


def test_average_ranks_handles_ties():
    values = np.asarray([30.0, 10.0, 10.0, 20.0])
    assert np.allclose(average_ranks(values), [3.0, 0.5, 0.5, 2.0])


def test_spearman_is_one_for_monotonic_inputs():
    assert np.isclose(
        spearman_correlation(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([10.0, 20.0, 30.0]),
        ),
        1.0,
    )


def test_rows_are_sorted_by_unsymmetrized_separation():
    template = {
        "correct_candidate_rscc": "1.0",
        "coverage_best_wrong_candidate": "B_alone",
        "coverage_best_wrong_rscc": "0.9",
        "coverage_margin": "0.1",
        "occupancy_best_wrong_candidate": "A0.50_B0.50",
        "occupancy_best_wrong_rscc": "0.99",
        "occupancy_margin": "0.01",
        "coverage_discriminability_sigma": "",
        "duplicate_max_absolute_density_error": "0",
        "duplicate_rscc_absolute_error": "0",
    }
    rows = build_rows([
        {
            **template,
            "site": "WIDE_A_TEST1",
            "local_unsym_AB_separation_A": "4.0",
        },
        {
            **template,
            "site": "CLOSE_A_TEST2",
            "local_unsym_AB_separation_A": "1.0",
        },
    ])
    assert [row["site"] for row in rows] == [
        "CLOSE_A_TEST2", "WIDE_A_TEST1"
    ]
