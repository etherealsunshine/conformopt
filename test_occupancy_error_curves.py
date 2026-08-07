import math

from scripts.summarize_frozen_v3_occupancy_error_curves import (
    build_rows,
    error_bin_label,
    spearman,
)


def test_spearman_monotonic_with_ties():
    assert math.isclose(spearman([0, 1, 1, 2], [0, 2, 2, 4]), 1.0)


def test_error_bin_boundaries():
    assert error_bin_label(0.0) == "[0.0, 0.1)"
    assert error_bin_label(0.0999) == "[0.0, 0.1)"
    assert error_bin_label(0.1) == "[0.1, 0.2)"
    assert error_bin_label(0.699) == "[0.6, 0.7)"


def test_build_rows_uses_same_conformer_fraction():
    candidates = [
        {
            "site": "TEST_A_RES1",
            "candidate": "correct",
            "candidate_class": "deposited",
            "target_A_occupancy": "0.28",
            "target_B_occupancy": "0.72",
            "occupancy_A": "0.28",
            "occupancy_B": "0.72",
            "rscc": "1.0",
        }
    ]
    for a in (0.25, 0.50, 0.75, 0.90):
        candidates.append(
            {
                "site": "TEST_A_RES1",
                "candidate": f"A{a:.2f}_B{1-a:.2f}",
                "candidate_class": "occupancy",
                "target_A_occupancy": "0.28",
                "target_B_occupancy": "0.72",
                "occupancy_A": str(a),
                "occupancy_B": str(1 - a),
                "matches_deposited_occupancy": "False",
                "rscc": "0.9",
            }
        )
    # build_rows has an intentional 20-site/80-row production guard.
    candidates = [
        {**row, "site": f"TEST{i:02d}_A_RES1"}
        for i in range(20)
        for row in candidates
    ]
    per_site = [
        {"site": f"TEST{i:02d}_A_RES1", "coverage_margin": "0.05"}
        for i in range(20)
    ]
    rows, thresholds = build_rows(candidates, per_site)
    first_site = [row for row in rows if row["site"] == "TEST00_A_RES1"]
    assert [round(row["occupancy_error_A_fraction"], 2) for row in first_site] == [
        0.03,
        0.22,
        0.47,
        0.62,
    ]
    assert all(math.isclose(row["margin"], 0.1) for row in first_site)
    assert len(thresholds) == 20
