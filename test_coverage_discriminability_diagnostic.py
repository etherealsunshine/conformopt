import numpy as np

from scripts.diagnose_frozen_v3_coverage_discriminability import (
    candidate_rsccs,
    fixed_label_rmsd,
)


def test_fixed_label_rmsd_is_conventional():
    left = np.zeros((2, 3), dtype=float)
    right = np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.isclose(fixed_label_rmsd(left, right), np.sqrt(12.5))


def test_correct_candidate_is_exact_and_duplicate_is_invariant():
    density_a = np.asarray([0.0, 1.0, 4.0, 0.2, 0.0])
    density_b = np.asarray([1.0, 0.0, 0.5, 3.0, 0.2])
    target_a, target_b = 0.37, 0.63
    target = target_a * density_a + target_b * density_b
    rows, summary = candidate_rsccs(
        target, density_a, density_b, target_a, target_b
    )
    by_name = {row["candidate"]: row for row in rows}
    assert np.isclose(by_name["correct"]["rscc"], 1.0)
    assert np.isclose(
        by_name["A_alone"]["rscc"], by_name["A_duplicated"]["rscc"]
    )
    assert summary["duplicate_max_absolute_density_error"] == 0.0
    assert summary["coverage_margin"] > 0.0
    assert summary["occupancy_margin"] >= 0.0


def test_candidate_classes_remain_separate():
    density_a = np.asarray([0.0, 1.0, 2.0, 0.0])
    density_b = np.asarray([1.0, 0.0, 0.0, 2.0])
    target = 0.42 * density_a + 0.58 * density_b
    _, summary = candidate_rsccs(
        target, density_a, density_b, 0.42, 0.58
    )
    assert str(summary["coverage_best_wrong_candidate"]).endswith("alone")
    assert str(summary["occupancy_best_wrong_candidate"]).startswith("A")


def test_deposited_occupancy_split_is_not_counted_as_wrong():
    density_a = np.asarray([0.0, 1.0, 2.0, 0.0])
    density_b = np.asarray([1.0, 0.0, 0.0, 2.0])
    target = 0.25 * density_a + 0.75 * density_b
    rows, summary = candidate_rsccs(
        target, density_a, density_b, 0.25, 0.75
    )
    deposited_row = next(
        row for row in rows if row["candidate"] == "A0.25_B0.75"
    )
    assert deposited_row["matches_deposited_occupancy"]
    assert summary["occupancy_best_wrong_candidate"] != "A0.25_B0.75"
    assert summary["occupancy_margin"] > 0.0
