import numpy as np

from density_denoiser.diagnose_frozen_tmol_gate import (
    matched_rmsd,
    pearson_and_spearman,
    rmsd_bin,
    summarize_margin_bins,
)


def test_matched_rmsd_uses_assignment_specific_reference():
    row = {
        "assignment": "B",
        "rmsd_to_A_conventional": "0.2",
        "rmsd_to_B_conventional": "0.7",
    }
    assert matched_rmsd(row) == 0.7


def test_rmsd_bins_have_requested_boundaries():
    assert rmsd_bin(0.1) == "<=0.1"
    assert rmsd_bin(0.10001) == "0.1-0.3"
    assert rmsd_bin(0.3) == "0.1-0.3"
    assert rmsd_bin(0.6) == "0.3-0.6"
    assert rmsd_bin(0.999) == "0.6-1.0"


def test_margin_bins_report_positive_and_half_unit_counts():
    rows = [
        {"rmsd_bin": "<=0.1", "tmol_margin": -0.2},
        {"rmsd_bin": "<=0.1", "tmol_margin": 0.2},
        {"rmsd_bin": "<=0.1", "tmol_margin": 0.8},
    ]
    summary = summarize_margin_bins(rows)[0]
    assert summary["conformers"] == 3
    assert summary["margin_positive"] == 2
    assert summary["margin_0_to_0_5"] == 1


def test_correlations_include_average_rank_tie_handling():
    pearson, spearman = pearson_and_spearman(
        np.asarray([1.0, 2.0, 2.0, 4.0]),
        np.asarray([1.0, 3.0, 2.0, 4.0]),
    )
    assert 0.92 < pearson < 0.93
    assert 0.94 < spearman < 0.95
