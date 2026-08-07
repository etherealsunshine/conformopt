from density_denoiser.diagnose_site_tmol_and_pair_metrics import (
    positive_reproduction_percentile,
)


def test_positive_reproduction_percentile_excludes_negative_and_far_rows():
    rows = [
        {
            "assignment": "A",
            "rmsd_to_A_conventional": "0.05",
            "rmsd_to_B_conventional": "2.0",
            "tmol_delta_vs_matched_AB": "-0.2",
        },
        {
            "assignment": "A",
            "rmsd_to_A_conventional": "0.08",
            "rmsd_to_B_conventional": "2.0",
            "tmol_delta_vs_matched_AB": "0.1",
        },
        {
            "assignment": "B",
            "rmsd_to_A_conventional": "2.0",
            "rmsd_to_B_conventional": "0.09",
            "tmol_delta_vs_matched_AB": "0.3",
        },
        {
            "assignment": "B",
            "rmsd_to_A_conventional": "2.0",
            "rmsd_to_B_conventional": "0.2",
            "tmol_delta_vs_matched_AB": "0.4",
        },
    ]
    near, positive, q99 = positive_reproduction_percentile(rows)
    assert near == 3
    assert positive == 2
    assert 0.29 < q99 < 0.31
