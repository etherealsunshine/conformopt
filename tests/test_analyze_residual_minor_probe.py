from scripts.analyze_residual_minor_probe import summarize


def test_summarize_separates_recovered_and_failed_signal():
    rows = [
        {
            "recovered_minor_lt_1A": "True",
            "initial_rmsd_to_minor": "2",
            "final_rmsd_to_minor": "0.5",
            "deposited_minor_occupancy": "0.3",
            "final_occupancy": "0.2",
            "minor_before_rms": "1.0",
            "minor_before_integrated_positive": "4",
            "major_before_integrated_positive": "1",
            "minor_after_rms": "0.5",
            "minor_before_overlap_fraction": "0.1",
        },
        {
            "recovered_minor_lt_1A": "False",
            "initial_rmsd_to_minor": "3",
            "final_rmsd_to_minor": "2",
            "deposited_minor_occupancy": "0.3",
            "final_occupancy": "0.1",
            "minor_before_rms": "2.0",
            "minor_before_integrated_positive": "8",
            "major_before_integrated_positive": "2",
            "minor_after_rms": "1.5",
            "minor_before_overlap_fraction": "0.2",
        },
    ]

    result = summarize(rows, separation=3.5)

    assert result["eligible"] == 2
    assert result["recovered"] == 1
    assert result["recovery_rate"] == 0.5
    assert result["minor_positive_integral_before_recovered_median"] == 4
    assert result["minor_positive_integral_before_failed_median"] == 8
    assert result["local_unsym_AB_separation_A"] == 3.5
