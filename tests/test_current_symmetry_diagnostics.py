import math

from density_denoiser.analyze_tmol_assignments import summarize_assignment
from density_denoiser.attribute_rotamer_floor_rejections import (
    attribute_rejections,
)
from density_denoiser.summarize_tmol_margin_sweep import build_sweep


def _conformer(start, assignment, margin, rotamer=True):
    return {
        "candidate_id": f"site__{start}__{assignment}",
        "site": "SITE",
        "start": str(start),
        "conformer": "0" if assignment == "A" else "1",
        "occupancy": "0.5",
        "assignment": assignment,
        "rmsd_to_A_conventional": "0.05" if assignment == "A" else "2.0",
        "rmsd_to_B_conventional": "0.05" if assignment == "B" else "2.0",
        "tmol_delta_vs_matched_AB": str(margin),
        "rotamer_within_allowed_width": str(rotamer),
        "canonical_like_30deg": str(rotamer),
        "no_direct_clash": "True",
        "no_symmetry_clash": "True",
    }


def _ensemble(start):
    return {
        "site": "SITE",
        "start": str(start),
        "both_found_conventional": "True",
        "geometric_occupancy_success": "True",
    }


def test_margin_sweep_is_monotone_and_uses_posthoc_tolerances():
    conformers = [
        _conformer(0, "A", -1.0),
        _conformer(0, "B", 0.25),
        _conformer(1, "A", -1.0),
        _conformer(1, "B", -0.5, rotamer=False),
    ]
    rows, q99 = build_sweep(
        conformers, [_ensemble(0), _ensemble(1)], {"SITE": 1}
    )
    row = rows[0]
    assert row["both_found"] == 2
    assert row["recovery_and_occupancy"] == 2
    assert row["plus_all_active_rotamer"] == 1
    assert row["plus_all_active_direct_clash"] == 1
    assert row["plus_all_active_symmetry_clash"] == 1
    assert row["plus_all_active_tmol_tol_0_0"] == 0
    assert row["plus_all_active_tmol_tol_0_5"] == 1
    assert row["both_found_delta_vs_stale"] == 1
    assert math.isclose(q99[0]["positive_margin_q99"], 0.25)


def test_assignment_summary_reports_margin_distribution():
    rows = [
        _conformer(0, "A", -1.0),
        _conformer(1, "A", 1.0),
    ]
    summary = summarize_assignment(rows)
    assert summary["conformers"] == 2
    assert summary["tmol_pass_at_0"] == 1
    assert summary["tmol_pass_rate_at_0"] == 0.5
    assert summary["margin_median"] == 0.0


def test_floor_attribution_reports_each_failed_chi_and_excess():
    rejected, rows = attribute_rejections([{
        "site": "X",
        "control": "deposited_A",
        "residue_name": "LYS",
        "rotamer_pass": "False",
        "chi_degrees": "10;20;30;40",
        "nearest_rotamer": "a/b/c/d",
        "rotamer_deviation_degrees": "46;10;70;5",
        "rotamer_allowed_width_degrees": "45;45;60;60",
    }])
    assert len(rejected) == 1
    assert [row["failed_chi"] for row in rows] == ["chi1", "chi3"]
    assert [row["excess_degrees"] for row in rows] == [1.0, 10.0]
