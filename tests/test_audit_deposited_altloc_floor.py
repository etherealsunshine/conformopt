from density_denoiser.audit_deposited_altloc_floor import (
    deterministic_site_priority,
    summarize_by_residue,
)


def test_site_priority_is_deterministic_and_key_specific():
    assert deterministic_site_priority("1ABC_A_SER1") == deterministic_site_priority(
        "1ABC_A_SER1"
    )
    assert deterministic_site_priority("1ABC_A_SER1") != deterministic_site_priority(
        "1ABC_A_SER2"
    )


def test_residue_summary_reports_conformer_and_pair_rejection():
    rows = [
        {
            "site": "s1",
            "residue_name": "SER",
            "rotamer_pass": True,
            "direct_clash_pass": True,
            "symmetry_clash_pass": True,
            "all_geometry_gates_pass": True,
        },
        {
            "site": "s1",
            "residue_name": "SER",
            "rotamer_pass": False,
            "direct_clash_pass": True,
            "symmetry_clash_pass": True,
            "all_geometry_gates_pass": False,
        },
    ]
    summary = summarize_by_residue(rows)[0]
    assert summary["deposited_pairs"] == 1
    assert summary["any_geometry_rejected_conformers"] == 1
    assert summary["conformer_false_rejection_rate"] == 0.5
    assert summary["pair_false_rejection_rate"] == 1.0
