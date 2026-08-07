import csv

import pytest

from scripts.plot_latest_synthetic_recovery_cascade import load_rows


def _write_source(path, *, total_strict=626):
    fields = [
        "site",
        "starts",
        "v3_both_found",
        "v3_occupancy",
        "v3_rotamer",
        "v3_direct_clash",
        "v3_symmetry_clash",
        "v3_tmol_0_44",
    ]
    rows = []
    for index in range(20):
        rows.append(
            {
                "site": f"S{index:02d}_A_RES1",
                "starts": 50,
                "v3_both_found": 40,
                "v3_occupancy": 35,
                "v3_rotamer": 34,
                "v3_direct_clash": 34,
                "v3_symmetry_clash": 34,
                "v3_tmol_0_44": 30 + (index == 19),
            }
        )
    rows.append(
        {
            "site": "TOTAL",
            "starts": 1000,
            "v3_both_found": 742,
            "v3_occupancy": 714,
            "v3_rotamer": 710,
            "v3_direct_clash": 710,
            "v3_symmetry_clash": 710,
            "v3_tmol_0_44": total_strict,
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_rows_guards_and_sorts(tmp_path):
    source = tmp_path / "cascade.csv"
    _write_source(source)
    rows, total = load_rows(source)
    assert len(rows) == 20
    assert rows[0]["site_label"] == "S19"
    assert rows[0]["plus_frozen_physical_audit"] == 31
    assert total["v3_tmol_0_44"] == 626


def test_load_rows_rejects_nonfrozen_total(tmp_path):
    source = tmp_path / "cascade.csv"
    _write_source(source, total_strict=625)
    with pytest.raises(RuntimeError, match="guard failed"):
        load_rows(source)
