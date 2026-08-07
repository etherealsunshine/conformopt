import csv

import torch

from density_denoiser.five_site_optimizer import (
    _lobe_statistics,
    _missed_minor_starts,
)


def test_missed_minor_starts_uses_occupancy_rank(tmp_path):
    path = tmp_path / "ensembles.csv"
    rows = [
        {
            "site": "SITE",
            "start": 0,
            "found_A_conventional": "False",
            "found_B_conventional": "True",
            "target_A_occupancy": "0.3",
            "target_B_occupancy": "0.7",
        },
        {
            "site": "SITE",
            "start": 1,
            "found_A_conventional": "True",
            "found_B_conventional": "False",
            "target_A_occupancy": "0.7",
            "target_B_occupancy": "0.3",
        },
        {
            "site": "SITE",
            "start": 2,
            "found_A_conventional": "True",
            "found_B_conventional": "True",
            "target_A_occupancy": "0.3",
            "target_B_occupancy": "0.7",
        },
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    selected = _missed_minor_starts(path, "SITE")

    assert set(selected) == {0, 1}
    assert selected[0]["minor"] == "A"
    assert selected[0]["major"] == "B"
    assert selected[1]["minor"] == "B"
    assert selected[1]["major"] == "A"


def test_lobe_statistics_reports_positive_and_overlap():
    grid = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    lobe = torch.tensor([[0.0, 0.0, 0.0]])
    other = torch.tensor([[0.5, 0.0, 0.0]])
    residual = torch.tensor([2.0, -1.0, 10.0])

    result = _lobe_statistics(
        residual,
        grid,
        lobe,
        other,
        radius=0.6,
        voxel_volume=0.125,
    )

    assert result["voxels"] == 2
    assert result["overlap_fraction"] == 1.0
    assert result["integrated_positive"] == 0.25
    assert result["integrated_signed"] == 0.125
