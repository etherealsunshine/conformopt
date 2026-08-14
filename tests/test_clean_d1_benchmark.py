import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clean_d1_benchmark import MOTION_BINS, collapse_altlocs  # noqa: E402


def test_motion_bins_are_fixed_and_nonoverlapping():
    assert MOTION_BINS == ((0.30, 0.80, "small"), (0.80, 1.50, "medium"), (1.50, None, "large"))


def test_collapse_altlocs_averages_coordinates_and_resets_identity(tmp_path):
    source = tmp_path / "source.pdb"
    source.write_text(
        f"ATOM  {1:5d}  CA AALA A{1:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{0.40:6.2f}{10.0:6.2f}          C  \n"
        f"ATOM  {2:5d}  CA BALA A{1:4d}    {1.0:8.3f}{0.0:8.3f}{0.0:8.3f}{0.60:6.2f}{20.0:6.2f}          C  \n"
        "END\n"
    )
    output = tmp_path / "collapsed.pdb"
    record = collapse_altlocs(source, output)
    text = output.read_text()
    assert record["collapsed_groups"] == 1
    assert text.count(" CA  ALA") == 1
    assert text[16] == " "
    assert float(text[30:38]) == 0.6
    assert float(text[54:60]) == 1.0
    assert float(text[60:66]) == 16.0
