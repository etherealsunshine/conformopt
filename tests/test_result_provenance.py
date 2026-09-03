from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.result_provenance import file_provenance, realized_fold_provenance


def test_file_provenance_records_sha256(tmp_path: Path):
    path = tmp_path / "input.dat"
    path.write_bytes(b"qfit provenance\n")
    record = file_provenance(path)
    assert record["path"] == str(path)
    assert record["sha256"] == "2fb0ba959597bb6979f89741b84dc98abe1e7cfe1fa53886241f0e6c5b69adbb"


def test_realized_folds_depend_on_mask_size_and_are_serializable():
    small = realized_fold_provenance(5978)
    large = realized_fold_provenance(8820)
    assert small["seed"] == large["seed"] == 20260805
    assert small["heldout_voxels_per_fold"] == 1196
    assert large["heldout_voxels_per_fold"] == 1764
    assert small["directions"] != large["directions"]
    json.dumps(small)
    assert np.isclose(np.linalg.norm(small["directions"][0]), 1.0)
