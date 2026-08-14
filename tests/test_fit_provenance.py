import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_provenance import assert_heldout_geometry_provenance, fit_voxel_provenance  # noqa: E402


def endpoint(indices, total=10):
    provenance = fit_voxel_provenance(indices, total)
    saved = {"fit_voxel_indices": (
        np.empty(0, dtype=np.int64) if provenance["indices"] is None else provenance["indices"]
    )}
    return {"fit_provenance": {key: value for key, value in provenance.items() if key != "indices"}}, saved


def test_heldout_requires_exact_training_voxel_set():
    result, saved = endpoint(np.array([9, 2, 5, 1]))
    assert_heldout_geometry_provenance(result, saved, np.array([1, 2, 5, 9]), 10)

    with pytest.raises(RuntimeError, match="different voxel set"):
        assert_heldout_geometry_provenance(result, saved, np.array([1, 2, 5, 8]), 10)


def test_heldout_rejects_full_mask_and_legacy_endpoints():
    result, saved = endpoint(None)
    with pytest.raises(RuntimeError, match="full mask"):
        assert_heldout_geometry_provenance(result, saved, np.array([1, 2]), 10)

    with pytest.raises(RuntimeError, match="no fit-provenance"):
        assert_heldout_geometry_provenance({}, saved, np.array([1, 2]), 10)
