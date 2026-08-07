from __future__ import annotations

import json

import gemmi
import numpy as np

from density_denoiser.export_density_triplet import export_triplet


def test_export_triplet_round_trips_ccp4_values(tmp_path):
    shape = (4, 4, 4)
    raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    targets = tmp_path / "targets.npz"
    np.savez_compressed(
        targets,
        raw=raw,
        denoised=raw + 100.0,
        denoiser_training_target=raw - 100.0,
        radial_mask=np.ones(shape, dtype=bool),
        metadata=np.asarray(json.dumps({
            "center": [1.0, 2.0, 3.0],
            "grid_spacing": 0.5,
            "patch_size": 4,
            "short_key": "TEST_A_MET1",
        })),
    )

    output = tmp_path / "export"
    manifest = export_triplet(targets, output)

    assert manifest["shape"] == [4, 4, 4]
    assert manifest["crystal_origin_xyz"] == [0.25, 1.25, 2.25]
    for label, expected in {
        "experimental_omit_mfo_dfc": raw,
        "denoised": raw + 100.0,
        "synthetic_ground_truth": raw - 100.0,
    }.items():
        ccp4 = gemmi.read_ccp4_map(str(output / manifest["maps"][label]))
        np.testing.assert_allclose(np.asarray(ccp4.grid), expected)
