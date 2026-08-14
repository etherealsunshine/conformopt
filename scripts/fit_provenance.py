"""Strict provenance records for density fits that will be evaluated held-out."""

from __future__ import annotations

import hashlib

import numpy as np


def voxel_hash(indices: np.ndarray) -> str:
    canonical = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def fit_voxel_provenance(training_indices, total_voxels: int) -> dict[str, object]:
    """Describe the exact voxel set supplied while an endpoint was fitted."""
    if training_indices is None:
        return {"fit_scope": "full-mask", "total_mask_voxels": int(total_voxels),
                "fitted_voxel_count": int(total_voxels), "indices": None,
                "sha256": None}
    indices = np.asarray(training_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= total_voxels):
        raise ValueError("training_indices must be a nonempty in-mask one-dimensional index array")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("training_indices contains duplicate voxels")
    canonical = np.sort(indices)
    return {"fit_scope": "training-only", "total_mask_voxels": int(total_voxels),
            "fitted_voxel_count": int(len(canonical)), "indices": canonical,
            "sha256": voxel_hash(canonical)}


def assert_heldout_geometry_provenance(candidate_result: dict[str, object], saved: object,
                                       expected_training_indices: np.ndarray,
                                       total_voxels: int) -> None:
    """Refuse held-out scoring unless geometry was fitted on this exact split."""
    provenance = candidate_result.get("fit_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("held-out evaluation refused: endpoint has no fit-provenance record")
    if provenance.get("fit_scope") != "training-only":
        raise RuntimeError("held-out evaluation refused: endpoint geometry was fitted on the full mask")
    expected = np.sort(np.asarray(expected_training_indices, dtype=np.int64))
    observed = np.asarray(saved["fit_voxel_indices"], dtype=np.int64)
    if int(provenance.get("total_mask_voxels", -1)) != int(total_voxels):
        raise RuntimeError("held-out evaluation refused: endpoint mask cardinality differs from scorer mask")
    if not np.array_equal(observed, expected):
        raise RuntimeError("held-out evaluation refused: endpoint geometry was fitted on a different voxel set")
    if provenance.get("sha256") != voxel_hash(expected):
        raise RuntimeError("held-out evaluation refused: endpoint voxel-index fingerprint is invalid")
