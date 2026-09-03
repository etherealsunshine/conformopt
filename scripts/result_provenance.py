"""Minimal, machine-readable provenance for density-fit result files."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


PROVENANCE_SCHEMA_VERSION = 1
DEFAULT_FOLD_SEED = 20260805
DEFAULT_FOLD_COUNT = 5
DEFAULT_TEST_FRACTION = 0.20


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one immutable input/output file."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def realized_fold_provenance(
    mask_voxels: int,
    seed: int = DEFAULT_FOLD_SEED,
    n_folds: int = DEFAULT_FOLD_COUNT,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> dict[str, object]:
    """Reproduce and record the fold RNG stream, including directions.

    The historical splitter consumes one random test-index draw per fold
    before drawing the spatial directions.  Recording the realized directions
    is therefore necessary: the same seed does not imply the same folds when
    the mask size changes.
    """
    n_voxels = int(mask_voxels)
    if n_voxels < 1:
        raise ValueError("mask_voxels must be positive")
    n_test = round(float(test_fraction) * n_voxels)
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_folds)):
        rng.choice(n_voxels, size=n_test, replace=False)
    directions = []
    for _ in range(int(n_folds)):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        directions.append(direction.tolist())
    return {
        "seed": int(seed),
        "n_folds": int(n_folds),
        "test_fraction": float(test_fraction),
        "mask_voxels": n_voxels,
        "heldout_voxels_per_fold": int(n_test),
        "directions": directions,
    }


def runner_provenance(
    runner,
    source_pdb: Path,
    map_path: Path,
    endpoint_npzs: Mapping[str, Path],
    fold_seed: int = DEFAULT_FOLD_SEED,
) -> dict[str, object]:
    """Build the required provenance block for a completed site result."""
    base = getattr(runner, "base", runner)
    full_mask_voxels = int(base.mask.sum())
    target_voxels = int(len(base.target))
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_pdb": file_provenance(source_pdb),
        "map_mtz": file_provenance(map_path),
        "mask_voxels": full_mask_voxels,
        "target_voxels": target_voxels,
        "endpoint_npz": {
            name: file_provenance(path) for name, path in endpoint_npzs.items()
        },
        "folds": realized_fold_provenance(full_mask_voxels, seed=fold_seed),
        "renderer": {
            "backend": getattr(base, "renderer_backend", None),
            "mask_scope": getattr(base, "mask_scope", None),
            "density_atom_scope": getattr(base, "density_atom_scope", None),
            "model_atom_count": int(len(base.model_atom_indices)),
            "resolution_A": float(base.resolution),
            "map_scaler_structure": getattr(base, "map_scaler_structure", None),
            "b_factor_mode": getattr(base, "b_factor_mode", None),
        },
    }
