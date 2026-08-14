#!/usr/bin/env python3
"""Read-only spatial and support audit for guarded A'' CV fold records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from run_d1_8d_sequential_poc import atomic_json
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_reachability import local_index
from run_d1_supported_panel import full_models, site_name


SITES = (
    ("4HFS", "A", 200), ("4ZXG", "B", 13), ("6ZWK", "B", 47),
    ("7ZTL", "A", 257), ("8AJK", "A", 240), ("8R7O", "C", 1681),
)


def describe_slab(base, indices: np.ndarray, direction: np.ndarray) -> dict[str, object]:
    # This deliberately duplicates ``blocked_splits`` rather than using the
    # renderer's Cartesian grid.  The latter wraps in unit-cell coordinates and
    # can make a single contiguous low-projection slab appear at two opposite
    # Cartesian edges.
    coordinates = (np.argwhere(base.mask)
                   * np.asarray(base.qfit.xmap.voxelspacing, dtype=float))
    all_projection = coordinates @ direction
    selected_projection = all_projection[indices]
    cart = coordinates[indices]
    ca = np.asarray([
        base.initial_window[local_index(base.window, residue, "CA")]
        for residue in base.window.residues
    ], dtype=float)
    peptide_axis = ca[-1] - ca[0]
    peptide_axis /= np.linalg.norm(peptide_axis)
    peptide_projection = coordinates @ peptide_axis
    selected_peptide_projection = peptide_projection[indices]
    return {
        "map_grid_centroid_A": cart.mean(axis=0).tolist(),
        "projection_A": {"min": float(selected_projection.min()), "max": float(selected_projection.max()),
                           "mean": float(selected_projection.mean())},
        "projection_percentiles_of_full_mask": np.percentile(
            all_projection, [0, 20, 50, 80, 100]
        ).tolist(),
        "peptide_window_axis_A": peptide_axis.tolist(),
        "peptide_axis_projection_A": {
            "min": float(selected_peptide_projection.min()),
            "max": float(selected_peptide_projection.max()),
            "mean": float(selected_peptide_projection.mean()),
            "mean_percentile_of_full_mask": float(
                100.0 * np.mean(peptide_projection <= selected_peptide_projection.mean())
            ),
        },
    }


def audit_fold(root: Path, site: tuple[str, str, int], fold: int, runner,
               treatment: str) -> dict[str, object]:
    name = site_name(site)
    fold_root = root / "sites" / name / f"fold_{fold}"
    result = json.loads((root / "sites" / name / f"fold_{fold}.json").read_text())
    treatment_record = result[treatment]
    saved = np.load(
        fold_root / f"candidates_{treatment}" / "D_null_axis2_30deg" / "final_slots.npz"
    )
    train, test, direction = blocked_splits(runner.base)[fold]
    observed = np.asarray(saved["fit_voxel_indices"], dtype=int)
    if not np.array_equal(np.sort(observed), np.sort(train)):
        raise RuntimeError(f"{name} fold {fold}: endpoint provenance does not match split")
    slots = np.stack((saved["slot1_window"], saved["slot2_window"]))
    a_models = full_models(runner.base, slots, float(saved["b_offset_A2"]))
    deposited_windows = np.stack((runner.base.initial_window, runner.base.window_for_deposited_b()))
    deposited_models = full_models(runner.base, deposited_windows,
                                   float(treatment_record["deposited_two_state"]["dB_A2"]))
    # The discriminating subset is pre-optimization deposited A/B contrast at
    # dB=0, using exactly the screen's 5%-of-mean-density definition.
    contrast_models = full_models(runner.base, deposited_windows, 0.0)
    discriminating = np.abs(contrast_models[0] - contrast_models[1]) > (0.05 * contrast_models.mean())
    weights_a = np.asarray(treatment_record["A_prime"]["occupancies"], dtype=float)
    weights_d = np.asarray(treatment_record["deposited_two_state"]["occupancies"], dtype=float)
    return {
        "site": name, "fold": fold, "treatment": treatment,
        "heldout_voxels": int(len(test)), "training_voxels": int(len(train)),
        "mean_target": float(runner.base.target[test].mean()),
        "mean_calculated_density": {
            "A_prime_density_only": float((weights_a @ a_models[:, test]).mean()),
            "deposited_density_only": float((weights_d @ deposited_models[:, test]).mean()),
            "A_prime_unweighted_slot_mean": float(a_models[:, test].mean()),
            "deposited_unweighted_slot_mean": float(deposited_models[:, test].mean()),
        },
        "A_prime_occupancies": weights_a.tolist(),
        "deposited_occupancies": weights_d.tolist(),
        "heldout_slab": describe_slab(runner.base, test, direction),
        "training_discriminating_voxels": int(discriminating[train].sum()),
        "training_discriminating_voxel_fraction": float(discriminating[train].mean()),
        "full_mask_discriminating_voxel_fraction": float(discriminating.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in SITES:
        runner = APrimeSequential(args.out / site_name(site) / "runner", 8, 6, *site,
                                   renderer_backend="torch", residual_scale_mode="none",
                                   map_scaler_structure="full", mask_scope="window", device="cuda")
        for fold in range(5):
            fold_path = args.cv_root / "sites" / site_name(site) / f"fold_{fold}.json"
            if not fold_path.is_file():
                continue
            for treatment in ("dB_fitted", "dB_zero"):
                rows.append(audit_fold(args.cv_root, site, fold, runner, treatment))
    atomic_json(args.out / "fold_spatial_support.json", {"status": "complete", "rows": rows})
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
