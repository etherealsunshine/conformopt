#!/usr/bin/env python3
"""Score fixed native-qFit MIQP selections on the frozen A-prime folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qfit  # noqa: F401  # load qFit/CCTBX before Torch/CUDA initialization
import numpy as np

from occupancy_selection import solve_affine_qp
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential


SITES = {
    "6ZWK_B_PHE47": ("6ZWK", "B", 47),
    "8R7O_C_THR1681": ("8R7O", "C", 1681),
}


def replace_central(base, window, candidate, candidate_names):
    result = np.asarray(window, dtype=float).copy()
    indices = {str(name): int(index) for name, index in zip(
        base.central.name.tolist(), base.central_indices
    )}
    by_name = {str(name): np.asarray(xyz, dtype=float)
               for name, xyz in zip(candidate_names, candidate)}
    for name, index in indices.items():
        if name in by_name:
            result[index] = by_name[name]
    return result


def score_site(site_key, selected_root, starts_root, temp_root):
    pdb_id, chain, resnum = SITES[site_key]
    start = starts_root / "sites" / site_key / "neutral_start_aprime_single_slot.pdb"
    data = np.load(selected_root / site_key / "selected.npz")
    selected = np.asarray(data["selected_indices"], dtype=int)
    candidates = np.asarray(data["candidate_coordinates"], dtype=float)
    names = [str(x) for x in data["candidate_atom_names"].tolist()]

    runner = APrimeSequential(
        temp_root / site_key, 40, 50, pdb_id, chain, resnum,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device="cuda",
        start_pdb=start, b_factor_mode="single_conformer",
    )
    base = runner.base
    windows = np.asarray([
        replace_central(base, base.initial_window, candidates[i], names)
        for i in selected
    ])
    slots = np.arange(len(windows), dtype=int) % 2
    folds = blocked_splits(base)
    rows = []
    for fold, (train, test, direction) in enumerate(folds):
        treatments = {}
        for treatment in ("dB_fitted", "dB_zero"):
            if treatment == "dB_fitted":
                fit = base.profile_affine_b_offset(
                    base.target[train], list(windows), slots.tolist(),
                    voxel_indices=train,
                )
                delta_b = float(fit["b_offset"])
            else:
                delta_b = 0.0
            models = base.model_density_batch(windows, slots=slots, b_offset=delta_b)
            weights, intercept, train_rss = solve_affine_qp(
                base.target[train], models[:, train]
            )
            heldout = base.target[test] - weights @ models[:, test] - intercept
            treatments[treatment] = {
                "delta_B_A2": delta_b,
                "occupancies": np.asarray(weights, dtype=float).tolist(),
                "intercept": float(intercept),
                "training_rss": float(train_rss),
                "heldout_rss": float(np.square(heldout).sum()),
            }
        rows.append({
            "fold": fold, "train_voxels": int(len(train)),
            "heldout_voxels": int(len(test)),
            "split_direction": direction.tolist(), "treatments": treatments,
        })

    out = {"status": "complete", "site": site_key,
           "selected_indices": selected.tolist(), "folds": rows}
    (selected_root / site_key / "scored_folds.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [score_site(key, args.selected_root, args.starts, args.output / "tmp")
            for key in SITES]
    result = {"status": "complete", "rows": rows}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
