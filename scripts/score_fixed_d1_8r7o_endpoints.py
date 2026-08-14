#!/usr/bin/env python3
"""Score the deposited pair and saved 8R7O A-prime endpoints on fixed folds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import qfit  # noqa: F401

from occupancy_selection import solve_affine_qp
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential


def main() -> None:
    output = Path("/home/dev/qfit_audit/clean_d1_fixed_endpoint_comparison_8r7o_v1")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    site = ("8R7O", "C", 1681)
    start = Path(
        "/home/dev/qfit_unet_data/qfit_audit/clean_d1_neutral_starts_v4/sites/"
        "8R7O_C_THR1681/neutral_start_aprime_single_slot.pdb"
    )
    base = APrimeSequential(
        output / "base", 80, 6, *site, renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", device="cuda", start_pdb=start,
        b_factor_mode="single_conformer",
    )
    folds = blocked_splits(base.base)
    models = {
        "deposited_pair": np.stack((
            base.base.window_for_deposited_a(),
            base.base.window_for_deposited_b(),
        ))
    }
    endpoint_root = Path(
        "/home/dev/qfit_unet_data/qfit_audit/"
        "clean_d1_guarded_cv_amplitude_prior_zero_converged_v2/"
        "8R7O_C_THR1681/dB_zero"
    )
    for endpoint_fold in range(5):
        saved = np.load(
            endpoint_root / f"fold_{endpoint_fold}/D_null_axis2_30deg/final_slots.npz"
        )
        models[f"aprime_endpoint_fold_{endpoint_fold}"] = np.stack((
            saved["slot1_window"], saved["slot2_window"],
        ))

    rows = []
    for model_name, coordinates in models.items():
        rendered = base.base.model_density_batch(
            coordinates, slots=np.array((0, 1)), b_offset=0.0,
        )
        endpoint_fold = (
            None if model_name == "deposited_pair"
            else int(model_name.rsplit("_", 1)[-1])
        )
        for fold, (train, test, direction) in enumerate(folds):
            weights, intercept, training_rss = solve_affine_qp(
                base.base.target[train], rendered[:, train]
            )
            heldout = base.base.target[test] - weights @ rendered[:, test] - intercept
            rows.append({
                "model": model_name, "endpoint_fold": endpoint_fold,
                "fold": fold, "training_rss": float(training_rss),
                "heldout_rss": float(np.square(heldout).sum()),
                "occupancies": weights.tolist(), "intercept": float(intercept),
                "split_direction": direction.tolist(),
            })

    report = {
        "status": "complete", "site": "8R7O_C_THR1681", "dB_A2": 0.0,
        "score": "fixed geometry; occupancies and intercept fitted on training voxels only",
        "rows": rows,
    }
    for model_name in models:
        values = np.asarray([
            row["heldout_rss"] for row in rows if row["model"] == model_name
        ])
        report[model_name] = {
            "mean_all_five": float(values.mean()),
            "mean_excluding_fold0": float(values[1:].mean()),
            "folds": values.tolist(),
        }
    diagonal = []
    for endpoint_fold in range(5):
        aprime = next(row for row in rows if (
            row["model"] == f"aprime_endpoint_fold_{endpoint_fold}"
            and row["fold"] == endpoint_fold
        ))
        oracle = next(row for row in rows if (
            row["model"] == "deposited_pair" and row["fold"] == endpoint_fold
        ))
        diagonal.append({
            "fold": endpoint_fold,
            "deposited_pair_heldout_rss": oracle["heldout_rss"],
            "aprime_endpoint_heldout_rss": aprime["heldout_rss"],
            "aprime_minus_deposited": aprime["heldout_rss"] - oracle["heldout_rss"],
        })
    report["diagonal_comparison"] = diagonal
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "deposited_pair", "diagonal_comparison",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
