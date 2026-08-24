#!/usr/bin/env python3
"""Score pre-rendered 5OHJ model arrays with the optimizer environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from occupancy_selection import solve_affine_qp


ROOT = Path(os.environ.get(
    "D1_SPLIT_ROOT",
    "/home/dev/qfit_unet_data/qfit_audit/d1_5ohj_aprime_phenix_closeout_v4_split",
))
MODE = os.environ.get("D1_SPLIT_MODE", "raw")


def score(target: np.ndarray, models: np.ndarray, folds: list[dict], pair: tuple[int, int]) -> dict:
    rows = []
    for fold_spec in folds:
        train = np.asarray(fold_spec["train"], dtype=int)
        test = np.asarray(fold_spec["test"], dtype=int)
        weights, intercept, train_rss = solve_affine_qp(target[train], models[list(pair)][:, train])
        residual = target[test] - weights @ models[list(pair)][:, test] - intercept
        rows.append({
            "fold": int(fold_spec["fold"]),
            "heldout_rss": float(np.square(residual).sum()),
            "train_rss": float(train_rss),
            "weights": np.asarray(weights).tolist(),
            "intercept": float(intercept),
            "heldout_voxels": int(len(test)),
        })
    values = np.asarray([row["heldout_rss"] for row in rows], dtype=float)
    return {"folds": rows, "mean": float(values.mean()), "median": float(np.median(values))}


def main() -> None:
    inputs = np.load(ROOT / f"render_inputs_{MODE}.npz")
    rendered = np.load(ROOT / f"rendered_models_{MODE}.npz")
    manifest = json.loads((ROOT / f"manifest_{MODE}.json").read_text())
    target = np.asarray(inputs["target"], dtype=float)
    models = np.asarray(rendered["models"], dtype=float)
    folds = manifest["folds"]
    report = {
        "status": "complete",
        "mode": MODE,
        "models": {
            "qfit_raw" if MODE == "raw" else "qfit_phenix": score(target, models, folds, (0, 1)),
            "aprime" if MODE == "raw" else "aprime_phenix": score(target, models, folds, (2, 3)),
        },
        "rendered_models_shape": list(models.shape),
        "n_voxels": int(target.size),
    }
    (ROOT / f"score_{MODE}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
