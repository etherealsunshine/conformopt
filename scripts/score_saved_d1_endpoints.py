#!/usr/bin/env python3
"""Score fixed deposited and saved A' endpoint geometries on clean-D1 folds."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import qfit  # noqa: F401
import numpy as np

from clean_d1_benchmark import site_key
from occupancy_selection import solve_affine_qp
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential


def score_one(task: dict[str, object]) -> dict[str, object]:
    site = tuple(task["site"])
    key = str(task["site_key"])
    base = APrimeSequential(
        Path(task["work_root"]), 80, 6, *site,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=task["device"],
        start_pdb=task["start_pdb"], b_factor_mode="single_conformer",
    )
    folds = blocked_splits(base.base)
    model = str(task["model"])
    if model == "deposited_pair":
        coordinates = np.stack((
            base.base.window_for_deposited_a(),
            base.base.window_for_deposited_b(),
        ))
    else:
        saved = np.load(task["endpoint"])
        coordinates = np.stack((saved["slot1_window"], saved["slot2_window"]))

    rows = []
    models = base.base.model_density_batch(
        coordinates, slots=np.array((0, 1)), b_offset=0.0,
    )
    for fold, (train, test, direction) in enumerate(folds):
        weights, intercept, train_rss = solve_affine_qp(
            base.base.target[train], models[:, train]
        )
        heldout = base.base.target[test] - weights @ models[:, test] - intercept
        rows.append({
            "fold": fold,
            "split_direction": direction.tolist(),
            "training_rss": float(train_rss),
            "heldout_rss": float(np.square(heldout).sum()),
            "occupancies": weights.tolist(),
            "intercept": float(intercept),
        })
    values = np.asarray([row["heldout_rss"] for row in rows])
    return {
        "site": key, "model": model, "dB_A2": 0.0, "folds": rows,
        "mean_all_five_heldout_rss": float(values.mean()),
        "mean_excluding_fold_0_heldout_rss": float(values[1:].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()

    manifest = {site_key(row): row for row in json.loads(args.manifest.read_text())}
    key = "6ZWK_B_PHE47"
    row = manifest[key]
    site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
    start = args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"
    endpoint_root = args.endpoints / key / "dB_zero"
    tasks = [
        {
            "site_key": key, "site": list(site), "start_pdb": str(start),
            "device": args.device, "model": "deposited_pair",
            "endpoint": "", "work_root": str(args.output / "deposited_pair"),
        },
        {
            "site_key": key, "site": list(site), "start_pdb": str(start),
            "device": args.device, "model": "aprime_fold_2",
            "endpoint": str(endpoint_root / "fold_2/D_null_axis2_30deg/final_slots.npz"),
            "work_root": str(args.output / "aprime_fold_2"),
        },
        {
            "site_key": key, "site": list(site), "start_pdb": str(start),
            "device": args.device, "model": "aprime_fold_4",
            "endpoint": str(endpoint_root / "fold_4/D_null_axis2_30deg/final_slots.npz"),
            "work_root": str(args.output / "aprime_fold_4"),
        },
    ]
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(score_one, tasks))
    report = {"status": "complete", "device": args.device, "models": results}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
