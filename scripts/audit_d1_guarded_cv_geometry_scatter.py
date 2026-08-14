#!/usr/bin/env python3
"""Read-only central-backbone geometry scatter audit for guarded CV endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
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
from run_d1_reachability import rmsd
from fit_provenance import assert_heldout_geometry_provenance
from run_d1_supported_panel import site_name


SITES = (
    ("4HFS", "A", 200),
    ("4ZXG", "B", 13),
    ("6ZWK", "B", 47),
    ("7ZTL", "A", 257),
    ("8AJK", "A", 240),
    ("8R7O", "C", 1681),
)


def distance_record(coordinates: np.ndarray, a: np.ndarray, b: np.ndarray, separation: float) -> dict[str, float]:
    to_a, to_b = float(rmsd(coordinates, a)), float(rmsd(coordinates, b))
    return {"to_A_A": to_a, "to_B_A": to_b,
            "to_A_fraction_of_A_B": to_a / separation,
            "to_B_fraction_of_A_B": to_b / separation}


def _treatment_scatter(cv_root: Path, name: str, runner: APrimeSequential,
                       separation: float, treatment: str) -> dict[str, object]:
    """Read saved train-only endpoints for one genuine ΔB treatment."""
    fold_slots = [[], []]
    folds = []
    for fold, (train, _, _) in enumerate(blocked_splits(runner.base)):
        root = cv_root / "sites" / name / f"fold_{fold}" / f"candidates_{treatment}"
        candidate = json.loads((root / "D_null_axis2_30deg" / "result.json").read_text())
        saved = np.load(root / "D_null_axis2_30deg" / "final_slots.npz")
        assert_heldout_geometry_provenance(candidate, saved, train, len(runner.base.target))
        pair = [runner.base.central_backbone(saved[f"slot{slot}_window"]) for slot in (1, 2)]
        for index, coordinates in enumerate(pair):
            fold_slots[index].append(coordinates)
        folds.append({"fold": fold, "slot1": distance_record(pair[0], runner.a_backbone, runner.b_backbone, separation),
                      "slot2": distance_record(pair[1], runner.a_backbone, runner.b_backbone, separation)})
    slot_summary = []
    for positions in fold_slots:
        values = np.asarray(positions)
        mean = values.mean(axis=0)
        deviations = np.asarray([rmsd(value, mean) for value in values])
        pairs = np.asarray([rmsd(left, right) for left, right in combinations(values, 2)])
        slot_summary.append({
            "fold_position_std_RMS_A": float(np.sqrt(np.mean(deviations ** 2))),
            "fold_position_std_fraction_of_A_B": float(np.sqrt(np.mean(deviations ** 2)) / separation),
            "mean_pairwise_RMSD_A": float(pairs.mean()), "max_pairwise_RMSD_A": float(pairs.max()),
            "mean_position": distance_record(mean, runner.a_backbone, runner.b_backbone, separation),
        })
    return {"folds": folds, "slot_summary": slot_summary}


def audit_site(cv_root: Path, site: tuple[str, str, int], out: Path) -> dict[str, object]:
    name = site_name(site)
    runner = APrimeSequential(out / name / "runner", 8, 6, *site,
                               renderer_backend="torch", residual_scale_mode="none",
                               map_scaler_structure="full", mask_scope="window", device="cuda")
    separation = float(rmsd(runner.a_backbone, runner.b_backbone))
    return {
        "site": name, "deposited_A_B_RMSD_A": separation,
        "dB_fitted": _treatment_scatter(cv_root, name, runner, separation, "dB_fitted"),
        "dB_zero": _treatment_scatter(cv_root, name, runner, separation, "dB_zero"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [audit_site(args.cv_root, site, args.out) for site in SITES]
    atomic_json(args.out / "geometry_scatter.json", {"status": "complete", "rows": rows})
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
