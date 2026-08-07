#!/usr/bin/env python3
"""Test whether the calibrated 7UTC Slot-1 intermediate is a local 8-D basin."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import null_space

from qfit.backbone import compute_jacobian
from run_d1_8d_sequential_poc import SequentialBackbonePOC
from run_d1_reachability import rmsd


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def local_null_basis(experiment: SequentialBackbonePOC, window: np.ndarray) -> np.ndarray:
    original = experiment.window.coor.copy()
    try:
        experiment.window.coor = window.copy()
        selection = np.sort(experiment.window.select("name", ("N", "CA", "C")))
        return null_space(compute_jacobian(experiment.window.get_xyz(selection)))
    finally:
        experiment.window.coor = original


def evaluate(experiment: SequentialBackbonePOC, window: np.ndarray) -> dict[str, float]:
    value = experiment.evaluate(window, experiment.target, 1.0, normalizer=1.0)
    backbone = experiment.central_backbone(window)
    return {
        "density_rss": float(value["density_rss"]), "objective": float(value["objective"]),
        "occupancy": float(value["occupancy"]), "rama_penalty": float(value["rama_penalty"]),
        "rmsd_to_A_A": rmsd(backbone, experiment.a_backbone),
        "rmsd_to_B_A": rmsd(backbone, experiment.b_backbone),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps-deg", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    slots = np.load(args.source_run / "final_slots.npz")
    slot1 = np.asarray(slots["slot1_window"], dtype=float)
    experiment = SequentialBackbonePOC("7UTC", "A", 52, args.output, 0.25, 2.0, 0.05, residual_scale_mode="deposited_ab")
    baseline = evaluate(experiment, slot1)
    basis = local_null_basis(experiment, slot1)
    rows = []
    for axis in range(basis.shape[1]):
        for step in args.steps_deg:
            for sign in (-1.0, 1.0):
                candidate = experiment.apply_increment(slot1, sign * step * basis[:, axis])
                value = evaluate(experiment, candidate)
                rows.append({"axis": axis, "step_deg": sign * step, **value,
                             "delta_density_rss": value["density_rss"] - baseline["density_rss"],
                             "delta_objective": value["objective"] - baseline["objective"]})
                atomic_csv(args.output / "axis_profile.csv", rows)
                atomic_json(args.output / "progress.json", {"status": "running", "samples_complete": len(rows), "samples_total": basis.shape[1] * len(args.steps_deg) * 2})
    near = [row for row in rows if abs(row["step_deg"]) == min(args.steps_deg)]
    result = {
        "status": "complete", "site": "7UTC_A_ARG52", "source_run": str(args.source_run),
        "parameterization": "all orthonormal local axes of null(compute_jacobian) at the converged Slot-1 intermediate",
        "null_dimension": int(basis.shape[1]), "baseline": baseline,
        "smallest_step_deg": min(args.steps_deg),
        "all_smallest_axis_perturbations_increase_density_rss": bool(all(row["delta_density_rss"] > 0 for row in near)),
        "all_smallest_axis_perturbations_increase_full_objective": bool(all(row["delta_objective"] > 0 for row in near)),
        "minimum_small_step_delta_density_rss": float(min(row["delta_density_rss"] for row in near)),
        "minimum_small_step_delta_objective": float(min(row["delta_objective"] for row in near)),
    }
    atomic_json(args.output / "result.json", result)
    atomic_json(args.output / "progress.json", {"status": "complete", "samples_complete": len(rows), "samples_total": len(rows)})


if __name__ == "__main__":
    main()
