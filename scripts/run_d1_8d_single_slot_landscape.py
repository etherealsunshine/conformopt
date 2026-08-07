#!/usr/bin/env python3
"""Map the one-slot 7UTC density profile along the deposited A->B null tangent.

The direction is the least-squares torsion displacement that moves central
N/CA/C/O from deposited A towards B, projected through qFit's actual
``compute_jacobian`` closure null space at A. This is a local chart, not an
assumption that a peptide flip is globally linear.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def central_jacobian(experiment: SequentialBackbonePOC, step_deg: float) -> np.ndarray:
    columns = []
    for index in range(14):
        plus = np.zeros(14)
        plus[index] = step_deg
        coor_plus = experiment.central_backbone(experiment.apply_increment(experiment.initial_window, plus))
        coor_minus = experiment.central_backbone(experiment.apply_increment(experiment.initial_window, -plus))
        columns.append(((coor_plus - coor_minus) / (2.0 * step_deg)).ravel())
    return np.column_stack(columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--points", type=int, default=81)
    parser.add_argument("--fd-step-deg", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    experiment = SequentialBackbonePOC(
        "7UTC", "A", 52, args.output, args.fd_step_deg, 2.0, 0.05,
        residual_scale_mode="deposited_ab",
    )
    jacobian = central_jacobian(experiment, args.fd_step_deg)
    delta_xyz = (experiment.b_backbone - experiment.a_backbone).ravel()
    unconstrained, *_ = np.linalg.lstsq(jacobian, delta_xyz, rcond=None)
    direction, null_dimension, null_residual = experiment.closure_project(experiment.initial_window, unconstrained)
    predicted = jacobian @ direction
    if float(np.dot(predicted, delta_xyz)) < 0:
        direction *= -1.0
        predicted *= -1.0
    direction /= float(np.max(np.abs(direction)))  # coordinate is max torsion degrees
    predicted = jacobian @ direction
    alpha_to_b = float(np.dot(predicted, delta_xyz) / np.dot(predicted, predicted))
    maximum = max(30.0, min(90.0, 1.5 * abs(alpha_to_b)))
    amplitudes = np.linspace(-0.25 * maximum, maximum, args.points)
    rows = []
    for point, amplitude in enumerate(amplitudes, start=1):
        window = experiment.apply_increment(experiment.initial_window, amplitude * direction)
        evaluation = experiment.evaluate(window, experiment.target, 1.0, normalizer=1.0)
        backbone = experiment.central_backbone(window)
        rows.append({
            "point": point, "max_torsion_displacement_deg": float(amplitude),
            "density_rss": float(evaluation["density_rss"]), "occupancy": float(evaluation["occupancy"]),
            "rama_penalty": float(evaluation["rama_penalty"]), "objective": float(evaluation["objective"]),
            "backbone_rmsd_to_A_A": rmsd(backbone, experiment.a_backbone),
            "backbone_rmsd_to_B_A": rmsd(backbone, experiment.b_backbone),
        })
        atomic_csv(args.output / "profile.csv", rows)
        atomic_json(args.output / "progress.json", {"status": "running", "points_complete": point, "points_total": len(amplitudes)})
    minima = [row for index, row in enumerate(rows[1:-1], start=1)
              if row["density_rss"] < rows[index - 1]["density_rss"] and row["density_rss"] < rows[index + 1]["density_rss"]]
    fig, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(amplitudes, [row["density_rss"] for row in rows], color="#4c72b0")
    axis.set(xlabel="A→B closure-null tangent amplitude (max torsion, degrees)", ylabel="single-slot RSS",
             title="7UTC A:ARG52: calibrated residual-map profile")
    fig.tight_layout()
    fig.savefig(args.output / "profile.png", dpi=180)
    plt.close(fig)
    atomic_json(args.output / "result.json", {
        "status": "complete", "site": "7UTC_A_ARG52",
        "parameterization": "14 torsions projected into 8-D null(compute_jacobian) at deposited A",
        "direction": {"finite_difference_step_deg": args.fd_step_deg, "null_dimension": null_dimension,
                      "J_times_unscaled_direction_norm": null_residual,
                      "linear_best_A_to_B_amplitude_deg": alpha_to_b,
                      "scan_amplitude_range_deg": [float(amplitudes.min()), float(amplitudes.max())]},
        "residual_map_scale": experiment.residual_scale_diagnostic,
        "global_density_minimum": min(rows, key=lambda row: row["density_rss"]),
        "interior_density_local_minima": minima,
        "points": len(rows),
    })
    atomic_json(args.output / "progress.json", {"status": "complete", "points_complete": len(rows), "points_total": len(rows)})


if __name__ == "__main__":
    main()
