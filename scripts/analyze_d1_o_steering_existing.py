#!/usr/bin/env python3
"""Refit the completed D1 O-steering control experiment on O-only metrics."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def ols(rows: list[dict[str, str]], y_key: str) -> dict[str, float | int | None]:
    x = np.asarray([float(row["deposited_A_to_B_O_distance_A"]) for row in rows])
    y = np.asarray([float(row[y_key]) for row in rows])
    design = np.column_stack((np.ones(len(rows)), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    error = y - design @ beta
    dof = len(rows) - 2
    return {
        "n": len(rows), "intercept_A": float(beta[0]), "slope": float(beta[1]),
        "slope_standard_error": (float(np.sqrt(error.dot(error) / dof * np.linalg.inv(design.T @ design)[1, 1])) if dof > 0 else None),
        "Pearson_r": (float(np.corrcoef(x, y)[0, 1]) if len(rows) > 2 else None),
        "residual_standard_error_A": (float(np.sqrt(error.dot(error) / dof)) if dof > 0 else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with args.input.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "complete"]
    anisou = [row for row in rows if row["CB_has_anisou"] == "True" and row["O_has_anisou"] == "True"]
    for row in rows:
        row["O_minus_CB_O_RMSD_A"] = float(row["tier_a_min_central_O_rmsd_A"]) - float(row["baseline_CB_central_O_rmsd_A"])
        row["comparison_class"] = "direct_anisou_steering_atom" if row in anisou else "geometric_fallback_frame_swap"
    atomic_json(args.output / "summary.json", {
        "status": "complete", "all_ten_O_only_regressions": {
            "CB_baseline": ols(rows, "baseline_CB_central_O_rmsd_A"),
            "O_steering": ols(rows, "tier_a_min_central_O_rmsd_A"),
        },
        "direct_anisou_comparison": {
            "n": len(anisou), "sites": [row["site"] for row in anisou],
            "CB_baseline_O_only_fit": ols(anisou, "baseline_CB_central_O_rmsd_A"),
            "O_steering_O_only_fit": ols(anisou, "tier_a_min_central_O_rmsd_A"),
            "O_minus_CB_O_RMSD_A_by_site": {row["site"]: row["O_minus_CB_O_RMSD_A"] for row in anisou},
            "interpretation": "n=2: slope standard errors and a generalizable regression inference are undefined",
        },
        "comparison_class_counts": {
            "direct_anisou_steering_atom": len(anisou),
            "geometric_fallback_frame_swap": len(rows) - len(anisou),
        },
    })
    fields = sorted({key for row in rows for key in row})
    with (args.output / "per_site.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
