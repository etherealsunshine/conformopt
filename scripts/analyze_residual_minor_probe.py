"""Summarize the diagnostic single-slot residual minor-conformer probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def median(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(np.median(values)) if values else math.nan


def summarize(rows: list[dict], separation: float | None = None) -> dict:
    recovered = [
        row for row in rows if row["recovered_minor_lt_1A"].lower() == "true"
    ]
    failed = [
        row for row in rows if row["recovered_minor_lt_1A"].lower() != "true"
    ]
    return {
        "eligible": len(rows),
        "recovered": len(recovered),
        "recovery_rate": len(recovered) / len(rows) if rows else math.nan,
        "local_unsym_AB_separation_A": (
            separation if separation is not None else math.nan
        ),
        "initial_rmsd_to_minor_median": median(
            rows, "initial_rmsd_to_minor"
        ),
        "final_rmsd_to_minor_median": median(
            rows, "final_rmsd_to_minor"
        ),
        "deposited_minor_occupancy_median": median(
            rows, "deposited_minor_occupancy"
        ),
        "final_occupancy_median": median(rows, "final_occupancy"),
        "absolute_occupancy_error_median": float(np.median([
            abs(
                float(row["final_occupancy"])
                - float(row["deposited_minor_occupancy"])
            )
            for row in rows
        ])) if rows else math.nan,
        "minor_residual_rms_before_median": median(
            rows, "minor_before_rms"
        ),
        "minor_positive_integral_before_median": median(
            rows, "minor_before_integrated_positive"
        ),
        "major_positive_integral_before_median": median(
            rows, "major_before_integrated_positive"
        ),
        "minor_positive_integral_before_recovered_median": median(
            recovered, "minor_before_integrated_positive"
        ),
        "minor_positive_integral_before_failed_median": median(
            failed, "minor_before_integrated_positive"
        ),
        "minor_residual_rms_after_median": median(
            rows, "minor_after_rms"
        ),
        "lobe_overlap_fraction_median": median(
            rows, "minor_before_overlap_fraction"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--separations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--separation-cutoff", type=float, default=2.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with args.separations.open(newline="") as handle:
        separation_rows = list(csv.DictReader(handle))
    separations = {
        row["site"]: float(row["local_unsym_rmsd_A"])
        for row in separation_rows
    }

    by_site_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_stratum_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        site = row["site"]
        mode = row["occupancy_mode"]
        separation = separations[site]
        stratum = (
            f"gt_{args.separation_cutoff:g}A"
            if separation > args.separation_cutoff
            else f"le_{args.separation_cutoff:g}A"
        )
        by_site_mode[(site, mode)].append(row)
        by_stratum_mode[(stratum, mode)].append(row)
        by_mode[mode].append(row)

    per_site = [
        {
            "site": site,
            "occupancy_mode": mode,
            **summarize(population, separations[site]),
        }
        for (site, mode), population in sorted(by_site_mode.items())
    ]
    per_stratum = [
        {
            "separation_stratum": stratum,
            "occupancy_mode": mode,
            **summarize(population),
        }
        for (stratum, mode), population in sorted(by_stratum_mode.items())
    ]
    overall = {
        mode: summarize(population)
        for mode, population in sorted(by_mode.items())
    }

    atomic_csv(args.output / "summary_by_site.csv", per_site)
    atomic_csv(args.output / "summary_by_separation.csv", per_stratum)
    atomic_json(args.output / "summary.json", {
        "diagnostic_only": True,
        "metric_changed": False,
        "separation_definition": "local fixed-label unsymmetrized A/B RMSD",
        "separation_cutoff_A": args.separation_cutoff,
        "overall": overall,
    })


if __name__ == "__main__":
    main()
