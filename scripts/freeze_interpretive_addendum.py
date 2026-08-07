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

from density_denoiser.summarize_endpoint_audit import as_bool, select_assigned_pair
from density_denoiser.summarize_tmol_margin_sweep import (
    _geometry_flags,
    _pair_pass,
    _tmol_pass,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformer-table", type=Path, action="append", required=True)
    parser.add_argument("--ensemble-table", type=Path, action="append", required=True)
    parser.add_argument("--margin-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    conformers = [row for path in args.conformer_table for row in read_csv(path)]
    ensembles = [row for path in args.ensemble_table for row in read_csv(path)]
    by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in conformers:
        by_start[(row["site"], int(row["start"]))].append(row)
    ensemble_by_start = {
        (row["site"], int(row["start"])): row for row in ensembles
    }
    if len(ensemble_by_start) != 1000:
        raise ValueError(f"expected 1000 starts, got {len(ensemble_by_start)}")

    tolerance_rows = []
    for tolerance in (0.36, 0.44):
        assigned = all_active = 0
        for key, ensemble in ensemble_by_start.items():
            active = by_start.get(key, [])
            occupancy = as_bool(ensemble["geometric_occupancy_success"])
            assigned += occupancy and _pair_pass(active, tolerance)
            all_active += (
                occupancy
                and _geometry_flags(active)[2]
                and _tmol_pass(active, tolerance)
            )
        tolerance_rows.append({
            "tmol_tolerance": tolerance,
            "assigned_pair": assigned,
            "all_active": all_active,
            "starts": len(ensemble_by_start),
        })

    margins = []
    for row in read_csv(args.margin_table):
        if row["assignment"] not in {"A", "B"}:
            continue
        try:
            rmsd = float(row["rmsd_to_matched_deposited"])
            margin = float(row["tmol_margin_candidate_minus_matched_deposited"])
        except ValueError:
            continue
        if math.isfinite(rmsd) and math.isfinite(margin):
            margins.append((row, rmsd, margin))
    q99_rows = []
    for cutoff in (0.05, 0.1, 0.15, 0.2, 0.3):
        near = [(row, margin) for row, rmsd, margin in margins if rmsd <= cutoff]
        positive = np.asarray([margin for _, margin in near if margin > 0.0])
        q99_rows.append({
            "rmsd_cutoff": cutoff,
            "matched_finite_n": len(near),
            "positive_margin_n": int(len(positive)),
            "positive_margin_q99": (
                float(np.quantile(positive, 0.99)) if len(positive) else ""
            ),
            "positive_margin_max": (
                float(positive.max()) if len(positive) else ""
            ),
        })
    near_exceptions = [
        {
            "candidate_id": row["candidate_id"],
            "site": row["site"],
            "assignment": row["assignment"],
            "rmsd_to_matched_deposited": rmsd,
            "tmol_margin": margin,
        }
        for row, rmsd, margin in margins
        if rmsd <= 0.1 and margin > 0.44
    ]

    extra_rows = []
    starts_with_extras = set()
    occupancy_qualified_starts_with_extras = set()
    headline_pass_starts_with_extras = set()
    per_site: dict[str, list[dict[str, object]]] = defaultdict(list)
    for key in sorted(ensemble_by_start):
        active = by_start.get(key, [])
        pair = select_assigned_pair(active)
        occupancy_qualified = as_bool(
            ensemble_by_start[key]["geometric_occupancy_success"]
        )
        headline_pass = occupancy_qualified and _pair_pass(active, 0.44)
        selected = (
            {row["candidate_id"] for row in pair.values()} if pair else set()
        )
        for row in active:
            if row["candidate_id"] in selected:
                continue
            starts_with_extras.add(key)
            if occupancy_qualified:
                occupancy_qualified_starts_with_extras.add(key)
            if headline_pass:
                headline_pass_starts_with_extras.add(key)
            entry = {
                "site": row["site"],
                "start": int(row["start"]),
                "candidate_id": row["candidate_id"],
                "occupancy": float(row["occupancy"]),
                "nearest_deposited_rmsd": min(
                    float(row["rmsd_to_A_conventional"]),
                    float(row["rmsd_to_B_conventional"]),
                ),
                "geometry_pass": all((
                    as_bool(row["rotamer_within_allowed_width"]),
                    as_bool(row["no_direct_clash"]),
                    as_bool(row["no_symmetry_clash"]),
                )),
                "assignment": row["assignment"],
                "occupancy_qualified_start": occupancy_qualified,
                "headline_pass_start": headline_pass,
            }
            extra_rows.append(entry)
            per_site[row["site"]].append(entry)

    def extra_summary(rows: list[dict[str, object]]) -> dict[str, object]:
        return {
            "extra_conformers": len(rows),
            "occupancy": describe([float(row["occupancy"]) for row in rows]),
            "nearest_deposited_rmsd": describe([
                float(row["nearest_deposited_rmsd"]) for row in rows
            ]),
            "geometry_pass": sum(bool(row["geometry_pass"]) for row in rows),
            "geometry_pass_rate": (
                sum(bool(row["geometry_pass"]) for row in rows) / len(rows)
                if rows else None
            ),
        }

    extra = {
        "starts": len(ensemble_by_start),
        "starts_with_extra_active_conformers": len(starts_with_extras),
        "start_rate": len(starts_with_extras) / len(ensemble_by_start),
        **extra_summary(extra_rows),
        "occupancy_qualified_population": {
            "starts": sum(
                as_bool(row["geometric_occupancy_success"])
                for row in ensemble_by_start.values()
            ),
            "starts_with_extra_active_conformers": len(
                occupancy_qualified_starts_with_extras
            ),
            **extra_summary([
                row for row in extra_rows if row["occupancy_qualified_start"]
            ]),
        },
        "frozen_headline_pass_population": {
            "starts": sum(
                as_bool(ensemble_by_start[key]["geometric_occupancy_success"])
                and _pair_pass(by_start.get(key, []), 0.44)
                for key in ensemble_by_start
            ),
            "starts_with_extra_active_conformers": len(
                headline_pass_starts_with_extras
            ),
            **extra_summary([
                row for row in extra_rows if row["headline_pass_start"]
            ]),
        },
        "per_site": {
            site: extra_summary(rows) for site, rows in sorted(per_site.items())
        },
        "causal_extra_occupancy_medians": {
            "2V05_A_HIS168": 0.1191,
            "8Q6Q_B_ASP81": 0.1525,
        },
    }

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "tolerance_robustness.csv", tolerance_rows)
    atomic_csv(args.output / "near_reproduction_q99.csv", q99_rows)
    atomic_csv(args.output / "near_reproduction_exceptions.csv", near_exceptions)
    atomic_csv(args.output / "extra_active_conformers_all_sites.csv", extra_rows)
    summary = {
        "tolerance_robustness": tolerance_rows,
        "near_reproduction_q99": q99_rows,
        "near_reproduction_exceptions": near_exceptions,
        "extra_active_conformers": extra,
    }
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
