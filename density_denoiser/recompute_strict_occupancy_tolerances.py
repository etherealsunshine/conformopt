"""Recompute strict success at alternative occupancy tolerances.

This consumes completed ensemble strict-audit tables and changes only the
occupancy tolerance. Recovery, RMSD, rotamer, clash, and tmol decisions are
reused exactly as recorded in the source audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def _boolean(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"unexpected boolean value: {value!r}")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["source_table"] = str(path)
                rows.append(row)
    return rows


def _evaluate(row: dict, tolerance: float) -> dict[str, bool]:
    both_found = _boolean(row["both_found_conventional"])
    physical = _boolean(row["all_active_strict_physical_valid"])
    occupancy = (
        abs(float(row["predicted_A_occupancy"]) - float(row["target_A_occupancy"]))
        <= tolerance
        and abs(float(row["predicted_B_occupancy"]) - float(row["target_B_occupancy"]))
        <= tolerance
    )
    return {
        "both_found": both_found,
        "occupancy_accurate": occupancy,
        "recovery_and_occupancy": both_found and occupancy,
        "all_active_strict_physical": physical,
        "strict_joint_success": both_found and occupancy and physical,
    }


def recompute(args: argparse.Namespace) -> dict:
    rows = _read_rows(args.ensemble_tables)
    if not rows:
        raise ValueError("no ensemble rows found")

    tolerance_summaries = []
    per_site_rows = []
    for tolerance in args.tolerances:
        site_metrics: dict[str, list[dict[str, bool]]] = defaultdict(list)
        evaluations = []
        for row in rows:
            result = _evaluate(row, tolerance)
            site_metrics[row["site"]].append(result)
            evaluations.append(result)

        for site in sorted(site_metrics):
            values = site_metrics[site]
            per_site_rows.append({
                "occupancy_tolerance": tolerance,
                "site": site,
                "ensembles": len(values),
                "both_found_conventional": sum(value["both_found"] for value in values),
                "recovery_and_occupancy": sum(
                    value["recovery_and_occupancy"] for value in values
                ),
                "all_active_strict_physical": sum(
                    value["all_active_strict_physical"] for value in values
                ),
                "strict_joint_success": sum(
                    value["strict_joint_success"] for value in values
                ),
            })
        tolerance_summaries.append({
            "occupancy_tolerance": tolerance,
            "ensembles": len(evaluations),
            "both_found_conventional": sum(value["both_found"] for value in evaluations),
            "recovery_and_occupancy": sum(
                value["recovery_and_occupancy"] for value in evaluations
            ),
            "all_active_strict_physical": sum(
                value["all_active_strict_physical"] for value in evaluations
            ),
            "strict_joint_success": sum(
                value["strict_joint_success"] for value in evaluations
            ),
        })
        _atomic_json(args.output / "progress.json", {
            "status": "running",
            "completed_tolerances": len(tolerance_summaries),
            "total_tolerances": len(args.tolerances),
            "latest_tolerance": tolerance,
        })

    baseline = next(
        (summary for summary in tolerance_summaries if summary["occupancy_tolerance"] == 0.2),
        None,
    )
    if baseline is not None and args.expected_baseline is not None:
        actual = baseline["strict_joint_success"]
        if actual != args.expected_baseline:
            raise RuntimeError(
                f"±0.20 validation failed: expected {args.expected_baseline}, got {actual}"
            )

    _atomic_csv(args.output / "per_site.csv", per_site_rows)
    _atomic_csv(args.output / "combined.csv", tolerance_summaries)
    payload = {
        "status": "complete",
        "source_tables": [str(path) for path in args.ensemble_tables],
        "criteria_held_fixed": [
            "both deposited A and B found at conventional RMSD < 1.0 A",
            "all active conformers canonical within 30 degrees",
            "no direct or crystallographic-symmetry contact below 2.0 A",
            "tmol energy within 10 units of the better deposited A/B control",
        ],
        "occupancy_rule": (
            "absolute predicted-versus-target error must be <= tolerance "
            "for both A and B"
        ),
        "summaries": tolerance_summaries,
        "baseline_validation_passed": (
            baseline is None
            or args.expected_baseline is None
            or baseline["strict_joint_success"] == args.expected_baseline
        ),
    }
    _atomic_json(args.output / "summary.json", payload)
    _atomic_json(args.output / "progress.json", {
        "status": "complete",
        "completed_tolerances": len(tolerance_summaries),
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ensemble-table", dest="ensemble_tables", action="append",
                        type=Path, required=True)
    result.add_argument("--tolerance", dest="tolerances", action="append",
                        type=float, required=True)
    result.add_argument("--expected-baseline", type=int)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    recompute(parser().parse_args())
