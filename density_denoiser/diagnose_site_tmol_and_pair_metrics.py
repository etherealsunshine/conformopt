from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .diagnose_frozen_tmol_gate import load_composite_rows, matched_rmsd
from .summarize_endpoint_audit import as_bool, select_assigned_pair


TMOL_RULE_VERSION = "frozen_matched_deposited_minstate_v1"


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def positive_reproduction_percentile(
    rows: list[dict[str, str]], percentile: float = 99.0
) -> tuple[int, int, float | None]:
    near = [
        row
        for row in rows
        if row["assignment"] in {"A", "B"}
        and math.isfinite(float(row["tmol_delta_vs_matched_AB"]))
        and matched_rmsd(row) <= 0.1
    ]
    positive = np.asarray([
        float(row["tmol_delta_vs_matched_AB"])
        for row in near
        if float(row["tmol_delta_vs_matched_AB"]) > 0.0
    ])
    return (
        len(near),
        len(positive),
        float(np.percentile(positive, percentile)) if len(positive) else None,
    )


def _strict_valid(row: dict[str, str], tolerance: float) -> bool:
    return (
        row["assignment"] in {"A", "B"}
        and as_bool(row["geometry_physical_valid"])
        and math.isfinite(float(row["tmol_delta_vs_matched_AB"]))
        and float(row["tmol_delta_vs_matched_AB"]) <= tolerance
    )


def strict_metrics(
    conformers: list[dict[str, str]],
    ensembles: list[dict[str, str]],
    tolerance: float,
) -> list[dict[str, object]]:
    result = []
    for site in sorted({row["site"] for row in ensembles}):
        site_ensembles = [row for row in ensembles if row["site"] == site]
        site_conformers = [row for row in conformers if row["site"] == site]
        by_start = {
            int(row["start"]): [
                item
                for item in site_conformers
                if int(item["start"]) == int(row["start"])
            ]
            for row in site_ensembles
        }
        extra_slots = sum(max(len(active) - 2, 0) for active in by_start.values())
        missing_slots = sum(max(2 - len(active), 0) for active in by_start.values())
        all_active_independent = 0
        all_active_joint = 0
        assigned_pair_joint = 0
        for ensemble in site_ensembles:
            active = by_start[int(ensemble["start"])]
            recovered = as_bool(ensemble["geometric_occupancy_success"])
            all_active = bool(active) and all(
                _strict_valid(row, tolerance) for row in active
            )
            pair = select_assigned_pair(active)
            pair_valid = pair is not None and all(
                _strict_valid(row, tolerance) for row in pair.values()
            )
            all_active_independent += all_active
            all_active_joint += recovered and all_active
            assigned_pair_joint += recovered and pair_valid
        result.append({
            "site": site,
            "tmol_tolerance": tolerance,
            "ensembles": len(site_ensembles),
            "active_conformers": len(site_conformers),
            "net_active_conformers_above_two_per_start": (
                len(site_conformers) - 2 * len(site_ensembles)
            ),
            "extra_active_slots_above_two_within_start": extra_slots,
            "missing_active_slots_below_two_within_start": missing_slots,
            "both_found": sum(
                as_bool(row["both_found_conventional"]) for row in site_ensembles
            ),
            "recovery_and_occupancy": sum(
                as_bool(row["geometric_occupancy_success"]) for row in site_ensembles
            ),
            "all_active_physical_independent": all_active_independent,
            "occupancy_conditioned_all_active_strict": all_active_joint,
            "assigned_pair_strict": assigned_pair_joint,
            "assigned_pair_minus_all_active": (
                assigned_pair_joint - all_active_joint
            ),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    conformers = load_composite_rows(
        args.baseline_root,
        args.replacement_root,
        "active_conformer_strict_audit.csv",
    )
    ensembles = load_composite_rows(
        args.baseline_root,
        args.replacement_root,
        "ensemble_geometry_audit.csv",
    )
    geometry_rules = set()
    for root, panels in (
        (args.baseline_root, ("original5", "expanded15")),
        (args.replacement_root, ("original2", "expanded1")),
    ):
        for panel in panels:
            summary = json.loads((root / panel / "strict_summary.json").read_text())
            geometry_rules.add(summary["audit_rule_version"])
    if len(geometry_rules) != 1:
        raise ValueError(f"inconsistent geometry rules: {sorted(geometry_rules)}")
    geometry_rule = geometry_rules.pop()
    percentile_rows = []
    for site in sorted({row["site"] for row in conformers}):
        near, positive, q99 = positive_reproduction_percentile([
            row for row in conformers if row["site"] == site
        ])
        percentile_rows.append({
            "site": site,
            "matched_conformers_rmsd_le_0_1": near,
            "positive_margins_rmsd_le_0_1": positive,
            "positive_margin_q99": "" if q99 is None else q99,
        })
    strict_rows = []
    for tolerance in (0.0, 0.5):
        strict_rows.extend(strict_metrics(conformers, ensembles, tolerance))

    args.output.mkdir(parents=True)
    _atomic_csv(args.output / "per_site_positive_margin_q99.csv", percentile_rows)
    _atomic_csv(args.output / "per_site_assigned_pair_strict.csv", strict_rows)
    q99_values = [
        float(row["positive_margin_q99"])
        for row in percentile_rows
        if row["positive_margin_q99"] != ""
    ]
    totals = {}
    for tolerance in (0.0, 0.5):
        rows = [row for row in strict_rows if row["tmol_tolerance"] == tolerance]
        totals[str(tolerance)] = {
            key: int(sum(int(row[key]) for row in rows))
            for key in (
                "ensembles",
                "active_conformers",
                "net_active_conformers_above_two_per_start",
                "extra_active_slots_above_two_within_start",
                "missing_active_slots_below_two_within_start",
                "both_found",
                "recovery_and_occupancy",
                "all_active_physical_independent",
                "occupancy_conditioned_all_active_strict",
                "assigned_pair_strict",
                "assigned_pair_minus_all_active",
            )
        }
    summary = {
        "geometry_rule": geometry_rule,
        "tmol_rule": TMOL_RULE_VERSION,
        "no_global_tolerance_promoted": True,
        "sites_with_positive_near_reproduction_margin": len(q99_values),
        "per_site_positive_q99_min": min(q99_values),
        "per_site_positive_q99_max": max(q99_values),
        "per_site_positive_q99_spread": max(q99_values) - min(q99_values),
        "totals": totals,
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
