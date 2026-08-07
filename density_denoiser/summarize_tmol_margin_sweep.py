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

from .summarize_endpoint_audit import as_bool, select_assigned_pair


FIXED_TOLERANCES = (0.0, 0.5, 1.0, 2.0)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
            fields = list(rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
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
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _matched_rmsd(row: dict[str, str]) -> float:
    assignment = row["assignment"]
    if assignment not in {"A", "B"}:
        return math.nan
    return float(row[f"rmsd_to_{assignment}_conventional"])


def _finite_margin(row: dict[str, str]) -> bool:
    try:
        return math.isfinite(float(row["tmol_delta_vs_matched_AB"]))
    except (KeyError, TypeError, ValueError):
        return False


def _geometry_flags(active: list[dict[str, str]]) -> tuple[bool, bool, bool]:
    rotamer = bool(active) and all(
        as_bool(row.get("rotamer_within_allowed_width", row["canonical_like_30deg"]))
        for row in active
    )
    direct = rotamer and all(as_bool(row["no_direct_clash"]) for row in active)
    symmetry = direct and all(as_bool(row["no_symmetry_clash"]) for row in active)
    return rotamer, direct, symmetry


def _tmol_pass(active: list[dict[str, str]], tolerance: float) -> bool:
    return bool(active) and all(
        row["assignment"] in {"A", "B"}
        and _finite_margin(row)
        and float(row["tmol_delta_vs_matched_AB"]) <= tolerance
        for row in active
    )


def _pair_pass(active: list[dict[str, str]], tolerance: float) -> bool:
    pair = select_assigned_pair(active)
    if pair is None:
        return False
    return all(
        as_bool(row.get("rotamer_within_allowed_width", row["canonical_like_30deg"]))
        and as_bool(row["no_direct_clash"])
        and as_bool(row["no_symmetry_clash"])
        and _finite_margin(row)
        and float(row["tmol_delta_vs_matched_AB"]) <= tolerance
        for row in pair.values()
    )


def _site_q99(rows: list[dict[str, str]]) -> dict[str, object]:
    reproduction = [
        float(row["tmol_delta_vs_matched_AB"])
        for row in rows
        if row["assignment"] in {"A", "B"}
        and _finite_margin(row)
        and _matched_rmsd(row) <= 0.1
    ]
    positive = np.asarray([value for value in reproduction if value > 0.0])
    defined = bool(len(positive))
    q99 = float(np.percentile(positive, 99.0)) if defined else None
    return {
        "matched_reproduction_conformers": len(reproduction),
        "positive_reproduction_margins": int(len(positive)),
        "positive_margin_q99": q99,
        "positive_margin_q99_defined": defined,
        "effective_q99_tolerance": q99 if defined else 0.0,
    }


def build_sweep(
    conformers: list[dict[str, str]],
    ensembles: list[dict[str, str]],
    stale_both: dict[str, int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in conformers:
        by_key[(row["site"], int(row["start"]))].append(row)

    per_site = []
    q99_rows = []
    for site in sorted({row["site"] for row in ensembles}):
        site_conformers = [row for row in conformers if row["site"] == site]
        site_ensembles = [row for row in ensembles if row["site"] == site]
        q99 = _site_q99(site_conformers)
        q99_rows.append({"site": site, **q99})
        result: dict[str, object] = {
            "site": site,
            "starts": len(site_ensembles),
            "active_conformers": len(site_conformers),
            "stale_both_found": stale_both.get(site, ""),
            "both_found": sum(
                as_bool(row["both_found_conventional"]) for row in site_ensembles
            ),
            "recovery_and_occupancy": sum(
                as_bool(row["geometric_occupancy_success"]) for row in site_ensembles
            ),
        }
        if site in stale_both:
            result["both_found_delta_vs_stale"] = (
                int(result["both_found"]) - stale_both[site]
            )
        else:
            result["both_found_delta_vs_stale"] = ""

        stage_counts = {"rotamer": 0, "direct": 0, "symmetry": 0}
        tmol_counts = {tolerance: 0 for tolerance in FIXED_TOLERANCES}
        pair_counts = {tolerance: 0 for tolerance in FIXED_TOLERANCES}
        q99_count = 0
        q99_pair_count = 0
        for ensemble in site_ensembles:
            recovered = as_bool(ensemble["geometric_occupancy_success"])
            active = by_key[(site, int(ensemble["start"]))]
            rotamer, direct, symmetry = _geometry_flags(active)
            stage_counts["rotamer"] += recovered and rotamer
            stage_counts["direct"] += recovered and direct
            stage_counts["symmetry"] += recovered and symmetry
            for tolerance in FIXED_TOLERANCES:
                tmol_counts[tolerance] += (
                    recovered and symmetry and _tmol_pass(active, tolerance)
                )
                pair_counts[tolerance] += (
                    recovered and _pair_pass(active, tolerance)
                )
            q99_tolerance = float(q99["effective_q99_tolerance"])
            q99_count += (
                recovered and symmetry and _tmol_pass(active, q99_tolerance)
            )
            q99_pair_count += recovered and _pair_pass(active, q99_tolerance)

        result.update({
            "plus_all_active_rotamer": stage_counts["rotamer"],
            "plus_all_active_direct_clash": stage_counts["direct"],
            "plus_all_active_symmetry_clash": stage_counts["symmetry"],
        })
        for tolerance in FIXED_TOLERANCES:
            suffix = str(tolerance).replace(".", "_")
            result[f"plus_all_active_tmol_tol_{suffix}"] = tmol_counts[tolerance]
            result[f"assigned_pair_tmol_tol_{suffix}"] = pair_counts[tolerance]
        result.update({
            "site_positive_margin_q99": (
                "" if q99["positive_margin_q99"] is None
                else q99["positive_margin_q99"]
            ),
            "site_q99_effective_tolerance": q99["effective_q99_tolerance"],
            "plus_all_active_tmol_site_q99": q99_count,
            "assigned_pair_tmol_site_q99": q99_pair_count,
        })
        per_site.append(result)
    return per_site, q99_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a post-hoc tmol-margin sweep and monotone cascade."
    )
    parser.add_argument("--audit-root", type=Path, action="append", required=True)
    parser.add_argument("--stale-baseline-table", type=Path, required=True)
    parser.add_argument("--comparison-table", type=Path)
    parser.add_argument("--comparison-label", default="comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    conformers = []
    ensembles = []
    geometry_rules = set()
    tmol_rules = set()
    for root in args.audit_root:
        conformers.extend(_read_csv(root / "active_conformer_strict_audit.csv"))
        ensembles.extend(_read_csv(root / "ensemble_geometry_audit.csv"))
        geometry_rules.add(
            json.loads((root / "strict_summary.json").read_text())[
                "audit_rule_version"
            ]
        )
        tmol_rules.update(
            row["tmol_environment_rule"]
            for row in _read_csv(root / "tmol_energies.csv")
        )
    if len(geometry_rules) != 1 or len(tmol_rules) != 1:
        raise ValueError(
            f"inconsistent rules: geometry={sorted(geometry_rules)} "
            f"tmol={sorted(tmol_rules)}"
        )
    if len(ensembles) != 1000 or len({row["site"] for row in ensembles}) != 20:
        raise ValueError("expected 20 sites and 1000 starts")

    stale_both = {
        row["site"]: int(row["both_found_conventional"])
        for row in _read_csv(args.stale_baseline_table)
    }
    per_site, q99_rows = build_sweep(conformers, ensembles, stale_both)
    if args.comparison_table:
        comparison = {
            row["site"]: int(
                row.get("both_found", row.get("both_found_conventional", ""))
            )
            for row in _read_csv(args.comparison_table)
            if row["site"] != "TOTAL"
        }
        if set(comparison) != {row["site"] for row in per_site}:
            raise ValueError("comparison table does not contain the same 20 sites")
        for row in per_site:
            row[f"{args.comparison_label}_both_found"] = comparison[row["site"]]
            row[f"both_found_delta_vs_{args.comparison_label}"] = (
                int(row["both_found"]) - comparison[row["site"]]
            )

    margin_rows = []
    for row in conformers:
        margin_rows.append({
            "candidate_id": row["candidate_id"],
            "site": row["site"],
            "start": int(row["start"]),
            "conformer": int(row["conformer"]),
            "occupancy": float(row["occupancy"]),
            "assignment": row["assignment"],
            "rmsd_to_matched_deposited": (
                _matched_rmsd(row)
                if row["assignment"] in {"A", "B"} else ""
            ),
            "tmol_energy": row["tmol_energy"],
            "tmol_reference_matched_AB": (
                row["tmol_reference_matched_AB"]
                if row["assignment"] in {"A", "B"} else ""
            ),
            "tmol_margin_candidate_minus_matched_deposited": (
                row["tmol_delta_vs_matched_AB"]
                if row["assignment"] in {"A", "B"} else ""
            ),
            "finite_matched_margin": (
                row["assignment"] in {"A", "B"} and _finite_margin(row)
            ),
        })

    totals: dict[str, object] = {"site": "TOTAL"}
    for key in per_site[0]:
        if key == "site":
            continue
        if key in {
            "site_positive_margin_q99",
            "site_q99_effective_tolerance",
        }:
            totals[key] = "per-site"
        elif key in {"stale_both_found", "both_found_delta_vs_stale"}:
            totals[key] = sum(
                int(row[key]) for row in per_site if row[key] != ""
            )
        else:
            totals[key] = sum(int(row[key]) for row in per_site)
    cascade_rows = [*per_site, totals]

    args.output.mkdir(parents=True)
    _atomic_csv(args.output / "per_conformer_tmol_margins.csv", margin_rows)
    _atomic_csv(args.output / "per_site_positive_margin_q99.csv", q99_rows)
    _atomic_csv(args.output / "per_site_cascade_and_tmol_sweep.csv", cascade_rows)
    _atomic_json(args.output / "summary.json", {
        "geometry_rule": next(iter(geometry_rules)),
        "tmol_rule": next(iter(tmol_rules)),
        "fixed_tolerances": list(FIXED_TOLERANCES),
        "site_q99_definition": (
            "99th percentile of positive candidate-minus-matched-deposited "
            "margins among finite A/B-matched conformers with RMSD <= 0.1 A"
        ),
        "site_q99_when_no_positive_margin": (
            "reported undefined; effective sweep tolerance is 0.0"
        ),
        "tolerance_promoted": False,
        "tmol_tolerance_is_post_hoc": True,
        "stale_baseline_not_model_progress_comparator": True,
        "totals": totals,
    })


if __name__ == "__main__":
    main()
