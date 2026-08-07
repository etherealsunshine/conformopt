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


TUNED_SITES = {"3A1C_B_ARG447", "6H59_B_ARG144", "3NY7_B_LYS19"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def correlation_rows(
    margins: list[dict[str, str]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for row in margins:
        assignment = row["assignment"]
        if assignment not in {"A", "B"}:
            continue
        try:
            rmsd = float(row["rmsd_to_matched_deposited"])
            margin = float(row["tmol_margin_candidate_minus_matched_deposited"])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(rmsd) and math.isfinite(margin)):
            continue
        groups[(row["site"], "all")].append((rmsd, margin))
        groups[(row["site"], assignment)].append((rmsd, margin))

    results = []
    for (site, assignment), values in sorted(groups.items()):
        rmsd = np.asarray([value[0] for value in values], dtype=float)
        margin = np.asarray([value[1] for value in values], dtype=float)
        pearson = _pearson(rmsd, margin)
        spearman = _pearson(_rankdata(rmsd), _rankdata(margin))
        results.append(
            {
                "site": site,
                "assignment": assignment,
                "n": len(values),
                "pearson_margin_vs_rmsd": (
                    None if pearson is None or math.isnan(pearson) else pearson
                ),
                "spearman_margin_vs_rmsd": (
                    None if spearman is None or math.isnan(spearman) else spearman
                ),
            }
        )
    return results


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite", type=Path, required=True)
    args = parser.parse_args()

    cascade = read_csv(args.composite / "per_site_cascade_and_tmol_sweep.csv")
    sites = [row for row in cascade if row["site"] != "TOTAL"]
    totals = next(row for row in cascade if row["site"] == "TOTAL")
    margins = read_csv(args.composite / "per_conformer_tmol_margins.csv")
    provenance = read_csv(args.composite / "site_rule_provenance.csv")

    correlations = correlation_rows(margins)
    atomic_csv(args.composite / "tmol_margin_rmsd_correlations.csv", correlations)

    full_geometry_zero_then_jump = [
        {
            "site": row["site"],
            "geometry": int(row["plus_all_active_symmetry_clash"]),
            "tol_0_0": int(row["plus_all_active_tmol_tol_0_0"]),
            "tol_0_5": int(row["plus_all_active_tmol_tol_0_5"]),
        }
        for row in sites
        if int(row["plus_all_active_symmetry_clash"]) > 0
        and int(row["plus_all_active_tmol_tol_0_0"]) == 0
        and int(row["plus_all_active_tmol_tol_0_5"]) > 0
    ]
    tuned = [row for row in sites if row["site"] in TUNED_SITES]
    tuned_tol0_all = sum(
        int(row["plus_all_active_tmol_tol_0_0"]) for row in tuned
    )
    tuned_tol0_pair = sum(int(row["assigned_pair_tmol_tol_0_0"]) for row in tuned)

    rules = sorted(
        {row["optimizer_physics_environment_rule"] for row in provenance}
    )
    optimizer_hashes = sorted(
        {row["five_site_optimizer_sha256"] for row in provenance}
    )
    clash_hashes = sorted(
        {row["clash_environment_sha256"] for row in provenance}
    )
    if len(provenance) != 20 or any(
        len(values) != 1 for values in (rules, optimizer_hashes, clash_hashes)
    ):
        raise ValueError("composite provenance is not uniform across 20 sites")

    lookup = {
        (row["site"], row["assignment"]): row for row in correlations
    }
    evidence = {
        "site_count": len(sites),
        "start_count": sum(int(row["starts"]) for row in sites),
        "uniform_provenance": {
            "optimizer_environment_rule": rules[0],
            "five_site_optimizer_sha256": optimizer_hashes[0],
            "clash_environment_sha256": clash_hashes[0],
        },
        "totals": totals,
        "tuned_sites": sorted(TUNED_SITES),
        "tuned_site_contribution_tol0_all_active": tuned_tol0_all,
        "tuned_site_contribution_tol0_assigned_pair": tuned_tol0_pair,
        "full_geometry_sites_zero_at_tol0_then_jump_at_tol0_5": (
            full_geometry_zero_then_jump
        ),
        "selected_margin_rmsd_correlations": {
            f"{site}_{assignment}": lookup.get((site, assignment))
            for site in ("3K8W_A_SER337", "8Q6Q_B_ASP81")
            for assignment in ("all", "A", "B")
        },
    }
    atomic_json(args.composite / "freeze_evidence.json", evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
