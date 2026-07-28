"""Analyze merge-then-assign sensitivity and hidden greedy duplicates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from density_denoiser.residue_geometry import symmetry_aware_rmsd
from scripts.analyze_one_to_one_metric_v2 import STAGES, load_version


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-audit-root", type=Path, action="append", required=True)
    parser.add_argument("--v2-audit-root", type=Path, action="append", required=True)
    parser.add_argument("--merge-0p3-root", type=Path, action="append", required=True)
    parser.add_argument("--merge-0p5-root", type=Path, action="append", required=True)
    parser.add_argument("--merge-0p8-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    versions = {
        "v1_greedy": load_version(args.v1_audit_root),
        "v2_one_to_one_no_merge": load_version(args.v2_audit_root),
        "merge_0p3": load_version(args.merge_0p3_root),
        "merge_0p5": load_version(args.merge_0p5_root),
        "merge_0p8": load_version(args.merge_0p8_root),
    }
    sites = sorted(versions["v1_greedy"])
    sensitivity_rows = []
    total_rows = []
    for version, counts in versions.items():
        total = {"version": version}
        for stage in STAGES:
            total[stage] = sum(counts[site][stage] for site in sites)
        total_rows.append(total)
    for site in sites:
        row: dict[str, object] = {"site": site, "starts": 50}
        for version, counts in versions.items():
            for stage in STAGES:
                row[f"{version}_{stage}"] = counts[site][stage]
        sensitivity_rows.append(row)
    write_csv(args.output / "threshold_sensitivity_totals.csv", total_rows)
    write_csv(args.output / "threshold_sensitivity_per_site.csv", sensitivity_rows)

    geometry_rows = []
    coordinates: dict[str, torch.Tensor] = {}
    site_metadata: dict[str, tuple[list[str], str]] = {}
    for root in args.v1_audit_root:
        geometry_rows.extend(read_csv(root / "active_conformer_geometry_audit.csv"))
        inputs = json.loads((root / "tmol_inputs.json").read_text())
        for site in inputs["sites"]:
            site_metadata[site["site"]] = (
                site["atom_names"],
                site["residue_type"],
            )
            for candidate in site["candidates"]:
                coordinates[candidate["candidate_id"]] = torch.tensor(
                    candidate["coordinates"], dtype=torch.float32
                )

    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in geometry_rows:
        if row["assignment"] in {"A", "B"}:
            grouped[(row["site"], int(row["start"]), row["assignment"])].append(row)

    group_rows, member_rows = [], []
    starts_with_duplicates: set[tuple[str, int]] = set()
    per_site_starts: dict[str, set[int]] = defaultdict(set)
    for (site, start, state), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        starts_with_duplicates.add((site, start))
        per_site_starts[site].add(start)
        primary = min(
            rows,
            key=lambda row: float(row[f"rmsd_to_{state}_conventional"]),
        )
        nonprimary = [row for row in rows if row is not primary]
        names, resname = site_metadata[site]
        pairwise = []
        for left_offset, left in enumerate(rows):
            for right in rows[left_offset + 1:]:
                pairwise.append(float(symmetry_aware_rmsd(
                    coordinates[left["candidate_id"]],
                    coordinates[right["candidate_id"]],
                    names,
                    resname,
                )))
        group_rows.append({
            "site": site,
            "start": start,
            "state": state,
            "members": len(rows),
            "primary_candidate": primary["candidate_id"],
            "primary_occupancy": float(primary["occupancy"]),
            "total_group_occupancy": sum(float(row["occupancy"]) for row in rows),
            "nonprimary_occupancies": ";".join(
                row["occupancy"] for row in nonprimary
            ),
            "pairwise_rmsd_A": ";".join(f"{value:.9g}" for value in pairwise),
            "pairwise_rmsd_min_A": min(pairwise),
            "pairwise_rmsd_median_A": percentile(pairwise, 0.5),
            "pairwise_rmsd_max_A": max(pairwise),
        })
        for row in nonprimary:
            member_rows.append({
                "site": site,
                "start": start,
                "state": state,
                "candidate_id": row["candidate_id"],
                "occupancy": float(row["occupancy"]),
                "rmsd_to_primary_A": float(symmetry_aware_rmsd(
                    coordinates[row["candidate_id"]],
                    coordinates[primary["candidate_id"]],
                    names,
                    resname,
                )),
                "rmsd_to_deposited_state_A": float(
                    row[f"rmsd_to_{state}_conventional"]
                ),
            })
    write_csv(args.output / "greedy_duplicate_groups.csv", group_rows)
    write_csv(args.output / "greedy_duplicate_nonprimary_members.csv", member_rows)

    per_site_rows = []
    for site in sites:
        site_groups = [row for row in group_rows if row["site"] == site]
        site_members = [row for row in member_rows if row["site"] == site]
        occupancies = [float(row["occupancy"]) for row in site_members]
        pairwise = [
            value
            for row in site_groups
            for value in (
                float(item) for item in str(row["pairwise_rmsd_A"]).split(";")
            )
        ]
        per_site_rows.append({
            "site": site,
            "starts_with_same_state_duplicates": len(per_site_starts[site]),
            "duplicate_groups": len(site_groups),
            "nonprimary_members": len(site_members),
            "nonprimary_occupancy_median": percentile(occupancies, 0.5),
            "nonprimary_occupancy_q25": percentile(occupancies, 0.25),
            "nonprimary_occupancy_q75": percentile(occupancies, 0.75),
            "nonprimary_occupancy_gt_0p20": sum(
                value > 0.20 for value in occupancies
            ),
            "within_group_pairwise_rmsd_median_A": percentile(pairwise, 0.5),
            "within_group_pairwise_rmsd_q25_A": percentile(pairwise, 0.25),
            "within_group_pairwise_rmsd_q75_A": percentile(pairwise, 0.75),
        })
    write_csv(args.output / "greedy_duplicate_per_site.csv", per_site_rows)

    all_occupancies = [float(row["occupancy"]) for row in member_rows]
    all_pairwise = [
        value
        for row in group_rows
        for value in (
            float(item) for item in str(row["pairwise_rmsd_A"]).split(";")
        )
    ]
    summary = {
        "threshold_sensitivity_totals": {
            row["version"]: {stage: row[stage] for stage in STAGES}
            for row in total_rows
        },
        "greedy_hidden_duplication": {
            "starts_with_two_or_more_same_state_conformers": len(
                starts_with_duplicates
            ),
            "duplicate_groups": len(group_rows),
            "nonprimary_members": len(member_rows),
            "nonprimary_occupancy": {
                "median": percentile(all_occupancies, 0.5),
                "q25": percentile(all_occupancies, 0.25),
                "q75": percentile(all_occupancies, 0.75),
                "minimum": min(all_occupancies),
                "maximum": max(all_occupancies),
                "count_gt_0p20": sum(value > 0.20 for value in all_occupancies),
            },
            "within_group_pairwise_rmsd_A": {
                "median": percentile(all_pairwise, 0.5),
                "q25": percentile(all_pairwise, 0.25),
                "q75": percentile(all_pairwise, 0.75),
                "minimum": min(all_pairwise),
                "maximum": max(all_pairwise),
            },
            "comparison_unmatched_extra_occupancy_median": 0.066,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
