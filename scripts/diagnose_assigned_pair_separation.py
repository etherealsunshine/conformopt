"""Check whether recovered A/B assignments double-count one geometry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from density_denoiser.residue_geometry import symmetry_aware_rmsd


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


def conventional_rmsd(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.sum((left - right).square(), dim=-1))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-audit-root", type=Path, action="append", required=True)
    parser.add_argument("--v1-audit-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    coordinates: dict[str, torch.Tensor] = {}
    site_metadata: dict[str, dict[str, object]] = {}
    ensembles = []
    for root in args.v3_audit_root:
        ensembles.extend(read_csv(root / "ensemble_strict_audit.csv"))
        inputs = json.loads((root / "tmol_inputs.json").read_text())
        for site in inputs["sites"]:
            key = site["site"]
            names = site["atom_names"]
            resname = site["residue_type"]
            deposited_a = torch.tensor(site["A"], dtype=torch.float32)
            deposited_b = torch.tensor(site["B"], dtype=torch.float32)
            site_metadata[key] = {
                "names": names,
                "resname": resname,
                "deposited_A": deposited_a,
                "deposited_B": deposited_b,
                "deposited_separation": float(symmetry_aware_rmsd(
                    deposited_a, deposited_b, names, resname
                )),
            }
            for candidate in site["candidates"]:
                coordinates[candidate["candidate_id"]] = torch.tensor(
                    candidate["coordinates"], dtype=torch.float32
                )

    separation_rows = []
    for row in ensembles:
        if str(row["both_found_conventional"]).lower() != "true":
            continue
        site = row["site"]
        candidate_a = row["assigned_pair_candidate_A"]
        candidate_b = row["assigned_pair_candidate_B"]
        if not candidate_a or not candidate_b:
            raise ValueError(f"recovered start lacks assigned pair: {site} {row['start']}")
        metadata = site_metadata[site]
        assigned_separation = float(symmetry_aware_rmsd(
            coordinates[candidate_a],
            coordinates[candidate_b],
            metadata["names"],
            metadata["resname"],
        ))
        deposited_separation = float(metadata["deposited_separation"])
        separation_rows.append({
            "site": site,
            "start": int(row["start"]),
            "candidate_A": candidate_a,
            "candidate_B": candidate_b,
            "assigned_pair_separation_A": assigned_separation,
            "deposited_A_B_separation_A": deposited_separation,
            "assigned_minus_deposited_A": (
                assigned_separation - deposited_separation
            ),
            "assigned_over_deposited": assigned_separation / deposited_separation,
        })
    if len(separation_rows) != 742:
        raise ValueError(f"expected 742 recovered starts, found {len(separation_rows)}")
    write_csv(args.output / "recovered_pair_separation.csv", separation_rows)

    per_site_rows = []
    for site in sorted(site_metadata):
        rows = [row for row in separation_rows if row["site"] == site]
        assigned = [float(row["assigned_pair_separation_A"]) for row in rows]
        ratios = [float(row["assigned_over_deposited"]) for row in rows]
        differences = [float(row["assigned_minus_deposited_A"]) for row in rows]
        deposited = float(site_metadata[site]["deposited_separation"])
        per_site_rows.append({
            "site": site,
            "recovered_starts": len(rows),
            "deposited_A_B_separation_A": deposited,
            "assigned_separation_median_A": percentile(assigned, 0.5),
            "assigned_separation_q25_A": percentile(assigned, 0.25),
            "assigned_separation_q75_A": percentile(assigned, 0.75),
            "assigned_separation_min_A": min(assigned) if assigned else float("nan"),
            "assigned_separation_max_A": max(assigned) if assigned else float("nan"),
            "assigned_over_deposited_median": percentile(ratios, 0.5),
            "assigned_over_deposited_q25": percentile(ratios, 0.25),
            "assigned_over_deposited_q75": percentile(ratios, 0.75),
            "assigned_minus_deposited_median_A": percentile(differences, 0.5),
            "assigned_separation_lt_half_deposited": sum(
                ratio < 0.5 for ratio in ratios
            ),
            "assigned_separation_lt_0p25_A": sum(
                value < 0.25 for value in assigned
            ),
        })
    write_csv(args.output / "pair_separation_per_site.csv", per_site_rows)

    v1_geometry = []
    v1_coordinates: dict[str, torch.Tensor] = {}
    v1_8q6q_metadata = None
    for root in args.v1_audit_root:
        v1_geometry.extend(read_csv(root / "active_conformer_geometry_audit.csv"))
        inputs = json.loads((root / "tmol_inputs.json").read_text())
        for site in inputs["sites"]:
            if site["site"] != "8Q6Q_B_ASP81":
                continue
            v1_8q6q_metadata = (site["atom_names"], site["residue_type"])
            for candidate in site["candidates"]:
                v1_coordinates[candidate["candidate_id"]] = torch.tensor(
                    candidate["coordinates"], dtype=torch.float32
                )
    if v1_8q6q_metadata is None:
        raise ValueError("8Q6Q metadata not found")
    names, resname = v1_8q6q_metadata
    groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in v1_geometry:
        if row["site"] == "8Q6Q_B_ASP81" and row["assignment"] in {"A", "B"}:
            groups[(int(row["start"]), row["assignment"])].append(row)

    swap_rows = []
    for (start, state), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        for left_offset, left in enumerate(rows):
            for right in rows[left_offset + 1:]:
                left_xyz = v1_coordinates[left["candidate_id"]]
                right_xyz = v1_coordinates[right["candidate_id"]]
                raw = conventional_rmsd(left_xyz, right_xyz)
                swapped = float(symmetry_aware_rmsd(
                    left_xyz, right_xyz, names, resname
                ))
                swap_rows.append({
                    "site": "8Q6Q_B_ASP81",
                    "start": start,
                    "state": state,
                    "candidate_1": left["candidate_id"],
                    "candidate_2": right["candidate_id"],
                    "identity_label_rmsd_A": raw,
                    "terminal_swap_minimized_rmsd_A": swapped,
                    "reduction_A": raw - swapped,
                    "swap_changed_value": swapped < raw - 1e-6,
                })
    write_csv(args.output / "8q6q_same_state_swap_check.csv", swap_rows)

    ratios = [float(row["assigned_over_deposited"]) for row in separation_rows]
    differences = [
        float(row["assigned_minus_deposited_A"]) for row in separation_rows
    ]
    raw_8q6q = [float(row["identity_label_rmsd_A"]) for row in swap_rows]
    swapped_8q6q = [
        float(row["terminal_swap_minimized_rmsd_A"]) for row in swap_rows
    ]
    site_5kwb = next(
        row for row in per_site_rows if row["site"] == "5KWB_A_PHE591"
    )
    summary = {
        "recovered_starts": len(separation_rows),
        "rmsd_definition": (
            "conventional heavy-atom RMSD minimized over valid terminal-group "
            "equivalent-atom permutations"
        ),
        "assigned_over_deposited": {
            "median": percentile(ratios, 0.5),
            "q25": percentile(ratios, 0.25),
            "q75": percentile(ratios, 0.75),
            "count_lt_0p5": sum(value < 0.5 for value in ratios),
        },
        "assigned_minus_deposited_A": {
            "median": percentile(differences, 0.5),
            "q25": percentile(differences, 0.25),
            "q75": percentile(differences, 0.75),
        },
        "5KWB_A_PHE591": site_5kwb,
        "8Q6Q_B_ASP81_terminal_swap": {
            "pairs": len(swap_rows),
            "identity_median_A": percentile(raw_8q6q, 0.5),
            "swap_minimized_median_A": percentile(swapped_8q6q, 0.5),
            "identity_q25_A": percentile(raw_8q6q, 0.25),
            "identity_q75_A": percentile(raw_8q6q, 0.75),
            "swap_minimized_q25_A": percentile(swapped_8q6q, 0.25),
            "swap_minimized_q75_A": percentile(swapped_8q6q, 0.75),
            "pairs_changed_by_swap": sum(
                str(row["swap_changed_value"]).lower() == "true"
                for row in swap_rows
            ),
            "pairs_below_0p25_after_swap": sum(
                value < 0.25 for value in swapped_8q6q
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
