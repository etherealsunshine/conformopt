"""Compile normalization follow-up properties and successful-probe travel."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

from density_denoiser.residue_geometry import CHI_SPECS


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def median(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else math.nan


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--travel", type=Path, required=True)
    parser.add_argument("--separations", type=Path, required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--neighbor-census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    fixed_rows = [
        row for row in read_csv(args.probe)
        if row["occupancy_mode"] == "fixed_minor"
    ]
    by_site: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fixed_rows:
        by_site[row["site"]].append(row)

    separations = {
        row["site"]: float(row["local_unsym_rmsd_A"])
        for row in read_csv(args.separations)
    }
    selections = {}
    for path in args.selection:
        for row in json.loads(path.read_text())["sites"]:
            selections[row["key"]] = row

    neighbor_census = {
        row["site"]: row for row in read_csv(args.neighbor_census)
    }

    property_rows = []
    for site, rows in sorted(by_site.items()):
        record = selections[site]
        structure = gemmi.read_structure(record["pdb_path"])
        residue = next(
            residue for chain in structure[0] if chain.name == record["chain"]
            for residue in chain
            if residue.seqid.num == int(record["residue_number"])
            and residue.seqid.icode == record["insertion_code"]
        )
        sidechain_b = [
            float(atom.b_iso) for atom in residue
            if atom.element.name != "H"
            and atom.name.strip() not in {"N", "CA", "C", "O"}
        ]
        success = sum(truth(row["recovered_minor_lt_1A"]) for row in rows)
        census = neighbor_census[site]
        minor_integral = median(
            row["minor_before_integrated_positive"] for row in rows
        )
        major_integral = median(
            row["major_before_integrated_positive"] for row in rows
        )
        property_rows.append({
            "site": site,
            "residue_type": record["residue_name"],
            "eligible": len(rows),
            "fixed_probe_recovered": success,
            "fixed_probe_recovery_rate": success / len(rows),
            "local_unsym_AB_separation_A": separations[site],
            "minor_state": rows[0]["minor_state"],
            "minor_occupancy": float(rows[0]["deposited_minor_occupancy"]),
            "n_chi": len(CHI_SPECS[record["residue_name"]]["rotations"]),
            "minor_positive_integral_median": minor_integral,
            "major_positive_integral_median": major_integral,
            "minor_to_major_positive_ratio": (
                minor_integral / major_integral
                if major_integral > 0 else math.inf
            ),
            "minor_residual_rms_median": median(
                row["minor_before_rms"] for row in rows
            ),
            "lobe_overlap_fraction_median": median(
                row["minor_before_overlap_fraction"] for row in rows
            ),
            "initial_rmsd_to_minor_median": median(
                row["initial_rmsd_to_minor"] for row in rows
            ),
            "pdb_resolution_A": float(structure.resolution),
            "target_sidechain_B_iso_median": median(sidechain_b),
            "total_altloc_neighbor_groups": int(
                census["total_altloc_groups"]
            ),
            "soft_vdw_nonzero_altloc_groups": int(
                census["nonzero_in_at_least_one_endpoint"]
            ),
            "soft_vdw_state_sensitive_altloc_groups": int(
                census[
                    "materially_state_dependent_in_at_least_one_endpoint"
                ]
            ),
            "soft_vdw_nonzero_by_category": census["nonzero_by_category"],
            "soft_vdw_state_sensitive_by_category": census[
                "material_by_category"
            ],
        })

    travel_rows = [
        row for row in read_csv(args.travel)
        if row["occupancy_mode"] == "fixed_minor"
        and truth(row["recovered_minor_lt_1A"])
    ]
    if len(travel_rows) != 26:
        raise RuntimeError(f"expected 26 fixed successes, got {len(travel_rows)}")
    travel_output = []
    travel_by_site = defaultdict(list)
    for row in travel_rows:
        separation = separations[row["site"]]
        value = {
            "site": row["site"],
            "start": int(row["start"]),
            "local_unsym_AB_separation_A": separation,
            "initial_rmsd_to_minor_A": float(row["initial_rmsd_to_minor"]),
            "final_rmsd_to_minor_A": float(row["final_rmsd_to_minor"]),
            "fixed_label_travel_rmsd_A": float(
                row["fixed_label_travel_rmsd_A"]
            ),
            "symmetry_aware_travel_rmsd_A": float(
                row["symmetry_aware_travel_rmsd_A"]
            ),
            "travel_to_separation_ratio": (
                float(row["fixed_label_travel_rmsd_A"]) / separation
            ),
            "chi_space_travel_degrees": float(
                row["chi_space_travel_degrees"]
            ),
        }
        travel_output.append(value)
        travel_by_site[row["site"]].append(value)

    travel_summary = []
    for site, rows in sorted(travel_by_site.items()):
        travel_summary.append({
            "site": site,
            "successes": len(rows),
            "local_unsym_AB_separation_A": separations[site],
            "fixed_label_travel_rmsd_median_A": median(
                row["fixed_label_travel_rmsd_A"] for row in rows
            ),
            "fixed_label_travel_rmsd_min_A": min(
                row["fixed_label_travel_rmsd_A"] for row in rows
            ),
            "fixed_label_travel_rmsd_max_A": max(
                row["fixed_label_travel_rmsd_A"] for row in rows
            ),
            "travel_to_separation_ratio_median": median(
                row["travel_to_separation_ratio"] for row in rows
            ),
            "travel_lt_0p5A": sum(
                row["fixed_label_travel_rmsd_A"] < 0.5 for row in rows
            ),
            "travel_lt_1p0A": sum(
                row["fixed_label_travel_rmsd_A"] < 1.0 for row in rows
            ),
        })

    atomic_csv(args.output / "site_properties.csv", property_rows)
    atomic_csv(args.output / "close_success_travel.csv", travel_output)
    atomic_csv(args.output / "close_success_travel_by_site.csv", travel_summary)
    atomic_json(args.output / "summary.json", {
        "diagnostic_only": True,
        "metric_changed": False,
        "close_successes": len(travel_output),
        "fixed_label_travel_rmsd_median_A": median(
            row["fixed_label_travel_rmsd_A"] for row in travel_output
        ),
        "fixed_label_travel_rmsd_q25_A": float(np.quantile(
            [row["fixed_label_travel_rmsd_A"] for row in travel_output], 0.25
        )),
        "fixed_label_travel_rmsd_q75_A": float(np.quantile(
            [row["fixed_label_travel_rmsd_A"] for row in travel_output], 0.75
        )),
        "travel_lt_0p5A": sum(
            row["fixed_label_travel_rmsd_A"] < 0.5
            for row in travel_output
        ),
        "travel_lt_1p0A": sum(
            row["fixed_label_travel_rmsd_A"] < 1.0
            for row in travel_output
        ),
        "successes_at_separation_le_1p5A": sum(
            row["local_unsym_AB_separation_A"] <= 1.5
            for row in travel_output
        ),
        "successes_at_separation_le_2p0A": sum(
            row["local_unsym_AB_separation_A"] <= 2.0
            for row in travel_output
        ),
    })


if __name__ == "__main__":
    main()
