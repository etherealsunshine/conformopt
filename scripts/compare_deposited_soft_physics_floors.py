from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--water-invariant-v1", type=Path, required=True)
    parser.add_argument("--final-rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--material-excess", type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    old_rows = read_csv(args.water_invariant_v1 / "soft_terms_per_conformer.csv")
    new_rows = read_csv(args.final_rule / "soft_terms_per_conformer.csv")
    old = {(row["site"], row["conformer"]): row for row in old_rows}
    new = {(row["site"], row["conformer"]): row for row in new_rows}
    if old.keys() != new.keys() or len(old) != 40:
        raise ValueError("expected matching 40-conformer floor tables")

    new_contacts = read_csv(args.final_rule / "contact_contributions.csv")
    final_contact_sums: dict[tuple[str, str], dict[str, float]] = {
        key: {"water": 0.0, "nonwater": 0.0} for key in new
    }
    for contact in new_contacts:
        if contact["term"] != "vdw":
            continue
        key = (contact["site"], contact["conformer"])
        is_water = any(
            marker in contact["environment_atom"]
            for marker in (":HOH:", ":WAT:", ":DOD:")
        )
        final_contact_sums[key]["water" if is_water else "nonwater"] += float(
            contact["raw_contribution"]
        )

    comparison = []
    for key in sorted(old):
        previous = old[key]
        current = new[key]
        old_vdw = float(previous["vdw_raw"])
        new_vdw = float(current["vdw_raw"])
        comparison.append(
            {
                "site": key[0],
                "residue": current["residue"],
                "conformer": key[1],
                "water_invariant_v1_vdw_raw": old_vdw,
                "final_rule_vdw_raw": new_vdw,
                "final_nonwater_vdw_raw": final_contact_sums[key]["nonwater"],
                "final_water_vdw_raw": final_contact_sums[key]["water"],
                "v1_excess_vdw_raw": old_vdw - new_vdw,
                "material_v1_excess": (
                    old_vdw - new_vdw >= args.material_excess
                ),
                "final_rotamer_raw": float(current["rotamer_raw"]),
                "final_symmetry_raw": float(current["symmetry_raw"]),
                "final_total_weighted": float(
                    current["total_weighted_soft_physics"]
                ),
            }
        )

    old_contacts = read_csv(
        args.water_invariant_v1 / "contact_contributions.csv"
    )
    contact_attribution = []
    for row in comparison:
        if not row["material_v1_excess"]:
            continue
        matching = [
            contact
            for contact in old_contacts
            if contact["site"] == row["site"]
            and contact["conformer"] == row["conformer"]
            and contact["term"] == "vdw"
            and any(
                marker in contact["environment_atom"]
                for marker in (":HOH:", ":WAT:", ":DOD:")
            )
        ]
        for contact in sorted(
            matching,
            key=lambda item: float(item["raw_contribution"]),
            reverse=True,
        ):
            contact_attribution.append(
                {
                    "site": row["site"],
                    "conformer": row["conformer"],
                    "v1_excess_vdw_raw": row["v1_excess_vdw_raw"],
                    "moving_atom": contact["moving_atom"],
                    "environment_atom": contact["environment_atom"],
                    "distance_A": float(contact["distance_A"]),
                    "v1_raw_contribution": float(contact["raw_contribution"]),
                }
            )

    final_vdw = [float(row["final_rule_vdw_raw"]) for row in comparison]
    baseline_values = [
        float(row["final_nonwater_vdw_raw"]) for row in comparison
    ]
    rotamer_nonzero = [
        row for row in comparison if float(row["final_rotamer_raw"]) > 1e-12
    ]
    payload = {
        "water_invariant_rule": json.loads(
            (args.water_invariant_v1 / "summary.json").read_text()
        ).get(
            "optimizer_physics_environment_rule",
            "2026-07-24-altloc-minstate-water-invariant-v1",
        ),
        "final_rule": json.loads(
            (args.final_rule / "summary.json").read_text()
        )["optimizer_physics_environment_rule"],
        "conformers": len(comparison),
        "material_v1_excess_threshold_raw": args.material_excess,
        "material_v1_excess_conformers": sum(
            bool(row["material_v1_excess"]) for row in comparison
        ),
        "all_final_vdw_floor": describe(final_vdw),
        "ordinary_cutoff_floor": {
            "definition": (
                "sum of final-rule 3.0 A squared-hinge VDW contributions from "
                "target backbone and non-water neighboring atoms"
            ),
            **describe(baseline_values),
        },
        "final_nonzero_symmetry_conformers": sum(
            float(row["final_symmetry_raw"]) > 1e-12 for row in comparison
        ),
        "final_rotamer_floor": {
            "nonzero_conformers": len(rotamer_nonzero),
            "maximum_raw": max(
                float(row["final_rotamer_raw"]) for row in comparison
            ),
        },
    }
    write_csv(args.output / "vdw_floor_side_by_side.csv", comparison)
    write_csv(
        args.output / "material_v1_excess_contact_attribution.csv",
        contact_attribution,
    )
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
