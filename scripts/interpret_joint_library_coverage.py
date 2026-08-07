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

from scripts.score_endpoint_dunbrack_joint_library import (
    backbone_angles,
    classify,
    read_csv,
)


SEMI_ROTAMERIC = {"ASN", "ASP", "GLN", "GLU", "HIS", "PHE", "TRP", "TYR"}


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
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def grouped_summary(rows: list[dict[str, object]], population: str) -> list[dict[str, object]]:
    output = []
    groupings = [
        *[(residue, lambda row, aa=residue: row["residue"] == aa)
          for residue in sorted({str(row["residue"]) for row in rows})],
        ("rotameric", lambda row: row["library_treatment"] == "rotameric"),
        ("semi_rotameric", lambda row: row["library_treatment"] == "semi_rotameric"),
        ("ALL", lambda row: True),
    ]
    for label, predicate in groupings:
        selected = [row for row in rows if predicate(row)]
        rejected = sum(as_bool(row["gate_library_disagreement"]) for row in selected)
        per_chi: dict[int, list[float]] = defaultdict(list)
        maximum = []
        for row in selected:
            distances = [
                float(value)
                for value in str(
                    row["nearest_qualifying_per_chi_distance_degrees"]
                ).split(";")
            ]
            for index, value in enumerate(distances, start=1):
                per_chi[index].append(value)
            maximum.append(
                float(row["nearest_qualifying_max_chi_distance_degrees"])
            )
        output.append({
            "population": population,
            "group": label,
            "library_treatment": (
                label if label in {"rotameric", "semi_rotameric"} else ""
            ),
            "conformers": len(selected),
            "disagreements": rejected,
            "disagreement_rate": rejected / len(selected) if selected else "",
            "max_over_chis_distribution_json": json.dumps(
                describe(maximum), sort_keys=True
            ),
            "per_chi_distribution_json": json.dumps(
                {str(key): describe(values) for key, values in sorted(per_chi.items())},
                sort_keys=True,
            ),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-table", type=Path, required=True)
    parser.add_argument("--deposited-audit", type=Path, required=True)
    parser.add_argument("--pdb-root", type=Path, required=True)
    parser.add_argument("--dunbrack-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.003)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

    database = DunbrackRotamerLibrary.from_file(str(args.dunbrack_bin))
    table_for_residue = {
        item.residue_name: item.dun_table_name for item in database.dun_lookup
    }
    libraries = {
        item.table_name: item
        for item in (*database.rotameric_libraries, *database.semi_rotameric_libraries)
    }

    endpoints = []
    for row in read_csv(args.endpoint_table):
        result = dict(row)
        result["library_treatment"] = (
            "semi_rotameric" if row["residue"] in SEMI_ROTAMERIC else "rotameric"
        )
        endpoints.append(result)

    backbone_cache = {}
    deposited = []
    for row in read_csv(args.deposited_audit):
        residue = row["residue_name"]
        site_key = (
            row["pdb_id"],
            row["chain"],
            int(row["residue_number"]),
        )
        if site_key not in backbone_cache:
            pdb = args.pdb_root / f"{row['pdb_id'].upper()}.pdb"
            if not pdb.exists():
                pdb = args.pdb_root / f"{row['pdb_id'].lower()}.pdb"
            backbone_cache[site_key] = backbone_angles(
                pdb, row["chain"], int(row["residue_number"])
            )
        chis = [float(value) for value in row["chi_degrees"].split(";")]
        widths = [
            float(value)
            for value in row["rotamer_allowed_width_degrees"].split(";")
        ]
        phi, psi = backbone_cache[site_key]
        deposited.append({
            "population": "deposited_340",
            "candidate_id": f"{row['site']}_{row['control']}",
            "site": row["site"],
            "residue": residue,
            "control": row["control"],
            "chi_degrees": row["chi_degrees"],
            "production_rotamer_pass": row["rotamer_pass"],
            "library_treatment": (
                "semi_rotameric" if residue in SEMI_ROTAMERIC else "rotameric"
            ),
            **classify(
                chis,
                widths,
                residue,
                phi,
                psi,
                libraries[table_for_residue[residue]],
                args.probability_threshold,
                3.0,
            ),
        })

    summary = [
        *grouped_summary(endpoints, "endpoint_gate_pass"),
        *grouped_summary(deposited, "deposited_340"),
    ]
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "deposited_340_joint_library.csv", deposited)
    atomic_csv(args.output / "coverage_summary.csv", summary)
    atomic_json(args.output / "summary.json", {
        "probability_threshold": args.probability_threshold,
        "semi_rotameric_types": sorted(SEMI_ROTAMERIC),
        "semi_rotameric_handling": (
            "The current coverage test enumerates the library's tabulated "
            "semi-rotameric terminal-chi bins as discrete joint states; it "
            "does not integrate the continuous terminal-chi density."
        ),
        "binary_rate_is_lower_bound": True,
        "coverage_summary": summary,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
