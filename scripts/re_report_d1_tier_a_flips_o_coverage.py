#!/usr/bin/env python3
"""Re-express existing D1 flip tier-(a) results as carbonyl-O A→B coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from qfit.structure import Structure
from run_d1_tier_a_flips import atom_local_index, source_path


def o_coordinate(residue: object) -> np.ndarray:
    return np.asarray(residue.coor[atom_local_index(residue, "O")], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.input / "per_site.csv").open() as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        path, _ = source_path(source["pdb_id"])
        residue_id = (int(source["resnum"]), "")
        a = Structure.fromfile(path).extract("altloc", ("", "A"))[source["chain"]].conformers[0][residue_id]
        b = Structure.fromfile(path).extract("altloc", ("", "B"))[source["chain"]].conformers[0][residue_id]
        separation = float(np.linalg.norm(o_coordinate(a) - o_coordinate(b)))
        residual = float(source["tier_a_min_central_O_rmsd_A"])
        rows.append({
            "site": source["site"], "qfit_result": source["sampling_outcome"],
            "candidate_count": int(source["candidate_count_including_input"]),
            "O_A_to_B_distance_A": separation, "best_candidate_O_residual_to_B_A": residual,
            "O_fraction_A_to_B_covered": 1.0 - residual / separation,
            "O_fraction_A_to_B_remaining": residual / separation,
        })
    with (args.output / "per_site.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    covered = np.asarray([row["O_fraction_A_to_B_covered"] for row in rows])
    full = np.asarray([row["O_fraction_A_to_B_covered"] for row in rows if row["candidate_count"] == 19])
    (args.output / "summary.json").write_text(json.dumps({
        "status": "complete", "sites": len(rows),
        "metric": "1 - best O-only residual-to-B / deposited A-to-B O distance",
        "median_fraction_covered": float(np.median(covered)),
        "range_fraction_covered": [float(covered.min()), float(covered.max())],
        "full_19_candidate_sites": int(len(full)),
        "full_19_candidate_median_fraction_covered": float(np.median(full)),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
