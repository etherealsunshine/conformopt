from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def describe(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--conformer", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", default="2VFP_A_TYR417")
    args = parser.parse_args()

    ensembles = {
        int(row["start"]): row
        for row in read_csv(args.ensemble)
        if row["site"] == args.site and as_bool(row["both_found_conventional"])
    }
    conformers: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.conformer):
        if row["site"] == args.site:
            conformers[int(row["start"])].append(row)
    raw_starts = {
        int(row["start"]): row
        for row in read_csv(args.starts)
        if int(row["start"]) in ensembles
    }

    rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    for start, ensemble in sorted(ensembles.items()):
        active = conformers[start]
        others = [row for row in active if row["assignment"] == "other"]
        other_occupancy = sum(float(row["occupancy"]) for row in others)
        predicted_a = float(ensemble["predicted_A_occupancy"])
        target_a = float(ensemble["target_A_occupancy"])
        target_b = float(ensemble["target_B_occupancy"])
        merged_a = predicted_a + other_occupancy
        closest_other_a = min(
            (float(row["rmsd_to_A_conventional"]) for row in others),
            default=math.nan,
        )
        closest_other_b = min(
            (float(row["rmsd_to_B_conventional"]) for row in others),
            default=math.nan,
        )
        rows.append(
            {
                "site": args.site,
                "start": start,
                "occupancy_pass": as_bool(ensemble["occupancy_accurate"]),
                "target_A_occupancy": target_a,
                "predicted_A_occupancy": predicted_a,
                "unmatched_active_count": len(others),
                "unmatched_active_occupancy": other_occupancy,
                "A_plus_unmatched_occupancy": merged_a,
                "A_absolute_error": abs(predicted_a - target_a),
                "merged_A_absolute_error": abs(merged_a - target_a),
                "merge_improves_A_error": (
                    abs(merged_a - target_a) < abs(predicted_a - target_a)
                ),
                "merged_A_within_0.20": abs(merged_a - target_a) <= 0.20,
                "closest_unmatched_rmsd_to_A": closest_other_a,
                "closest_unmatched_rmsd_to_B": closest_other_b,
            }
        )
        raw = raw_starts[start]
        occupancies = [float(value) for value in raw["occupancies"].split(";")]
        rmsd_a = [float(value) for value in raw["rmsd_to_A"].split(";")]
        rmsd_b = [float(value) for value in raw["rmsd_to_B"].split(";")]
        slot_assignments = []
        for slot, (occupancy, distance_a, distance_b) in enumerate(
            zip(occupancies, rmsd_a, rmsd_b)
        ):
            if occupancy > 0.10 and min(distance_a, distance_b) <= 1.0:
                assignment = "A" if distance_a <= distance_b else "B"
            elif occupancy > 0.05:
                assignment = "other_active"
            else:
                assignment = "inactive"
            slot_assignments.append(assignment)
            slot_rows.append(
                {
                    "site": args.site,
                    "start": start,
                    "slot": slot,
                    "occupancy": occupancy,
                    "rmsd_to_A": distance_a,
                    "rmsd_to_B": distance_b,
                    "assignment": assignment,
                }
            )
        allocation = {
            label: sum(
                occupancy
                for occupancy, assignment in zip(occupancies, slot_assignments)
                if assignment == label
            )
            for label in ("A", "B", "other_active", "inactive")
        }
        rows[-1].update(
            {
                "all_K_A_occupancy": allocation["A"],
                "all_K_B_occupancy": allocation["B"],
                "all_K_other_active_occupancy": allocation["other_active"],
                "all_K_inactive_occupancy": allocation["inactive"],
                "all_K_occupancy_sum": sum(occupancies),
                "missing_A_vs_0.42": target_a - allocation["A"],
                "excess_B_vs_0.58": allocation["B"] - target_b,
                "other_plus_inactive": (
                    allocation["other_active"] + allocation["inactive"]
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_recovered_start.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "all_K4_slots.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(slot_rows[0]))
        writer.writeheader()
        writer.writerows(slot_rows)

    with_others = [row for row in rows if row["unmatched_active_count"]]
    payload = {
        "site": args.site,
        "recovered_starts": len(rows),
        "starts_with_unmatched_active": len(with_others),
        "target_A_occupancy": (
            rows[0]["target_A_occupancy"] if rows else math.nan
        ),
        "predicted_A_occupancy": describe(
            [float(row["predicted_A_occupancy"]) for row in rows]
        ),
        "unmatched_active_occupancy": describe(
            [float(row["unmatched_active_occupancy"]) for row in with_others]
        ),
        "A_plus_unmatched_occupancy": describe(
            [float(row["A_plus_unmatched_occupancy"]) for row in with_others]
        ),
        "closest_unmatched_rmsd_to_A": describe(
            [float(row["closest_unmatched_rmsd_to_A"]) for row in with_others]
        ),
        "merge_improves_A_error": sum(
            bool(row["merge_improves_A_error"]) for row in with_others
        ),
        "merged_A_within_0.20": sum(
            bool(row["merged_A_within_0.20"]) for row in with_others
        ),
        "all_K4_allocation": {
            key: describe([float(row[key]) for row in rows])
            for key in (
                "all_K_A_occupancy",
                "all_K_B_occupancy",
                "all_K_other_active_occupancy",
                "all_K_inactive_occupancy",
                "missing_A_vs_0.42",
                "excess_B_vs_0.58",
                "other_plus_inactive",
            )
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
