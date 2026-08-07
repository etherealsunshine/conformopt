"""Measure target-density support at unmatched slots for a completed arm."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.diagnose_frozen_v3_occupancy_pooling import (
    ACTIVE_THRESHOLD,
    FOUND_THRESHOLD,
    atomic_csv,
    atomic_json,
    describe,
    identify_single_recovery,
    raw_slot_rows,
)
from scripts.diagnose_unmatched_target_density import (
    native_target_evaluator,
    safe_ratio,
)


def load_arm(root: Path):
    optimizer_rows = {}
    for path in root.glob("shards/*/*/synthetic/*_starts.csv"):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                optimizer_rows[(row["site"], int(row["start"]))] = row
    if len(optimizer_rows) != 1000:
        raise ValueError(
            f"expected 1000 optimizer rows, found {len(optimizer_rows)}"
        )

    sites = {}
    coordinates = {}
    for panel in ("original5", "expanded15"):
        audit_root = root / "audit" / panel
        payload = json.loads((audit_root / "tmol_inputs.json").read_text())
        for site in payload["sites"]:
            record = dict(site)
            record["_audit_root"] = str(audit_root)
            sites[site["site"]] = record
            for candidate in site["candidates"]:
                coordinates[(
                    site["site"],
                    int(candidate["start"]),
                    int(candidate["conformer"]),
                )] = np.asarray(candidate["coordinates"], dtype=np.float64)
    if len(sites) != 20:
        raise ValueError(f"expected 20 audited sites, found {len(sites)}")
    return optimizer_rows, sites, coordinates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    optimizer_rows, sites, coordinates = load_arm(args.arm_root)
    slots_by_start = raw_slot_rows(optimizer_rows)
    rows = []
    counts = defaultdict(int)
    for key, optimizer in sorted(optimizer_rows.items()):
        identity = identify_single_recovery(optimizer)
        if identity is None:
            continue
        counts[str(identity["recovery_rank"])] += 1
        recovered = str(identity["recovered_state"])
        start_slots = slots_by_start[key]
        recovered_candidates = [
            slot for slot in start_slots
            if float(slot["occupancy"]) > FOUND_THRESHOLD
            and slot["raw_assignment"] == recovered
        ]
        if not recovered_candidates:
            raise RuntimeError(f"missing recovered representative for {key}")
        representative = min(
            recovered_candidates,
            key=lambda slot: float(slot[f"rmsd_to_{recovered}"]),
        )
        unmatched = [
            slot for slot in start_slots
            if float(slot["occupancy"]) > ACTIVE_THRESHOLD
            and int(slot["slot"]) != int(representative["slot"])
        ]
        site = sites[key[0]]
        evaluate = native_target_evaluator(
            site, float(identity["target_A"]), float(identity["target_B"])
        )
        mean_a = float(evaluate(np.asarray(site["A"])).mean())
        mean_b = float(evaluate(np.asarray(site["B"])).mean())
        mean_ab = 0.5 * (mean_a + mean_b)
        for slot in unmatched:
            slot_number = int(slot["slot"])
            mean_slot = float(
                evaluate(coordinates[(key[0], key[1], slot_number)]).mean()
            )
            rows.append({
                "site": key[0],
                "start": key[1],
                "recovery_rank": identity["recovery_rank"],
                "recovered_state": recovered,
                "missed_state": identity["missed_state"],
                "slot": slot_number,
                "occupancy": float(slot["occupancy"]),
                "target_density_mean_at_slot_atoms": mean_slot,
                "target_density_mean_at_deposited_AB_mean": mean_ab,
                "slot_over_deposited_AB_mean": safe_ratio(
                    mean_slot, mean_ab
                ),
            })

    ratios = [float(row["slot_over_deposited_AB_mean"]) for row in rows]
    args.output.mkdir(parents=True)
    atomic_csv(args.output / "unmatched_slot_target_density.csv", rows)
    atomic_json(args.output / "summary.json", {
        "arm_root": str(args.arm_root),
        "single_recovery_counts": dict(counts),
        "unmatched_active_slots": len(rows),
        "slot_over_deposited_AB_mean": describe(ratios),
        "fraction_at_least_0p5": (
            sum(value >= 0.5 for value in ratios) / len(ratios)
            if ratios else 0.0
        ),
    })


if __name__ == "__main__":
    main()
