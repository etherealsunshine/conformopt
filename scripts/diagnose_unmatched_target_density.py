"""Measure frozen-target density at unmatched active conformer positions.

This is a read-only prerequisite diagnostic for the merge-and-respawn
experiment.  It preserves the historical raw-greedy 142 major-only / 45
minor-only partition and evaluates the exact native Gaussian-mixture target
used by the synthetic optimizer at saved endpoint atom positions.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable

import gemmi
import numpy as np

from scripts.diagnose_frozen_v3_occupancy_pooling import (
    ACTIVE_THRESHOLD,
    FOUND_THRESHOLD,
    atomic_csv,
    atomic_json,
    describe,
    identify_single_recovery,
    kinematic_control_coordinates,
    load_optimizer_rows,
    load_v3_payload,
    raw_slot_rows,
)


def native_target_evaluator(
    site: dict[str, object],
    target_a: float,
    target_b: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return target density at arbitrary points for one frozen site."""
    audit_root = Path(str(site["_audit_root"]))
    structure = gemmi.read_structure(
        str(audit_root / str(site["base_pdb_B"]))
    )
    residue = next(
        residue
        for chain in structure[0]
        if chain.name == site["chain"]
        for residue in chain
        if residue.seqid.num == int(site["residue_number"])
    )
    atom_lookup = {atom.name.strip(): atom for atom in residue}
    names = list(site["atom_names"])
    sigma2 = np.asarray(
        [
            max(
                float(atom_lookup[name].b_iso) / (8.0 * math.pi**2),
                0.04,
            )
            for name in names
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [float(atom_lookup[name].element.atomic_number) for name in names],
        dtype=np.float64,
    )
    normalization = np.power(2.0 * math.pi * sigma2, -1.5)
    control_a, control_b = kinematic_control_coordinates(site, audit_root)

    def conformer_density(
        points: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        distance2 = np.square(
            np.asarray(points, dtype=np.float64)[:, None, :]
            - coordinates[None, :, :]
        ).sum(axis=-1)
        return (
            weights[None, :]
            * normalization[None, :]
            * np.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(axis=1)

    def evaluate(points: np.ndarray) -> np.ndarray:
        return (
            target_a * conformer_density(points, control_a)
            + target_b * conformer_density(points, control_b)
        )

    return evaluate


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    metric_root = (
        args.frozen_root
        / "analysis"
        / "metric_v3_protected_merge_sweep"
        / "0p5"
    )
    optimizer_rows, _target_paths = load_optimizer_rows(
        args.baseline_root, args.replacement_root
    )
    (
        _active_by_slot,
        coordinates,
        sites,
        _ensembles,
        _strict_by_start,
    ) = load_v3_payload(metric_root)
    slots_by_start = raw_slot_rows(optimizer_rows)

    rows: list[dict[str, object]] = []
    start_rows: list[dict[str, object]] = []
    counts: dict[str, int] = defaultdict(int)
    site_references: dict[str, dict[str, float]] = {}

    for key, optimizer in sorted(optimizer_rows.items()):
        identity = identify_single_recovery(optimizer)
        if identity is None:
            continue
        rank = str(identity["recovery_rank"])
        counts[rank] += 1
        recovered = str(identity["recovered_state"])
        start_slots = slots_by_start[key]
        recovered_candidates = [
            slot
            for slot in start_slots
            if float(slot["occupancy"]) > FOUND_THRESHOLD
            and slot["raw_assignment"] == recovered
        ]
        if not recovered_candidates:
            raise RuntimeError(f"missing recovered representative for {key}")
        representative = min(
            recovered_candidates,
            key=lambda slot: float(slot[f"rmsd_to_{recovered}"]),
        )
        representative_slot = int(representative["slot"])
        unmatched = [
            slot
            for slot in start_slots
            if float(slot["occupancy"]) > ACTIVE_THRESHOLD
            and int(slot["slot"]) != representative_slot
        ]

        site = sites[key[0]]
        evaluate = native_target_evaluator(
            site,
            float(identity["target_A"]),
            float(identity["target_B"]),
        )
        deposited_a = np.asarray(site["A"], dtype=np.float64)
        deposited_b = np.asarray(site["B"], dtype=np.float64)
        mean_a = float(evaluate(deposited_a).mean())
        mean_b = float(evaluate(deposited_b).mean())
        mean_ab = 0.5 * (mean_a + mean_b)
        lower_ab = min(mean_a, mean_b)
        site_references.setdefault(
            key[0],
            {
                "target_density_mean_at_deposited_A": mean_a,
                "target_density_mean_at_deposited_B": mean_b,
                "target_density_mean_at_deposited_AB_mean": mean_ab,
                "target_density_mean_at_deposited_AB_lower": lower_ab,
            },
        )

        start_slot_means: list[float] = []
        start_occupancies: list[float] = []
        for slot in unmatched:
            slot_number = int(slot["slot"])
            mean_slot = float(
                evaluate(coordinates[(key[0], key[1], slot_number)]).mean()
            )
            occupancy = float(slot["occupancy"])
            start_slot_means.append(mean_slot)
            start_occupancies.append(occupancy)
            rows.append(
                {
                    "site": key[0],
                    "start": key[1],
                    "recovery_rank": rank,
                    "recovered_state": recovered,
                    "missed_state": identity["missed_state"],
                    "slot": slot_number,
                    "occupancy": occupancy,
                    "target_density_mean_at_slot_atoms": mean_slot,
                    "target_density_mean_at_deposited_A": mean_a,
                    "target_density_mean_at_deposited_B": mean_b,
                    "target_density_mean_at_deposited_AB_mean": mean_ab,
                    "target_density_mean_at_deposited_AB_lower": lower_ab,
                    "slot_over_deposited_A": safe_ratio(mean_slot, mean_a),
                    "slot_over_deposited_B": safe_ratio(mean_slot, mean_b),
                    "slot_over_deposited_AB_mean": safe_ratio(
                        mean_slot, mean_ab
                    ),
                    "slot_over_deposited_AB_lower": safe_ratio(
                        mean_slot, lower_ab
                    ),
                }
            )

        occupancy_sum = sum(start_occupancies)
        weighted_mean = (
            float(np.average(start_slot_means, weights=start_occupancies))
            if occupancy_sum > 0.0
            else math.nan
        )
        unweighted_mean = (
            float(np.mean(start_slot_means))
            if start_slot_means
            else math.nan
        )
        start_rows.append(
            {
                "site": key[0],
                "start": key[1],
                "recovery_rank": rank,
                "unmatched_active_slot_count": len(unmatched),
                "unmatched_active_occupancy": occupancy_sum,
                "unmatched_slot_density_mean_unweighted": unweighted_mean,
                "unmatched_slot_density_mean_occupancy_weighted": weighted_mean,
                "deposited_AB_density_mean": mean_ab,
                "unweighted_slot_over_deposited_AB_mean": safe_ratio(
                    unweighted_mean, mean_ab
                ),
                "occupancy_weighted_slot_over_deposited_AB_mean": safe_ratio(
                    weighted_mean, mean_ab
                ),
            }
        )

    if counts != {"major_only": 142, "minor_only": 45}:
        raise RuntimeError(f"unexpected single-recovery counts: {dict(counts)}")
    if len(rows) != 259:
        raise RuntimeError(f"expected 259 unmatched active slots, got {len(rows)}")

    ratio_values = [
        float(row["slot_over_deposited_AB_mean"]) for row in rows
    ]
    summary = {
        "provenance": {
            "metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
            "partition": "historical raw-greedy single-recovery partition",
            "target": (
                "exact native Gaussian A/B mixture used to construct the "
                "saved normalized synthetic optimizer target"
            ),
            "sampling": (
                "target evaluated at every saved unmatched active slot atom; "
                "deposited references evaluated at deposited A/B atoms"
            ),
            "active_threshold": ACTIVE_THRESHOLD,
            "control_rerun": False,
        },
        "counts": {
            **counts,
            "single_recovery_starts": len(start_rows),
            "unmatched_active_slots": len(rows),
        },
        "unmatched_slot_target_density": describe(
            float(row["target_density_mean_at_slot_atoms"]) for row in rows
        ),
        "slot_over_deposited_A": describe(
            float(row["slot_over_deposited_A"]) for row in rows
        ),
        "slot_over_deposited_B": describe(
            float(row["slot_over_deposited_B"]) for row in rows
        ),
        "slot_over_deposited_AB_mean": describe(ratio_values),
        "slot_over_deposited_AB_lower": describe(
            float(row["slot_over_deposited_AB_lower"]) for row in rows
        ),
        "ratio_threshold_counts": {
            "below_0p05": sum(value < 0.05 for value in ratio_values),
            "below_0p10": sum(value < 0.10 for value in ratio_values),
            "at_least_0p25": sum(value >= 0.25 for value in ratio_values),
            "at_least_0p50": sum(value >= 0.50 for value in ratio_values),
            "at_least_0p75": sum(value >= 0.75 for value in ratio_values),
        },
        "per_recovery_rank": {
            rank: {
                "slots": sum(row["recovery_rank"] == rank for row in rows),
                "slot_over_deposited_AB_mean": describe(
                    float(row["slot_over_deposited_AB_mean"])
                    for row in rows
                    if row["recovery_rank"] == rank
                ),
            }
            for rank in ("major_only", "minor_only")
        },
        "per_site": {
            site_key: {
                "starts": sum(
                    row["site"] == site_key for row in start_rows
                ),
                "slots": sum(row["site"] == site_key for row in rows),
                **site_references[site_key],
                "slot_over_deposited_AB_mean": describe(
                    float(row["slot_over_deposited_AB_mean"])
                    for row in rows
                    if row["site"] == site_key
                ),
            }
            for site_key in sorted(
                {str(row["site"]) for row in start_rows}
            )
        },
    }

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "unmatched_slot_target_density.csv", rows)
    atomic_csv(args.output / "single_recovery_start_density.csv", start_rows)
    atomic_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
