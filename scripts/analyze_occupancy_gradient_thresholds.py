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

from density_denoiser.summarize_endpoint_audit import as_bool, select_assigned_pair


THRESHOLDS = (0.05, 0.075, 0.10, 0.15)
ZERO_EPSILON = 1.0e-12
TMOL_TOLERANCE = 0.44


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def auc(positive: list[float], negative: list[float]) -> float | None:
    if not positive or not negative:
        return None
    left = np.asarray(positive, dtype=float)
    right = np.asarray(negative, dtype=float)
    ranks = average_ranks(np.concatenate([left, right]))
    rank_sum = ranks[: len(left)].sum()
    u = rank_sum - len(left) * (len(left) + 1) / 2.0
    return float(u / (len(left) * len(right)))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, newline=""
    ) as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def sign_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    values = [float(row["grad_occ"]) for row in rows]
    negative = sum(value < -ZERO_EPSILON for value in values)
    zero = sum(abs(value) <= ZERO_EPSILON for value in values)
    positive = sum(value > ZERO_EPSILON for value in values)
    return {
        **describe(values),
        "negative": negative,
        "zero_abs_le_1e_12": zero,
        "positive": positive,
        "negative_fraction": negative / len(values) if values else None,
        "zero_fraction": zero / len(values) if values else None,
        "positive_fraction": positive / len(values) if values else None,
    }


def conformer_passes(row: dict[str, str]) -> bool:
    if not as_bool(row["rotamer_within_allowed_width"]):
        return False
    if not as_bool(row["no_direct_clash"]):
        return False
    if not as_bool(row["no_symmetry_clash"]):
        return False
    if row["assignment"] not in {"A", "B"}:
        return False
    try:
        margin = float(row["tmol_delta_vs_matched_AB"])
    except ValueError:
        return False
    return math.isfinite(margin) and margin <= TMOL_TOLERANCE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--strict-table", type=Path, action="append", required=True)
    parser.add_argument("--ensemble-table", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    derivatives = read_csv(
        args.analysis_root / "active_conformer_density_derivatives.csv"
    )
    slots = read_csv(args.analysis_root / "k4_slot_occupancies.csv")
    starts = read_csv(args.analysis_root / "k4_start_occupancy_summary.csv")
    strict = [row for path in args.strict_table for row in read_csv(path)]
    ensembles = [row for path in args.ensemble_table for row in read_csv(path)]

    if len(slots) != 4000 or len(starts) != 1000:
        raise ValueError(
            f"expected 4000 slots and 1000 starts, got {len(slots)} and {len(starts)}"
        )
    if len({(row["site"], row["start"]) for row in ensembles}) != 1000:
        raise ValueError("ensemble tables do not contain exactly 1000 unique starts")

    # Signed occupancy-gradient distributions.
    signed_rows: list[dict[str, object]] = []
    for site in ["ALL", *sorted({row["site"] for row in derivatives})]:
        selected = (
            derivatives if site == "ALL"
            else [row for row in derivatives if row["site"] == site]
        )
        for population in ("matched", "extra"):
            population_rows = [
                row for row in selected if row["population"] == population
            ]
            signed_rows.append(
                {
                    "site": site,
                    "population": population,
                    **sign_summary(population_rows),
                }
            )

    # Quantify mask proximity and active-slot KKT imbalance.
    extras = [row for row in derivatives if row["population"] == "extra"]
    matched = [row for row in derivatives if row["population"] == "matched"]
    by_active_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in derivatives:
        by_active_start[(row["site"], int(row["start"]))].append(row)
    kkt_ranges = [
        max(float(row["grad_occ"]) for row in rows)
        - min(float(row["grad_occ"]) for row in rows)
        for rows in by_active_start.values()
        if len(rows) >= 2
    ]
    mask_proximity = {
        "extra_occupancy": describe([float(row["occupancy"]) for row in extras]),
        "matched_occupancy": describe([float(row["occupancy"]) for row in matched]),
        "extra_counts": {
            "0.05_to_0.055": sum(
                0.05 < float(row["occupancy"]) <= 0.055 for row in extras
            ),
            "0.05_to_0.06": sum(
                0.05 < float(row["occupancy"]) <= 0.06 for row in extras
            ),
            "0.05_to_0.075": sum(
                0.05 < float(row["occupancy"]) <= 0.075 for row in extras
            ),
            "0.05_to_0.10": sum(
                0.05 < float(row["occupancy"]) <= 0.10 for row in extras
            ),
        },
        "active_slot_grad_occ_range_within_start": describe(kkt_ranges),
        "active_starts_with_at_least_two_slots": len(kkt_ranges),
    }

    strict_by_slot = {
        (row["site"], int(row["start"]), int(row["conformer"])): row
        for row in strict
    }
    strict_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in strict:
        strict_by_start[(row["site"], int(row["start"]))].append(row)
    ensemble_by_start = {
        (row["site"], int(row["start"])): row for row in ensembles
    }
    slots_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in slots:
        slots_by_start[(row["site"], int(row["start"]))].append(row)

    threshold_rows: list[dict[str, object]] = []
    baseline_extras = [
        row for row in slots if row["population"] == "extra"
    ]
    baseline_matched = [
        row for row in slots if row["population"] == "matched"
    ]
    headline_keys = set()
    for key, ensemble in ensemble_by_start.items():
        if not as_bool(ensemble["geometric_occupancy_success"]):
            continue
        pair = select_assigned_pair(strict_by_start[key])
        if pair and all(conformer_passes(row) for row in pair.values()):
            headline_keys.add(key)
    if len(headline_keys) != 621:
        raise ValueError(
            f"expected 621 frozen headline starts, found {len(headline_keys)}"
        )
    headline_extras = [
        row for row in baseline_extras
        if (row["site"], int(row["start"])) in headline_keys
    ]
    headline_matched = [
        row for row in baseline_matched
        if (row["site"], int(row["start"])) in headline_keys
    ]
    for threshold in THRESHOLDS:
        extra_remaining = [
            row for row in baseline_extras
            if float(row["occupancy"]) > threshold
        ]
        matched_remaining = [
            row for row in baseline_matched
            if float(row["occupancy"]) > threshold
        ]
        extra_bearing = {
            (row["site"], int(row["start"])) for row in extra_remaining
        }
        headline_extra_remaining = [
            row for row in headline_extras
            if float(row["occupancy"]) > threshold
        ]
        headline_matched_remaining = [
            row for row in headline_matched
            if float(row["occupancy"]) > threshold
        ]
        headline_extra_bearing = {
            (row["site"], int(row["start"]))
            for row in headline_extra_remaining
        }
        all_active_success = 0
        pair_available = 0
        for key, start_slots in slots_by_start.items():
            ensemble = ensemble_by_start[key]
            active = [
                row for row in start_slots
                if float(row["occupancy"]) > threshold
            ]
            selected = [
                row for row in start_slots
                if as_bool(row["selected_assigned_pair"])
            ]
            selected_active = all(
                float(row["occupancy"]) > threshold for row in selected
            ) and len(selected) == 2
            if selected_active:
                pair_available += 1
            if not (
                as_bool(ensemble["geometric_occupancy_success"])
                and selected_active
                and active
            ):
                continue
            if all(
                conformer_passes(
                    strict_by_slot[
                        (key[0], key[1], int(row["slot"]))
                    ]
                )
                for row in active
            ):
                all_active_success += 1
        threshold_rows.append(
            {
                "activity_threshold": threshold,
                "extras_baseline": len(baseline_extras),
                "extras_eliminated": len(baseline_extras) - len(extra_remaining),
                "extras_remaining": len(extra_remaining),
                "matched_baseline": len(baseline_matched),
                "matched_conformers_lost": (
                    len(baseline_matched) - len(matched_remaining)
                ),
                "matched_conformers_remaining": len(matched_remaining),
                "extra_bearing_starts": len(extra_bearing),
                "selected_pairs_remaining": pair_available,
                "all_active_composite_tmol_0_44": all_active_success,
                "headline_starts_baseline": len(headline_keys),
                "headline_extras_baseline": len(headline_extras),
                "headline_extras_eliminated": (
                    len(headline_extras) - len(headline_extra_remaining)
                ),
                "headline_extras_remaining": len(headline_extra_remaining),
                "headline_matched_baseline": len(headline_matched),
                "headline_matched_lost": (
                    len(headline_matched) - len(headline_matched_remaining)
                ),
                "headline_extra_bearing_starts": len(headline_extra_bearing),
            }
        )

    # Occupancy-gate failures versus the structural extra + sub-mask mass.
    start_lookup = {
        (row["site"], int(row["start"])): row for row in starts
    }
    occupancy_pass_mass: list[float] = []
    occupancy_fail_mass: list[float] = []
    occupancy_fail_rows: list[dict[str, object]] = []
    for key, ensemble in ensemble_by_start.items():
        if not as_bool(ensemble["both_found_conventional"]):
            continue
        row = start_lookup[key]
        mass = (
            float(row["extra_active_occupancy_mass"])
            + float(row["submask_occupancy_mass"])
        )
        if as_bool(ensemble["geometric_occupancy_success"]):
            occupancy_pass_mass.append(mass)
        else:
            occupancy_fail_mass.append(mass)
            occupancy_fail_rows.append(
                {
                    "site": key[0],
                    "start": key[1],
                    "extra_active_occupancy_mass": float(
                        row["extra_active_occupancy_mass"]
                    ),
                    "submask_occupancy_mass": float(
                        row["submask_occupancy_mass"]
                    ),
                    "extra_plus_submask_mass": mass,
                    "matched_occupancy_sum": float(
                        row["matched_occupancy_sum"]
                    ),
                    "target_AB_occupancy_sum": float(
                        row["target_AB_occupancy_sum"]
                    ),
                    "matched_deficit": -float(
                        row["matched_minus_target_AB_sum"]
                    ),
                }
            )
    occupancy_gate = {
        "both_found_starts": len(occupancy_pass_mass) + len(occupancy_fail_mass),
        "occupancy_pass_starts": len(occupancy_pass_mass),
        "occupancy_fail_starts": len(occupancy_fail_mass),
        "extra_plus_submask_mass_pass": describe(occupancy_pass_mass),
        "extra_plus_submask_mass_fail": describe(occupancy_fail_mass),
        "auc_higher_mass_predicts_occupancy_failure": auc(
            occupancy_fail_mass, occupancy_pass_mass
        ),
    }

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "signed_grad_occ_by_site.csv", signed_rows)
    atomic_csv(args.output / "activity_threshold_sweep.csv", threshold_rows)
    atomic_csv(args.output / "occupancy_gate_failures.csv", occupancy_fail_rows)
    atomic_json(
        args.output / "summary.json",
        {
            "definitions": {
                "signed_gradient": "post-softmax ambient partial dL_density/docc_k",
                "positive_sign": "adding occupancy increases density MSE",
                "zero_epsilon": ZERO_EPSILON,
                "activity_rule": "occupancy > threshold",
                "tmol_tolerance": TMOL_TOLERANCE,
            },
            "signed_gradient_overall": {
                row["population"]: row
                for row in signed_rows if row["site"] == "ALL"
            },
            "mask_proximity_and_kkt": mask_proximity,
            "activity_threshold_sweep": threshold_rows,
            "occupancy_gate": occupancy_gate,
        },
    )
    print(
        json.dumps(
            {
                "signed_gradient_overall": {
                    row["population"]: row
                    for row in signed_rows if row["site"] == "ALL"
                },
                "mask_proximity_and_kkt": mask_proximity,
                "activity_threshold_sweep": threshold_rows,
                "occupancy_gate": occupancy_gate,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
