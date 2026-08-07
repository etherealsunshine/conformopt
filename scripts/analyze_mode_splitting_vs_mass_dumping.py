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


ACTIVE_THRESHOLD = 0.05
REPORTED_THRESHOLD = 0.10
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
        "fraction_lt_0_5A": float((array < 0.5).mean()),
        "fraction_lt_1_0A": float((array < 1.0).mean()),
        "fraction_lt_1_5A": float((array < 1.5).mean()),
        "fraction_lt_2_0A": float((array < 2.0).mean()),
    }


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


def independent_representatives(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    selected = {}
    for state in ("A", "B"):
        candidates = [
            row
            for row in rows
            if row["assignment"] == state
            and float(row["occupancy"]) > REPORTED_THRESHOLD
        ]
        if candidates:
            selected[state] = min(
                candidates,
                key=lambda row: float(row[f"rmsd_to_{state}"]),
            )
    return selected


def distribution_rows(
    rows: list[dict[str, object]],
    distance_key: str,
    population: str,
) -> list[dict[str, object]]:
    output = []
    sites = sorted({str(row["site"]) for row in rows})
    for site in ["ALL", *sites]:
        selected = rows if site == "ALL" else [
            row for row in rows if row["site"] == site
        ]
        for occupancy_bin, predicate in (
            ("0.05-0.10", lambda value: ACTIVE_THRESHOLD < value <= REPORTED_THRESHOLD),
            (">0.10", lambda value: value > REPORTED_THRESHOLD),
        ):
            distances = [
                float(row[distance_key])
                for row in selected
                if predicate(float(row["occupancy"]))
                and math.isfinite(float(row[distance_key]))
            ]
            output.append(
                {
                    "population": population,
                    "site": site,
                    "extra_occupancy_bin": occupancy_bin,
                    **describe(distances),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-table", type=Path, required=True)
    parser.add_argument("--strict-table", type=Path, action="append", required=True)
    parser.add_argument("--ensemble-table", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    slots = read_csv(args.slot_table)
    strict = [row for path in args.strict_table for row in read_csv(path)]
    ensembles = [row for path in args.ensemble_table for row in read_csv(path)]
    if len(slots) != 4000:
        raise ValueError(f"expected 4000 K=4 slot rows, found {len(slots)}")

    slots_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in slots:
        slots_by_start[(row["site"], int(row["start"]))].append(row)
    strict_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in strict:
        strict_by_start[(row["site"], int(row["start"]))].append(row)
    ensemble_by_start = {
        (row["site"], int(row["start"])): row for row in ensembles
    }
    if len(slots_by_start) != 1000 or len(ensemble_by_start) != 1000:
        raise ValueError("expected exactly 1000 starts")

    headline = set()
    for key, ensemble in ensemble_by_start.items():
        if not as_bool(ensemble["geometric_occupancy_success"]):
            continue
        pair = select_assigned_pair(strict_by_start[key])
        if pair and all(conformer_passes(row) for row in pair.values()):
            headline.add(key)
    if len(headline) != 621:
        raise ValueError(f"expected 621 headline starts, found {len(headline)}")

    failed_extra_rows: list[dict[str, object]] = []
    headline_extra_rows: list[dict[str, object]] = []
    one_miss_rows: list[dict[str, object]] = []
    failed_start_rows: list[dict[str, object]] = []

    for key, start_slots in sorted(slots_by_start.items()):
        site, start = key
        representatives = independent_representatives(start_slots)
        found_states = sorted(representatives)
        missed_states = [state for state in ("A", "B") if state not in representatives]
        selected_slots = {int(row["slot"]) for row in representatives.values()}
        active_extras = [
            row for row in start_slots
            if float(row["occupancy"]) > ACTIVE_THRESHOLD
            and int(row["slot"]) not in selected_slots
        ]

        if missed_states:
            missed_label = "".join(missed_states)
            recovered_states = [state for state in ("A", "B") if state in representatives]
            failed_start_rows.append(
                {
                    "site": site,
                    "start": start,
                    "missed_state": (
                        missed_states[0] if len(missed_states) == 1 else "both"
                    ),
                    "found_states": "".join(found_states) or "none",
                    "active_extras": len(active_extras),
                    "extras_above_0_10": sum(
                        float(row["occupancy"]) > REPORTED_THRESHOLD
                        for row in active_extras
                    ),
                }
            )
            for row in active_extras:
                rmsd_a = float(row["rmsd_to_A"])
                rmsd_b = float(row["rmsd_to_B"])
                missed_distances = {
                    state: rmsd_a if state == "A" else rmsd_b
                    for state in missed_states
                }
                recovered_distances = {
                    state: rmsd_a if state == "A" else rmsd_b
                    for state in recovered_states
                }
                failed_extra_rows.append(
                    {
                        "site": site,
                        "start": start,
                        "missed_state": (
                            missed_states[0] if len(missed_states) == 1 else "both"
                        ),
                        "recovered_state": (
                            recovered_states[0]
                            if len(recovered_states) == 1 else "none"
                        ),
                        "slot": int(row["slot"]),
                        "assignment": row["assignment"],
                        "occupancy": float(row["occupancy"]),
                        "occupancy_bin": (
                            ">0.10"
                            if float(row["occupancy"]) > REPORTED_THRESHOLD
                            else "0.05-0.10"
                        ),
                        "rmsd_to_A": rmsd_a,
                        "rmsd_to_B": rmsd_b,
                        "rmsd_to_missed_A": (
                            rmsd_a if "A" in missed_states else float("nan")
                        ),
                        "rmsd_to_missed_B": (
                            rmsd_b if "B" in missed_states else float("nan")
                        ),
                        "rmsd_to_nearest_missed_state": min(
                            missed_distances.values()
                        ),
                        "rmsd_to_recovered_state": (
                            min(recovered_distances.values())
                            if recovered_distances else float("nan")
                        ),
                        "rmsd_to_nearest_deposited_state": min(rmsd_a, rmsd_b),
                    }
                )

            if len(missed_states) == 1:
                recovered_state = recovered_states[0]
                recovered_occupancy = float(
                    representatives[recovered_state]["occupancy"]
                )
                extra_mass = sum(float(row["occupancy"]) for row in active_extras)
                near_missed_mass = sum(
                    float(row["occupancy"])
                    for row in active_extras
                    if float(row[f"rmsd_to_{missed_states[0]}"]) < 1.0
                )
                submask_mass = sum(
                    float(row["occupancy"])
                    for row in start_slots
                    if float(row["occupancy"]) <= ACTIVE_THRESHOLD
                )
                target_sum = float(start_slots[0]["target_AB_occupancy_sum"])
                one_miss_rows.append(
                    {
                        "site": site,
                        "start": start,
                        "missed_state": missed_states[0],
                        "recovered_state": recovered_state,
                        "recovered_occupancy": recovered_occupancy,
                        "all_active_extra_occupancy": extra_mass,
                        "high_extra_occupancy": sum(
                            float(row["occupancy"])
                            for row in active_extras
                            if float(row["occupancy"]) > REPORTED_THRESHOLD
                        ),
                        "near_missed_extra_occupancy_rmsd_lt_1A": near_missed_mass,
                        "submask_occupancy": submask_mass,
                        "recovered_plus_all_extras": recovered_occupancy + extra_mass,
                        "target_AB_occupancy_sum": target_sum,
                        "all_extra_mass_absolute_error": abs(
                            target_sum - recovered_occupancy - extra_mass
                        ),
                        "recovered_plus_near_missed_extra": (
                            recovered_occupancy + near_missed_mass
                        ),
                        "near_missed_mass_absolute_error": abs(
                            target_sum - recovered_occupancy - near_missed_mass
                        ),
                        "has_high_extra_near_missed_lt_1A": any(
                            float(row["occupancy"]) > REPORTED_THRESHOLD
                            and float(row[f"rmsd_to_{missed_states[0]}"]) < 1.0
                            for row in active_extras
                        ),
                    }
                )

        if key in headline:
            pair = select_assigned_pair(strict_by_start[key])
            if pair is None:
                raise RuntimeError(f"headline start lacks pair: {key}")
            pair_candidate_ids = {
                pair["A"]["candidate_id"], pair["B"]["candidate_id"]
            }
            pair_slots = {
                int(row["conformer"])
                for row in strict_by_start[key]
                if row["candidate_id"] in pair_candidate_ids
            }
            comparison_extras = [
                row for row in start_slots
                if float(row["occupancy"]) > ACTIVE_THRESHOLD
                and int(row["slot"]) not in pair_slots
            ]
            if any(
                float(row["occupancy"]) > REPORTED_THRESHOLD
                for row in comparison_extras
            ):
                for row in comparison_extras:
                    rmsd_a = float(row["rmsd_to_A"])
                    rmsd_b = float(row["rmsd_to_B"])
                    headline_extra_rows.append(
                        {
                            "site": site,
                            "start": start,
                            "slot": int(row["slot"]),
                            "assignment": row["assignment"],
                            "occupancy": float(row["occupancy"]),
                            "occupancy_bin": (
                                ">0.10"
                                if float(row["occupancy"]) > REPORTED_THRESHOLD
                                else "0.05-0.10"
                            ),
                            "rmsd_to_A": rmsd_a,
                            "rmsd_to_B": rmsd_b,
                            "rmsd_to_nearest_deposited_state": min(rmsd_a, rmsd_b),
                        }
                    )

    failed_distributions = distribution_rows(
        failed_extra_rows,
        "rmsd_to_nearest_missed_state",
        "failed_recovery",
    )
    headline_distributions = distribution_rows(
        headline_extra_rows,
        "rmsd_to_nearest_deposited_state",
        "headline_with_high_extra",
    )

    exact_one = [row for row in failed_start_rows if row["missed_state"] != "both"]
    both_missed = [row for row in failed_start_rows if row["missed_state"] == "both"]
    headline_high_keys = {
        (str(row["site"]), int(row["start"]))
        for row in headline_extra_rows
        if float(row["occupancy"]) > REPORTED_THRESHOLD
    }
    high_failed = [
        row for row in failed_extra_rows
        if float(row["occupancy"]) > REPORTED_THRESHOLD
    ]
    low_failed = [
        row for row in failed_extra_rows
        if float(row["occupancy"]) <= REPORTED_THRESHOLD
    ]
    high_headline = [
        row for row in headline_extra_rows
        if float(row["occupancy"]) > REPORTED_THRESHOLD
    ]
    high_exact_one = [
        row for row in high_failed if row["missed_state"] in {"A", "B"}
    ]
    high_both_missed = [
        row for row in high_failed if row["missed_state"] == "both"
    ]
    low_exact_one = [
        row for row in low_failed if row["missed_state"] in {"A", "B"}
    ]
    exact_one_high_keys = {
        (str(row["site"]), int(row["start"])) for row in high_exact_one
    }
    exact_one_high_near_keys = {
        (str(row["site"]), int(row["start"]))
        for row in high_exact_one
        if float(row["rmsd_to_nearest_missed_state"]) < 1.0
    }

    one_miss_all_error = [
        float(row["all_extra_mass_absolute_error"]) for row in one_miss_rows
    ]
    one_miss_near_error = [
        float(row["near_missed_mass_absolute_error"]) for row in one_miss_rows
    ]
    summary = {
        "definitions": {
            "active_extra": "occupancy >0.05 outside independently selected recovered representatives",
            "recovered_representative": "best RMSD assignment with occupancy >0.10, selected independently for A and B",
            "high_extra": "occupancy >0.10",
            "mode_splitting_neighborhood": "RMSD to missed deposited state <1.0 A",
        },
        "headline_starts": len(headline),
        "failed_recovery_starts": len(failed_start_rows),
        "failed_exactly_one_state": len(exact_one),
        "failed_both_states": len(both_missed),
        "missed_state_counts": {
            state: sum(row["missed_state"] == state for row in failed_start_rows)
            for state in ("A", "B", "both")
        },
        "failed_starts_with_any_extra": sum(
            int(row["active_extras"]) > 0 for row in failed_start_rows
        ),
        "failed_starts_with_high_extra": sum(
            int(row["extras_above_0_10"]) > 0 for row in failed_start_rows
        ),
        "failed_extra_conformers": len(failed_extra_rows),
        "failed_high_extra_rmsd_to_missed": describe(
            [float(row["rmsd_to_nearest_missed_state"]) for row in high_failed]
        ),
        "failed_low_extra_rmsd_to_missed": describe(
            [float(row["rmsd_to_nearest_missed_state"]) for row in low_failed]
        ),
        "failed_exactly_one_state_high_extras": {
            "conformers": len(high_exact_one),
            "starts": len(exact_one_high_keys),
            "starts_with_any_high_extra_near_missed_lt_1A": len(
                exact_one_high_near_keys
            ),
            "rmsd_to_missed": describe(
                [
                    float(row["rmsd_to_nearest_missed_state"])
                    for row in high_exact_one
                ]
            ),
            "rmsd_to_recovered": describe(
                [
                    float(row["rmsd_to_recovered_state"])
                    for row in high_exact_one
                ]
            ),
            "fraction_closer_to_recovered_than_missed": (
                float(np.mean([
                    float(row["rmsd_to_recovered_state"])
                    < float(row["rmsd_to_nearest_missed_state"])
                    for row in high_exact_one
                ]))
                if high_exact_one else None
            ),
        },
        "failed_exactly_one_state_low_extras": {
            "conformers": len(low_exact_one),
            "rmsd_to_missed": describe(
                [
                    float(row["rmsd_to_nearest_missed_state"])
                    for row in low_exact_one
                ]
            ),
            "rmsd_to_recovered": describe(
                [
                    float(row["rmsd_to_recovered_state"])
                    for row in low_exact_one
                ]
            ),
        },
        "failed_both_states_high_extras": {
            "conformers": len(high_both_missed),
            "rmsd_to_nearest_missed": describe(
                [
                    float(row["rmsd_to_nearest_missed_state"])
                    for row in high_both_missed
                ]
            ),
        },
        "headline_starts_with_high_extra": len(headline_high_keys),
        "headline_high_extra_conformers": len(high_headline),
        "headline_high_extra_rmsd_to_nearest_deposited": describe(
            [
                float(row["rmsd_to_nearest_deposited_state"])
                for row in high_headline
            ]
        ),
        "one_missed_state_mass_accounting": {
            "starts": len(one_miss_rows),
            "recovered_plus_all_active_extras_absolute_error": describe(
                one_miss_all_error
            ),
            "fraction_within_0_05": float(
                (np.asarray(one_miss_all_error) <= 0.05).mean()
            ) if one_miss_all_error else None,
            "fraction_within_0_10": float(
                (np.asarray(one_miss_all_error) <= 0.10).mean()
            ) if one_miss_all_error else None,
            "fraction_within_0_20": float(
                (np.asarray(one_miss_all_error) <= 0.20).mean()
            ) if one_miss_all_error else None,
            "recovered_plus_near_missed_extra_absolute_error": describe(
                one_miss_near_error
            ),
            "starts_with_high_extra_near_missed_lt_1A": sum(
                bool(row["has_high_extra_near_missed_lt_1A"])
                for row in one_miss_rows
            ),
        },
    }

    per_site_rows = []
    for site in sorted({row["site"] for row in failed_start_rows}):
        starts = [row for row in failed_start_rows if row["site"] == site]
        extras = [row for row in failed_extra_rows if row["site"] == site]
        high = [
            row for row in extras
            if float(row["occupancy"]) > REPORTED_THRESHOLD
        ]
        per_site_rows.append(
            {
                "site": site,
                "failed_recovery_starts": len(starts),
                "missed_A": sum(row["missed_state"] == "A" for row in starts),
                "missed_B": sum(row["missed_state"] == "B" for row in starts),
                "missed_both": sum(row["missed_state"] == "both" for row in starts),
                "starts_with_high_extra": sum(
                    int(row["extras_above_0_10"]) > 0 for row in starts
                ),
                "high_extras": len(high),
                **{
                    f"high_extra_rmsd_to_missed_{key}": value
                    for key, value in describe(
                        [
                            float(row["rmsd_to_nearest_missed_state"])
                            for row in high
                        ]
                    ).items()
                },
            }
        )

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "failed_recovery_starts.csv", failed_start_rows)
    atomic_csv(
        args.output / "failed_recovery_extra_conformers.csv",
        failed_extra_rows,
    )
    atomic_csv(
        args.output / "failed_recovery_rmsd_distributions.csv",
        failed_distributions,
    )
    atomic_csv(
        args.output / "headline_high_extra_conformers.csv",
        headline_extra_rows,
    )
    atomic_csv(
        args.output / "headline_comparison_rmsd_distributions.csv",
        headline_distributions,
    )
    atomic_csv(
        args.output / "one_missed_state_mass_accounting.csv",
        one_miss_rows,
    )
    atomic_csv(args.output / "per_site_summary.csv", per_site_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
