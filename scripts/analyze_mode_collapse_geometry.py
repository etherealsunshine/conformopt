from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from density_denoiser.residue_geometry import symmetry_aware_rmsd


DOMINANT_SITES = {
    "1ZV8_E_ASN1",
    "2V05_A_HIS168",
    "7UO8_A_GLN53",
    "4C16_A_MET258",
}


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


def conventional_rmsd(
    left: np.ndarray,
    right: np.ndarray,
    names: list[str],
    resname: str,
) -> float:
    return float(
        symmetry_aware_rmsd(
            torch.tensor(left, dtype=torch.float32),
            torch.tensor(right, dtype=torch.float32),
            names,
            resname,
        )
    )


def load_coordinates(
    paths: list[Path],
) -> tuple[dict[str, dict[str, object]], dict[tuple[str, int, int], np.ndarray]]:
    sites = {}
    candidates = {}
    for path in paths:
        payload = json.loads(path.read_text())
        for site in payload["sites"]:
            key = site["site"]
            if key in sites:
                raise ValueError(f"duplicate tmol-input site {key}")
            sites[key] = site
            for candidate in site["candidates"]:
                candidates[
                    (
                        key,
                        int(candidate["start"]),
                        int(candidate["conformer"]),
                    )
                ] = np.asarray(candidate["coordinates"], dtype=np.float32)
    if len(sites) != 20:
        raise ValueError(f"expected 20 tmol-input sites, found {len(sites)}")
    return sites, candidates


def independent_representatives(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    output = {}
    for state in ("A", "B"):
        candidates = [
            row
            for row in rows
            if row["assignment"] == state and float(row["occupancy"]) > 0.10
        ]
        if candidates:
            output[state] = min(
                candidates, key=lambda row: float(row[f"rmsd_to_{state}"])
            )
    return output


def optimal_pair(rows: list[dict[str, str]]) -> dict[str, object] | None:
    active = [row for row in rows if float(row["occupancy"]) > 0.10]
    options = []
    for row_a, row_b in itertools.permutations(active, 2):
        rmsd_a = float(row_a["rmsd_to_A"])
        rmsd_b = float(row_b["rmsd_to_B"])
        if rmsd_a < 1.0 and rmsd_b < 1.0:
            options.append((rmsd_a + rmsd_b, row_a, row_b, rmsd_a, rmsd_b))
    if not options:
        return None
    total, row_a, row_b, rmsd_a, rmsd_b = min(options, key=lambda item: item[0])
    return {
        "slot_A": int(row_a["slot"]),
        "slot_B": int(row_b["slot"]),
        "rmsd_A": rmsd_a,
        "rmsd_B": rmsd_b,
        "total_rmsd": total,
    }


def collapse_category(row: dict[str, object]) -> str:
    recovered = float(row["rmsd_to_recovered_state"])
    missed = float(row["rmsd_to_nearest_missed_state"])
    if recovered < 1.0 and missed < 1.0:
        return "near_both"
    if recovered < 1.0:
        return "near_recovered_only"
    if missed < 1.0:
        return "near_missed_only"
    return "far_from_both"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-analysis", type=Path, required=True)
    parser.add_argument("--slot-table", type=Path, required=True)
    parser.add_argument("--tmol-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    failed_extras = read_csv(
        args.previous_analysis / "failed_recovery_extra_conformers.csv"
    )
    failed_starts = read_csv(
        args.previous_analysis / "failed_recovery_starts.csv"
    )
    slots = read_csv(args.slot_table)
    slots_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in slots:
        slots_by_start[(row["site"], int(row["start"]))].append(row)
    site_data, coordinates = load_coordinates(args.tmol_input)

    separation_rows = []
    separations = {}
    for site, record in sorted(site_data.items()):
        value = conventional_rmsd(
            np.asarray(record["A"], dtype=np.float32),
            np.asarray(record["B"], dtype=np.float32),
            list(record["atom_names"]),
            str(record["residue_type"]),
        )
        separations[site] = value
        separation_rows.append(
            {
                "site": site,
                "residue_type": record["residue_type"],
                "deposited_A_B_rmsd": value,
                "atom_count": len(record["atom_names"]),
            }
        )

    high_exact_one = []
    for raw in failed_extras:
        if (
            float(raw["occupancy"]) <= 0.10
            or raw["missed_state"] not in {"A", "B"}
        ):
            continue
        missed = float(raw["rmsd_to_nearest_missed_state"])
        recovered = float(raw["rmsd_to_recovered_state"])
        row = {
            **raw,
            "deposited_A_B_rmsd": separations[raw["site"]],
            "recovered_over_missed_rmsd_ratio": (
                recovered / missed if missed > 0 else float("inf")
            ),
            "missed_over_A_B_separation_ratio": (
                missed / separations[raw["site"]]
            ),
        }
        row["collapse_category"] = collapse_category(row)
        high_exact_one.append(row)

    recovered_distribution_rows = []
    for site in ["ALL", *sorted({row["site"] for row in high_exact_one})]:
        selected = high_exact_one if site == "ALL" else [
            row for row in high_exact_one if row["site"] == site
        ]
        recovered_values = [
            float(row["rmsd_to_recovered_state"]) for row in selected
        ]
        missed_values = [
            float(row["rmsd_to_nearest_missed_state"]) for row in selected
        ]
        ratios = [
            float(row["recovered_over_missed_rmsd_ratio"]) for row in selected
        ]
        recovered_distribution_rows.append(
            {
                "site": site,
                **{
                    f"recovered_rmsd_{key}": value
                    for key, value in describe(recovered_values).items()
                },
                **{
                    f"missed_rmsd_{key}": value
                    for key, value in describe(missed_values).items()
                },
                **{
                    f"recovered_over_missed_ratio_{key}": value
                    for key, value in describe(ratios).items()
                },
                "fraction_closer_to_recovered": (
                    float(np.mean(np.asarray(ratios) < 1.0))
                    if ratios else None
                ),
            }
        )

    # Pairwise conventional RMSDs among extras when both deposited states were missed.
    both_rows = [
        row for row in failed_extras if row["missed_state"] == "both"
    ]
    both_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in both_rows:
        both_by_start[(row["site"], int(row["start"]))].append(row)
    pairwise_rows = []
    pairwise_start_rows = []
    for key, rows in sorted(both_by_start.items()):
        site, start = key
        record = site_data[site]
        names = list(record["atom_names"])
        resname = str(record["residue_type"])
        all_values = []
        high_values = []
        for left, right in itertools.combinations(rows, 2):
            left_key = (site, start, int(left["slot"]))
            right_key = (site, start, int(right["slot"]))
            value = conventional_rmsd(
                coordinates[left_key],
                coordinates[right_key],
                names,
                resname,
            )
            both_high = (
                float(left["occupancy"]) > 0.10
                and float(right["occupancy"]) > 0.10
            )
            all_values.append(value)
            if both_high:
                high_values.append(value)
            pairwise_rows.append(
                {
                    "site": site,
                    "start": start,
                    "left_slot": int(left["slot"]),
                    "right_slot": int(right["slot"]),
                    "left_occupancy": float(left["occupancy"]),
                    "right_occupancy": float(right["occupancy"]),
                    "both_above_0_10": both_high,
                    "pairwise_conventional_rmsd": value,
                }
            )
        high_extras = [row for row in rows if float(row["occupancy"]) > 0.10]
        pairwise_start_rows.append(
            {
                "site": site,
                "start": start,
                "active_extras": len(rows),
                "high_extras": len(high_extras),
                "nearest_deposited_rmsd_min": min(
                    float(row["rmsd_to_nearest_deposited_state"]) for row in rows
                ),
                "all_pair_count": len(all_values),
                "all_pair_median_rmsd": (
                    float(np.median(all_values)) if all_values else float("nan")
                ),
                "all_pair_max_rmsd": (
                    max(all_values) if all_values else float("nan")
                ),
                "high_pair_count": len(high_values),
                "high_pair_median_rmsd": (
                    float(np.median(high_values)) if high_values else float("nan")
                ),
                "high_pair_max_rmsd": (
                    max(high_values) if high_values else float("nan")
                ),
                "high_wrong_mode_tight_cluster": (
                    len(high_extras) >= 2
                    and bool(high_values)
                    and float(np.median(high_values)) < 1.0
                ),
            }
        )

    # Compare frozen greedy-nearest labeling with a distinct-slot A/B assignment.
    flagged_keys = {
        (row["site"], int(row["start"]))
        for row in failed_extras
        if float(row["occupancy"]) > 0.10
        and float(row["rmsd_to_nearest_missed_state"]) < 1.0
    }
    assignment_rows = []
    rescued_all = set()
    for raw in failed_starts:
        key = (raw["site"], int(raw["start"]))
        best = optimal_pair(slots_by_start[key])
        if best:
            rescued_all.add(key)
        if key in flagged_keys:
            assignment_rows.append(
                {
                    "site": key[0],
                    "start": key[1],
                    "greedy_missed_state": raw["missed_state"],
                    "hungarian_pair_exists": best is not None,
                    "hungarian_A_slot": best["slot_A"] if best else -1,
                    "hungarian_B_slot": best["slot_B"] if best else -1,
                    "hungarian_A_rmsd": best["rmsd_A"] if best else float("nan"),
                    "hungarian_B_rmsd": best["rmsd_B"] if best else float("nan"),
                    "hungarian_total_rmsd": (
                        best["total_rmsd"] if best else float("nan")
                    ),
                }
            )

    # Low-occupancy near-missed cases requested individually.
    low_near_rows = []
    for raw in failed_extras:
        if not (
            0.05 < float(raw["occupancy"]) <= 0.10
            and float(raw["rmsd_to_nearest_missed_state"]) < 1.0
        ):
            continue
        recovered = float(raw["rmsd_to_recovered_state"])
        missed = float(raw["rmsd_to_nearest_missed_state"])
        low_near_rows.append(
            {
                **raw,
                "deposited_A_B_rmsd": separations[raw["site"]],
                "recovered_over_missed_rmsd_ratio": (
                    recovered / missed if math.isfinite(recovered) and missed > 0
                    else float("nan")
                ),
            }
        )

    # Dominant-site sensitivity.
    concentration_rows = []
    both_start_lookup = {
        (row["site"], int(row["start"])): row for row in pairwise_start_rows
    }
    for label, exclude_dominant in (
        ("all_sites", False),
        ("excluding_1ZV8_2V05_7UO8_4C16", True),
    ):
        exact = [
            row for row in high_exact_one
            if not (exclude_dominant and row["site"] in DOMINANT_SITES)
        ]
        both_starts = [
            row for row in pairwise_start_rows
            if not (exclude_dominant and row["site"] in DOMINANT_SITES)
        ]
        categories = {
            category: sum(row["collapse_category"] == category for row in exact)
            for category in (
                "near_recovered_only",
                "near_missed_only",
                "near_both",
                "far_from_both",
            )
        }
        concentration_rows.append(
            {
                "population": label,
                "exact_one_high_extras": len(exact),
                **categories,
                "both_missed_starts": len(both_starts),
                "both_missed_starts_with_at_least_two_high_extras": sum(
                    int(row["high_extras"]) >= 2 for row in both_starts
                ),
                "both_missed_tight_wrong_mode_clusters": sum(
                    bool(row["high_wrong_mode_tight_cluster"])
                    for row in both_starts
                ),
            }
        )

    all_pair_values = [
        float(row["pairwise_conventional_rmsd"]) for row in pairwise_rows
    ]
    high_pair_values = [
        float(row["pairwise_conventional_rmsd"])
        for row in pairwise_rows if row["both_above_0_10"]
    ]
    high_pair_starts = [
        row for row in pairwise_start_rows if int(row["high_extras"]) >= 2
    ]
    summary = {
        "matching_rule": {
            "frozen": (
                "greedy-nearest independently per conformer: A if rmsd_A<1 "
                "and rmsd_A<=rmsd_B, else B if rmsd_B<1; no one-to-one constraint"
            ),
            "hungarian_test": (
                "two distinct occupancy>0.10 slots assigned one-to-one to A/B, "
                "each required to have RMSD<1.0"
            ),
            "flagged_starts": len(flagged_keys),
            "flagged_recovered_by_one_to_one": sum(
                row["hungarian_pair_exists"] for row in assignment_rows
            ),
            "all_271_failures_recovered_by_one_to_one": len(rescued_all),
        },
        "deposited_A_B_separation": describe(list(separations.values())),
        "high_extras_exactly_one_state_missed": {
            "conformers": len(high_exact_one),
            "rmsd_to_recovered": describe(
                [float(row["rmsd_to_recovered_state"]) for row in high_exact_one]
            ),
            "rmsd_to_missed": describe(
                [
                    float(row["rmsd_to_nearest_missed_state"])
                    for row in high_exact_one
                ]
            ),
            "recovered_over_missed_ratio": describe(
                [
                    float(row["recovered_over_missed_rmsd_ratio"])
                    for row in high_exact_one
                ]
            ),
            "collapse_categories": {
                category: sum(
                    row["collapse_category"] == category for row in high_exact_one
                )
                for category in (
                    "near_recovered_only",
                    "near_missed_only",
                    "near_both",
                    "far_from_both",
                )
            },
        },
        "both_states_missed_mutual_clustering": {
            "starts": len(pairwise_start_rows),
            "all_extra_pairwise_rmsd": describe(all_pair_values),
            "high_extra_pairwise_rmsd": describe(high_pair_values),
            "starts_with_at_least_two_high_extras": len(high_pair_starts),
            "tight_high_wrong_mode_clusters_median_pairwise_lt_1A": sum(
                bool(row["high_wrong_mode_tight_cluster"])
                for row in high_pair_starts
            ),
            "high_pair_start_median_rmsd": describe(
                [float(row["high_pair_median_rmsd"]) for row in high_pair_starts]
            ),
        },
        "low_occupancy_near_missed_cases": len(low_near_rows),
        "dominant_site_sensitivity": concentration_rows,
    }

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "deposited_A_B_separation.csv", separation_rows)
    atomic_csv(
        args.output / "high_extra_recovered_vs_missed.csv", high_exact_one
    )
    atomic_csv(
        args.output / "high_extra_recovered_distributions.csv",
        recovered_distribution_rows,
    )
    atomic_csv(args.output / "both_missed_extra_pairs.csv", pairwise_rows)
    atomic_csv(
        args.output / "both_missed_pairwise_by_start.csv",
        pairwise_start_rows,
    )
    atomic_csv(args.output / "greedy_vs_hungarian.csv", assignment_rows)
    atomic_csv(args.output / "low_occupancy_near_missed.csv", low_near_rows)
    atomic_csv(args.output / "dominant_site_sensitivity.csv", concentration_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
