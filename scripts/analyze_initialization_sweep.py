"""Compile the initialization sweep under frozen metric v3."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.analyze_fixed_occupancy_sweep import (
    STAGES,
    TAIL_SITES,
    atomic_csv,
    atomic_json,
    cascade,
    duplication_summary,
    extra_summary,
    load_audits,
    load_optimizer_rows,
    missed_rank_optimizer_rows,
    missed_rank_rows,
    occupancy_summary,
    percentile,
    separation_rows,
)


ARM_PATHS = {
    "canonical_free": "canonical_free",
    "canonical_a_anchor": "canonical_a_anchor",
    "deposited_a_cloud_120": "deposited_a_cloud_120",
}


def split_vector(value: str) -> list[float]:
    return [float(item) for item in value.split(";")]


def split_matrix(value: str) -> list[list[float]]:
    return [split_vector(row) for row in value.split("|")]


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def load_control_initializations(root: Path) -> dict[tuple[str, int, int], dict]:
    rows = {}
    for path in sorted(root.glob("*/*/initialization_only.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row["site"], int(row["start"]), int(row["slot"]))
                if key in rows:
                    raise RuntimeError(f"duplicate control initialization {key}")
                rows[key] = row
    if len(rows) != 20 * 50 * 4:
        raise RuntimeError(
            f"expected 4000 control initialization rows, found {len(rows)}"
        )
    return rows


def mechanism_rows(
    arm: str,
    optimizer_rows: list[dict[str, str]],
    *,
    control_initializations: dict[tuple[str, int, int], dict] | None = None,
) -> list[dict]:
    output = []
    for row in optimizer_rows:
        site = row["site"]
        start = int(row["start"])
        final_delta = np.asarray(split_matrix(row["final_chi_radians"]))
        final_rmsd_a = np.asarray(split_vector(row["rmsd_to_A"]))
        final_rmsd_b = np.asarray(split_vector(row["rmsd_to_B"]))
        occupancies = np.asarray(split_vector(row["occupancies"]))
        if control_initializations is None:
            initial_physical = np.asarray(split_matrix(
                row["initial_physical_chi_degrees"]
            ))
            final_physical = np.asarray(split_matrix(
                row["final_physical_chi_degrees"]
            ))
            initial_rmsd_a = np.asarray(split_vector(
                row["initial_rmsd_to_A"]
            ))
            initial_rmsd_b = np.asarray(split_vector(
                row["initial_rmsd_to_B"]
            ))
            initial_delta = np.asarray(split_matrix(
                row["initial_delta_chi_radians"]
            ))
        else:
            initialization = [
                control_initializations[(site, start, slot)]
                for slot in range(len(final_delta))
            ]
            initial_physical = np.asarray([
                split_vector(value["initial_physical_chi_degrees"])
                for value in initialization
            ])
            initial_delta = np.asarray([
                split_vector(value["initial_delta_chi_radians"])
                for value in initialization
            ])
            initial_rmsd_a = np.asarray([
                float(value["initial_rmsd_to_A"]) for value in initialization
            ])
            initial_rmsd_b = np.asarray([
                float(value["initial_rmsd_to_B"]) for value in initialization
            ])
            base = np.asarray(split_vector(
                initialization[0]["base_physical_chi_degrees"]
            ))
            direction = np.asarray(split_vector(
                initialization[0]["delta_direction"]
            ))
            final_physical = wrap_degrees(
                base[None, :] + direction[None, :] * np.degrees(final_delta)
            )

        target_a = float(row["target_A_occupancy"])
        target_b = float(row["target_B_occupancy"])
        if target_a < target_b:
            minor = "A"
            initial_minor = initial_rmsd_a
            final_minor = final_rmsd_a
        elif target_b < target_a:
            minor = "B"
            initial_minor = initial_rmsd_b
            final_minor = final_rmsd_b
        else:
            minor = "equal"
            initial_minor = np.full(len(final_delta), np.nan)
            final_minor = np.full(len(final_delta), np.nan)

        chi_delta = wrap_degrees(final_physical - initial_physical)
        net_distance = np.linalg.norm(chi_delta, axis=1)
        for slot in range(len(final_delta)):
            output.append({
                "arm": arm,
                "site": site,
                "start": start,
                "slot": slot,
                "occupancy": occupancies[slot],
                "minor_state": minor,
                "initial_delta_chi_radians": ";".join(
                    f"{value:.9g}" for value in initial_delta[slot]
                ),
                "initial_physical_chi_degrees": ";".join(
                    f"{value:.9g}" for value in initial_physical[slot]
                ),
                "final_physical_chi_degrees": ";".join(
                    f"{value:.9g}" for value in final_physical[slot]
                ),
                "chi_space_net_distance_degrees": net_distance[slot],
                "initial_rmsd_to_A": initial_rmsd_a[slot],
                "initial_rmsd_to_B": initial_rmsd_b[slot],
                "final_rmsd_to_A": final_rmsd_a[slot],
                "final_rmsd_to_B": final_rmsd_b[slot],
                "final_rmsd_to_nearest_deposited": min(
                    final_rmsd_a[slot], final_rmsd_b[slot]
                ),
                "initial_rmsd_to_minor": initial_minor[slot],
                "final_rmsd_to_minor": final_minor[slot],
                "ended_nearer_minor_than_initial": (
                    bool(final_minor[slot] < initial_minor[slot])
                    if minor != "equal" else False
                ),
                "started_within_1A_of_minor": (
                    bool(initial_minor[slot] < 1.0)
                    if minor != "equal" else False
                ),
                "started_near_minor_then_moved_away": (
                    bool(
                        initial_minor[slot] < 1.0
                        and final_minor[slot] > initial_minor[slot]
                    )
                    if minor != "equal" else False
                ),
            })
    return output


def mechanism_summary(rows: list[dict]) -> tuple[dict, list[dict]]:
    by_start = defaultdict(list)
    by_site = defaultdict(list)
    for row in rows:
        by_start[(row["site"], row["start"])].append(row)
        by_site[row["site"]].append(row)

    def summarize(population: list[dict], starts: dict) -> dict:
        net = [float(row["chi_space_net_distance_degrees"]) for row in population]
        nearest = [
            float(row["final_rmsd_to_nearest_deposited"]) for row in population
        ]
        started_near = [
            row for row in population if row["started_within_1A_of_minor"]
        ]
        moved_away = [
            row for row in started_near
            if row["started_near_minor_then_moved_away"]
        ]
        return {
            "starts": len(starts),
            "slots": len(population),
            "starts_with_any_slot_ending_nearer_minor": sum(
                any(row["ended_nearer_minor_than_initial"] for row in values)
                for values in starts.values()
                if values[0]["minor_state"] != "equal"
            ),
            "equal_occupancy_starts_excluded": sum(
                values[0]["minor_state"] == "equal"
                for values in starts.values()
            ),
            "slots_ending_nearer_minor": sum(
                row["ended_nearer_minor_than_initial"] for row in population
            ),
            "slots_starting_within_1A_of_minor": len(started_near),
            "near_minor_slots_moving_away": len(moved_away),
            "near_minor_move_away_rate": (
                len(moved_away) / len(started_near) if started_near else math.nan
            ),
            "net_chi_distance_degrees_median": percentile(net, 0.5),
            "net_chi_distance_degrees_q25": percentile(net, 0.25),
            "net_chi_distance_degrees_q75": percentile(net, 0.75),
            "final_nearest_rmsd_A_median": percentile(nearest, 0.5),
        }

    overall = summarize(rows, by_start)
    per_site = []
    for site, site_rows in sorted(by_site.items()):
        site_starts = {
            key: values for key, values in by_start.items() if key[0] == site
        }
        per_site.append({"site": site, **summarize(site_rows, site_starts)})
    return overall, per_site


def load_local_separations(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {
        row["site"]: float(row["local_unsym_rmsd_A"]) for row in rows
    }
    if len(output) != 20:
        raise RuntimeError(f"expected 20 deposited separations, found {len(output)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-audit-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--control-results-root", type=Path, action="append", required=True
    )
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--separation-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    populations = {
        "control": (
            load_audits(args.control_audit_root),
            load_optimizer_rows(args.control_results_root),
        )
    }
    for arm, directory in ARM_PATHS.items():
        root = args.sweep_root / directory
        populations[arm] = (
            load_audits([
                root / "audit" / "original5",
                root / "audit" / "expanded15",
            ]),
            load_optimizer_rows([
                root / "shards" / "original5",
                root / "shards" / "expanded15",
            ]),
        )

    separations = load_local_separations(args.separation_csv)
    control_initializations = load_control_initializations(
        args.sweep_root / "control_initialization"
    )
    summary = {
        "metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
        "metric_changed": False,
        "separation_definition": "local fixed-label A/B RMSD",
        "separation_cutoff_A": 2.5,
        "arms": {},
    }
    cascade_output = []
    primary_output = []
    tail_output = []
    separation_output = []
    mechanism_output = []
    mechanism_per_site_output = []
    control_counts = None

    for arm, (audit, optimizer_rows) in populations.items():
        cascade_rows, cascade_counts = cascade(
            audit["ensembles"], audit["active"]
        )
        raw_primary_rows, raw_primary_totals = missed_rank_optimizer_rows(
            optimizer_rows
        )
        v3_primary_rows, v3_primary_totals = missed_rank_rows(
            audit["ensembles"]
        )
        duplication = duplication_summary(optimizer_rows, audit)
        extras = extra_summary(audit["active"])
        occupancy = occupancy_summary(audit["ensembles"])
        assigned_separation = separation_rows(audit)
        ratios = [
            float(row["assigned_over_deposited"])
            for row in assigned_separation
        ]
        mechanism = mechanism_rows(
            arm,
            optimizer_rows,
            control_initializations=(
                control_initializations if arm == "control" else None
            ),
        )
        mechanism_totals, mechanism_per_site = mechanism_summary(mechanism)
        mechanism_output.extend(mechanism)
        mechanism_per_site_output.extend(
            {"arm": arm, **row} for row in mechanism_per_site
        )

        raw_by_site = {row["site"]: row for row in raw_primary_rows}
        separation_groups = {}
        for group, predicate in {
            "below_or_equal_2p5A": lambda value: value <= 2.5,
            "above_2p5A": lambda value: value > 2.5,
        }.items():
            sites = [site for site, value in separations.items() if predicate(value)]
            values = [raw_by_site.get(site, {}) for site in sites]
            separation_groups[group] = {
                "sites": len(sites),
                "minor_missed": sum(int(row.get("minor_missed", 0)) for row in values),
                "major_missed": sum(int(row.get("major_missed", 0)) for row in values),
            }
            separation_output.append({
                "arm": arm,
                "group": group,
                **separation_groups[group],
            })

        totals = {
            stage: sum(counts[stage] for counts in cascade_counts.values())
            for stage in STAGES
        }
        summary["arms"][arm] = {
            "cascade": totals,
            "single_state_failures_raw_greedy": raw_primary_totals,
            "single_state_failures_frozen_v3": v3_primary_totals,
            "single_state_failures_by_local_AB_separation": separation_groups,
            "same_state_duplication": duplication,
            "unmatched_extras": extras,
            "matched_occupancy": occupancy,
            "assigned_pair_separation": {
                "recovered_starts": len(ratios),
                "ratio_median": percentile(ratios, 0.5),
                "count_below_0p5": sum(value < 0.5 for value in ratios),
            },
            "initialization_mechanism": mechanism_totals,
        }
        cascade_output.extend({"arm": arm, **row} for row in cascade_rows)
        primary_output.extend({
            "arm": arm,
            "matching": "raw_greedy_control_comparable",
            "local_unsym_AB_separation_A": separations[row["site"]],
            **row,
        } for row in raw_primary_rows)
        primary_output.extend({
            "arm": arm,
            "matching": "frozen_v3_one_to_one",
            "local_unsym_AB_separation_A": separations[row["site"]],
            **row,
        } for row in v3_primary_rows)
        if arm == "control":
            control_counts = cascade_counts

    if control_counts is None:
        raise RuntimeError("control cascade missing")
    control = summary["arms"]["control"]
    if control["cascade"]["found"] != 742 or control["cascade"]["strict"] != 626:
        raise RuntimeError(f"frozen control mismatch: {control['cascade']}")
    raw = control["single_state_failures_raw_greedy"]
    if raw["minor"] != 142 or raw["major"] != 45:
        raise RuntimeError(f"frozen missed-rank mismatch: {raw}")

    for row in cascade_output:
        if row["arm"] == "control" or row["site"] not in TAIL_SITES:
            continue
        baseline = control_counts[row["site"]]
        tail_output.append({
            **row,
            **{
                f"delta_{stage}": int(row[stage]) - baseline[stage]
                for stage in STAGES
            },
        })

    atomic_csv(args.output / "cascade_per_site.csv", cascade_output)
    atomic_csv(
        args.output / "single_state_failures_per_site.csv", primary_output
    )
    atomic_csv(args.output / "tail_site_deltas.csv", tail_output)
    atomic_csv(
        args.output / "missed_rank_by_local_AB_separation.csv",
        separation_output,
    )
    atomic_csv(
        args.output / "initialization_slot_mechanism.csv", mechanism_output
    )
    atomic_csv(
        args.output / "initialization_mechanism_per_site.csv",
        mechanism_per_site_output,
    )
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
