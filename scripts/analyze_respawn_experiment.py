"""Compile merge-and-respawn arms against the reused frozen-v3 control."""

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
import torch

from density_denoiser.residue_geometry import (
    reference_permutations,
    symmetry_aware_rmsd,
)
from scripts.analyze_fixed_occupancy_sweep import (
    STAGES,
    TAIL_SITES,
    cascade,
    duplication_summary,
    missed_rank_optimizer_rows,
    missed_rank_rows,
    occupancy_summary,
)


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def describe(values) -> dict[str, float | int]:
    array = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=float,
    )
    if not len(array):
        return {"n": 0}
    return {
        "n": int(len(array)),
        "min": float(array.min()),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(array.max()),
    }


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, newline="", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        return None
    return value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        json.dump(
            finite_json(payload),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def load_audits(roots: list[Path]) -> dict[str, object]:
    ensembles, active = [], []
    coordinates: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict[str, object]] = {}
    for root in roots:
        ensembles.extend(read_csv(root / "ensemble_strict_audit.csv"))
        active.extend(read_csv(root / "active_conformer_strict_audit.csv"))
        payload = json.loads((root / "tmol_inputs.json").read_text())
        for site in payload["sites"]:
            deposited_a = torch.tensor(site["A"], dtype=torch.float32)
            deposited_b = torch.tensor(site["B"], dtype=torch.float32)
            names = list(site["atom_names"])
            resname = site["residue_type"]
            permutations = reference_permutations(names, resname)
            permutation = min(
                permutations,
                key=lambda order: float(
                    (deposited_a - deposited_b[order])
                    .square().sum(dim=-1).mean()
                ),
            )
            metadata[site["site"]] = {
                "names": names,
                "resname": resname,
                "A": deposited_a,
                "B": deposited_b,
                "midpoint": (
                    deposited_a + deposited_b[permutation]
                ) / 2.0,
                "deposited_separation": float(symmetry_aware_rmsd(
                    deposited_a, deposited_b, names, resname
                )),
            }
            for candidate in site["candidates"]:
                coordinates[candidate["candidate_id"]] = torch.tensor(
                    candidate["coordinates"], dtype=torch.float32
                )
    if len(ensembles) != 1000:
        raise ValueError(f"expected 1000 ensembles, found {len(ensembles)}")
    return {
        "ensembles": ensembles,
        "active": active,
        "coordinates": coordinates,
        "metadata": metadata,
    }


def load_optimizer_rows(roots: list[Path]) -> list[dict[str, str]]:
    rows = []
    for root in roots:
        for path in sorted(root.glob("*/synthetic/*_starts.csv")):
            rows.extend(read_csv(path))
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 optimizer rows, found {len(rows)}")
    return rows


def parse_values(value: str) -> list[float]:
    return [float(item) for item in value.split(";")]


def single_recovery_unmatched(
    optimizer_rows: list[dict[str, str]],
    audit: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output = []
    start_count = 0
    for row in optimizer_rows:
        found_a, found_b = truth(row["found_A"]), truth(row["found_B"])
        if found_a == found_b:
            continue
        target_a = float(row["target_A_occupancy"])
        target_b = float(row["target_B_occupancy"])
        if math.isclose(target_a, target_b, abs_tol=1e-6):
            continue
        start_count += 1
        recovered = "A" if found_a else "B"
        missed = "B" if found_a else "A"
        occupancies = parse_values(row["occupancies"])
        rmsd_a = parse_values(row["rmsd_to_A"])
        rmsd_b = parse_values(row["rmsd_to_B"])
        assignments = row["assignments"].split(";")
        recovered_candidates = [
            index for index in range(len(occupancies))
            if occupancies[index] > 0.10
            and assignments[index] == recovered
        ]
        if not recovered_candidates:
            raise RuntimeError(
                f"missing recovered representative: {row['site']} {row['start']}"
            )
        representative = min(
            recovered_candidates,
            key=lambda index: (
                rmsd_a[index] if recovered == "A" else rmsd_b[index]
            ),
        )
        info = audit["metadata"][row["site"]]
        for slot, occupancy in enumerate(occupancies):
            if occupancy <= 0.05 or slot == representative:
                continue
            candidate_id = (
                f"{row['site']}__{int(row['start']):03d}__{slot}"
            )
            xyz = audit["coordinates"][candidate_id]
            literal_distance_a = float(symmetry_aware_rmsd(
                xyz, info["A"], info["names"], info["resname"]
            ))
            literal_distance_b = float(symmetry_aware_rmsd(
                xyz, info["B"], info["names"], info["resname"]
            ))
            # Preserve the exact optimizer-control definition underlying the
            # frozen 213/259 (82.2%) comparison.  Literal deposited-coordinate
            # distances are retained separately as a sensitivity diagnostic.
            distance_a = rmsd_a[slot]
            distance_b = rmsd_b[slot]
            distance_midpoint = float(symmetry_aware_rmsd(
                xyz, info["midpoint"], info["names"], info["resname"]
            ))
            output.append({
                "site": row["site"],
                "start": int(row["start"]),
                "slot": slot,
                "recovered_state": recovered,
                "missed_state": missed,
                "occupancy": occupancy,
                "rmsd_to_A_A": distance_a,
                "rmsd_to_B_A": distance_b,
                "literal_deposited_rmsd_to_A_A": literal_distance_a,
                "literal_deposited_rmsd_to_B_A": literal_distance_b,
                "rmsd_to_midpoint_A": distance_midpoint,
                "no_reference_A_B_or_midpoint_within_1A": (
                    min(distance_a, distance_b, distance_midpoint) >= 1.0
                ),
                "no_kinematic_deposited_A_or_B_within_1A": (
                    min(distance_a, distance_b) >= 1.0
                ),
                "no_literal_deposited_reference_within_1A": (
                    min(literal_distance_a, literal_distance_b) >= 1.0
                ),
            })
    no_reference = sum(
        bool(row["no_reference_A_B_or_midpoint_within_1A"])
        for row in output
    )
    no_kinematic_deposited = sum(
        bool(row["no_kinematic_deposited_A_or_B_within_1A"])
        for row in output
    )
    no_literal_reference = sum(
        bool(row["no_literal_deposited_reference_within_1A"])
        for row in output
    )
    summary = {
        "single_recovery_starts": start_count,
        "unmatched_active_slots": len(output),
        "unmatched_active_occupancy": describe(
            row["occupancy"] for row in output
        ),
        "no_reference_A_B_or_midpoint_within_1A": no_reference,
        "no_reference_A_B_or_midpoint_within_1A_fraction": (
            no_reference / len(output) if output else 0.0
        ),
        "no_kinematic_deposited_A_or_B_within_1A": (
            no_kinematic_deposited
        ),
        "no_kinematic_deposited_A_or_B_within_1A_fraction": (
            no_kinematic_deposited / len(output) if output else 0.0
        ),
        "no_literal_deposited_reference_within_1A": no_literal_reference,
        "no_literal_deposited_reference_within_1A_fraction": (
            no_literal_reference / len(output) if output else 0.0
        ),
    }
    return output, summary


def geometry_loss_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        key: describe(float(row[key]) for row in rows)
        for key in (
            "final_density_loss",
            "final_vdw_loss",
            "final_rotamer_loss",
            "final_symmetry_loss",
        )
    }


def load_respawn_logs(arm_root: Path) -> tuple[
    list[dict[str, str]], list[dict[str, str]]
]:
    checks, events = [], []
    for path in sorted(
        arm_root.glob("shards/*/*/synthetic/*_respawn_checks.csv")
    ):
        checks.extend(read_csv(path))
    for path in sorted(
        arm_root.glob("shards/*/*/synthetic/*_respawn_events.csv")
    ):
        events.extend(read_csv(path))
    return checks, events


def event_mechanism_summary(
    checks: list[dict[str, str]],
    events: list[dict[str, str]],
    endpoint_duplicate_starts: set[tuple[str, int]],
) -> dict[str, object]:
    starts_with_events = {
        (row["site"], int(row["start"])) for row in events
    }
    event_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in events:
        event_counts[(row["site"], int(row["start"]))] += 1
    failures = [
        row for row in events if not truth(row["peak_reached_within_0p5_A"])
    ]
    worse = [
        row for row in events if truth(row["endpoint_worse_than_replaced"])
    ]
    endpoint_single = [
        row for row in events if row["endpoint_missed_state"] in {"A", "B"}
    ]

    def peak_class(row: dict[str, str]) -> str:
        a = float(row["peak_to_A_nearest_atom_distance_A"])
        b = float(row["peak_to_B_nearest_atom_distance_A"])
        midpoint = float(
            row["peak_to_midpoint_nearest_atom_distance_A"]
        )
        missed = row["endpoint_missed_state"]
        missed_distance = a if missed == "A" else b if missed == "B" else math.inf
        recovered_distance = (
            b if missed == "A" else a if missed == "B" else math.inf
        )
        if missed_distance < 1.0 and missed_distance <= midpoint:
            return "missed"
        if midpoint < 1.0 and midpoint < min(a, b):
            return "midpoint"
        if recovered_distance < 1.0:
            return "recovered"
        return "unrelated"

    def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
        counts = {
            "missed": 0,
            "recovered": 0,
            "midpoint": 0,
            "unrelated": 0,
        }
        for row in rows:
            counts[peak_class(row)] += 1
        return counts

    def region_counts(rows: list[dict[str, str]]) -> dict[str, int]:
        counts = {"deposited": 0, "midpoint": 0, "unrelated": 0}
        for row in rows:
            a = float(row["peak_to_A_nearest_atom_distance_A"])
            b = float(row["peak_to_B_nearest_atom_distance_A"])
            midpoint = float(
                row["peak_to_midpoint_nearest_atom_distance_A"]
            )
            if midpoint < 1.0 and midpoint < min(a, b):
                counts["midpoint"] += 1
            elif min(a, b) < 1.0:
                counts["deposited"] += 1
            else:
                counts["unrelated"] += 1
        return counts

    first_by_start: dict[tuple[str, int, str], int] = {}
    for row in checks:
        key = (row["site"], int(row["start"]))
        for criterion, field in (
            ("gram", "gram_condition_triggered"),
            ("rmsd_0p3", "rmsd_below_0p3"),
            ("rmsd_0p5", "rmsd_below_0p5"),
            ("rmsd_0p8", "rmsd_below_0p8"),
        ):
            if truth(row[field]):
                first_by_start.setdefault(
                    (*key, criterion), int(row["step"])
                )
    gram_vs_rmsd = {}
    for criterion in ("rmsd_0p3", "rmsd_0p5", "rmsd_0p8"):
        keys = {
            (row["site"], int(row["start"])) for row in checks
        }
        both = [
            key for key in keys
            if (*key, "gram") in first_by_start
            and (*key, criterion) in first_by_start
        ]
        gram_vs_rmsd[criterion] = {
            "gram_detected_starts": sum(
                (*key, "gram") in first_by_start for key in keys
            ),
            "rmsd_detected_starts": sum(
                (*key, criterion) in first_by_start for key in keys
            ),
            "both_detected_starts": len(both),
            "gram_earlier": sum(
                first_by_start[(*key, "gram")]
                < first_by_start[(*key, criterion)]
                for key in both
            ),
            "same_step": sum(
                first_by_start[(*key, "gram")]
                == first_by_start[(*key, criterion)]
                for key in both
            ),
            "rmsd_earlier": sum(
                first_by_start[(*key, "gram")]
                > first_by_start[(*key, criterion)]
                for key in both
            ),
        }

    detector_reliability = {}
    for criterion, field in (
        ("gram_condition_100", "gram_condition_triggered"),
        ("rmsd_0p3", "rmsd_below_0p3"),
        ("rmsd_0p5", "rmsd_below_0p5"),
        ("rmsd_0p8", "rmsd_below_0p8"),
    ):
        detected = {
            (row["site"], int(row["start"]))
            for row in checks if truth(row[field])
        }
        true_positive = len(detected & endpoint_duplicate_starts)
        detector_reliability[criterion] = {
            "detected_starts": len(detected),
            "endpoint_duplicate_starts": len(endpoint_duplicate_starts),
            "true_positive": true_positive,
            "false_positive": len(detected - endpoint_duplicate_starts),
            "false_negative": len(endpoint_duplicate_starts - detected),
            "precision": (
                true_positive / len(detected) if detected else 0.0
            ),
            "recall": (
                true_positive / len(endpoint_duplicate_starts)
                if endpoint_duplicate_starts else 0.0
            ),
        }

    separation_groups = {}
    for label, predicate in (
        ("below_2p5_A", lambda row: float(row["site_separation_A"]) < 2.5),
        ("at_least_2p5_A", lambda row: float(row["site_separation_A"]) >= 2.5),
    ):
        subset = [row for row in events if predicate(row)]
        separation_groups[label] = {
            "events": len(subset),
            "peak_to_midpoint_A": describe(
                float(row[
                    "peak_to_midpoint_nearest_atom_distance_A"
                ]) for row in subset
            ),
            "midpoint_region": sum(
                truth(row["peak_in_midpoint_region_within_1A"])
                for row in subset
            ),
        }

    per_site = {}
    for site in sorted({row["site"] for row in checks}):
        site_checks = [row for row in checks if row["site"] == site]
        site_events = [row for row in events if row["site"] == site]
        site_worse = [
            row for row in site_events
            if truth(row["endpoint_worse_than_replaced"])
        ]
        per_site[site] = {
            "checks": len(site_checks),
            "events": len(site_events),
            "event_bearing_starts": len({
                int(row["start"]) for row in site_events
            }),
            "torsion_inversion_failures_gt_0p5_A": sum(
                not truth(row["peak_reached_within_0p5_A"])
                for row in site_events
            ),
            "endpoint_worse_than_replaced": len(site_worse),
            "endpoint_survived_above_0p10": sum(
                truth(row["endpoint_slot_survived_above_0p10"])
                for row in site_events
            ),
            "worse_replacement_peak_regions": region_counts(site_worse),
        }

    return {
        "cadence_checks": len(checks),
        "respawn_events": len(events),
        "starts_with_events": len(starts_with_events),
        "events_per_event_bearing_start": describe(event_counts.values()),
        "torsion_inversion_failures_gt_0p5_A": len(failures),
        "torsion_inversion_failure_residual_A": describe(
            float(row["peak_residual_distance_A"]) for row in failures
        ),
        "respawned_events_endpoint_within_1A": sum(
            truth(row["endpoint_slot_within_1A_of_deposited"])
            for row in events
        ),
        "respawned_events_survived_above_0p10": sum(
            truth(row["endpoint_slot_survived_above_0p10"])
            for row in events
        ),
        "merged_away_near_deposited": sum(
            truth(row["merged_away_was_near_deposited"]) for row in events
        ),
        "endpoint_worse_than_replaced": len(worse),
        "peak_classes_endpoint_single_recovery": class_counts(endpoint_single),
        "worse_replacement_peak_classes": class_counts(worse),
        "worse_replacement_peak_regions": region_counts(worse),
        "worse_replacement_peak_to_midpoint_A": describe(
            float(row[
                "peak_to_midpoint_nearest_atom_distance_A"
            ]) for row in worse
        ),
        "respawned_slot_geometry": {
            "endpoint_direct_min_distance_A": describe(
                float(row["endpoint_slot_direct_min_distance_A"])
                for row in events
            ),
            "endpoint_symmetry_min_distance_A": describe(
                float(row["endpoint_slot_symmetry_min_distance_A"])
                for row in events
            ),
            "endpoint_rotamer_max_deviation_degrees": describe(
                float(row[
                    "endpoint_slot_rotamer_max_deviation_degrees"
                ]) for row in events
            ),
            "endpoint_canonical": sum(
                truth(row["endpoint_slot_canonical"]) for row in events
            ),
            "direct_clash_lt_2A": sum(
                math.isfinite(float(
                    row["endpoint_slot_direct_min_distance_A"]
                ))
                and float(row["endpoint_slot_direct_min_distance_A"]) < 2.0
                for row in events
            ),
            "symmetry_clash_lt_2A": sum(
                math.isfinite(float(
                    row["endpoint_slot_symmetry_min_distance_A"]
                ))
                and float(row["endpoint_slot_symmetry_min_distance_A"]) < 2.0
                for row in events
            ),
            "noncanonical": sum(
                not truth(row["endpoint_slot_canonical"]) for row in events
            ),
            "any_geometry_failure": sum(
                (
                    math.isfinite(float(
                        row["endpoint_slot_direct_min_distance_A"]
                    ))
                    and float(
                        row["endpoint_slot_direct_min_distance_A"]
                    ) < 2.0
                )
                or (
                    math.isfinite(float(
                        row["endpoint_slot_symmetry_min_distance_A"]
                    ))
                    and float(
                        row["endpoint_slot_symmetry_min_distance_A"]
                    ) < 2.0
                )
                or not truth(row["endpoint_slot_canonical"])
                for row in events
            ),
        },
        "gram_vs_rmsd_detection": gram_vs_rmsd,
        "detector_reliability_against_endpoint_duplication": (
            detector_reliability
        ),
        "peak_midpoint_by_site_separation": separation_groups,
        "per_site": per_site,
    }


def respawn_start_summary(rows: list[dict[str, str]]) -> dict[str, object]:
    event_counts = [int(row["respawn_event_count"]) for row in rows]
    return {
        "starts": len(rows),
        "starts_with_respawn": sum(value > 0 for value in event_counts),
        "events_per_start": describe(event_counts),
        "total_events": sum(event_counts),
        "unique_respawned_slots_per_start": describe(
            int(row["respawned_unique_slot_count"]) for row in rows
        ),
        "total_unique_respawned_slots": sum(
            int(row["respawned_unique_slot_count"]) for row in rows
        ),
        "total_unique_respawned_endpoint_slots_above_0p10": sum(
            int(row["respawned_endpoint_slots_above_0p10"])
            for row in rows
        ),
    }


def endpoint_duplicate_starts(
    rows: list[dict[str, str]],
) -> set[tuple[str, int]]:
    output = set()
    for row in rows:
        occupancies = parse_values(row["occupancies"])
        assignments = row["assignments"].split(";")
        if any(
            sum(
                occupancy > 0.05 and assignment == state
                for occupancy, assignment in zip(
                    occupancies, assignments
                )
            ) >= 2
            for state in ("A", "B")
        ):
            output.add((row["site"], int(row["start"])))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-frozen-root", type=Path, required=True)
    parser.add_argument("--control-baseline-root", type=Path, required=True)
    parser.add_argument("--control-replacement-root", type=Path, required=True)
    parser.add_argument(
        "--arm", action="append", required=True,
        help="LABEL=/absolute/arm/root",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    metric = (
        args.control_frozen_root
        / "analysis/metric_v3_protected_merge_sweep/0p5"
    )
    populations = {
        "control": (
            load_audits([
                metric / "original5",
                metric / "expanded13",
                metric / "water2",
            ]),
            load_optimizer_rows([
                args.control_baseline_root / "shards/original5",
                args.control_baseline_root / "shards/expanded15",
                args.control_replacement_root / "shards",
            ]),
            None,
        )
    }
    for item in args.arm:
        label, raw_path = item.split("=", 1)
        root = Path(raw_path)
        populations[label] = (
            load_audits([
                root / "audit/original5",
                root / "audit/expanded15",
            ]),
            load_optimizer_rows([
                root / "shards/original5",
                root / "shards/expanded15",
            ]),
            root,
        )

    summary = {
        "metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
        "metric_changed": False,
        "control_rerun": False,
        "arms": {},
    }
    cascade_rows, primary_rows, unmatched_rows, tail_rows = [], [], [], []
    control_per_site = None
    for label, (audit, optimizer_rows, arm_root) in populations.items():
        per_site_rows, per_site = cascade(audit["ensembles"], audit["active"])
        raw_primary_rows, raw_primary = missed_rank_optimizer_rows(
            optimizer_rows
        )
        v3_primary_rows, v3_primary = missed_rank_rows(audit["ensembles"])
        unmatched, unmatched_summary = single_recovery_unmatched(
            optimizer_rows, audit
        )
        totals = {
            stage: sum(site[stage] for site in per_site.values())
            for stage in STAGES
        }
        arm_summary = {
            "cascade": totals,
            "single_state_failures_raw_greedy": raw_primary,
            "single_state_failures_frozen_v3": v3_primary,
            "same_state_duplication": duplication_summary(
                optimizer_rows, audit
            ),
            "single_recovery_unmatched": unmatched_summary,
            "matched_occupancy": occupancy_summary(audit["ensembles"]),
            "geometry_losses": geometry_loss_summary(optimizer_rows),
        }
        if arm_root is not None:
            checks, events = load_respawn_logs(arm_root)
            for event in events:
                event["site_separation_A"] = audit["metadata"][
                    event["site"]
                ]["deposited_separation"]
            arm_summary["respawn_mechanism"] = event_mechanism_summary(
                checks,
                events,
                endpoint_duplicate_starts(optimizer_rows),
            )
            arm_summary["respawn_per_start"] = respawn_start_summary(
                optimizer_rows
            )
            atomic_csv(
                args.output / f"{label}_respawn_checks.csv", checks
            )
            atomic_csv(
                args.output / f"{label}_respawn_events.csv", events
            )
        summary["arms"][label] = arm_summary
        for row in per_site_rows:
            cascade_rows.append({"arm": label, **row})
        for matching, rows in (
            ("raw_greedy_control_comparable", raw_primary_rows),
            ("frozen_v3_one_to_one", v3_primary_rows),
        ):
            for row in rows:
                primary_rows.append({
                    "arm": label, "matching": matching, **row
                })
        for row in unmatched:
            unmatched_rows.append({"arm": label, **row})
        if label == "control":
            control_per_site = per_site
        else:
            for row in per_site_rows:
                if row["site"] in TAIL_SITES:
                    tail_rows.append({
                        "arm": label,
                        **row,
                        **{
                            f"delta_{stage}": (
                                int(row[stage])
                                - control_per_site[row["site"]][stage]
                            )
                            for stage in STAGES
                        },
                    })

    expected_cascade = {
        "found": 742,
        "occupancy": 714,
        "rotamer": 710,
        "direct": 710,
        "symmetry": 710,
        "strict": 626,
    }
    actual_cascade = summary["arms"]["control"]["cascade"]
    if actual_cascade != expected_cascade:
        raise RuntimeError(
            f"frozen control cascade guard failed: {actual_cascade}"
        )
    raw_control = summary["arms"]["control"][
        "single_state_failures_raw_greedy"
    ]
    if raw_control["minor"] != 142 or raw_control["major"] != 45:
        raise RuntimeError(
            f"frozen control 142/45 guard failed: {raw_control}"
        )
    unmatched_control = summary["arms"]["control"][
        "single_recovery_unmatched"
    ]
    if (
        unmatched_control["unmatched_active_slots"] != 259
        or unmatched_control[
            "no_reference_A_B_or_midpoint_within_1A"
        ] != 213
    ):
        raise RuntimeError(
            f"frozen control 213/259 guard failed: {unmatched_control}"
        )
    summary["control_guards_passed"] = True

    atomic_csv(args.output / "cascade_per_site.csv", cascade_rows)
    atomic_csv(
        args.output / "single_state_failures_per_site.csv", primary_rows
    )
    atomic_csv(
        args.output / "single_recovery_unmatched_slots.csv", unmatched_rows
    )
    atomic_csv(args.output / "tail_site_deltas.csv", tail_rows)
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(
        finite_json(summary),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
