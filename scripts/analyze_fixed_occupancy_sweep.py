"""Compile fixed-occupancy sweep endpoints under frozen metric v3."""

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

from density_denoiser.residue_geometry import symmetry_aware_rmsd


STAGES = ("found", "occupancy", "rotamer", "direct", "symmetry", "strict")
TAIL_SITES = {
    "1ZV8_E_ASN1",
    "2VFP_A_TYR417",
    "5Z8H_A_MET730",
    "7UO8_A_GLN53",
    "4C16_A_MET258",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else float("nan")


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


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
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
        inputs = json.loads((root / "tmol_inputs.json").read_text())
        for site in inputs["sites"]:
            key = site["site"]
            deposited_a = torch.tensor(site["A"], dtype=torch.float32)
            deposited_b = torch.tensor(site["B"], dtype=torch.float32)
            metadata[key] = {
                "names": site["atom_names"],
                "resname": site["residue_type"],
                "deposited_separation": float(symmetry_aware_rmsd(
                    deposited_a,
                    deposited_b,
                    site["atom_names"],
                    site["residue_type"],
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


def assigned_by_start(
    active: list[dict[str, str]],
) -> dict[tuple[str, int], dict[str, dict[str, str]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in active:
        if row["assignment"] in {"A", "B"} and truth(
            row["merge_cluster_representative"]
        ):
            grouped[(row["site"], int(row["start"]))][row["assignment"]] = row
    return grouped


def cascade(
    ensembles: list[dict[str, str]],
    active: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    pairs = assigned_by_start(active)
    per_site: dict[str, dict[str, int]] = defaultdict(
        lambda: {stage: 0 for stage in STAGES}
    )
    for row in ensembles:
        site = row["site"]
        key = (site, int(row["start"]))
        found = truth(row["both_found_conventional"])
        occupancy = found and truth(row["geometric_occupancy_success"])
        pair = pairs.get(key, {})
        pair_complete = set(pair) == {"A", "B"}
        rotamer = occupancy and pair_complete and all(
            truth(pair[state]["rotamer_within_allowed_width"])
            for state in ("A", "B")
        )
        direct = rotamer and all(
            truth(pair[state]["no_direct_clash"]) for state in ("A", "B")
        )
        symmetry = direct and all(
            truth(pair[state]["no_symmetry_clash"]) for state in ("A", "B")
        )
        strict = symmetry and all(
            math.isfinite(float(pair[state]["tmol_delta_vs_matched_AB"]))
            and float(pair[state]["tmol_delta_vs_matched_AB"]) <= 0.44
            for state in ("A", "B")
        )
        for stage, passed in zip(
            STAGES, (found, occupancy, rotamer, direct, symmetry, strict)
        ):
            per_site[site][stage] += int(passed)
    rows = [
        {"site": site, **per_site[site]}
        for site in sorted(per_site)
    ]
    return rows, per_site


def missed_rank_rows(
    ensembles: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals = {"minor": 0, "major": 0, "equal_excluded": 0}
    for row in ensembles:
        found_a = truth(row["found_A_conventional"])
        found_b = truth(row["found_B_conventional"])
        if found_a == found_b:
            continue
        occupancy_a = float(row["target_A_occupancy"])
        occupancy_b = float(row["target_B_occupancy"])
        if math.isclose(occupancy_a, occupancy_b, abs_tol=1e-6):
            totals["equal_excluded"] += 1
            continue
        missed = "B" if found_a else "A"
        minor = "A" if occupancy_a < occupancy_b else "B"
        rank = "minor" if missed == minor else "major"
        counts[(row["site"], rank)] += 1
        totals[rank] += 1
    sites = sorted({site for site, _rank in counts})
    return [
        {
            "site": site,
            "minor_missed": counts[(site, "minor")],
            "major_missed": counts[(site, "major")],
            "single_state_failures": (
                counts[(site, "minor")] + counts[(site, "major")]
            ),
        }
        for site in sites
    ], totals


def missed_rank_optimizer_rows(
    optimizer_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Historical raw greedy diagnostic that defines the 142/45 control."""
    normalized = [
        {
            "site": row["site"],
            "found_A_conventional": row["found_A"],
            "found_B_conventional": row["found_B"],
            "target_A_occupancy": row["target_A_occupancy"],
            "target_B_occupancy": row["target_B_occupancy"],
        }
        for row in optimizer_rows
    ]
    return missed_rank_rows(normalized)


def parse_values(value: str) -> list[float]:
    return [float(item) for item in value.split(";")]


def duplication_summary(
    optimizer_rows: list[dict[str, str]],
    audit: dict[str, object],
) -> dict[str, object]:
    starts: set[tuple[str, int]] = set()
    group_count = 0
    nonprimary_occupancies = []
    nonprimary_boundary_occupancies = []
    nonprimary_stage1_occupancies = []
    pairwise_distances = []
    coordinates = audit["coordinates"]
    metadata = audit["metadata"]
    per_site_starts: dict[str, set[int]] = defaultdict(set)
    per_site_groups: dict[str, int] = defaultdict(int)
    per_site_nonprimary: dict[str, list[float]] = defaultdict(list)

    for row in optimizer_rows:
        site, start = row["site"], int(row["start"])
        occupancies = parse_values(row["occupancies"])
        boundary_occupancies = parse_values(
            row["fixed_boundary_occupancies"]
        )
        stage1_occupancies = parse_values(row["stage1_occupancies"])
        rmsd_a = parse_values(row["rmsd_to_A"])
        rmsd_b = parse_values(row["rmsd_to_B"])
        groups: dict[str, list[int]] = defaultdict(list)
        for index, (occupancy, distance_a, distance_b) in enumerate(
            zip(occupancies, rmsd_a, rmsd_b)
        ):
            if occupancy <= 0.05:
                continue
            if distance_a < 1.0 and distance_a <= distance_b:
                groups["A"].append(index)
            elif distance_b < 1.0:
                groups["B"].append(index)
        for state, members in groups.items():
            if len(members) < 2:
                continue
            starts.add((site, start))
            per_site_starts[site].add(start)
            group_count += 1
            per_site_groups[site] += 1
            primary = min(
                members,
                key=lambda index: (
                    rmsd_a[index] if state == "A" else rmsd_b[index]
                ),
            )
            for index in members:
                if index == primary:
                    continue
                nonprimary_occupancies.append(occupancies[index])
                per_site_nonprimary[site].append(occupancies[index])
                if math.isfinite(boundary_occupancies[index]):
                    nonprimary_boundary_occupancies.append(
                        boundary_occupancies[index]
                    )
                if math.isfinite(stage1_occupancies[index]):
                    nonprimary_stage1_occupancies.append(
                        stage1_occupancies[index]
                    )
            names = metadata[site]["names"]
            resname = metadata[site]["resname"]
            for offset, left in enumerate(members):
                for right in members[offset + 1:]:
                    left_id = f"{site}__{start:03d}__{left}"
                    right_id = f"{site}__{start:03d}__{right}"
                    pairwise_distances.append(float(symmetry_aware_rmsd(
                        coordinates[left_id],
                        coordinates[right_id],
                        names,
                        resname,
                    )))

    representative_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in audit["active"]:
        if (
            truth(row["merge_cluster_representative"])
            and float(row["assigned_occupancy"]) > 0.05
        ):
            representative_counts[(row["site"], int(row["start"]))] += 1
    final_group_sizes = list(representative_counts.values())
    return {
        "starts_with_same_state_duplicates": len(starts),
        "duplicate_groups": group_count,
        "nonprimary_members": len(nonprimary_occupancies),
        "nonprimary_occupancy_median": percentile(
            nonprimary_occupancies, 0.5
        ),
        "nonprimary_occupancy_q25": percentile(nonprimary_occupancies, 0.25),
        "nonprimary_occupancy_q75": percentile(nonprimary_occupancies, 0.75),
        "nonprimary_occupancy_gt_0p20": sum(
            value > 0.20 for value in nonprimary_occupancies
        ),
        "nonprimary_boundary_occupancy_median": percentile(
            nonprimary_boundary_occupancies, 0.5
        ),
        "nonprimary_stage1_occupancy_median": percentile(
            nonprimary_stage1_occupancies, 0.5
        ),
        "nonprimary_final_within_0p05_of_quarter": sum(
            abs(value - 0.25) <= 0.05 for value in nonprimary_occupancies
        ),
        "nonprimary_final_below_0p10": sum(
            value < 0.10 for value in nonprimary_occupancies
        ),
        "within_group_pairwise_rmsd_median_A": percentile(
            pairwise_distances, 0.5
        ),
        "post_merge_distinct_conformers_median": percentile(
            final_group_sizes, 0.5
        ),
        "post_merge_distinct_conformers_mean": (
            float(np.mean(final_group_sizes)) if final_group_sizes else float("nan")
        ),
        "per_site": {
            site: {
                "starts_with_same_state_duplicates": len(
                    per_site_starts[site]
                ),
                "duplicate_groups": per_site_groups[site],
                "nonprimary_members": len(per_site_nonprimary[site]),
                "nonprimary_occupancy_median": percentile(
                    per_site_nonprimary[site], 0.5
                ),
                "nonprimary_occupancy_gt_0p20": sum(
                    value > 0.20 for value in per_site_nonprimary[site]
                ),
            }
            for site in sorted(metadata)
        },
    }


def extra_summary(active: list[dict[str, str]]) -> dict[str, object]:
    extras = [
        float(row["assigned_occupancy"])
        for row in active
        if truth(row["merge_cluster_representative"])
        and row["assignment"] == "other"
    ]
    starts_005 = {
        (row["site"], int(row["start"]))
        for row in active
        if truth(row["merge_cluster_representative"])
        and row["assignment"] == "other"
        and float(row["assigned_occupancy"]) > 0.05
    }
    starts_010 = {
        (row["site"], int(row["start"]))
        for row in active
        if truth(row["merge_cluster_representative"])
        and row["assignment"] == "other"
        and float(row["assigned_occupancy"]) > 0.10
    }
    return {
        "unmatched_extra_conformers": len(extras),
        "extra_bearing_starts_gt_0p05": len(starts_005),
        "extra_bearing_starts_gt_0p10": len(starts_010),
        "extra_occupancy_median": percentile(extras, 0.5),
        "extra_occupancy_q25": percentile(extras, 0.25),
        "extra_occupancy_q75": percentile(extras, 0.75),
    }


def occupancy_summary(ensembles: list[dict[str, str]]) -> dict[str, object]:
    deficits = [
        (
            float(row["target_A_occupancy"])
            + float(row["target_B_occupancy"])
            - float(row["predicted_A_occupancy"])
            - float(row["predicted_B_occupancy"])
        )
        for row in ensembles
        if truth(row["both_found_conventional"])
    ]
    errors_a = [
        abs(
            float(row["predicted_A_occupancy"])
            - float(row["target_A_occupancy"])
        )
        for row in ensembles
        if truth(row["both_found_conventional"])
    ]
    errors_b = [
        abs(
            float(row["predicted_B_occupancy"])
            - float(row["target_B_occupancy"])
        )
        for row in ensembles
        if truth(row["both_found_conventional"])
    ]
    return {
        "recovered_starts": len(deficits),
        "all_recovered_undershoot": sum(value > 0 for value in deficits),
        "A_plus_B_deficit_median": percentile(deficits, 0.5),
        "A_plus_B_deficit_q95": percentile(deficits, 0.95),
        "A_plus_B_deficit_max": max(deficits) if deficits else float("nan"),
        "matched_A_absolute_error_median": percentile(errors_a, 0.5),
        "matched_B_absolute_error_median": percentile(errors_b, 0.5),
    }


def separation_rows(audit: dict[str, object]) -> list[dict[str, object]]:
    output = []
    coordinates = audit["coordinates"]
    metadata = audit["metadata"]
    for row in audit["ensembles"]:
        if not truth(row["both_found_conventional"]):
            continue
        candidate_a = row["assigned_pair_candidate_A"]
        candidate_b = row["assigned_pair_candidate_B"]
        site = row["site"]
        info = metadata[site]
        assigned = float(symmetry_aware_rmsd(
            coordinates[candidate_a],
            coordinates[candidate_b],
            info["names"],
            info["resname"],
        ))
        deposited = float(info["deposited_separation"])
        output.append({
            "site": site,
            "start": int(row["start"]),
            "assigned_pair_separation_A": assigned,
            "deposited_A_B_separation_A": deposited,
            "assigned_over_deposited": assigned / deposited,
        })
    return output


def unfreeze_summary(optimizer_rows: list[dict[str, str]]) -> dict[str, object]:
    pre = [float(row["unfreeze_pre_density_loss"]) for row in optimizer_rows]
    post = [float(row["unfreeze_post_density_loss"]) for row in optimizer_rows]
    finite = [
        (left, right)
        for left, right in zip(pre, post)
        if math.isfinite(left) and math.isfinite(right)
    ]
    deltas = [right - left for left, right in finite]
    relative = [
        (right - left) / max(abs(left), 1e-12) for left, right in finite
    ]
    return {
        "starts": len(finite),
        "density_loss_delta_median": percentile(deltas, 0.5),
        "density_loss_delta_q25": percentile(deltas, 0.25),
        "density_loss_delta_q75": percentile(deltas, 0.75),
        "density_loss_increased": sum(value > 0 for value in deltas),
        "relative_increase_gt_0p10": sum(value > 0.10 for value in relative),
        "relative_delta_median": percentile(relative, 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-audit-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--control-results-root", type=Path, action="append", required=True
    )
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    populations: dict[str, tuple[dict[str, object], list[dict[str, str]]]] = {}
    populations["fixed_0_control"] = (
        load_audits(args.control_audit_root),
        load_optimizer_rows(args.control_results_root),
    )
    for fixed in (100, 200, 300):
        arm = args.sweep_root / f"fixed_{fixed}"
        populations[f"fixed_{fixed}"] = (
            load_audits([
                arm / "audit" / "original5",
                arm / "audit" / "expanded15",
            ]),
            load_optimizer_rows([
                arm / "shards" / "original5",
                arm / "shards" / "expanded15",
            ]),
        )

    cascade_output, primary_output = [], []
    summary: dict[str, object] = {
        "metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
        "metric_changed": False,
        "arms": {},
        "control_trajectory_available": False,
    }
    control_counts = None
    for label, (audit, optimizer_rows) in populations.items():
        cascade_rows, cascade_counts = cascade(
            audit["ensembles"], audit["active"]
        )
        primary_rows, primary_totals = missed_rank_optimizer_rows(
            optimizer_rows
        )
        v3_primary_rows, v3_primary_totals = missed_rank_rows(
            audit["ensembles"]
        )
        duplication = duplication_summary(optimizer_rows, audit)
        extras = extra_summary(audit["active"])
        occupancy = occupancy_summary(audit["ensembles"])
        separation = separation_rows(audit)
        ratios = [float(row["assigned_over_deposited"]) for row in separation]
        separation_summary = {
            "recovered_starts": len(separation),
            "ratio_median": percentile(ratios, 0.5),
            "ratio_q25": percentile(ratios, 0.25),
            "ratio_q75": percentile(ratios, 0.75),
            "count_below_0p5": sum(value < 0.5 for value in ratios),
        }
        boundary = (
            {}
            if label == "fixed_0_control"
            else unfreeze_summary(optimizer_rows)
        )
        totals = {
            stage: sum(counts[stage] for counts in cascade_counts.values())
            for stage in STAGES
        }
        summary["arms"][label] = {
            "cascade": totals,
            "single_state_failures_by_missed_rank_greedy_diagnostic": (
                primary_totals
            ),
            "single_state_failures_by_missed_rank_frozen_v3": (
                v3_primary_totals
            ),
            "same_state_duplication": duplication,
            "unmatched_extras": extras,
            "matched_occupancy": occupancy,
            "assigned_pair_separation": separation_summary,
            "unfreeze_boundary": boundary,
        }
        for row in cascade_rows:
            cascade_output.append({"arm": label, **row})
        for row in primary_rows:
            primary_output.append({
                "arm": label,
                "matching": "raw_greedy_control_comparable",
                **row,
            })
        for row in v3_primary_rows:
            primary_output.append({
                "arm": label,
                "matching": "frozen_v3_one_to_one",
                **row,
            })
        if label == "fixed_0_control":
            control_counts = cascade_counts
        for site in sorted({row["site"] for row in separation}):
            site_ratios = [
                float(row["assigned_over_deposited"])
                for row in separation if row["site"] == site
            ]
            summary.setdefault("separation_per_site", []).append({
                "arm": label,
                "site": site,
                "recovered_starts": len(site_ratios),
                "ratio_median": percentile(site_ratios, 0.5),
                "count_below_0p5": sum(value < 0.5 for value in site_ratios),
            })

    if control_counts is None:
        raise RuntimeError("control cascade missing")
    tail_rows = []
    for row in cascade_output:
        if row["arm"] == "fixed_0_control" or row["site"] not in TAIL_SITES:
            continue
        control = control_counts[row["site"]]
        tail_rows.append({
            **row,
            **{
                f"delta_{stage}": int(row[stage]) - control[stage]
                for stage in STAGES
            },
        })

    control_primary = summary["arms"]["fixed_0_control"][
        "single_state_failures_by_missed_rank_greedy_diagnostic"
    ]
    if (
        control_primary["minor"] != 142
        or control_primary["major"] != 45
    ):
        raise RuntimeError(
            "frozen control missed-rank counts do not reproduce 142/45: "
            f"{control_primary}"
        )
    control_cascade = summary["arms"]["fixed_0_control"]["cascade"]
    if control_cascade["found"] != 742 or control_cascade["strict"] != 626:
        raise RuntimeError(
            f"frozen cascade mismatch: {control_cascade}"
        )

    atomic_csv(args.output / "cascade_per_site.csv", cascade_output)
    atomic_csv(
        args.output / "single_state_failures_per_site.csv", primary_output
    )
    atomic_csv(args.output / "tail_site_deltas.csv", tail_rows)
    atomic_csv(
        args.output / "assigned_pair_separation_per_site.csv",
        summary.pop("separation_per_site"),
    )
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
