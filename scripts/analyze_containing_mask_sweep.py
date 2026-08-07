"""Compile containing-mask arms under frozen synthetic metric v3."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
)


ARMS = ("containing_uniform", "containing_variance_weighted")
EXPECTED_CONTROL_CASCADE = {
    "found": 742,
    "occupancy": 714,
    "rotamer": 710,
    "direct": 710,
    "symmetry": 710,
    "strict": 626,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_control_optimizer_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for source in read_csv(path):
        shard = Path(source["shard"])
        starts = shard / "synthetic" / f"{source['site']}_starts.csv"
        rows.extend(read_csv(starts))
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 control rows, found {len(rows)}")
    return rows


def physics_loss_summary(
    rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    fields = (
        "final_vdw_loss",
        "final_rotamer_loss",
        "final_symmetry_loss",
    )

    def summarize(population: list[dict[str, str]]) -> dict[str, object]:
        result = {}
        for field in fields:
            values = [float(row[field]) for row in population]
            result[f"{field}_median"] = percentile(values, 0.5)
            result[f"{field}_q25"] = percentile(values, 0.25)
            result[f"{field}_q75"] = percentile(values, 0.75)
        return result

    sites = sorted({row["site"] for row in rows})
    return summarize(rows), [
        {
            "site": site,
            **summarize([row for row in rows if row["site"] == site]),
        }
        for site in sites
    ]


def realized_signal(
    arm_root: Path,
    optimizer_rows: list[dict[str, str]],
    weighted: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    masks = {}
    calibration = []
    for path in sorted((arm_root / "shards").glob("*/*/run_config.json")):
        config = json.loads(path.read_text())
        masks.update(config["resolved_density_masks"])
    for path in sorted((arm_root / "calibration").glob("*/calibration.csv")):
        calibration.extend(
            row for row in read_csv(path) if row["target"] == "synthetic"
        )
    occupancy = {}
    for row in optimizer_rows:
        occupancy.setdefault(row["site"], (
            float(row["target_A_occupancy"]),
            float(row["target_B_occupancy"]),
        ))

    rows = []
    for row in calibration:
        site = row["site"]
        occ_a, occ_b = occupancy[site]
        major = "A" if occ_a >= occ_b else "B"
        gap = (
            float(row[f"loss_{major}_only"])
            - float(row["loss_A_plus_B"])
        )
        n_voxels = int(masks[site]["voxel_count"])
        value = (
            gap
            if weighted
            else gap * n_voxels / (2.0 * (n_voxels - 1))
        )
        rows.append({
            "site": site,
            "major_state": major,
            "mask_voxels": n_voxels,
            "reachable_atoms_outside_mask": masks[site][
                "reachable_atoms_outside_mask"
            ],
            "deposited_A_atoms_outside_mask": masks[site][
                "deposited_A_atoms_outside_mask"
            ],
            "deposited_B_atoms_outside_mask": masks[site][
                "deposited_B_atoms_outside_mask"
            ],
            "signal_definition": (
                "weighted zscore MSE gap"
                if weighted else "1 - Pearson correlation"
            ),
            "realized_major_collapse_signal": value,
        })
    values = [float(row["realized_major_collapse_signal"]) for row in rows]
    return {
        "definition": rows[0]["signal_definition"],
        "median": percentile(values, 0.5),
        "mean": float(np.mean(values)),
        "q25": percentile(values, 0.25),
        "q75": percentile(values, 0.75),
        "minimum": min(values),
        "maximum": max(values),
        "voxel_count_min": min(int(row["mask_voxels"]) for row in rows),
        "voxel_count_max": max(int(row["mask_voxels"]) for row in rows),
        "reachable_atoms_outside": sum(
            int(row["reachable_atoms_outside_mask"]) for row in rows
        ),
        "deposited_atoms_outside": sum(
            int(row["deposited_A_atoms_outside_mask"])
            + int(row["deposited_B_atoms_outside_mask"])
            for row in rows
        ),
    }, rows


def weight_distribution(arm_root: Path) -> list[dict[str, object]]:
    masks = {}
    for path in sorted((arm_root / "shards").glob("*/*/run_config.json")):
        config = json.loads(path.read_text())
        masks.update(config["resolved_density_masks"])
    rows = []
    for site, mask in sorted(masks.items()):
        voxel_count = int(mask["voxel_count"])
        weight_mean = float(mask["density_weight_mean"])
        weight_total = voxel_count * weight_mean
        weight_max = float(mask["density_weight_max"])
        rows.append({
            "site": site,
            "weight_unit": "voxel",
            "voxel_count": voxel_count,
            "weight_min": float(mask["density_weight_min"]),
            "weight_mean": weight_mean,
            "weight_max": weight_max,
            "total_weight": weight_total,
            "max_voxel_weight_fraction": weight_max / weight_total,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-audit-root", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--control-source-provenance", type=Path, required=True
    )
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    populations = {
        "production_control": (
            load_audits(args.control_audit_root),
            load_control_optimizer_rows(args.control_source_provenance),
        )
    }
    for arm in ARMS:
        root = args.sweep_root / arm
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

    summary = {
        "metric": "qfit-synth20-merge050-one-to-one-tmol044-v3",
        "metric_changed": False,
        "control_audit_roots": [
            str(path) for path in args.control_audit_root
        ],
        "control": {
            "found": 742,
            "strict": 626,
            "raw_minor_major_misses": [142, 45],
        },
        "arms": {},
    }
    cascade_output = []
    primary_output = []
    tail_output = []
    physics_output = []
    signal_output = []
    weight_output = []
    control_counts = None

    for arm, (audit, optimizer_rows) in populations.items():
        cascade_rows, cascade_counts = cascade(
            audit["ensembles"], audit["active"]
        )
        raw_rows, raw_totals = missed_rank_optimizer_rows(optimizer_rows)
        v3_rows, v3_totals = missed_rank_rows(audit["ensembles"])
        physics, physics_by_site = physics_loss_summary(optimizer_rows)
        totals = {
            stage: sum(values[stage] for values in cascade_counts.values())
            for stage in STAGES
        }
        arm_summary = {
            "cascade": totals,
            "single_state_failures_raw_optimizer": raw_totals,
            "single_state_failures_frozen_v3": v3_totals,
            "same_state_duplication": duplication_summary(
                optimizer_rows, audit
            ),
            "unmatched_extras": extra_summary(audit["active"]),
            "matched_occupancy": occupancy_summary(audit["ensembles"]),
            "geometry_stage_losses": physics,
        }
        if arm != "production_control":
            signal, signal_rows = realized_signal(
                args.sweep_root / arm,
                optimizer_rows,
                weighted=arm.endswith("variance_weighted"),
            )
            arm_summary["realized_signal"] = signal
            signal_output.extend({"arm": arm, **row} for row in signal_rows)
            if arm.endswith("variance_weighted"):
                weight_output.extend(
                    {"arm": arm, **row}
                    for row in weight_distribution(args.sweep_root / arm)
                )
        summary["arms"][arm] = arm_summary
        cascade_output.extend({"arm": arm, **row} for row in cascade_rows)
        primary_output.extend({
            "arm": arm,
            "matching": "raw_optimizer",
            **row,
        } for row in raw_rows)
        primary_output.extend({
            "arm": arm,
            "matching": "frozen_v3",
            **row,
        } for row in v3_rows)
        physics_output.extend(
            {"arm": arm, **row} for row in physics_by_site
        )
        if arm == "production_control":
            if totals != EXPECTED_CONTROL_CASCADE:
                raise ValueError(
                    "control audit roots do not reproduce frozen v3: "
                    f"expected {EXPECTED_CONTROL_CASCADE}, found {totals}"
                )
            control_counts = cascade_counts
        else:
            for site in sorted(TAIL_SITES):
                tail_output.append({
                    "arm": arm,
                    "site": site,
                    **{
                        f"{stage}_delta": (
                            cascade_counts[site][stage]
                            - control_counts[site][stage]
                        )
                        for stage in STAGES
                    },
                })

    args.output.mkdir(parents=True)
    atomic_json(args.output / "summary.json", summary)
    atomic_csv(args.output / "cascade_by_site.csv", cascade_output)
    atomic_csv(args.output / "minor_major_misses_by_site.csv", primary_output)
    atomic_csv(args.output / "tail_site_deltas.csv", tail_output)
    atomic_csv(args.output / "geometry_losses_by_site.csv", physics_output)
    atomic_csv(args.output / "realized_signal_by_site.csv", signal_output)
    atomic_csv(
        args.output / "weight_distribution_by_site.csv", weight_output
    )


if __name__ == "__main__":
    main()
