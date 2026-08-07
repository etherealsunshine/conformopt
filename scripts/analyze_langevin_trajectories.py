"""Analyze saved Stage-1 Langevin trajectories without rerunning optimization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from density_denoiser.five_site_optimizer import (
    _unique_canonical_centers_radians,
)
from scripts.diagnose_frozen_v3_occupancy_pooling import (
    atomic_csv,
    atomic_json,
    describe,
    identify_single_recovery,
    truth,
)


def parse_vector(value: str) -> list[float]:
    return [float(item) for item in value.split(";")]


def wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def residue_type_from_site(site: str) -> str:
    match = re.search(r"_([A-Z]{3})-?\d+$", site)
    if match is None:
        raise ValueError(f"cannot infer residue type from {site}")
    return match.group(1)


def nearest_well_indices(
    physical_chi: np.ndarray, residue_type: str
) -> np.ndarray:
    """Classify (..., n_chi) physical angles by production marginal wells."""
    output = np.empty(physical_chi.shape, dtype=np.int16)
    for chi_index in range(physical_chi.shape[-1]):
        centers = np.asarray(
            _unique_canonical_centers_radians(residue_type, chi_index),
            dtype=np.float64,
        )
        distance = np.abs(wrap(
            physical_chi[..., chi_index, None] - centers
        ))
        output[..., chi_index] = np.argmin(distance, axis=-1)
    return output


def trajectory_slot_metrics(
    chi_radians: np.ndarray,
    base_physical_chi: np.ndarray,
    delta_direction: np.ndarray,
    residue_type: str,
) -> list[dict[str, float | int]]:
    physical = wrap(
        base_physical_chi[None, None, :]
        + delta_direction[None, None, :] * chi_radians
    )
    increments = wrap(np.diff(physical, axis=0))
    travelled = np.linalg.norm(increments, axis=-1).sum(axis=0)
    wells = nearest_well_indices(physical, residue_type)
    changed = wells[1:] != wells[:-1]
    return [
        {
            "chi_space_distance_travelled_degrees": float(
                math.degrees(float(travelled[slot]))
            ),
            "rotamer_well_boundary_crossings": int(
                changed[:, slot, :].sum()
            ),
            "steps_with_any_well_boundary_crossing": int(
                changed[:, slot, :].any(axis=1).sum()
            ),
        }
        for slot in range(chi_radians.shape[1])
    ]


def loss_at_step(
    steps: np.ndarray,
    phases: np.ndarray,
    losses: np.ndarray,
    wanted_step: int,
) -> float:
    loss_steps = steps[1:]
    loss_phases = phases[1:]
    selected = np.flatnonzero(
        (loss_steps == wanted_step) & (loss_phases == 1)
    )
    return float(losses[selected[0]]) if len(selected) else math.nan


def read_optimizer_rows(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    output = {}
    for path in sorted(root.glob("shards/*/*/synthetic/*_starts.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                output[(row["site"], int(row["start"]))] = row
    return output


def summarize_arm(rows: list[dict[str, object]]) -> dict[str, object]:
    failed = [row for row in rows if bool(row["failed_pair_recovery"])]
    single = [row for row in rows if bool(row["single_recovery"])]
    comparable = [
        row for row in failed
        if math.isfinite(float(row["stage1_density_loss_step_300"]))
        and math.isfinite(float(row["stage1_density_loss_step_500"]))
    ]
    return {
        "starts": len(rows),
        "failed_pair_recovery_starts": len(failed),
        "single_recovery_starts": len(single),
        "failed_starts_with_any_well_crossing": sum(
            bool(row["any_slot_crossed_rotamer_well"]) for row in failed
        ),
        "failed_starts_with_any_well_crossing_fraction": (
            sum(bool(row["any_slot_crossed_rotamer_well"]) for row in failed)
            / len(failed) if failed else 0.0
        ),
        "failed_stage1_loss_step300": describe(
            float(row["stage1_density_loss_step_300"]) for row in comparable
        ),
        "failed_stage1_loss_step500": describe(
            float(row["stage1_density_loss_step_500"]) for row in comparable
        ),
        "failed_stage1_relative_decline_300_to_500": describe(
            float(row["stage1_relative_loss_decline_300_to_500"])
            for row in comparable
        ),
        "failed_flat_by_step300_count": sum(
            bool(row["stage1_loss_flat_300_to_500_within_1pct"])
            for row in comparable
        ),
        "failed_flat_by_step300_fraction": (
            sum(
                bool(row["stage1_loss_flat_300_to_500_within_1pct"])
                for row in comparable
            ) / len(comparable) if comparable else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", action="append", required=True,
        help="LABEL=/absolute/arm/root",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    start_rows: list[dict[str, object]] = []
    slot_rows: list[dict[str, object]] = []
    arm_summaries = {}
    for arm_item in args.arm:
        arm, raw_root = arm_item.split("=", 1)
        root = Path(raw_root)
        optimizer_rows = read_optimizer_rows(root)
        if not optimizer_rows:
            raise ValueError(f"no optimizer rows under {root}")
        arm_start_rows = []
        for key, endpoint in sorted(optimizer_rows.items()):
            trajectory_path = (
                root / "shards"
                / ("original5" if key[0] in {
                    "4C16_A_MET258", "7F72_A_MET103", "3A1C_B_ARG447",
                    "6H59_B_ARG144", "8Q6Q_B_ASP81",
                } else "expanded15")
                / key[0] / "trajectories"
                / f"synthetic_start_{key[1]:03d}.npz"
            )
            if not trajectory_path.exists():
                raise FileNotFoundError(trajectory_path)
            payload = np.load(trajectory_path, allow_pickle=False)
            phase1 = np.asarray(payload["phase"]) == 1
            chi_stage1 = np.asarray(payload["chi_radians"])[phase1]
            slot_metrics = trajectory_slot_metrics(
                chi_stage1,
                np.asarray(payload["base_physical_chi_radians"]),
                np.asarray(payload["delta_direction"]),
                residue_type_from_site(key[0]),
            )
            identity = identify_single_recovery(endpoint)
            found_a, found_b = truth(endpoint["found_A"]), truth(
                endpoint["found_B"]
            )
            failed = not (found_a and found_b)
            rmsd_a = parse_vector(endpoint["rmsd_to_A"])
            rmsd_b = parse_vector(endpoint["rmsd_to_B"])
            occupancies = parse_vector(endpoint["occupancies"])
            missed = str(identity["missed_state"]) if identity else ""
            loss300 = loss_at_step(
                payload["step"], payload["phase"],
                payload["density_loss_pre"], 300
            )
            loss500 = loss_at_step(
                payload["step"], payload["phase"],
                payload["density_loss_pre"], 500
            )
            relative_decline = (
                (loss300 - loss500) / max(abs(loss300), 1e-12)
                if math.isfinite(loss300) and math.isfinite(loss500)
                else math.nan
            )
            start_record = {
                "arm": arm,
                "site": key[0],
                "start": key[1],
                "found_A": found_a,
                "found_B": found_b,
                "failed_pair_recovery": failed,
                "single_recovery": identity is not None,
                "recovery_rank": (
                    identity["recovery_rank"] if identity else ""
                ),
                "missed_state": missed,
                "stage1_steps": int(phase1.sum() - 1),
                "stage1_density_loss_step_300": loss300,
                "stage1_density_loss_step_500": loss500,
                "stage1_endpoint_density_loss": float(
                    endpoint["stage1_density_loss"]
                ),
                "stage1_relative_loss_decline_300_to_500": relative_decline,
                "stage1_loss_flat_300_to_500_within_1pct": (
                    math.isfinite(relative_decline)
                    and abs(relative_decline) <= 0.01
                ),
                "slots_crossing_rotamer_wells": sum(
                    int(record["rotamer_well_boundary_crossings"]) > 0
                    for record in slot_metrics
                ),
                "any_slot_crossed_rotamer_well": any(
                    int(record["rotamer_well_boundary_crossings"]) > 0
                    for record in slot_metrics
                ),
                "total_rotamer_well_boundary_crossings": sum(
                    int(record["rotamer_well_boundary_crossings"])
                    for record in slot_metrics
                ),
                "configured_initial_noise_sd_degrees": float(
                    endpoint["stage1_chi_noise_initial_sd_degrees"]
                ),
                "recorded_first_step_noise_sd_degrees": float(
                    payload["chi_noise_sd_degrees"][1]
                ),
                "recorded_final_stage1_noise_sd_degrees": float(
                    payload["chi_noise_sd_degrees"][
                        np.flatnonzero(phase1)[-1]
                    ]
                ),
            }
            start_rows.append(start_record)
            arm_start_rows.append(start_record)
            for slot, metrics in enumerate(slot_metrics):
                nearest = min(rmsd_a[slot], rmsd_b[slot])
                missed_rmsd = (
                    rmsd_a[slot] if missed == "A"
                    else rmsd_b[slot] if missed == "B"
                    else math.nan
                )
                slot_rows.append({
                    "arm": arm,
                    "site": key[0],
                    "start": key[1],
                    "slot": slot,
                    "occupancy": occupancies[slot],
                    **metrics,
                    "escaped_rotamer_well": (
                        int(metrics["rotamer_well_boundary_crossings"]) > 0
                    ),
                    "final_rmsd_to_A_A": rmsd_a[slot],
                    "final_rmsd_to_B_A": rmsd_b[slot],
                    "final_nearest_deposited_rmsd_A": nearest,
                    "final_rmsd_to_missed_A": missed_rmsd,
                    "ended_within_1A_of_missed": (
                        math.isfinite(missed_rmsd) and missed_rmsd < 1.0
                    ),
                    "single_recovery": identity is not None,
                    "missed_state": missed,
                })
        arm_summaries[arm] = summarize_arm(arm_start_rows)

    for arm in arm_summaries:
        escaped = [
            row for row in slot_rows
            if row["arm"] == arm and bool(row["escaped_rotamer_well"])
            and bool(row["single_recovery"])
        ]
        arm_summaries[arm]["escaped_slots_on_single_recovery_starts"] = len(
            escaped
        )
        arm_summaries[arm]["escaped_slots_ending_within_1A_of_missed"] = sum(
            bool(row["ended_within_1A_of_missed"]) for row in escaped
        )
        arm_summaries[arm][
            "escaped_slots_final_nearest_deposited_rmsd_A"
        ] = describe(
            float(row["final_nearest_deposited_rmsd_A"]) for row in escaped
        )

    atomic_csv(args.output / "trajectory_per_start.csv", start_rows)
    atomic_csv(args.output / "trajectory_per_slot.csv", slot_rows)
    atomic_json(args.output / "summary.json", {
        "noise_schedule": (
            "linear_first_step_initial_to_final_stage1_step_zero"
        ),
        "well_definition": (
            "nearest physically unique production marginal canonical center"
        ),
        "flat_definition": (
            "absolute relative density-loss change from step 300 to 500 <=1%"
        ),
        "arms": arm_summaries,
    })


if __name__ == "__main__":
    main()
