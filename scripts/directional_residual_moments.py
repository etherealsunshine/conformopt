#!/usr/bin/env python3
"""Measure neutral-start residual first moments for the clean-D1 sites."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import qfit  # noqa: F401
import numpy as np

from clean_d1_benchmark import site_key
from run_d1_aprime_sequential import APrimeSequential, rmsd


def angle_deg(left: np.ndarray, right: np.ndarray) -> float | None:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    cosine = np.dot(left, right) / (left_norm * right_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def measure_site(task: dict[str, object]) -> dict[str, object]:
    key = str(task["site_key"])
    site = tuple(task["site"])
    base = APrimeSequential(
        Path(task["work_root"]), 80, 6, *site,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=task["device"],
        start_pdb=task["start_pdb"], b_factor_mode="single_conformer",
    )
    neutral = base.base.initial_window
    neutral_density = base.base.model_density_batch(
        neutral, slots=np.asarray([0]), b_offset=0.0
    )[0]
    residual = base.base.target - neutral_density
    grid = base.base._renderer_grid.detach().cpu().numpy()  # pylint: disable=protected-access
    a_window = base.base.window_for_deposited_a()
    b_window = base.base.window_for_deposited_b()
    occupancy = np.asarray(base.base.deposited_occupancies, dtype=float)
    minor_index = int(np.argmin(occupancy))
    atom_indices = np.asarray(base.base.model_atom_indices, dtype=int)
    names = base.base.window.name
    residue_ids = base.base.window.residues
    rows = []
    for model_index in atom_indices:
        atom_xyz = neutral[model_index]
        distances = np.linalg.norm(grid - atom_xyz[None, :], axis=1)
        nearby = distances <= 2.0
        displacement = grid[nearby] - atom_xyz[None, :]
        weights = residual[nearby]
        moment = np.sum(weights[:, None] * displacement, axis=0)
        a_to_b = b_window[model_index] - a_window[model_index]
        neutral_to_minor = (
            (a_window if minor_index == 0 else b_window)[model_index] - atom_xyz
        )
        rows.append({
            "atom_index": int(model_index),
            "atom": str(names[model_index]),
            "neutral_xyz": atom_xyz.tolist(),
            "nearby_voxels": int(np.count_nonzero(nearby)),
            "residual_sum": float(np.sum(weights)),
            "moment_vector": moment.tolist(),
            "moment_magnitude": float(np.linalg.norm(moment)),
            "moment_direction": (
                (moment / np.linalg.norm(moment)).tolist()
                if np.linalg.norm(moment) > 0.0 else None
            ),
            "deposited_A_to_B_vector": a_to_b.tolist(),
            "angle_to_A_to_B_deg": angle_deg(moment, a_to_b),
            "neutral_to_minor_vector": neutral_to_minor.tolist(),
            "angle_to_neutral_to_minor_deg": angle_deg(moment, neutral_to_minor),
            "neutral_to_A_distance": float(np.linalg.norm(a_window[model_index] - atom_xyz)),
            "neutral_to_B_distance": float(np.linalg.norm(b_window[model_index] - atom_xyz)),
        })
    return {
        "site": key,
        "deposited_occupancies_A_B": occupancy.tolist(),
        "minor_state": "A" if minor_index == 0 else "B",
        "mask_voxels": int(len(base.base.target)),
        "moment_definition": "sum_{||x-x_atom||<=2A} (target-rho_neutral)(x) * (x-x_atom)",
        "backbone_atoms": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    manifest = {site_key(row): row for row in json.loads(args.manifest.read_text())}
    tasks = []
    for key in ("6ZWK_B_PHE47", "8R7O_C_THR1681"):
        row = manifest[key]
        site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
        tasks.append({
            "site_key": key, "site": list(site),
            "start_pdb": str(args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"),
            "work_root": str(args.output / key), "device": args.device,
        })
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(measure_site, tasks))
    report = {"status": "complete", "device": args.device, "sites": results}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
