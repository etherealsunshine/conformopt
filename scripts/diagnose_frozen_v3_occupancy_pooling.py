"""Diagnose occupancy structure and pooled NNLS refits on frozen v3 endpoints.

This script never optimizes coordinates and never changes the frozen metric.  It
uses the historical raw-greedy 142/45 single-recovery partition requested for
interpretation, the frozen-v3 coordinate payload for geometry, and the frozen
0.5 A protected single-linkage merge rule for site-wide conformer pooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import gemmi
import numpy as np
import torch

from density_denoiser.audit_five_site_endpoints import (
    merge_then_assign_conformers,
)
from density_denoiser.data_pipeline import _grid_coordinates
from density_denoiser.residue_geometry import (
    CHI_SPECS,
    reference_permutations,
    symmetry_aware_rmsd,
)
from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles


ACTIVE_THRESHOLD = 0.05
FOUND_THRESHOLD = 0.10
RMSD_CUTOFF = 1.0
MERGE_THRESHOLD = 0.5
OCCUPANCY_TOLERANCE = 0.20
NNLS_NONZERO_THRESHOLD = 1e-6


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_vector(value: str) -> list[float]:
    return [float(item) for item in value.split(";")]


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
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


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, newline="", delete=False
    ) as handle:
        if rows:
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
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def conventional_rmsd(
    left: np.ndarray,
    right: np.ndarray,
    names: list[str],
    residue_type: str,
) -> float:
    return float(symmetry_aware_rmsd(
        torch.as_tensor(left, dtype=torch.float32),
        torch.as_tensor(right, dtype=torch.float32),
        names,
        residue_type,
    ))


def deposited_midpoint(
    deposited_a: np.ndarray,
    deposited_b: np.ndarray,
    names: list[str],
    residue_type: str,
) -> tuple[np.ndarray, list[int]]:
    """Use the equivalent-label B permutation closest to A, then average."""
    permutations = reference_permutations(names, residue_type)
    best = min(
        permutations,
        key=lambda permutation: float(np.square(
            deposited_a - deposited_b[permutation]
        ).sum(axis=1).mean()),
    )
    return (deposited_a + deposited_b[best]) / 2.0, best


def pairwise_rmsd_matrix(
    coordinates: list[np.ndarray],
    names: list[str],
    residue_type: str,
) -> np.ndarray:
    size = len(coordinates)
    output = np.zeros((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left + 1, size):
            value = conventional_rmsd(
                coordinates[left],
                coordinates[right],
                names,
                residue_type,
            )
            output[left, right] = output[right, left] = value
    return output


def load_optimizer_rows(
    baseline_root: Path,
    replacement_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, str]], dict[str, Path]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    target_paths: dict[str, Path] = {}
    for root in (baseline_root, replacement_root):
        for path in sorted(root.glob("shards/**/synthetic/*_starts.csv")):
            for row in read_csv(path):
                rows[(row["site"], int(row["start"]))] = row
            site = path.name.removesuffix("_starts.csv")
            target_paths[site] = path.parent.parent / f"{site}_targets.npz"
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 frozen optimizer rows, found {len(rows)}")
    if len(target_paths) != 20:
        raise ValueError(f"expected 20 target paths, found {len(target_paths)}")
    return rows, target_paths


def load_v3_payload(
    metric_root: Path,
) -> tuple[
    dict[tuple[str, int, int], dict[str, str]],
    dict[tuple[str, int, int], np.ndarray],
    dict[str, dict[str, object]],
    dict[tuple[str, int], dict[str, str]],
    dict[tuple[str, int], list[dict[str, str]]],
]:
    active_by_slot: dict[tuple[str, int, int], dict[str, str]] = {}
    coordinates: dict[tuple[str, int, int], np.ndarray] = {}
    sites: dict[str, dict[str, object]] = {}
    ensembles: dict[tuple[str, int], dict[str, str]] = {}
    strict_by_start: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for panel in ("original5", "expanded13", "water2"):
        root = metric_root / panel
        for row in read_csv(root / "active_conformer_strict_audit.csv"):
            key = (row["site"], int(row["start"]), int(row["conformer"]))
            active_by_slot[key] = row
            strict_by_start[key[:2]].append(row)
        for row in read_csv(root / "ensemble_strict_audit.csv"):
            ensembles[(row["site"], int(row["start"]))] = row
        payload = json.loads((root / "tmol_inputs.json").read_text())
        for site in payload["sites"]:
            record = dict(site)
            record["_audit_root"] = str(root)
            sites[site["site"]] = record
            for candidate in site["candidates"]:
                key = (
                    site["site"],
                    int(candidate["start"]),
                    int(candidate["conformer"]),
                )
                coordinates[key] = np.asarray(
                    candidate["coordinates"], dtype=np.float32
                )
    if len(active_by_slot) != len(coordinates):
        raise ValueError("every v3 active slot must have saved coordinates")
    if len(ensembles) != 1000 or len(sites) != 20:
        raise ValueError("incomplete v3 audit payload")
    return active_by_slot, coordinates, sites, ensembles, strict_by_start


def raw_slot_rows(
    optimizer_rows: dict[tuple[str, int], dict[str, str]],
) -> dict[tuple[str, int], list[dict[str, object]]]:
    output: dict[tuple[str, int], list[dict[str, object]]] = {}
    for key, row in optimizer_rows.items():
        occupancies = parse_vector(row["occupancies"])
        rmsd_a = parse_vector(row["rmsd_to_A"])
        rmsd_b = parse_vector(row["rmsd_to_B"])
        assignments = row["assignments"].split(";")
        if not (len(occupancies) == len(rmsd_a) == len(rmsd_b) == len(assignments) == 4):
            raise ValueError(f"malformed K=4 row {key}")
        output[key] = [
            {
                "site": key[0],
                "start": key[1],
                "slot": slot,
                "occupancy": occupancies[slot],
                "rmsd_to_A": rmsd_a[slot],
                "rmsd_to_B": rmsd_b[slot],
                "raw_assignment": assignments[slot],
            }
            for slot in range(4)
        ]
    return output


def identify_single_recovery(
    row: dict[str, str],
) -> dict[str, object] | None:
    found_a, found_b = truth(row["found_A"]), truth(row["found_B"])
    if found_a == found_b:
        return None
    target_a = float(row["target_A_occupancy"])
    target_b = float(row["target_B_occupancy"])
    if math.isclose(target_a, target_b, abs_tol=1e-6):
        return None
    minor = "A" if target_a < target_b else "B"
    major = "B" if minor == "A" else "A"
    recovered = "A" if found_a else "B"
    missed = "B" if found_a else "A"
    return {
        "minor_state": minor,
        "major_state": major,
        "recovered_state": recovered,
        "missed_state": missed,
        "recovery_rank": "major_only" if recovered == major else "minor_only",
        "target_A": target_a,
        "target_B": target_b,
    }


def render_density_columns(
    site: dict[str, object],
    representatives: list[np.ndarray],
    target_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    target_npz = np.load(target_path, allow_pickle=False)
    metadata = json.loads(str(target_npz["metadata"].item()))
    mask = np.asarray(target_npz["radial_mask"], dtype=bool)
    center = np.asarray(metadata["center"], dtype=np.float32)
    grid = _grid_coordinates(center, 32, 0.5, None)[mask].reshape(-1, 3)

    audit_root = Path(str(site["_audit_root"]))
    base_b_path = audit_root / str(site["base_pdb_B"])
    structure = gemmi.read_structure(str(base_b_path))
    residue = next(
        residue
        for chain in structure[0] if chain.name == site["chain"]
        for residue in chain
        if residue.seqid.num == int(site["residue_number"])
    )
    atom_lookup = {atom.name.strip(): atom for atom in residue}
    names = list(site["atom_names"])
    sigma2 = np.asarray([
        max(float(atom_lookup[name].b_iso) / (8.0 * math.pi**2), 0.04)
        for name in names
    ], dtype=np.float64)
    weights = np.asarray([
        float(atom_lookup[name].element.atomic_number) for name in names
    ], dtype=np.float64)
    normalization = np.power(2.0 * math.pi * sigma2, -1.5)

    def render(xyz: np.ndarray) -> np.ndarray:
        distance2 = np.square(
            grid[:, None, :] - np.asarray(xyz, dtype=np.float64)[None, :, :]
        ).sum(axis=-1)
        return (
            weights[None, :]
            * normalization[None, :]
            * np.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(axis=1)

    columns = np.column_stack([render(xyz) for xyz in representatives])
    deposited_a, deposited_b = kinematic_control_coordinates(
        site, audit_root
    )
    calibration = (
        float(site["_target_A"]) * render(deposited_a)
        + float(site["_target_B"]) * render(deposited_b)
    )
    # Production replaces the pair target with this renderer-consistent
    # deposited A/B mixture and z-scores it.  Fit the pre-z-score mixture so q
    # remains in occupancy units, then verify against the saved normalized
    # optimizer target vector.
    target = calibration
    saved_optimizer_target = np.asarray(
        np.load(
            target_path.parent
            / f"{site['site']}_optimizer_synthetic_vector.npy"
        ),
        dtype=np.float64,
    )
    normalized_calibration = (
        calibration - calibration.mean()
    ) / max(calibration.std(ddof=1), 1e-6)
    residual = normalized_calibration - saved_optimizer_target
    centered_left = normalized_calibration - normalized_calibration.mean()
    centered_right = saved_optimizer_target - saved_optimizer_target.mean()
    correlation = float(
        np.dot(centered_left, centered_right)
        / max(np.linalg.norm(centered_left) * np.linalg.norm(centered_right), 1e-15)
    )
    diagnostics = {
        "target_calibration_relative_l2": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(saved_optimizer_target), 1e-15)
        ),
        "target_calibration_correlation": correlation,
        "target_calibration_max_absolute_error": float(
            np.max(np.abs(residual))
        ),
        "target_native_sum": float(target.sum()),
        "target_native_norm": float(np.linalg.norm(target)),
        "mask_voxels": int(mask.sum()),
    }
    return columns, target, diagnostics


def kinematic_control_coordinates(
    site: dict[str, object],
    audit_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the exact optimizer A/B controls used for its synthetic target."""
    structures = {}
    for state, field in (("A", "base_pdb_A"), ("B", "base_pdb_B")):
        structure = gemmi.read_structure(str(audit_root / str(site[field])))
        residue = next(
            residue
            for chain in structure[0] if chain.name == site["chain"]
            for residue in chain
            if residue.seqid.num == int(site["residue_number"])
        )
        structures[state] = {
            atom.name.strip(): torch.tensor(
                atom.pos.tolist(), dtype=torch.float32
            )
            for atom in residue
            if atom.element.name != "H"
        }
    names = list(site["atom_names"])
    residue_type = str(site["residue_type"])
    spec = CHI_SPECS[residue_type]
    map_a, map_b = structures["A"], structures["B"]
    chi_a = torch.stack([
        dihedral(*(map_a[name] for name in quartet))
        for quartet in spec["dihedrals"]
    ])
    chi_b = torch.stack([
        dihedral(*(map_b[name] for name in quartet))
        for quartet in spec["dihedrals"]
    ])
    true_delta = wrap_angles(chi_b - chi_a)
    template = torch.stack([map_a[name] for name in names])
    fixed_lookup = {
        name: value for name, value in map_a.items() if name not in names
    }

    def from_delta(delta: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            template,
            names,
            delta,
            list(spec["rotations"]),
            fixed_lookup,
        )

    deposited_b = torch.stack([map_b[name] for name in names])
    plus, minus = from_delta(true_delta), from_delta(-true_delta)
    if symmetry_aware_rmsd(
        minus, deposited_b, names, residue_type
    ) < symmetry_aware_rmsd(plus, deposited_b, names, residue_type):
        true_delta = -true_delta
    return (
        from_delta(torch.zeros_like(true_delta)).numpy().astype(np.float64),
        from_delta(true_delta).numpy().astype(np.float64),
    )


def solve_gram_nnls(
    columns: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    gram = columns.T @ columns
    right = columns.T @ target
    condition = float(np.linalg.cond(gram))
    q, iterations, optimality = lawson_hanson_nnls(gram, right)
    diagnostics: dict[str, object] = {
        "gram_condition_number": condition,
        "gram_condition_number_finite": math.isfinite(condition),
        "solver": (
            "deterministic Lawson-Hanson active-set NNLS on "
            "min ||Gq-b||^2, q>=0"
        ),
        "solver_status": 1,
        "solver_message": "KKT conditions satisfied",
        "solver_iterations": iterations,
        "solver_optimality": optimality,
        "gram_objective_l2": float(np.linalg.norm(gram @ q - right)),
        "density_residual_relative_l2": float(
            np.linalg.norm(columns @ q - target)
            / max(np.linalg.norm(target), 1e-15)
        ),
    }
    return q, gram, right, diagnostics


def lawson_hanson_nnls(
    matrix: np.ndarray,
    right: np.ndarray,
    *,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """Solve ``min ||Ax-b||`` with ``x>=0`` using an active-set method."""
    matrix = np.asarray(matrix, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    rows, columns = matrix.shape
    if right.shape != (rows,):
        raise ValueError("NNLS matrix and right-hand side do not align")
    if tolerance is None:
        tolerance = (
            10.0
            * np.finfo(np.float64).eps
            * max(rows, columns)
            * max(float(np.linalg.norm(matrix, ord=1)), 1.0)
            * max(float(np.linalg.norm(right)), 1.0)
        )
    if max_iterations is None:
        max_iterations = max(30 * columns, 1000)

    solution = np.zeros(columns, dtype=np.float64)
    passive = np.zeros(columns, dtype=bool)
    gradient = matrix.T @ (right - matrix @ solution)
    iterations = 0
    while np.any((~passive) & (gradient > tolerance)):
        if iterations >= max_iterations:
            raise RuntimeError("Lawson-Hanson NNLS exceeded outer iteration limit")
        eligible = np.where(~passive, gradient, -np.inf)
        passive[int(np.argmax(eligible))] = True
        candidate = solution.copy()
        while True:
            iterations += 1
            if iterations >= max_iterations:
                raise RuntimeError(
                    "Lawson-Hanson NNLS exceeded inner iteration limit"
                )
            candidate.fill(0.0)
            candidate[passive] = np.linalg.lstsq(
                matrix[:, passive], right, rcond=None
            )[0]
            if np.all(candidate[passive] > tolerance):
                break
            violating = passive & (candidate <= tolerance)
            denominators = solution[violating] - candidate[violating]
            valid = denominators > 0
            alpha = (
                min(solution[violating][valid] / denominators[valid])
                if np.any(valid) else 0.0
            )
            solution += alpha * (candidate - solution)
            drop = passive & (solution <= tolerance)
            passive[drop] = False
            solution[drop] = 0.0
        solution = candidate
        gradient = matrix.T @ (right - matrix @ solution)

    solution[solution <= tolerance] = 0.0
    residual_gradient = matrix.T @ (right - matrix @ solution)
    kkt_violation = max(
        float(np.max(np.abs(residual_gradient[passive])))
        if np.any(passive) else 0.0,
        float(np.max(np.maximum(residual_gradient[~passive], 0.0)))
        if np.any(~passive) else 0.0,
        float(np.max(np.maximum(-solution, 0.0))),
    )
    return solution, iterations, kkt_violation


def nearest_state_assignment(
    rmsd_a: float,
    rmsd_b: float,
) -> str:
    if rmsd_a < RMSD_CUTOFF and rmsd_a <= rmsd_b:
        return "A"
    if rmsd_b < RMSD_CUTOFF:
        return "B"
    return "other"


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
    optimizer_rows, target_paths = load_optimizer_rows(
        args.baseline_root, args.replacement_root
    )
    (
        active_by_slot,
        coordinates,
        sites,
        ensembles,
        strict_by_start,
    ) = load_v3_payload(metric_root)
    slots_by_start = raw_slot_rows(optimizer_rows)

    # Part 1 and 2: exact historical raw-greedy 142/45 cohorts.
    start_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    population_counts: dict[str, int] = defaultdict(int)
    midpoint_by_site: dict[str, np.ndarray] = {}
    midpoint_permutation_by_site: dict[str, list[int]] = {}
    for site_key, site in sites.items():
        midpoint, permutation = deposited_midpoint(
            np.asarray(site["A"], dtype=np.float32),
            np.asarray(site["B"], dtype=np.float32),
            list(site["atom_names"]),
            str(site["residue_type"]),
        )
        midpoint_by_site[site_key] = midpoint
        midpoint_permutation_by_site[site_key] = permutation

    for key, optimizer in sorted(optimizer_rows.items()):
        identity = identify_single_recovery(optimizer)
        if identity is None:
            continue
        population = str(identity["recovery_rank"])
        population_counts[population] += 1
        recovered = str(identity["recovered_state"])
        missed = str(identity["missed_state"])
        target_recovered = float(identity[f"target_{recovered}"])
        target_missed = float(identity[f"target_{missed}"])
        target_major = float(identity[f"target_{identity['major_state']}"])
        target_minor = float(identity[f"target_{identity['minor_state']}"])
        recovered_state_total_occupancy = float(
            optimizer[f"predicted_{recovered}_occupancy"]
        )
        start_slots = slots_by_start[key]
        recovered_candidates = [
            slot for slot in start_slots
            if float(slot["occupancy"]) > FOUND_THRESHOLD
            and slot["raw_assignment"] == recovered
        ]
        if not recovered_candidates:
            raise RuntimeError(f"single-recovery row lacks representative: {key}")
        recovered_representative = min(
            recovered_candidates,
            key=lambda slot: float(slot[f"rmsd_to_{recovered}"]),
        )
        recovered_representative_occupancy = float(
            recovered_representative["occupancy"]
        )
        representative_slot = int(recovered_representative["slot"])
        representative_active = active_by_slot[
            (key[0], key[1], representative_slot)
        ]
        recovered_v3_cluster_occupancy = float(
            representative_active["merge_cluster_occupancy"]
        )
        unmatched = [
            slot for slot in start_slots
            if float(slot["occupancy"]) > ACTIVE_THRESHOLD
            and int(slot["slot"]) != representative_slot
        ]
        submask = sum(
            float(slot["occupancy"])
            for slot in start_slots
            if float(slot["occupancy"]) <= ACTIVE_THRESHOLD
        )
        difference_deposited = abs(
            recovered_representative_occupancy - target_recovered
        )
        difference_one = abs(recovered_representative_occupancy - 1.0)
        near_major = (
            abs(recovered_representative_occupancy - target_major)
            <= OCCUPANCY_TOLERANCE
        )
        closer_major_than_minor = (
            abs(recovered_representative_occupancy - target_major)
            < abs(recovered_representative_occupancy - target_minor)
        )
        minor_only_major_weight_label_swap = (
            population == "minor_only"
            and near_major
            and closer_major_than_minor
        )
        start_rows.append({
            "site": key[0],
            "start": key[1],
            **identity,
            "recovered_representative_slot": representative_slot,
            "recovered_representative_occupancy": (
                recovered_representative_occupancy
            ),
            "recovered_state_total_occupancy_raw_greedy": (
                recovered_state_total_occupancy
            ),
            "recovered_representative_v3_merge_cluster_occupancy": (
                recovered_v3_cluster_occupancy
            ),
            "deposited_recovered_occupancy": target_recovered,
            "deposited_missed_occupancy": target_missed,
            "deposited_major_occupancy": target_major,
            "deposited_minor_occupancy": target_minor,
            "absolute_error_to_deposited_recovered": difference_deposited,
            "absolute_error_to_1": difference_one,
            "closer_to_deposited_recovered_than_1": (
                difference_deposited < difference_one
            ),
            "equidistant_deposited_recovered_and_1": math.isclose(
                difference_deposited, difference_one, abs_tol=1e-12
            ),
            "unmatched_active_slots": len(unmatched),
            "unmatched_active_occupancy_total": sum(
                float(slot["occupancy"]) for slot in unmatched
            ),
            "submask_occupancy_total": submask,
            "recovered_plus_unmatched_plus_submask": (
                recovered_representative_occupancy
                + sum(float(slot["occupancy"]) for slot in unmatched)
                + submask
            ),
            "recovered_near_deposited_major_within_0p20": near_major,
            "recovered_closer_to_deposited_major_than_minor": (
                closer_major_than_minor
            ),
            "minor_only_major_weight_label_swap": (
                minor_only_major_weight_label_swap
            ),
        })
        site = sites[key[0]]
        for slot in unmatched:
            slot_key = (key[0], key[1], int(slot["slot"]))
            xyz = coordinates[slot_key]
            active = active_by_slot[slot_key]
            unmatched_rows.append({
                "site": key[0],
                "start": key[1],
                "recovery_rank": population,
                "recovered_state": recovered,
                "missed_state": missed,
                "slot": int(slot["slot"]),
                "occupancy": float(slot["occupancy"]),
                "rmsd_to_missed_A": float(slot[f"rmsd_to_{missed}"]),
                "rmsd_to_recovered_A": float(slot[f"rmsd_to_{recovered}"]),
                "rmsd_to_A_A": float(slot["rmsd_to_A"]),
                "rmsd_to_B_A": float(slot["rmsd_to_B"]),
                "rmsd_to_AB_midpoint_A": conventional_rmsd(
                    xyz,
                    midpoint_by_site[key[0]],
                    list(site["atom_names"]),
                    str(site["residue_type"]),
                ),
                "raw_assignment": slot["raw_assignment"],
                "v3_merge_cluster": int(active["merge_cluster"]),
                "v3_merge_cluster_occupancy": float(
                    active["merge_cluster_occupancy"]
                ),
                "v3_merge_cluster_assignment": active[
                    "merge_cluster_assignment"
                ],
            })

    if population_counts != {"major_only": 142, "minor_only": 45}:
        raise RuntimeError(
            f"raw-greedy single-recovery mismatch: {dict(population_counts)}"
        )

    # Exact-v3 count is recorded only as provenance; it does not redefine 142/45.
    v3_single_counts = {"major_only": 0, "minor_only": 0, "equal_excluded": 0}
    for row in ensembles.values():
        found_a = truth(row["found_A_conventional"])
        found_b = truth(row["found_B_conventional"])
        if found_a == found_b:
            continue
        target_a = float(row["target_A_occupancy"])
        target_b = float(row["target_B_occupancy"])
        if math.isclose(target_a, target_b, abs_tol=1e-6):
            v3_single_counts["equal_excluded"] += 1
            continue
        recovered = "A" if found_a else "B"
        major = "A" if target_a > target_b else "B"
        v3_single_counts[
            "major_only" if recovered == major else "minor_only"
        ] += 1

    # Comparator: raw-greedy unmatched active slots on both-recovered starts.
    successful_extra_distances = []
    successful_high_extra_distances = []
    successful_extra_rows = []
    for key, optimizer in sorted(optimizer_rows.items()):
        if not (truth(optimizer["found_A"]) and truth(optimizer["found_B"])):
            continue
        representatives = {}
        for state in ("A", "B"):
            candidates = [
                slot for slot in slots_by_start[key]
                if float(slot["occupancy"]) > FOUND_THRESHOLD
                and slot["raw_assignment"] == state
            ]
            representatives[state] = min(
                candidates, key=lambda slot: float(slot[f"rmsd_to_{state}"])
            )
        selected_slots = {
            int(row["slot"]) for row in representatives.values()
        }
        for slot in slots_by_start[key]:
            if not (
                float(slot["occupancy"]) > ACTIVE_THRESHOLD
                and int(slot["slot"]) not in selected_slots
            ):
                continue
            distance = min(
                float(slot["rmsd_to_A"]), float(slot["rmsd_to_B"])
            )
            successful_extra_distances.append(distance)
            if float(slot["occupancy"]) > FOUND_THRESHOLD:
                successful_high_extra_distances.append(distance)
            successful_extra_rows.append({
                "site": key[0],
                "start": key[1],
                "slot": int(slot["slot"]),
                "occupancy": float(slot["occupancy"]),
                "rmsd_to_nearest_deposited_A": distance,
            })

    # Part 3: site-wide protected clustering and fixed-position Gram NNLS.
    pool_site_rows: list[dict[str, object]] = []
    pool_cluster_rows: list[dict[str, object]] = []
    pool_member_rows: list[dict[str, object]] = []
    for site_key, site in sorted(sites.items()):
        site_active = [
            (key, active_by_slot[key], coordinates[key])
            for key in sorted(active_by_slot)
            if key[0] == site_key
        ]
        names = list(site["atom_names"])
        residue_type = str(site["residue_type"])
        candidate_coordinates = [row[2] for row in site_active]
        occupancies = [float(row[1]["occupancy"]) for row in site_active]
        deposited_a = np.asarray(site["A"], dtype=np.float32)
        deposited_b = np.asarray(site["B"], dtype=np.float32)
        rmsd_a = [
            conventional_rmsd(xyz, deposited_a, names, residue_type)
            for xyz in candidate_coordinates
        ]
        rmsd_b = [
            conventional_rmsd(xyz, deposited_b, names, residue_type)
            for xyz in candidate_coordinates
        ]
        pairwise = pairwise_rmsd_matrix(
            candidate_coordinates, names, residue_type
        )
        merged = merge_then_assign_conformers(
            occupancies,
            rmsd_a,
            rmsd_b,
            pairwise.tolist(),
            active_occupancy=ACTIVE_THRESHOLD,
            found_occupancy=FOUND_THRESHOLD,
            rmsd_cutoff=RMSD_CUTOFF,
            merge_rmsd_threshold=MERGE_THRESHOLD,
        )
        representatives = [
            candidate_coordinates[index] for index in merged["representatives"]
        ]
        target_a = float(next(
            row["target_A_occupancy"]
            for key, row in optimizer_rows.items() if key[0] == site_key
        ))
        target_b = float(next(
            row["target_B_occupancy"]
            for key, row in optimizer_rows.items() if key[0] == site_key
        ))
        site["_target_A"] = target_a
        site["_target_B"] = target_b
        columns, target, target_diagnostics = render_density_columns(
            site, representatives, target_paths[site_key]
        )
        q, gram, right, solver_diagnostics = solve_gram_nnls(columns, target)
        nonzero = q > NNLS_NONZERO_THRESHOLD
        representative_rmsd_a = [
            rmsd_a[index] for index in merged["representatives"]
        ]
        representative_rmsd_b = [
            rmsd_b[index] for index in merged["representatives"]
        ]
        greedy_labels = [
            nearest_state_assignment(left, right)
            for left, right in zip(representative_rmsd_a, representative_rmsd_b)
        ]
        greedy_a = float(sum(
            q[index] for index, label in enumerate(greedy_labels)
            if label == "A"
        ))
        greedy_b = float(sum(
            q[index] for index, label in enumerate(greedy_labels)
            if label == "B"
        ))
        nnls_assignments = merge_then_assign_conformers(
            q.tolist(),
            representative_rmsd_a,
            representative_rmsd_b,
            pairwise[np.ix_(
                merged["representatives"], merged["representatives"]
            )].tolist(),
            active_occupancy=NNLS_NONZERO_THRESHOLD,
            found_occupancy=NNLS_NONZERO_THRESHOLD,
            rmsd_cutoff=RMSD_CUTOFF,
            merge_rmsd_threshold=0.0,
        )
        one_to_one = {"A": 0.0, "B": 0.0}
        for cluster_index, label in enumerate(
            nnls_assignments["cluster_assignments"]
        ):
            if label in one_to_one:
                one_to_one[label] = float(q[cluster_index])

        minor = "A" if target_a < target_b else (
            "B" if target_b < target_a else "equal"
        )
        minor_contributing_starts = (
            len({
                key[1]
                for index, (key, _row, _xyz) in enumerate(site_active)
                if minor in {"A", "B"}
                and (
                    rmsd_a[index] if minor == "A" else rmsd_b[index]
                ) < RMSD_CUTOFF
                and occupancies[index] > FOUND_THRESHOLD
            })
            if minor != "equal" else 0
        )
        raw_minor_found_starts = (
            sum(
                truth(row[f"found_{minor}"])
                for key, row in optimizer_rows.items()
                if key[0] == site_key
            )
            if minor != "equal" else 0
        )
        fitted_pair_sum = greedy_a + greedy_b
        deficit = (target_a + target_b) - fitted_pair_sum
        pool_site_rows.append({
            "site": site_key,
            "residue_type": residue_type,
            "active_candidates_pooled": len(site_active),
            "clusters_0p5_protected": len(representatives),
            "nnls_nonzero_clusters": int(nonzero.sum()),
            "nnls_nonzero_exceeds_two": int(nonzero.sum()) > 2,
            "nnls_total_weight": float(q.sum()),
            "target_A_occupancy": target_a,
            "target_B_occupancy": target_b,
            "nnls_greedy_A_occupancy": greedy_a,
            "nnls_greedy_B_occupancy": greedy_b,
            "nnls_one_to_one_A_occupancy": one_to_one["A"],
            "nnls_one_to_one_B_occupancy": one_to_one["B"],
            "nnls_greedy_A_error": greedy_a - target_a,
            "nnls_greedy_B_error": greedy_b - target_b,
            "nnls_greedy_AB_deficit": deficit,
            "nnls_greedy_AB_absolute_deficit": abs(deficit),
            "deficit_below_reference_per_start_median_0p048": abs(deficit) < 0.048,
            "minor_state": minor,
            "minor_contributing_starts_active_gt_0p10_within_1A": (
                minor_contributing_starts
            ),
            "raw_optimizer_minor_found_starts": raw_minor_found_starts,
            **target_diagnostics,
            **solver_diagnostics,
        })
        for cluster_index, (members, representative_index) in enumerate(zip(
            merged["clusters"], merged["representatives"]
        )):
            representative_key = site_active[representative_index][0]
            pool_cluster_rows.append({
                "site": site_key,
                "cluster": cluster_index,
                "members": len(members),
                "representative_start": representative_key[1],
                "representative_slot": representative_key[2],
                "rmsd_to_A_A": representative_rmsd_a[cluster_index],
                "rmsd_to_B_A": representative_rmsd_b[cluster_index],
                "nearest_state_assignment": greedy_labels[cluster_index],
                "nnls_weight": float(q[cluster_index]),
                "nnls_nonzero": bool(nonzero[cluster_index]),
                "gram_diagonal": float(gram[cluster_index, cluster_index]),
                "b_value": float(right[cluster_index]),
            })
            for member in members:
                key, active, _xyz = site_active[member]
                pool_member_rows.append({
                    "site": site_key,
                    "cluster": cluster_index,
                    "start": key[1],
                    "slot": key[2],
                    "endpoint_occupancy": float(active["occupancy"]),
                    "rmsd_to_A_A": rmsd_a[member],
                    "rmsd_to_B_A": rmsd_b[member],
                    "is_representative": member == representative_index,
                })

    major_rows = [
        row for row in start_rows if row["recovery_rank"] == "major_only"
    ]
    minor_rows = [
        row for row in start_rows if row["recovery_rank"] == "minor_only"
    ]
    unmatched_distance = [
        float(row["rmsd_to_missed_A"]) for row in unmatched_rows
    ]
    midpoint_distance = [
        float(row["rmsd_to_AB_midpoint_A"]) for row in unmatched_rows
    ]
    recovered_distance = [
        float(row["rmsd_to_recovered_A"]) for row in unmatched_rows
    ]
    high_unmatched_rows = [
        row for row in unmatched_rows
        if float(row["occupancy"]) > FOUND_THRESHOLD
    ]
    high_unmatched_distance = [
        float(row["rmsd_to_missed_A"]) for row in high_unmatched_rows
    ]
    summary = {
        "scope": {
            "endpoint_source": "frozen v3 endpoints only",
            "optimizer_or_audit_rerun": False,
            "metric_changed": False,
            "single_recovery_population": (
                "historical raw-greedy found_A/found_B partition defining "
                "142 major-only and 45 minor-only"
            ),
            "exact_v3_single_recovery_counts_for_provenance": v3_single_counts,
            "pooling": (
                "all optimizer-active (>0.05) saved conformers across 50 starts; "
                "frozen 0.5 A protected single-linkage; positions held fixed"
            ),
            "nnls": "min ||Gq-b||^2 with q>=0; G=D.T@D and b=D.T@target",
            "native_target": (
                "pre-z-score deposited A/B mixture from the exact production "
                "Gaussian atom model; normalized reconstruction checked against "
                "the saved optimizer synthetic target vector"
            ),
            "nnls_nonzero_threshold": NNLS_NONZERO_THRESHOLD,
            "midpoint": (
                "atomwise midpoint after applying the valid equivalent-atom "
                "B permutation minimizing deposited A-B RMSD"
            ),
        },
        "single_recovery": {
            "counts": dict(population_counts),
            "major_only_recovered_occupancy": describe(
                float(row["recovered_representative_occupancy"])
                for row in major_rows
            ),
            "major_only_recovered_state_total_occupancy": describe(
                float(row["recovered_state_total_occupancy_raw_greedy"])
                for row in major_rows
            ),
            "major_only_recovered_v3_cluster_occupancy": describe(
                float(row["recovered_representative_v3_merge_cluster_occupancy"])
                for row in major_rows
            ),
            "major_only_error_to_deposited_major": describe(
                float(row["absolute_error_to_deposited_recovered"])
                for row in major_rows
            ),
            "major_only_closer_to_deposited_than_1": sum(
                bool(row["closer_to_deposited_recovered_than_1"])
                for row in major_rows
            ),
            "major_only_unmatched_active_mass": describe(
                float(row["unmatched_active_occupancy_total"])
                for row in major_rows
            ),
            "major_only_submask_mass": describe(
                float(row["submask_occupancy_total"]) for row in major_rows
            ),
            "minor_only_recovered_occupancy": describe(
                float(row["recovered_representative_occupancy"])
                for row in minor_rows
            ),
            "minor_only_recovered_state_total_occupancy": describe(
                float(row["recovered_state_total_occupancy_raw_greedy"])
                for row in minor_rows
            ),
            "minor_only_near_deposited_major_within_0p20": sum(
                bool(row["recovered_near_deposited_major_within_0p20"])
                for row in minor_rows
            ),
            "minor_only_closer_to_major_than_minor": sum(
                bool(row["recovered_closer_to_deposited_major_than_minor"])
                for row in minor_rows
            ),
            "minor_only_major_weight_label_swaps_both_criteria": sum(
                bool(row["minor_only_major_weight_label_swap"])
                for row in minor_rows
            ),
        },
        "unmatched_geometry": {
            "unmatched_active_slots": len(unmatched_rows),
            "rmsd_to_missed": describe(unmatched_distance),
            "rmsd_to_recovered": describe(recovered_distance),
            "rmsd_to_AB_midpoint": describe(midpoint_distance),
            "within_1A_of_missed": sum(value < 1.0 for value in unmatched_distance),
            "within_1A_of_recovered": sum(
                value < 1.0 for value in recovered_distance
            ),
            "within_1A_of_midpoint": sum(
                value < 1.0 for value in midpoint_distance
            ),
            "closest_of_missed_recovered_midpoint": {
                label: sum(
                    min(
                        (
                            float(row["rmsd_to_missed_A"]), "missed"
                        ),
                        (
                            float(row["rmsd_to_recovered_A"]), "recovered"
                        ),
                        (
                            float(row["rmsd_to_AB_midpoint_A"]), "midpoint"
                        ),
                    )[1] == label
                    for row in unmatched_rows
                )
                for label in ("missed", "recovered", "midpoint")
            },
            "outcome_categories": {
                "near_missed_just_outside_1p0_to_1p5A": sum(
                    1.0 <= float(row["rmsd_to_missed_A"]) < 1.5
                    for row in unmatched_rows
                ),
                "midpoint_within_1A_and_closest": sum(
                    float(row["rmsd_to_AB_midpoint_A"]) < 1.0
                    and float(row["rmsd_to_AB_midpoint_A"])
                    < min(
                        float(row["rmsd_to_missed_A"]),
                        float(row["rmsd_to_recovered_A"]),
                    )
                    for row in unmatched_rows
                ),
                "recovered_duplicate_within_1A": sum(
                    float(row["rmsd_to_recovered_A"]) < 1.0
                    for row in unmatched_rows
                ),
                "neither_missed_midpoint_nor_recovered_within_1A": sum(
                    min(
                        float(row["rmsd_to_missed_A"]),
                        float(row["rmsd_to_recovered_A"]),
                        float(row["rmsd_to_AB_midpoint_A"]),
                    ) >= 1.0
                    for row in unmatched_rows
                ),
            },
            "high_unmatched_occupancy_gt_0p10": {
                "slots": len(high_unmatched_rows),
                "rmsd_to_missed": describe(high_unmatched_distance),
                "within_1A_of_missed": sum(
                    value < 1.0 for value in high_unmatched_distance
                ),
            },
            "same_statistic_successful_raw_both_found_extras": {
                "slots": len(successful_extra_distances),
                "rmsd_to_nearest_deposited": describe(
                    successful_extra_distances
                ),
                "within_1A": sum(
                    value < 1.0 for value in successful_extra_distances
                ),
                "fraction_within_1A": (
                    sum(value < 1.0 for value in successful_extra_distances)
                    / len(successful_extra_distances)
                    if successful_extra_distances else None
                ),
                "high_occupancy_gt_0p10": {
                    "slots": len(successful_high_extra_distances),
                    "rmsd_to_nearest_deposited": describe(
                        successful_high_extra_distances
                    ),
                    "within_1A": sum(
                        value < 1.0 for value in successful_high_extra_distances
                    ),
                    "fraction_within_1A": (
                        sum(
                            value < 1.0
                            for value in successful_high_extra_distances
                        ) / len(successful_high_extra_distances)
                        if successful_high_extra_distances else None
                    ),
                },
            },
        },
        "pooling": {
            "sites": len(pool_site_rows),
            "sites_with_more_than_two_nonzero_clusters": sum(
                bool(row["nnls_nonzero_exceeds_two"]) for row in pool_site_rows
            ),
            "nnls_nonzero_clusters": describe(
                int(row["nnls_nonzero_clusters"]) for row in pool_site_rows
            ),
            "absolute_AB_deficit": describe(
                float(row["nnls_greedy_AB_absolute_deficit"])
                for row in pool_site_rows
            ),
            "sites_below_reference_per_start_median_0p048": sum(
                bool(row["deficit_below_reference_per_start_median_0p048"])
                for row in pool_site_rows
            ),
            "condition_number": describe(
                float(row["gram_condition_number"])
                for row in pool_site_rows
                if bool(row["gram_condition_number_finite"])
            ),
            "infinite_condition_numbers": sum(
                not bool(row["gram_condition_number_finite"])
                for row in pool_site_rows
            ),
            "target_calibration_relative_l2": describe(
                float(row["target_calibration_relative_l2"])
                for row in pool_site_rows
            ),
            "target_calibration_correlation": describe(
                float(row["target_calibration_correlation"])
                for row in pool_site_rows
            ),
        },
    }

    args.output.mkdir(parents=True)
    atomic_csv(args.output / "single_recovery_per_start.csv", start_rows)
    atomic_csv(args.output / "single_recovery_unmatched_active_slots.csv", unmatched_rows)
    atomic_csv(args.output / "successful_start_unmatched_active_slots.csv", successful_extra_rows)
    atomic_csv(args.output / "pooled_nnls_per_site.csv", pool_site_rows)
    atomic_csv(args.output / "pooled_nnls_clusters.csv", pool_cluster_rows)
    atomic_csv(args.output / "pooled_cluster_members.csv", pool_member_rows)
    atomic_json(
        args.output / "midpoint_equivalent_atom_permutations.json",
        midpoint_permutation_by_site,
    )
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
