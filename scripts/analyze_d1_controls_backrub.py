#!/usr/bin/env python3
"""Classify the fixed D1 controls and replay Cβ recovery only where needed."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np

from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.structure import Structure

from run_d1_tier_a_flips import BACKBONE_NAMES, atom_local_index, get_sampler_xmap, source_path


BASELINE_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/d1_tier_a_controls33_crankshaft_v2")
MANIFEST = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")
CRANKSHAFT_RATIO_UPPER = 0.5


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"


def coordinate(residue, name: str) -> np.ndarray:
    return np.asarray(residue.coor[atom_local_index(residue, name)], dtype=float)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


def kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    left, _, right = np.linalg.svd((a - ac).T @ (b - bc))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = bc - ac @ rotation
    return rotation, translation, rmsd(a @ rotation + translation, b)


def rotation_axis(rotation: np.ndarray) -> tuple[np.ndarray | None, float]:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))
    sine = np.sin(np.radians(angle))
    if abs(sine) < 1e-8:
        return None, angle
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]]) / (2.0 * sine)
    return axis / np.linalg.norm(axis), angle


def rotate_about_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    vector = np.asarray(points) - origin
    return origin + vector * np.cos(angle_rad) + np.cross(axis, vector) * np.sin(angle_rad) + np.outer(vector @ axis, axis) * (1.0 - np.cos(angle_rad))


def best_axis_angle(a: np.ndarray, b: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    av = a - origin
    bv = b - origin
    aperp = av - np.outer(av @ axis, axis)
    bperp = bv - np.outer(bv @ axis, axis)
    cosine = float(np.sum(aperp * bperp))
    sine = float(np.sum(np.cross(axis, aperp) * bperp))
    return float(np.arctan2(sine, cosine))


def a_b_context(site: dict[str, object]):
    path, split = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
    a_chain = a_structure[str(site["chain"])].conformers[0]
    b_chain = b_structure[str(site["chain"])].conformers[0]
    a_segment = next(segment for segment in a_chain.segments if any(residue.id == residue_id for residue in segment.residues))
    b_segment = next(segment for segment in b_chain.segments if any(residue.id == residue_id for residue in segment.residues))
    a_index = [residue.id for residue in a_segment.residues].index(residue_id)
    b_index = [residue.id for residue in b_segment.residues].index(residue_id)
    return split, a_structure, b_structure, a_segment, b_segment, a_index, b_index


def geometry(site: dict[str, object]) -> dict[str, object]:
    split, _, _, a_segment, b_segment, index, b_index = a_b_context(site)
    a = a_segment.residues[index]
    b = b_segment.residues[b_index]
    o_displacement = float(np.linalg.norm(coordinate(a, "O") - coordinate(b, "O")))
    base = {"site": site_key(site), "source_split": split, "resname": site["resname"],
            "O_deposited_displacement_A": o_displacement}
    if str(site["resname"]) == "GLY":
        return {**base, "classification": "no_CB_Gly", "CB_deposited_displacement_A": None, "CB_to_O_ratio": None}
    cb_displacement = float(np.linalg.norm(coordinate(a, "CB") - coordinate(b, "CB")))
    ratio = cb_displacement / o_displacement if o_displacement > 1e-12 else float("inf")
    classification = "backrub_like" if ratio >= 1.0 else ("crankshaft_like" if ratio < CRANKSHAFT_RATIO_UPPER else "intermediate")
    row: dict[str, object] = {**base, "CB_deposited_displacement_A": cb_displacement, "CB_to_O_ratio": ratio,
                              "classification": classification}
    if classification != "backrub_like" or index < 1 or index + 1 >= len(a_segment.residues) or b_index < 1 or b_index + 1 >= len(b_segment.residues):
        return row
    # The canonical backrub axis is the deposited-A Cα(i-1) → Cα(i+1) line.
    a_residues = [a_segment.residues[index - 1], a_segment.residues[index], a_segment.residues[index + 1]]
    b_residues = [b_segment.residues[b_index - 1], b_segment.residues[b_index], b_segment.residues[b_index + 1]]
    names = [name for residue_index in range(3) for name in BACKBONE_NAMES]
    a_points = np.asarray([coordinate(residue, name) for residue in a_residues for name in BACKBONE_NAMES] + [coordinate(a_residues[1], "CB")])
    b_points = np.asarray([coordinate(residue, name) for residue in b_residues for name in BACKBONE_NAMES] + [coordinate(b_residues[1], "CB")])
    origin = coordinate(a_residues[0], "CA")
    ca_axis = coordinate(a_residues[2], "CA") - origin
    ca_axis /= np.linalg.norm(ca_axis)
    theta = best_axis_angle(a_points, b_points, origin, ca_axis)
    fixed = rotate_about_axis(a_points, origin, ca_axis, theta)
    unrotated = rmsd(a_points, b_points)
    fixed_rmsd = rmsd(fixed, b_points)
    rotation, _, kabsch_rmsd = kabsch(a_points, b_points)
    axis, kabsch_angle = rotation_axis(rotation)
    alignment = None if axis is None else float(np.degrees(np.arccos(np.clip(abs(axis @ ca_axis), -1.0, 1.0))))
    endpoint_rmsd = rmsd(np.asarray([coordinate(a_residues[0], "CA"), coordinate(a_residues[2], "CA")]),
                         np.asarray([coordinate(b_residues[0], "CA"), coordinate(b_residues[2], "CA")]))
    row.update({
        "backrub_CA_axis_origin_A": origin.tolist(), "backrub_CA_axis_unit_vector": ca_axis.tolist(),
        "best_fixed_CA_axis_angle_deg": float(np.degrees(theta)), "fixed_CA_axis_rotation_rmsd_A": fixed_rmsd,
        "unrotated_tripeptide_plus_CB_rmsd_A": unrotated,
        "fixed_CA_axis_fraction_SSE_explained": float(1.0 - (fixed_rmsd / unrotated) ** 2) if unrotated else None,
        "CA_axis_endpoint_rmsd_A": endpoint_rmsd, "best_rigid_rotation_angle_deg": kabsch_angle,
        "best_rigid_rotation_axis_unit_vector": None if axis is None else axis.tolist(),
        "best_rigid_axis_to_CA_axis_angle_deg": alignment, "best_rigid_rotation_rmsd_A": kabsch_rmsd,
    })
    return row


def cb_candidate_minimum(site: dict[str, object], expected_o: float) -> float:
    _, a_structure, b_structure, _, _, _, _ = a_b_context(site)
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_residue = a_structure[str(site["chain"])].conformers[0][residue_id]
    b_residue = b_structure[str(site["chain"])].conformers[0][residue_id]
    options = QFitOptions(); options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(a_residue, a_structure, get_sampler_xmap(), options)
    qfit._sample_backbone()
    if len(qfit._coor_set) != 19:
        raise RuntimeError(f"expected 19 candidates, got {len(qfit._coor_set)}")
    cb_index, o_index = atom_local_index(a_residue, "CB"), atom_local_index(a_residue, "O")
    b_cb, b_o = coordinate(b_residue, "CB"), coordinate(b_residue, "O")
    minimum_o = min(float(np.linalg.norm(candidate[o_index] - b_o)) for candidate in qfit._coor_set)
    if not np.isclose(minimum_o, expected_o, rtol=0, atol=1e-12):
        raise RuntimeError(f"CB-default sampler replay changed stored O minimum: {minimum_o} != {expected_o}")
    return min(float(np.linalg.norm(candidate[cb_index] - b_cb)) for candidate in qfit._coor_set)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--measure-backrub-candidates", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (BASELINE_ROOT / "per_site.csv").open() as handle:
        baseline = {row["site"]: row for row in csv.DictReader(handle) if row["status"] == "complete"}
    sites = [site for site in json.loads(MANIFEST.read_text()) if site["panel"] == "nonflip_control"]
    rows = []
    for site in sites:
        row = {**geometry(site), **{f"baseline_{key}": value for key, value in baseline[site_key(site)].items()}}
        row["O_only_fraction_A_to_B_covered"] = 1.0 - float(row["baseline_tier_a_min_central_O_rmsd_A"]) / float(row["O_deposited_displacement_A"])
        if args.measure_backrub_candidates and row["classification"] == "backrub_like":
            minimum_cb = cb_candidate_minimum(site, float(row["baseline_tier_a_min_central_O_rmsd_A"]))
            row["tier_a_min_central_CB_rmsd_A"] = minimum_cb
            row["CB_only_fraction_A_to_B_covered"] = 1.0 - minimum_cb / float(row["CB_deposited_displacement_A"])
        rows.append(row)
        atomic_csv(args.output / "per_site.csv", rows)
        atomic_json(args.output / "progress.json", {"sites_recorded": len(rows), "sites_total": len(sites), "last_site": site_key(site)})
    counts = {label: sum(row["classification"] == label for row in rows) for label in ("backrub_like", "crankshaft_like", "intermediate", "no_CB_Gly")}
    nongly = [float(row["CB_to_O_ratio"]) for row in rows if row["CB_to_O_ratio"] is not None]
    atomic_json(args.output / "summary.json", {
        "status": "complete", "sites": len(rows), "classification_thresholds": {"backrub_like": "CB/O >= 1", "crankshaft_like": "CB/O < 0.5", "intermediate": "0.5 <= CB/O < 1"},
        "counts": counts, "CB_to_O_ratio_range": [min(nongly), max(nongly)],
        "backrub_candidate_replay_performed": args.measure_backrub_candidates,
    })
    atomic_json(args.output / "progress.json", {"status": "complete", "sites": len(rows)})


if __name__ == "__main__":
    main()
