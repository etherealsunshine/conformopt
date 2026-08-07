#!/usr/bin/env python3
"""Closure feasibility diagnostics for deposited peptide-flip A/B windows.

This is measurement code.  It never runs qFit sampling or density fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.samplers import BackboneRotator
from qfit.structure import Structure

from run_d1_reachability import BACKBONE_NAMES, dihedrals, rmsd, strict_window, wrapped_delta
from run_d1_tier_a_flips import get_sampler_xmap, source_path


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


def atom_coordinate(residue, name: str) -> np.ndarray:
    selection = residue.select("name", name)
    if len(selection) != 1:
        raise ValueError(f"{residue.id} lacks unique {name}")
    global_index = int(selection[0])
    local = int(np.searchsorted(residue.selection, global_index))
    return np.asarray(residue.coor[local], dtype=float)


def residue_backbone(residue) -> np.ndarray:
    return np.asarray([atom_coordinate(residue, name) for name in BACKBONE_NAMES])


def window_data(site: dict[str, object]):
    path, split = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
    a_residue = a_structure[str(site["chain"])].conformers[0][residue_id]
    options = QFitOptions(); options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(a_residue, a_structure, get_sampler_xmap(), options)
    index = qfit.segment.find(qfit.residue.id)
    required = qfit.options.neighbor_residues_required
    # Do not call Segment.__getitem__ for a known-truncated seven-residue
    # window. qFit's own `>` guard admits an upper-end equality case where
    # the slicer cannot build a concatenated selection; closure is not
    # defined there in any event.
    if index < required or index + required >= len(qfit.segment):
        return a_structure, b_structure, a_residue, qfit, None, {
            "reason": "no_complete_qfit_seven_residue_window",
            "qfit_guard_accepts": not (index < required or index + required > len(qfit.segment)),
            "strict_complete_window": False,
        }, split
    windows, preflight = strict_window(qfit, b_structure, site)
    return a_structure, b_structure, a_residue, qfit, windows, preflight, split


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"


def anchors(a_window, b_window) -> dict[str, object]:
    per_position = {}
    for index, label in enumerate(range(-3, 4)):
        differences = np.abs(residue_backbone(a_window.residues[index]) - residue_backbone(b_window.residues[index]))
        per_position[str(label)] = float(differences.max())
    return {
        "per_window_position_max_abs_coordinate_difference_A": per_position,
        "anchor_minus3_A": per_position["-3"], "anchor_plus3_A": per_position["3"],
        "anchor_plus_minus3_max_A": max(per_position["-3"], per_position["3"]),
        "neighbour_plus_minus1_max_A": max(per_position["-1"], per_position["1"]),
        "neighbour_plus_minus2_max_A": max(per_position["-2"], per_position["2"]),
    }


def backbone_geometry(a_window, b_window) -> dict[str, object]:
    bonds, angles, triangle_floors = [], [], []
    def kabsch_rmsd(first, second):
        first = np.asarray(first, dtype=float); second = np.asarray(second, dtype=float)
        first_centered = first - first.mean(axis=0); second_centered = second - second.mean(axis=0)
        left, _, right = np.linalg.svd(first_centered.T @ second_centered)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1.0
            rotation = left @ right
        return rmsd(first_centered @ rotation, second_centered)
    for index in range(7):
        for label, names in (("N-CA", ((index, "N"), (index, "CA"))),
                             ("CA-C", ((index, "CA"), (index, "C"))),
                             ("C-O", ((index, "C"), (index, "O")))):
            a = [atom_coordinate(a_window.residues[i], name) for i, name in names]
            b = [atom_coordinate(b_window.residues[i], name) for i, name in names]
            bonds.append((f"{index - 3}:{label}", abs(np.linalg.norm(a[0] - a[1]) - np.linalg.norm(b[0] - b[1]))))
        for label, names in (("N-CA-C", ((index, "N"), (index, "CA"), (index, "C"))),
                             ("CA-C-O", ((index, "CA"), (index, "C"), (index, "O")))):
            def angle(values):
                left, center, right = values
                u, v = left - center, right - center
                return np.degrees(np.arccos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
            a = [atom_coordinate(a_window.residues[i], name) for i, name in names]
            b = [atom_coordinate(b_window.residues[i], name) for i, name in names]
            angles.append((f"{index - 3}:{label}", abs(angle(a) - angle(b))))
            triangle_floors.append((f"{index - 3}:{label}", kabsch_rmsd(a, b)))
        if index < 6:
            a_c, a_n = atom_coordinate(a_window.residues[index], "C"), atom_coordinate(a_window.residues[index + 1], "N")
            b_c, b_n = atom_coordinate(b_window.residues[index], "C"), atom_coordinate(b_window.residues[index + 1], "N")
            bonds.append((f"{index - 3}:C-N:{index - 2}", abs(np.linalg.norm(a_c - a_n) - np.linalg.norm(b_c - b_n))))
            for label, names in (("CA-C-N", ((index, "CA"), (index, "C"), (index + 1, "N"))),
                                 ("O-C-N", ((index, "O"), (index, "C"), (index + 1, "N"))),
                                 ("C-N-CA", ((index, "C"), (index + 1, "N"), (index + 1, "CA")))):
                a = [atom_coordinate(a_window.residues[i], name) for i, name in names]
                b = [atom_coordinate(b_window.residues[i], name) for i, name in names]
                angles.append((f"{index - 3}:{label}:{index - 2}", abs(angle(a) - angle(b))))
                triangle_floors.append((f"{index - 3}:{label}:{index - 2}", kabsch_rmsd(a, b)))
    max_bond_label, max_bond = max(bonds, key=lambda item: item[1])
    max_angle_label, max_angle = max(angles, key=lambda item: item[1])
    max_triangle_label, max_triangle = max(triangle_floors, key=lambda item: item[1])
    # Any one pair-distance discrepancy d imposes RMSD >= d / sqrt(2N) over N atoms.
    n_atoms = 7 * len(BACKBONE_NAMES)
    return {"max_backbone_bond_length_difference_A": float(max_bond), "max_bond_label": max_bond_label,
            "max_backbone_bond_angle_difference_deg": float(max_angle), "max_angle_label": max_angle_label,
            "bond_length_implied_window_backbone_rmsd_floor_A": float(max_bond / np.sqrt(2 * n_atoms)),
            "max_local_backbone_triangle_rigid_rmsd_A": float(max_triangle), "max_triangle_label": max_triangle_label,
            "angle_and_bond_implied_window_backbone_rmsd_floor_A": float(max_triangle * np.sqrt(3 / n_atoms))}


def full_checks(site: dict[str, object]) -> dict[str, object]:
    _, _, _, _, windows, preflight, split = window_data(site)
    base = {"site": site_key(site), "source_split": split, **preflight}
    if windows is None:
        return {**base, "status": "no_complete_closure_window"}
    a_window, b_window = windows
    anchor_data = anchors(a_window, b_window)
    a_phi_psi, a_omega = dihedrals(a_window)
    b_phi_psi, b_omega = dihedrals(b_window)
    raw_delta = b_phi_psi - a_phi_psi
    wrapped = wrapped_delta(a_phi_psi, b_phi_psi)
    initial = a_window.coor.copy()
    try:
        BackboneRotator(a_window)(wrapped)
        forward_central = residue_backbone(a_window.residues[3])
        forward_window = np.vstack([residue_backbone(residue) for residue in a_window.residues])
    finally:
        a_window.coor = initial
    b_central = residue_backbone(b_window.residues[3])
    b_full = np.vstack([residue_backbone(residue) for residue in b_window.residues])
    omega = wrapped_delta(a_omega, b_omega)
    return {
        **base, "status": "complete", **anchor_data,
        "raw_phi_psi_delta_deg": raw_delta.tolist(), "wrapped_phi_psi_delta_deg": wrapped.tolist(),
        "forward_kinematics_central_N_CA_C_O_rmsd_to_B_A": rmsd(forward_central, b_central),
        "forward_kinematics_full_window_N_CA_C_O_rmsd_to_B_A": rmsd(forward_window, b_full),
        "omega_abs_delta_deg_by_peptide": {f"{i - 3}_to_{i - 2}": float(abs(value)) for i, value in enumerate(omega)},
        "omega_max_abs_delta_deg": float(np.max(np.abs(omega))),
        **backbone_geometry(a_window, b_window),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--site", default=None, help="Exact site key; omit for all 33 frozen flip sites.")
    parser.add_argument("--anchors-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")
    sites = [site for site in json.loads(manifest.read_text()) if site["panel"] == "flip_filter"]
    if args.site:
        sites = [site for site in sites if site_key(site) == args.site]
    if not sites:
        raise ValueError("no matching flip sites")
    rows = []
    for site in sites:
        if args.anchors_only:
            _, _, _, _, windows, preflight, split = window_data(site)
            row = {"site": site_key(site), "source_split": split, **preflight}
            row.update({"status": "no_complete_closure_window"} if windows is None else {"status": "complete", **anchors(*windows)})
        else:
            row = full_checks(site)
        rows.append(row)
        atomic_json(args.output / "site_checkpoints" / f"{row['site']}.json", row)
        atomic_csv(args.output / "per_site.csv", rows)
        atomic_json(args.output / "progress.json", {"sites_complete": len(rows), "sites_total": len(sites), "last_site": row["site"]})
    complete = [row for row in rows if row["status"] == "complete"]
    if len(rows) > 1 and complete:
        values = np.asarray([row["anchor_plus_minus3_max_A"] for row in complete])
        fig, axis = plt.subplots(figsize=(6.2, 4.3)); axis.hist(values, bins=min(12, len(values)), color="#4c72b0")
        axis.axvline(.1, color="#c44e52", linestyle="--", label="0.1 Å")
        axis.set(xlabel="max anchor disagreement at centre±3 (Å)", ylabel="sites", title="Flip panel closure-anchor agreement")
        axis.legend(frameon=False); fig.tight_layout(); fig.savefig(args.output / "anchor_distribution.png", dpi=180); plt.close(fig)
    summary = {"status": "complete", "sites_requested": len(rows), "sites_with_complete_closure_window": len(complete),
               "sites_without_complete_closure_window": len(rows) - len(complete)}
    if complete:
        values = np.asarray([row["anchor_plus_minus3_max_A"] for row in complete])
        summary["anchor_plus_minus3_max_A_distribution"] = {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()), "n_over_0p1_A": int(np.sum(values > .1))}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", {"status": "complete", **summary})


if __name__ == "__main__":
    main()
