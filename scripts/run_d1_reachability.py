#!/usr/bin/env python3
"""D1: qFit-backbone reachability decomposition.

This is a measurement wrapper around qFit, not a replacement sampler.  Tier A
calls ``QFitRotamericResidue._sample_backbone`` directly; all other tiers use
qFit's ``BackboneRotator`` with centred finite differences at the deposited-A
window.  Every completed site is committed atomically so a long pod job can
resume without silently changing its frozen panel.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qfit.backbone import compute_jacobian
from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.samplers import BackboneRotator
from qfit.structure import Structure
from qfit.structure.math import dihedral_angle

from run_d6_tier2_realmap import make_map


BACKBONE_NAMES = ("N", "CA", "C", "O")
FD_STEPS_DEG = (0.01, 0.001)
LINEARIZATION_LIMIT_DEG = 30.0
SAMPLER_XMAP = None


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(temporary, path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def atomic_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atom_index(residue, name):
    selection = residue.select("name", name)
    if len(selection) != 1:
        raise ValueError(f"Expected exactly one {name} in residue {residue.id}")
    return int(selection[0])


def residue_local_index(residue, name):
    """Return an atom's coordinate-array index in a residue object."""
    global_index = atom_index(residue, name)
    position = np.searchsorted(residue.selection, global_index)
    if position == len(residue.selection) or residue.selection[position] != global_index:
        raise ValueError(f"Atom {name} from {residue.id} not in residue selection")
    return int(position)


def local_index(segment, residue, name):
    global_index = atom_index(residue, name)
    position = np.searchsorted(segment.selection, global_index)
    if position == len(segment.selection) or segment.selection[position] != global_index:
        raise ValueError(f"Atom {name} from {residue.id} not in D1 window")
    return int(position)


def coords(residue, names=BACKBONE_NAMES):
    return np.asarray([residue.coor[residue_local_index(residue, name)] for name in names])


def wrapped_delta(a, b):
    """Map b-a to (-180, 180], treating -180 as +180 deterministically."""
    delta = (np.asarray(b) - np.asarray(a) + 180.0) % 360.0 - 180.0
    return np.where(np.isclose(delta, -180.0), 180.0, delta)


def dihedrals(window):
    """Return qFit-order phi/psi plus the separately-fixed omega vector."""
    phi_psi, omega = [], []
    for i, residue in enumerate(window.residues):
        if i == 0:
            phi_psi.append(0.0)
        else:
            previous = window.residues[i - 1]
            phi_psi.append(dihedral_angle(np.asarray([
                previous.coor[residue_local_index(previous, "C")],
                residue.coor[residue_local_index(residue, "N")],
                residue.coor[residue_local_index(residue, "CA")],
                residue.coor[residue_local_index(residue, "C")],
            ])))
        if i == len(window.residues) - 1:
            phi_psi.append(0.0)
        else:
            following = window.residues[i + 1]
            phi_psi.append(dihedral_angle(np.asarray([
                residue.coor[residue_local_index(residue, "N")],
                residue.coor[residue_local_index(residue, "CA")],
                residue.coor[residue_local_index(residue, "C")],
                following.coor[residue_local_index(following, "N")],
            ])))
            omega.append(dihedral_angle(np.asarray([
                residue.coor[residue_local_index(residue, "CA")],
                residue.coor[residue_local_index(residue, "C")],
                following.coor[residue_local_index(following, "N")],
                following.coor[residue_local_index(following, "CA")],
            ])))
    return np.asarray(phi_psi), np.asarray(omega)


def svd_rank_and_basis(matrix):
    u, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    scale = singular[0] if singular.size else 1.0
    tolerance = max(matrix.shape) * np.finfo(float).eps * scale * 100.0
    rank = int(np.sum(singular > tolerance))
    return rank, vh[rank:].T.copy(), singular, float(tolerance)


def row_space_projector(jacobian, null_basis):
    """Projector in torsion space onto row(J @ N), with its image dimension."""
    restricted = jacobian @ null_basis
    rank, _, singular, tolerance = svd_rank_and_basis(restricted)
    if rank == 0:
        return np.zeros((null_basis.shape[0], null_basis.shape[0])), rank, singular
    _, _, vh = np.linalg.svd(restricted, full_matrices=False)
    vectors = vh[:rank].T
    projector = null_basis @ vectors @ vectors.T @ null_basis.T
    return projector, rank, singular


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((np.asarray(a) - np.asarray(b)) ** 2, axis=1))))


def target_indices(window, central_atom):
    central = window.residues[3]
    central_target = [local_index(window, central, central_atom)]
    oxygen_indices = [local_index(window, residue, "O") for residue in window.residues]
    bb_indices = [
        local_index(window, residue, atom)
        for residue in window.residues for atom in BACKBONE_NAMES
    ]
    central_bb = [local_index(window, central, atom) for atom in BACKBONE_NAMES]
    return central_target, oxygen_indices, bb_indices, central_bb


def finite_difference_jacobian(window, local_atoms, step_deg):
    # BackboneRotator mutates its segment.  This wrapper must leave the shared
    # deposited-A window exactly unchanged: each derivative is at A, not at a
    # previous +/- finite-difference displacement.
    starting_coor = window.coor.copy()
    try:
        rotator = BackboneRotator(window)
        jacobian = np.empty((3 * len(local_atoms), rotator.ndofs))
        for column in range(rotator.ndofs):
            plus = np.zeros(rotator.ndofs)
            minus = np.zeros(rotator.ndofs)
            plus[column] = step_deg
            minus[column] = -step_deg
            rotator(plus)
            plus_coor = window.coor[local_atoms].copy()
            rotator(minus)
            minus_coor = window.coor[local_atoms].copy()
            jacobian[:, column] = ((plus_coor - minus_coor) / (2.0 * step_deg)).ravel()
        return jacobian
    finally:
        window.coor = starting_coor


def apply_rotator(window, delta_q, local_atoms):
    # Each tier is an independent projection from deposited A.  Never let a
    # preceding metric turn the next tier into a second (or third) rotation.
    starting_coor = window.coor.copy()
    try:
        rotator = BackboneRotator(window)
        rotator(delta_q)
        return window.coor[local_atoms].copy()
    finally:
        window.coor = starting_coor


def qfit_candidates_with_validation(qfit, central_bb_local, central_candidate_indices):
    """Call qFit Tier A and independently observe the final 18 rotator calls."""
    calls = []
    original = BackboneRotator.__call__

    def traced(self, torsions):
        original(self, torsions)
        if len(self.segment) == 7:
            calls.append(self.segment.coor.copy())

    BackboneRotator.__call__ = traced
    try:
        qfit._sample_backbone()
    finally:
        BackboneRotator.__call__ = original
    final = calls[-18:]
    # The wrapper must reproduce qFit's central coordinates exactly, otherwise
    # the recorded Tier A is not trustworthy.
    expected = qfit._coor_set[1:]
    if len(final) != len(expected):
        raise RuntimeError(f"Expected 18 final sampler calls, got {len(final)}")
    discrepancy = max(
        float(np.max(np.abs(coor[central_bb_local] - candidate[central_candidate_indices])))
        for coor, candidate in zip(final, expected)
    )
    return qfit._coor_set, discrepancy


def strict_window(qfit, b_structure, site):
    index = qfit.segment.find(qfit.residue.id)
    nn = qfit.options.neighbor_residues_required
    qfit_guard_accepts = not (index < nn or index + nn > len(qfit.segment))
    a_window = qfit.segment[index - nn:index + nn + 1]
    if len(a_window.residues) != 7:
        return None, {"reason": "truncated_A_window", "qfit_guard_accepts": qfit_guard_accepts}
    b_chain = b_structure[site["chain"]].conformers[0]
    b_segment = next(
        (segment for segment in b_chain.segments
         if any(residue.id == qfit.residue.id for residue in segment.residues)),
        None,
    )
    if b_segment is None:
        return None, {"reason": "B_target_not_in_segment", "qfit_guard_accepts": qfit_guard_accepts}
    b_index = b_segment.find(qfit.residue.id)
    b_window = b_segment[b_index - nn:b_index + nn + 1]
    if len(b_window.residues) != 7:
        return None, {"reason": "truncated_B_window", "qfit_guard_accepts": qfit_guard_accepts}
    if [r.id for r in a_window.residues] != [r.id for r in b_window.residues]:
        return None, {"reason": "A_B_window_ids_differ", "qfit_guard_accepts": qfit_guard_accepts}
    for label, window in (("A", a_window), ("B", b_window)):
        for residue in window.residues:
            missing = [name for name in BACKBONE_NAMES if len(residue.select("name", name)) != 1]
            if missing:
                return None, {"reason": f"missing_{label}_{','.join(missing)}", "qfit_guard_accepts": qfit_guard_accepts}
    return (a_window, b_window), {"qfit_guard_accepts": qfit_guard_accepts}


def analyze_site(site):
    global SAMPLER_XMAP
    pdb = site["pdb_id"]
    pdb_path = f"/home/dev/qfit_unet_data/train/{pdb.lower()}.pdb"
    a_structure = Structure.fromfile(pdb_path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(pdb_path).extract("altloc", ("", "B"))
    residue_id = (int(site["resnum"]), site.get("insertion_code", ""))
    a_residue = a_structure[site["chain"]].conformers[0][residue_id]
    # `_sample_backbone` never reads density.  It nevertheless lives on a
    # QFit object whose constructor needs an XMap.  Reuse one valid panel map
    # strictly as constructor plumbing; geometry and Tier A candidates are map
    # independent on this code path.
    if SAMPLER_XMAP is None:
        SAMPLER_XMAP, _, _, _ = make_map("/home/dev/qfit_unet_data/cache/train/mtz/4HVN.mtz")
    options = QFitOptions()
    options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(a_residue, a_structure, SAMPLER_XMAP, options)
    windows, preflight = strict_window(qfit, b_structure, site)
    base = {
        "site": f"{pdb}_{site['chain']}_{site['resname']}{site['resnum']}",
        "pdb_id": pdb, "chain": site["chain"], "resname": site["resname"],
        "resnum": int(site["resnum"]), "resolution_A": float(site["resolution"]),
        "backbone_deviation_A": float(site["max_backbone_deviation"]),
        "panel": site["panel"], **preflight,
    }
    if windows is None:
        base["status"] = "excluded"
        return base
    a_window, b_window = windows
    a_phi_psi, a_omega = dihedrals(a_window)
    b_phi_psi, b_omega = dihedrals(b_window)
    delta_q = wrapped_delta(a_phi_psi, b_phi_psi)
    delta_omega = wrapped_delta(a_omega, b_omega)
    bb_coor = a_window.get_xyz(np.sort(a_window.select("name", ("N", "CA", "C"))))
    closure = compute_jacobian(bb_coor)
    closure_rank, null_basis, closure_singular, closure_tolerance = svd_rank_and_basis(closure)
    central_atom = "O" if site["resname"] == "GLY" else "CB"
    target, oxygens, all_bb, central_bb = target_indices(a_window, central_atom)
    j_target = finite_difference_jacobian(a_window, target, FD_STEPS_DEG[0])
    j_o = finite_difference_jacobian(a_window, oxygens, FD_STEPS_DEG[0])
    j_target_fine = finite_difference_jacobian(a_window, target, FD_STEPS_DEG[1])
    j_o_fine = finite_difference_jacobian(a_window, oxygens, FD_STEPS_DEG[1])
    fd_target_relative = float(np.linalg.norm(j_target - j_target_fine) / np.linalg.norm(j_target_fine))
    fd_o_relative = float(np.linalg.norm(j_o - j_o_fine) / np.linalg.norm(j_o_fine))
    p_target, dim_target, target_singular = row_space_projector(j_target, null_basis)
    p_joint, dim_joint, joint_singular = row_space_projector(np.vstack((j_target, j_o)), null_basis)
    p_full = null_basis @ null_basis.T
    null_dim = int(null_basis.shape[1])

    a_central_bb = coords(a_window.residues[3])
    b_central_bb = coords(b_window.residues[3])
    a_full_bb = a_window.coor[all_bb].copy()
    b_full_bb = b_window.coor[all_bb].copy()
    b_o = b_central_bb[3]
    candidate_indices = [residue_local_index(a_residue, name) for name in BACKBONE_NAMES]
    candidate_coor, wrapper_error = qfit_candidates_with_validation(
        qfit, central_bb, candidate_indices
    )
    a_metrics = []
    for candidate in candidate_coor:
        candidate_bb = candidate[candidate_indices]
        a_metrics.append((rmsd(candidate_bb, b_central_bb), float(np.linalg.norm(candidate_bb[3] - b_o))))
    row = {
        **base, "status": "complete", "central_target_atom": central_atom,
        "delta_q_inf_deg": float(np.max(np.abs(delta_q))),
        "omega_max_abs_delta_deg": float(np.max(np.abs(delta_omega))),
        "omega_count_abs_delta_gt_10deg": int(np.sum(np.abs(delta_omega) > 10.0)),
        "linearization_suspect": bool(np.max(np.abs(delta_q)) > LINEARIZATION_LIMIT_DEG),
        "closure_rank": closure_rank, "null_dim": null_dim,
        "closure_singular_values": json.dumps(closure_singular.tolist()),
        "closure_rank_tolerance": closure_tolerance,
        "dim_target_image": dim_target, "dim_target_plus_O_image": dim_joint,
        "forfeited_dimension": null_dim - dim_target,
        "fd_target_relative_difference_0.01_vs_0.001": fd_target_relative,
        "fd_O_relative_difference_0.01_vs_0.001": fd_o_relative,
        "tier_a_qfit_candidate_count": len(candidate_coor),
        "tier_a_wrapper_max_abs_coord_error_A": wrapper_error,
        "tier_a_central_backbone_rmsd_A": min(metric[0] for metric in a_metrics),
        "tier_a_central_O_rmsd_A": min(metric[1] for metric in a_metrics),
    }
    tier_specs = (("b_target", p_target), ("c_target_plus_O", p_joint), ("d_full_null", p_full))
    for label, projector in tier_specs:
        projected = projector @ delta_q
        predicted_central = apply_rotator(a_window, projected, central_bb)
        predicted_full = apply_rotator(a_window, projected, all_bb)
        row[f"tier_{label}_central_backbone_rmsd_A"] = rmsd(predicted_central, b_central_bb)
        row[f"tier_{label}_central_O_rmsd_A"] = float(np.linalg.norm(predicted_central[3] - b_o))
        row[f"tier_{label}_window_backbone_rmsd_A"] = rmsd(predicted_full, b_full_bb)
        ratio = np.linalg.norm(projected) / max(np.linalg.norm(delta_q), 1e-12)
        row[f"tier_{label}_projection_norm_fraction"] = float(ratio)
        cosine = np.linalg.norm(projector @ delta_q) / max(np.linalg.norm(delta_q), 1e-12)
        row[f"tier_{label}_subspace_angle_deg"] = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    # Scale-free O movement available in the CB-forfeited directions.
    _, forfeited_basis, _, _ = svd_rank_and_basis(j_target @ null_basis)
    # ``forfeited_basis`` is in the small (nullspace-coordinate) basis here.
    if forfeited_basis.shape[1]:
        j_o_forfeited = j_o @ null_basis @ forfeited_basis
        j_o_all = j_o @ null_basis
        row["O_frobenius_forfeited_over_all"] = float(np.linalg.norm(j_o_forfeited) / np.linalg.norm(j_o_all))
        row["O_forfeited_singular_values"] = json.dumps(np.linalg.svd(j_o_forfeited, compute_uv=False).tolist())
        row["dim_CB_null_within_closure_null"] = int(forfeited_basis.shape[1])
    else:
        row["O_frobenius_forfeited_over_all"] = 0.0
        row["O_forfeited_singular_values"] = "[]"
        row["dim_CB_null_within_closure_null"] = 0
    return row


def make_figure(rows, output):
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return
    columns = [
        ("qFit samples", "tier_a_central_backbone_rmsd_A"),
        ("target image", "tier_b_target_central_backbone_rmsd_A"),
        ("target + O", "tier_c_target_plus_O_central_backbone_rmsd_A"),
        ("full null", "tier_d_full_null_central_backbone_rmsd_A"),
    ]
    values = [[float(row[column]) for row in complete] for _, column in columns]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.boxplot(values, labels=[label for label, _ in columns], showmeans=True)
    for x, series in enumerate(values, start=1):
        ax.scatter(np.full(len(series), x), series, color="#1f77b4", alpha=.7, s=23, zorder=3)
    ax.set_ylabel("central {N, CA, C, O} RMSD to deposited B (Å)")
    ax.set_title("D1 controls: qFit backbone reachability tiers")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel", default="nonflip_control", choices=("nonflip_control", "flip_filter"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume)
    progress_path = output / "progress.json"
    table_path = output / "per_site.csv"
    manifest_path = "/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json"
    with open(manifest_path) as handle:
        panel = [site for site in json.load(handle) if site["panel"] == args.panel]
    rows = []
    if args.resume and table_path.exists():
        with table_path.open() as handle:
            rows = list(csv.DictReader(handle))
    completed_sites = {row["site"] for row in rows}
    atomic_json(output / "run_config.json", {
        "panel_manifest": manifest_path, "panel": args.panel, "limit": args.limit,
        "fd_steps_deg": FD_STEPS_DEG, "linearization_limit_deg": LINEARIZATION_LIMIT_DEG,
        "qfit_tier_a": "actual _sample_backbone; input plus 18 displacement solutions",
    })
    for site in panel:
        key = f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"
        if key in completed_sites:
            continue
        if sum(row.get("status") == "complete" for row in rows) >= args.limit:
            break
        try:
            row = analyze_site(site)
        except Exception as error:  # Preserve failures as evidence, then continue selection.
            row = {"site": key, "pdb_id": site["pdb_id"], "panel": site["panel"],
                   "status": "error", "error": repr(error)}
        rows.append(row)
        atomic_csv(table_path, rows)
        make_figure(rows, output / "tier_rmsd_distribution.png")
        atomic_json(progress_path, {
            "complete": sum(row.get("status") == "complete" for row in rows),
            "excluded_or_error": sum(row.get("status") != "complete" for row in rows),
            "last_site": key, "target_complete": args.limit,
        })
    make_figure(rows, output / "tier_rmsd_distribution.png")
    atomic_json(progress_path, {
        "complete": sum(row.get("status") == "complete" for row in rows),
        "excluded_or_error": sum(row.get("status") != "complete" for row in rows),
        "target_complete": args.limit,
        "status": "complete" if sum(row.get("status") == "complete" for row in rows) >= args.limit else "incomplete",
    })


if __name__ == "__main__":
    main()
