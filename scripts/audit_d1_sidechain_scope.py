#!/usr/bin/env python3
"""Read-only deposited A/B backbone-versus-sidechain scope audit for A'' sites."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from run_d1_8d_sequential_poc import BACKBONE_NAMES, atomic_json, rmsd, window_backbone_indices
from run_d1_aprime_sequential import APrimeSequential


SITES = (
    ("4HFS", "A", 200),
    ("4ZXG", "B", 13),
    ("6ZWK", "B", 47),
    ("7ZTL", "A", 257),
    ("8AJK", "A", 240),
    ("8R7O", "C", 1681),
)


def name(site: tuple[str, str, int]) -> str:
    return f"{site[0]}_{site[1]}_{site[2]}"


def residue_coordinates(residue) -> dict[str, np.ndarray]:
    return {atom: np.asarray(coordinate, dtype=float)
            for atom, coordinate in zip(residue.name.tolist(), residue.coor)}


def component(a: dict[str, np.ndarray], b: dict[str, np.ndarray], names: list[str]) -> dict[str, object]:
    shared = [atom for atom in names if atom in a and atom in b]
    if not shared:
        return {"matched_atom_count": 0, "rmsd_A": None, "atoms": []}
    return {"matched_atom_count": len(shared),
            "rmsd_A": rmsd(np.asarray([a[atom] for atom in shared]), np.asarray([b[atom] for atom in shared])),
            "atoms": shared}


def paired_windows(base):
    # ``base.window`` is qFit's map-box working segment and can have a
    # different crystallographic image from the raw deposited model.  This is
    # an A/B deposition comparison, so extract both windows from their raw
    # altloc structures rather than mixing coordinate frames.
    a_chain = base.a_structure[base.chain].conformers[0]
    b_chain = base.b_structure[base.chain].conformers[0]
    a_segment = next(segment for segment in a_chain.segments
                     if any(residue.id == base.a_residue.id for residue in segment.residues))
    b_segment = next(segment for segment in b_chain.segments
                     if any(residue.id == base.b_residue.id for residue in segment.residues))
    a_index, b_index = a_segment.find(base.a_residue.id), b_segment.find(base.b_residue.id)
    a_window, b_window = a_segment[a_index - 3:a_index + 4], b_segment[b_index - 3:b_index + 4]
    if len(a_window.residues) != 7 or len(b_window.residues) != 7:
        raise RuntimeError("deposited A/B strict seven-residue window unavailable")
    if [residue.id for residue in a_window.residues] != [residue.id for residue in b_window.residues]:
        raise RuntimeError("deposited A/B seven-residue windows do not align")
    return list(a_window.residues), list(b_window.residues)


def audit_site(site: tuple[str, str, int]) -> dict[str, object]:
    runner = APrimeSequential(Path("/tmp") / f"sidechain_scope_{name(site)}", 8, 6, *site,
                               renderer_backend="torch", residual_scale_mode="none",
                               map_scaler_structure="full", mask_scope="window", device="cuda")
    base = runner.base
    a_window, b_window = paired_windows(base)
    expected_b = {
        (int(residue.id[0]), atom): float(value)
        for residue in b_window for atom, value in zip(residue.name.tolist(), residue.b)
    }
    expected_b_vector = np.asarray([
        expected_b[(int(residue.id[0]), atom)]
        for residue in base.window.residues for atom in residue.name.tolist()
    ])
    per_residue = []
    window_a, window_b = {}, {}
    for offset, (a_residue, b_residue) in enumerate(zip(a_window, b_window), start=-3):
        a, b = residue_coordinates(a_residue), residue_coordinates(b_residue)
        shared = [atom for atom in a if atom in b]
        sidechain = [atom for atom in shared if atom not in BACKBONE_NAMES]
        row = {
            "offset": offset,
            "residue": f"{a_residue.resn[0]}{int(a_residue.id[0])}",
            "backbone": component(a, b, list(BACKBONE_NAMES)),
            "sidechain": component(a, b, sidechain),
            "all_matched_atoms": component(a, b, shared),
            "atoms_only_in_A": sorted(set(a) - set(b)),
            "atoms_only_in_B": sorted(set(b) - set(a)),
        }
        per_residue.append(row)
        for atom in shared:
            key = f"{offset}:{atom}"
            window_a[key], window_b[key] = a[atom], b[atom]

    central = per_residue[3]
    whole_backbone = [key for key in window_a if key.rsplit(":", 1)[1] in BACKBONE_NAMES]
    whole_sidechain = [key for key in window_a if key.rsplit(":", 1)[1] not in BACKBONE_NAMES]
    whole_all = list(window_a)
    whole = {
        "backbone": component(window_a, window_b, whole_backbone),
        "sidechain": component(window_a, window_b, whole_sidechain),
        "all_matched_atoms": component(window_a, window_b, whole_all),
    }
    for scope in (central, whole):
        all_rmsd = scope["all_matched_atoms"]["rmsd_A"]
        backbone_rmsd = scope["backbone"]["rmsd_A"]
        scope["backbone_RMSD_fraction_of_all_atom_RMSD"] = (
            None if all_rmsd in (None, 0.0) else backbone_rmsd / all_rmsd
        )

    mask_backbone = window_backbone_indices(base.window)
    zero_torsions = np.zeros((2, runner.rotator.ndofs), dtype=float)
    seed_slots = runner.torch_forward(zero_torsions).detach().cpu().numpy()
    seed_deviation = float(np.max(np.abs(seed_slots - base.initial_window[None, :, :])))
    return {
        "site": name(site),
        "decomposition_metric": "conventional RMSD over matched deposited A/B atoms",
        "central_residue": central,
        "whole_seven_residue_window": whole,
        "per_residue": per_residue,
        "slot_sidechain_assignment": {
            "slot_1_coordinates": "deposited-A window at zero torsions",
            "slot_2_coordinates": "deposited-A window at zero torsions",
            "chi_parameters": "none; sidechains move only rigidly with backbone rotations",
            "slot_1_B_factors": "deposited A",
            "slot_2_B_factors": "deposited B",
            "slot_2_B_factor_vector_matches_requested_chain_B": bool(np.allclose(
                base.b_factors_b, expected_b_vector, atol=1e-8, rtol=0.0
            )),
            "zero_torsion_max_coordinate_deviation_from_A_A": seed_deviation,
        },
        "mask_and_renderer": {
            "mask_scope": base.mask_scope,
            "mask_radius_A": 0.5 + base.resolution / 3.0,
            "mask_support_centres": "N/CA/C/O only, across both deposited A and B seven-residue windows",
            "mask_has_sidechain_centres": False,
            "model_atom_set": "all atoms of the deposited-A seven-residue window",
            "model_atom_count": int(len(base.model_atom_indices)),
            "backbone_model_atom_count": int(len(mask_backbone)),
            "nonbackbone_model_atom_count": int(len(base.model_atom_indices) - len(mask_backbone)),
            "mask_voxels": int(base.mask.sum()),
            "sidechain_density_note": "all model atoms are rendered; sidechain density can enter backbone-centred voxels by overlap/tails, but sidechains do not expand the mask",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoints = args.output / "site_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    rows = []
    for site in SITES:
        path = checkpoints / f"{name(site)}.json"
        if args.resume and path.is_file():
            row = json.loads(path.read_text())
        else:
            row = audit_site(site)
            atomic_json(path, row)
        rows.append(row)
        atomic_json(args.output / "progress.json", {"status": "running", "completed_sites": len(rows)})
    atomic_json(args.output / "sidechain_scope.json", {"status": "complete", "rows": rows})
    atomic_json(args.output / "progress.json", {"status": "complete", "completed_sites": len(rows)})
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
