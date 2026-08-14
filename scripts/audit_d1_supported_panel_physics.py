#!/usr/bin/env python3
"""Read-only MolProbity/Probe audit of the fixed-init supported-site panel.

The recovery controller deliberately leaves physical validation separate from
the GPU fits.  This script reconstructs each saved seven-residue endpoint in a
canonical PDB hierarchy, then measures all window Rama classifications and
the two requested local clash partitions.  It never changes endpoint geometry.
"""

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
    import torch  # noqa: F401  # initialize CUDA before qFit extension imports
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

import libtbx.load_env  # noqa: E402,F401
import libtbx  # noqa: E402
from iotbx import pdb  # noqa: E402
from mmtbx.validation import clashscore, omegalyze, ramalyze  # noqa: E402

from run_d1_aprime_sequential import APrimeSequential  # noqa: E402
from run_d1_8d_sequential_poc import atomic_json  # noqa: E402
from run_d1_molprobity_validation import as_int, classify_rama, residue_row  # noqa: E402
from run_d1_supported_panel import SUPPORTED_SITES, site_name  # noqa: E402


def register_probe_module(probe_root: Path) -> None:
    """Make the independently installed Probe module visible to mmtbx."""
    if libtbx.env.has_module("probe"):
        return
    module_type = libtbx.env.module_dict["mmtbx"].__class__
    libtbx.env.register_module(None, module_type(libtbx.env, "probe", str(probe_root)))


def atom_key(base, global_index: int) -> tuple[str, int, str]:
    return (
        str(base.full_structure.chain[global_index]).strip(),
        int(base.full_structure.resi[global_index]),
        str(base.full_structure.name[global_index]).strip(),
    )


def window_coordinates(base, coordinates: np.ndarray) -> dict[tuple[str, int, str], np.ndarray]:
    return {
        atom_key(base, int(global_index)): np.asarray(coordinates[local], dtype=float)
        for local, global_index in enumerate(base.window.selection)
    }


def rewrite_atom(raw: str, xyz: np.ndarray, altloc: str) -> str:
    line = raw[:16] + altloc + raw[17:30]
    return line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:]


def endpoint_lines(base, first: np.ndarray, second: np.ndarray | None = None) -> list[str]:
    """Use blank/A scaffold atoms and explicitly insert endpoint window slots."""
    first_coords = window_coordinates(base, first)
    second_coords = None if second is None else window_coordinates(base, second)
    emitted: set[tuple[str, int, str]] = set()
    lines: list[str] = []
    for raw in base.a_structure._pdb_hierarchy.as_pdb_string().splitlines():  # pylint: disable=protected-access
        if not raw.startswith(("ATOM", "HETATM")):
            lines.append(raw)
            continue
        key = (raw[21].strip(), as_int(raw[22:26]), raw[12:16].strip())
        if key in first_coords:
            if key in emitted:
                continue
            emitted.add(key)
            lines.append(rewrite_atom(raw, first_coords[key], "A"))
            if second_coords is not None:
                lines.append(rewrite_atom(raw, second_coords[key], "B"))
        elif raw[16].strip() in ("", "A"):
            lines.append(raw)
    return lines


def parse(lines: list[str]):
    return pdb.input(source_info="A-prime endpoint", lines=lines).construct_hierarchy()


def info_in_window(info: object, residues: set[tuple[str, int]]) -> bool:
    return (str(info.chain_id).strip(), as_int(info.resseq)) in residues


def describe_clashes(items: list[object]) -> list[dict[str, object]]:
    return [{
        "overlap_A": float(item.overlap),
        "atoms": [{
            "chain": str(info.chain_id).strip(), "resseq": as_int(info.resseq),
            "resname": str(info.resname).strip(), "altloc": str(info.altloc).strip(),
            "name": str(info.name).strip(),
        } for info in item.atoms_info],
    } for item in items]


def clash_partitions(result: object, residues: set[tuple[str, int]]) -> dict[str, object]:
    between_slots, versus_neighbours = [], []
    for item in result.results:
        infos = list(item.atoms_info)
        inside = [info_in_window(info, residues) for info in infos]
        altlocs = {str(info.altloc).strip() for info in infos if info_in_window(info, residues)}
        if all(inside) and {"A", "B"}.issubset(altlocs):
            between_slots.append(item)
        elif any(inside) and not all(inside):
            versus_neighbours.append(item)
    return {
        "probe_clashes_between_slots": len(between_slots),
        "probe_clashes_slot_vs_frozen_neighbours": len(versus_neighbours),
        "slot_slot_details": describe_clashes(between_slots),
        "slot_neighbour_details": describe_clashes(versus_neighbours),
    }


def hierarchy_audit(hierarchy: object, residues: set[tuple[str, int]]) -> dict[str, object]:
    clash = clashscore.clashscore(hierarchy.deep_copy(), keep_hydrogens=False)
    rama = ramalyze.ramalyze(hierarchy.deep_copy(), outliers_only=False, quiet=True)
    omega = omegalyze.omegalyze(hierarchy.deep_copy(), quiet=True)
    rama_window = []
    for item in rama.results:
        if (str(item.chain_id).strip(), as_int(item.resseq)) in residues:
            row = residue_row(item)
            row.update({"probability": float(item.score) / 100.0,
                        "classification": classify_rama(item)})
            rama_window.append(row)
    omega_window = []
    for item in omega.results:
        if (str(item.chain_id).strip(), as_int(item.resseq)) in residues:
            row = residue_row(item)
            row.update({"omega_deg": float(item.omega), "outlier": bool(item.outlier),
                        "classification": {0: "trans", 1: "cis", 2: "twisted"}.get(
                            int(item.omega_type), str(item.omega_type))})
            omega_window.append(row)
    return {
        "clashscore": float(clash.clashscore), "clash_count": int(len(clash.results)),
        "rama_window": rama_window, "omega_window": omega_window,
    }


def audit_site(panel_root: Path, site: tuple[str, str, int], out: Path) -> dict[str, object]:
    name = site_name(site)
    endpoint = panel_root / "sites" / name / "full_candidates" / "D_null_axis2_30deg" / "final_slots.npz"
    saved = np.load(endpoint)
    runner = APrimeSequential(out / name / "runner", 8, 6, *site,
                               renderer_backend="torch", residual_scale_mode="none",
                               map_scaler_structure="full", mask_scope="window", device="cuda")
    first, second = saved["slot1_window"], saved["slot2_window"]
    # qFit residue metadata are arrays repeated over the residue's atoms.
    # The first entry identifies the entire residue for this PDB-level audit.
    residues = {
        (str(residue.chain[0]).strip(), int(residue.resi[0]))
        for residue in runner.base.window.residues
    }
    first_h = parse(endpoint_lines(runner.base, first))
    second_h = parse(endpoint_lines(runner.base, second))
    pair_h = parse(endpoint_lines(runner.base, first, second))
    pair_clash = clashscore.clashscore(pair_h.deep_copy(), keep_hydrogens=False)
    return {
        "site": name, "window_residues": sorted([list(row) for row in residues]),
        "slot1": hierarchy_audit(first_h, residues),
        "slot2": hierarchy_audit(second_h, residues),
        "pair": {**hierarchy_audit(pair_h, residues),
                 **clash_partitions(pair_clash, residues)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--probe-root", required=True, type=Path)
    args = parser.parse_args()
    register_probe_module(args.probe_root)
    args.out.mkdir(parents=True, exist_ok=True)
    progress = {"status": "running", "sites": []}
    for site in SUPPORTED_SITES:
        path = args.out / f"{site_name(site)}.json"
        record = json.loads(path.read_text()) if path.is_file() else audit_site(args.panel_root, site, args.out)
        atomic_json(path, record)
        progress["sites"].append({"site": site_name(site), "status": "complete"})
        atomic_json(args.out / "progress.json", progress)
    progress["status"] = "complete"
    atomic_json(args.out / "summary.json", progress)
    atomic_json(args.out / "progress.json", progress)


if __name__ == "__main__":
    main()
