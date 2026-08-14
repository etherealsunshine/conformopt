#!/usr/bin/env python3
"""Read-only MolProbity validation for the 5OHJ slot-coordination endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import libtbx.load_env  # noqa: F401  # installs libtbx.env for mmtbx
import libtbx
import numpy as np
from iotbx import pdb
from mmtbx.validation import clashscore, omegalyze, ramalyze, rotalyze

from run_d1_aprime_sequential import APrimeSequential


def register_probe_module(probe_root: Path) -> None:
    """Make the locally built Probe visible to mmtbx's module registry."""
    if libtbx.env.has_module("probe"):
        return
    module_type = libtbx.env.module_dict["mmtbx"].__class__
    module = module_type(libtbx.env, "probe", str(probe_root))
    libtbx.env.register_module(None, module)


def as_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def atom_info_is_central(info: object) -> bool:
    return getattr(info, "chain_id", "").strip() == "A" and as_int(
        getattr(info, "resseq", None)
    ) == 540


def classify_rama(result: object) -> str:
    probability = float(result.score) / 100.0
    value = ramalyze.ramalyze.evalScore(int(result.res_type), probability)
    return {2: "favoured", 1: "allowed", 0: "outlier"}[int(value)]


def residue_row(result: object) -> dict[str, object]:
    return {
        "chain": str(result.chain_id).strip(),
        "resseq": as_int(result.resseq),
        "altloc": str(result.altloc).strip(),
        "resname": str(result.resname).strip(),
    }


def filtered_pdb_lines(base: object, coordinates: np.ndarray) -> list[str]:
    """Build a valid A/blank scaffold by filtering PDB text before parsing.

    Editing hierarchy atom-group ownership in place can leave CCTBX's internal
    atom-group index arrays inconsistent.  Text filtering is slower but keeps
    the MolProbity hierarchy canonical and avoids that failure mode.
    """
    window_coords = {}
    for local, global_index in enumerate(base.window.selection):
        key = (
            str(base.full_structure.chain[int(global_index)]).strip(),
            int(base.full_structure.resi[int(global_index)]),
            str(base.full_structure.name[int(global_index)]).strip(),
        )
        window_coords[key] = coordinates[local]
    window_residues = {key[1] for key in window_coords if key[0] == "A"}
    emitted_window_atoms = set()
    lines = []
    for raw in base.a_structure._pdb_hierarchy.as_pdb_string().splitlines():  # pylint: disable=protected-access
        if not raw.startswith(("ATOM", "HETATM")):
            lines.append(raw)
            continue
        chain_id = raw[21].strip()
        resseq = as_int(raw[22:26])
        name = raw[12:16].strip()
        window_key = (chain_id, resseq, name)
        if chain_id == "A" and resseq in window_residues and window_key in window_coords:
            # Use the exact endpoint-window coordinates and a single A-like
            # atom group even if the deposited text has only B/D templates.
            if window_key in emitted_window_atoms:
                continue
            emitted_window_atoms.add(window_key)
            xyz = window_coords[window_key]
            raw = raw[:16] + "A" + raw[17:30]
            raw = raw[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + raw[54:]
            lines.append(raw)
            continue
        if raw[16].strip() not in ("", "A"):
            continue
        lines.append(raw)
    return lines


def parse_lines(lines: list[str]):
    return pdb.input(source_info="endpoint", lines=lines).construct_hierarchy()


def endpoint_hierarchy(base: object, coordinates: np.ndarray):
    hierarchy = parse_lines(filtered_pdb_lines(base, coordinates))
    central = coordinates[base.central_indices]
    by_name = {
        str(name).strip(): xyz for name, xyz in zip(base.central.name.tolist(), central)
    }
    return hierarchy


def add_second_slot(base: object, first: np.ndarray, second: np.ndarray):
    first_lines = filtered_pdb_lines(base, first)
    central = second[base.central_indices]
    by_name = {
        str(name).strip(): xyz for name, xyz in zip(base.central.name.tolist(), central)
    }
    second_lines = []
    for raw in first_lines:
        if raw.startswith(("ATOM", "HETATM")) and raw[21].strip() == "A" and as_int(raw[22:26]) == 540:
            name = raw[12:16].strip()
            xyz = by_name[name]
            line = raw[:16] + "B" + raw[17:30]
            line = line[:30] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[54:]
            second_lines.append(line)
    insert_at = max(
        index for index, raw in enumerate(first_lines)
        if raw.startswith(("ATOM", "HETATM")) and raw[21].strip() == "A" and as_int(raw[22:26]) == 540
    ) + 1
    return parse_lines(first_lines[:insert_at] + second_lines + first_lines[insert_at:])


def local_clash_partition(result: object) -> dict[str, object]:
    central_frozen = []
    slot_slot = []
    for clash in result.results:
        infos = list(clash.atoms_info)
        central_flags = [atom_info_is_central(info) for info in infos]
        if all(central_flags):
            slot_slot.append(clash)
        elif any(central_flags):
            central_frozen.append(clash)

    def describe(items: list[object]) -> list[dict[str, object]]:
        output = []
        for clash in items:
            output.append({
                "overlap_A": float(clash.overlap),
                "atoms": [
                    {
                        "chain": str(info.chain_id).strip(),
                        "resseq": as_int(info.resseq),
                        "resname": str(info.resname).strip(),
                        "altloc": str(info.altloc).strip(),
                        "name": str(info.name).strip(),
                    }
                    for info in clash.atoms_info
                ],
            })
        return output

    by_slot: dict[str, int] = {}
    for clash in central_frozen:
        for info in clash.atoms_info:
            if atom_info_is_central(info):
                by_slot[str(info.altloc).strip() or "A"] = by_slot.get(
                    str(info.altloc).strip() or "A", 0
                ) + 1
    return {
        "probe_clashes_between_slots": len(slot_slot),
        "probe_clashes_slot_vs_frozen_neighbours": len(central_frozen),
        "by_slot_vs_frozen": by_slot,
        "slot_slot_details": describe(slot_slot),
        "slot_frozen_details": describe(central_frozen),
    }


def validate_hierarchy(hierarchy: object) -> dict[str, object]:
    clash = clashscore.clashscore(hierarchy.deep_copy(), keep_hydrogens=False)
    rama = ramalyze.ramalyze(hierarchy.deep_copy(), outliers_only=False, quiet=True)
    rota = rotalyze.rotalyze(hierarchy.deep_copy(), outliers_only=False, quiet=True)
    omega = omegalyze.omegalyze(hierarchy.deep_copy(), quiet=True)

    window_rama = []
    for result in rama.results:
        if str(result.chain_id).strip() == "A" and 538 <= (as_int(result.resseq) or -1) <= 542:
            row = residue_row(result)
            row.update({
                "probability": float(result.score) / 100.0,
                "classification": classify_rama(result),
            })
            window_rama.append(row)
    window_rota = []
    for result in rota.results:
        if str(result.chain_id).strip() == "A" and 538 <= (as_int(result.resseq) or -1) <= 542:
            row = residue_row(result)
            row.update({
                "evaluation": str(result.evaluation),
                "rotamer": str(result.rotamer_name),
                "score": float(result.score),
            })
            window_rota.append(row)
    window_omega = []
    for result in omega.results:
        if str(result.chain_id).strip() == "A" and 538 <= (as_int(result.resseq) or -1) <= 542:
            row = residue_row(result)
            row.update({
                "omega_deg": float(result.omega),
                "classification": {
                    0: "trans", 1: "cis", 2: "twisted"
                }.get(int(result.omega_type), str(result.omega_type)),
                "outlier": bool(result.outlier),
            })
            window_omega.append(row)
    return {
        "clashscore": float(clash.clashscore),
        "clash_count": int(len(clash.results)),
        "rama_window": window_rama,
        "rama_counts_full": {
            "favoured": int(rama.n_favored),
            "allowed": int(rama.n_allowed),
            "outlier": int(rama.n_outliers),
        },
        "rotamer_window": window_rota,
        "rotamer_outliers_full": int(rota.n_outliers),
        "rotamer_outlier_residues_full": [
            residue_row(result) for result in rota.results if bool(result.outlier)
        ],
        "omega_window": window_omega,
        "omega_outliers_full": int(omega.n_outliers),
    }


def deposited_b_rama(base: object) -> list[dict[str, object]]:
    rama = ramalyze.ramalyze(
        base.full_structure._pdb_hierarchy.deep_copy(),  # pylint: disable=protected-access
        outliers_only=False,
        quiet=True,
    )
    return [
        {
            **residue_row(result),
            "probability": float(result.score) / 100.0,
            "classification": classify_rama(result),
        }
        for result in rama.results
        if str(result.chain_id).strip() == "A"
        and as_int(result.resseq) == 541
        and str(result.altloc).strip() == "B"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    args = parser.parse_args()
    register_probe_module(args.probe_root)
    runner = APrimeSequential(
        args.out / "runner", 8, 6, "5OHJ", "A", 540,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full",
    )
    output: dict[str, object] = {"endpoints": {}, "deposited_5ohj_ser541_altloc_b": deposited_b_rama(runner.base)}
    for label in ("D_null_axis2_30deg", "D_null_axis3_30deg"):
        path = args.endpoint_root / label / "final_slots.npz"
        saved = np.load(path)
        first = saved["slot1_window"]
        second = saved["slot2_window"]
        first_h = endpoint_hierarchy(runner.base, first)
        second_h = endpoint_hierarchy(runner.base, second)
        pair_h = add_second_slot(runner.base, first, second)
        output["endpoints"][label] = {
            "slot1": validate_hierarchy(first_h),
            "slot2": validate_hierarchy(second_h),
            "pair": {
                **validate_hierarchy(pair_h),
                **local_clash_partition(
                    clashscore.clashscore(pair_h.deep_copy(), keep_hydrogens=False)
                ),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
