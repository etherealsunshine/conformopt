#!/usr/bin/env python3
"""Measure qFit Tier-A backbone-candidate coverage at the six guarded-CV sites.

This is deliberately a sampler audit, not a qFit fit or refinement.  Each
site starts from deposited altloc A, calls qFit's native ``_sample_backbone``
once, and compares all 19 returned central-backbone candidates with deposited
altloc A and B in the same N/CA/C/O RMSD metric used for A'' slot reports.
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
    import torch  # noqa: F401
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.structure import Structure

from run_d1_8d_sequential_poc import atomic_json
from run_d1_tier_a_flips import (
    BACKBONE_NAMES,
    atom_local_index,
    backbone_coordinates,
    get_sampler_xmap,
    rmsd,
    source_path,
)


SITES = (
    ("4HFS", "A", 200),
    ("4ZXG", "B", 13),
    ("6ZWK", "B", 47),
    ("7ZTL", "A", 257),
    ("8AJK", "A", 240),
    ("8R7O", "C", 1681),
)
PANEL_MANIFEST = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")


def site_name(site: tuple[str, str, int]) -> str:
    return f"{site[0]}_{site[1]}_{site[2]}"


def panel_record(site: tuple[str, str, int], manifest: list[dict[str, object]]) -> dict[str, object]:
    pdb_id, chain, resnum = site
    found = [row for row in manifest if row["pdb_id"] == pdb_id and row["chain"] == chain
             and int(row["resnum"]) == resnum]
    if len(found) != 1:
        raise ValueError(f"Expected one manifest record for {site_name(site)}, found {len(found)}")
    return found[0]


def audit_site(site: tuple[str, str, int], manifest: list[dict[str, object]]) -> dict[str, object]:
    pdb_id, chain, resnum = site
    record = panel_record(site, manifest)
    path, split = source_path(pdb_id)
    residue_id = (resnum, record.get("insertion_code", ""))
    a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
    a_residue = a_structure[chain].conformers[0][residue_id]
    b_residue = b_structure[chain].conformers[0][residue_id]
    a_backbone, b_backbone = backbone_coordinates(a_residue), backbone_coordinates(b_residue)
    separation = rmsd(a_backbone, b_backbone)
    options = QFitOptions()
    options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(a_residue, a_structure, get_sampler_xmap(), options)
    qfit._sample_backbone()
    indices = [atom_local_index(a_residue, name) for name in BACKBONE_NAMES]
    candidates = []
    for number, coordinates in enumerate(qfit._coor_set):
        central = coordinates[indices]
        candidates.append({
            "candidate_index": number,
            "to_deposited_A_RMSD_A": rmsd(central, a_backbone),
            "to_deposited_B_RMSD_A": rmsd(central, b_backbone),
        })
    best_a = min(candidates, key=lambda row: row["to_deposited_A_RMSD_A"])
    best_b = min(candidates, key=lambda row: row["to_deposited_B_RMSD_A"])
    return {
        "site": site_name(site), "pdb_id": pdb_id, "chain": chain, "resnum": resnum,
        "panel": record["panel"], "source_split": split,
        "metric": "central residue N/CA/C/O conventional RMSD",
        "started_from": "deposited altloc A",
        "candidate_count_including_deposited_A_input": len(candidates),
        "deposited_A_B_RMSD_A": separation,
        "minimum_to_deposited_A": {
            **best_a,
            "fraction_of_A_B_separation": best_a["to_deposited_A_RMSD_A"] / separation,
        },
        "minimum_to_deposited_B": {
            **best_b,
            "fraction_of_A_B_separation": best_b["to_deposited_B_RMSD_A"] / separation,
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoints = args.output / "site_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    manifest = json.loads(PANEL_MANIFEST.read_text())
    rows = []
    for site in SITES:
        checkpoint = checkpoints / f"{site_name(site)}.json"
        if args.resume and checkpoint.is_file():
            row = json.loads(checkpoint.read_text())
        else:
            row = audit_site(site, manifest)
            atomic_json(checkpoint, row)
        rows.append(row)
        atomic_json(args.output / "progress.json", {
            "status": "running", "completed_sites": len(rows),
            "last_site": row["site"],
        })
    panel_counts = {panel: sum(row["panel"] == panel for row in rows)
                    for panel in sorted({row["panel"] for row in rows})}
    atomic_json(args.output / "qfit_sample_backbone_baseline.json", {
        "status": "complete",
        "operation": "native qFit _sample_backbone only; no qFit scoring or refinement",
        "panel_counts": panel_counts,
        "rows": rows,
    })
    atomic_json(args.output / "progress.json", {
        "status": "complete", "completed_sites": len(rows),
        "last_site": rows[-1]["site"],
    })
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
