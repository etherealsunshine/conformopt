#!/usr/bin/env python3
"""Export 5OHJ closeout render inputs in the optimizer/CCTBX environment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from analyze_d1_qfit_selected_aprime_closeout import replace_central
from run_d1_5ohj_aprime_phenix_closeout import (
    APRIME_SITE,
    CHAIN,
    PDB_ID,
    QFIT_ROOT,
    RESNUM,
    SITE,
    read_pair_pdb,
    write_pair_pdb,
)
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential


OUT = Path(os.environ.get(
    "D1_SPLIT_ROOT",
    "/home/dev/qfit_unet_data/qfit_audit/d1_5ohj_aprime_phenix_closeout_v4_split",
))
MODE = os.environ.get("D1_SPLIT_MODE", "raw")


def raw_pairs(runner: APrimeSequential) -> tuple[np.ndarray, np.ndarray]:
    qroot = QFIT_ROOT / SITE
    qz = np.load(qroot / "selected.npz", allow_pickle=False)
    selected = np.asarray(qz["selected_indices"], dtype=int)
    if len(selected) != 1:
        raise RuntimeError(f"expected one qFit alternate, got {selected.tolist()}")
    candidates = np.asarray(qz["candidate_coordinates"], dtype=float)
    names = [str(x) for x in qz["candidate_atom_names"].tolist()]
    qfit = np.stack((runner.initial, replace_central(
        runner.initial, runner, candidates[int(selected[0])], names)))
    with np.load(APRIME_SITE / "geometry_outer_500.npz") as saved:
        aprime = np.stack((saved["slot1_window"], saved["slot2_window"]))
    return qfit, aprime


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    qroot = QFIT_ROOT / SITE
    neutral = Path(json.loads((qroot / "result.json").read_text())["start_pdb"])
    runner = APrimeSequential(
        OUT / "context", 1, 1, PDB_ID, CHAIN, RESNUM,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device="cpu",
        start_pdb=neutral, b_factor_mode="single_conformer", density_atom_scope="all",
    )
    if MODE == "raw":
        qfit, aprime = raw_pairs(runner)
        qfit_pdb, aprime_pdb = OUT / "qfit_selected.pdb", OUT / "aprime_endpoint.pdb"
        qz = np.load(qroot / "selected.npz", allow_pickle=False)
        selected = np.asarray(qz["selected_indices"], dtype=int)
        qfit_occ = np.asarray([
            max(1.0 - float(qz["occupancies"][selected].sum()), 0.0),
            float(qz["occupancies"][selected[0]]),
        ])
        with np.load(APRIME_SITE / "resume_state.npz") as saved:
            aprime_occ = np.asarray(saved["occupancy_weights"], dtype=float)
        write_pair_pdb(neutral, runner, qfit, qfit_occ, qfit_pdb)
        write_pair_pdb(neutral, runner, aprime, aprime_occ, aprime_pdb)
        pairs = np.stack((qfit, aprime))
        pdbs = {"qfit_raw": str(qfit_pdb), "aprime": str(aprime_pdb)}
    elif MODE == "refined":
        qfit_pdb = Path(os.environ["D1_QFIT_REFINED_PDB"])
        aprime_pdb = Path(os.environ["D1_APRIME_REFINED_PDB"])
        qfit = read_pair_pdb(qfit_pdb, runner)
        aprime = read_pair_pdb(aprime_pdb, runner)
        pairs = np.stack((qfit, aprime))
        pdbs = {"qfit_phenix": str(qfit_pdb), "aprime_phenix": str(aprime_pdb)}
    else:
        raise ValueError(f"unknown D1_SPLIT_MODE={MODE}")

    model_indices = np.asarray(runner.model_atom_indices, dtype=int)
    coords = np.stack([pair[:, model_indices] for pair in pairs]).reshape(4, len(model_indices), 3)
    b_factors = np.stack([
        runner.slot_b_factors(0), runner.slot_b_factors(1),
        runner.slot_b_factors(0), runner.slot_b_factors(1),
    ])
    np.savez_compressed(
        OUT / f"render_inputs_{MODE}.npz",
        coordinates=coords,
        b_factors=b_factors,
        renderer_grid=runner._renderer_grid.detach().cpu().numpy(),
        renderer_cell=runner._renderer_cell.detach().cpu().numpy(),
        renderer_coefficients=runner._renderer_coefficients.detach().cpu().numpy(),
        renderer_u_base=np.asarray([runner._renderer_u_base], dtype=float),
        target=np.asarray(runner.target, dtype=float),
        model_atom_indices=model_indices,
    )
    folds = []
    for fold, (train, test, direction) in enumerate(blocked_splits(runner.base)):
        folds.append({"fold": fold, "train": train.tolist(), "test": test.tolist(), "direction": np.asarray(direction).tolist()})
    (OUT / f"manifest_{MODE}.json").write_text(json.dumps({
        "site": SITE, "mode": MODE, "pdbs": pdbs,
        "renderer": {"mask_scope": runner.base.mask_scope, "density_atom_scope": runner.base.density_atom_scope,
                     "smooth_exp": True, "u_base": float(runner._renderer_u_base)},
        "n_voxels": int(runner.target.size), "n_model_atoms": int(len(model_indices)),
        "folds": folds,
    }, indent=2) + "\n")
    print(json.dumps({"status": "complete", "mode": MODE, "out": str(OUT), "n_voxels": int(runner.target.size), "n_model_atoms": int(len(model_indices))}, indent=2))


if __name__ == "__main__":
    main()
