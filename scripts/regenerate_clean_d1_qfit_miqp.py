#!/usr/bin/env python3
"""Regenerate native qFit MIQP selections from the frozen neutral starts.

This deliberately performs only qFit's native backbone candidate generation and
MIQP selection.  It does not optimize A-prime or score held-out folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qfit  # noqa: F401  # keep qFit/CCTBX imports first
import numpy as np

from clean_d1_benchmark import source_path
from run_d1_tier_a_flips import BACKBONE_NAMES, atom_local_index
from run_d6_tier2_realmap import make_map


SITES = {
    "6ZWK_B_PHE47": ("6ZWK", "B", 47),
    "8R7O_C_THR1681": ("8R7O", "C", 1681),
}


def conventional_rmsd(candidate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((candidate - truth) ** 2, axis=1))))


def run_site(site_key: str, start_pdb: Path, output: Path) -> dict[str, object]:
    from qfit.qfit import QFitOptions, QFitRotamericResidue
    from qfit.structure import Structure

    pdb_id, chain, resnum = SITES[site_key]
    _, split = source_path(pdb_id)
    mtz = Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{pdb_id}.mtz")
    xmap, _, _, _ = make_map(mtz)
    structure = Structure.fromfile(str(start_pdb)).extract("altloc", ("", "A"))
    residue_id = (resnum, "")
    residue = structure[chain].conformers[0][residue_id]

    options = QFitOptions()
    options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit_residue = QFitRotamericResidue(residue, structure, xmap, options)
    qfit_residue._sample_backbone()
    if len(qfit_residue._coor_set) != 19:
        raise RuntimeError(f"{site_key}: expected 19 candidates, got {len(qfit_residue._coor_set)}")

    truth_structure = Structure.fromfile(source_path(pdb_id)[0]).extract("altloc", ("", "A"))
    truth_a = truth_structure[chain].conformers[0][residue_id]
    truth_structure_b = Structure.fromfile(source_path(pdb_id)[0]).extract("altloc", ("", "B"))
    truth_b = truth_structure_b[chain].conformers[0][residue_id]
    names = [atom_local_index(residue, name) for name in BACKBONE_NAMES]
    a = np.asarray([truth_a.coor[atom_local_index(truth_a, name)] for name in BACKBONE_NAMES])
    b = np.asarray([truth_b.coor[atom_local_index(truth_b, name)] for name in BACKBONE_NAMES])
    candidates = np.asarray(qfit_residue._coor_set, dtype=float)
    b_factors = np.asarray(qfit_residue._bs, dtype=float)
    candidate_rows = []
    for index, coordinates in enumerate(candidates):
        central = np.asarray(coordinates[names], dtype=float)
        candidate_rows.append({
            "candidate": index,
            "rmsd_to_A_A": conventional_rmsd(central, a),
            "rmsd_to_B_A": conventional_rmsd(central, b),
            "fraction_to_A": conventional_rmsd(central, a) / conventional_rmsd(a, b),
            "fraction_to_B": conventional_rmsd(central, b) / conventional_rmsd(a, b),
        })

    # Native qFit fixed-geometry selection: score the 19 candidates, then run
    # qFit's own BIC/MIQP path.  No A-prime or held-out data are involved.
    qfit_residue._convert()
    qfit_residue._solve_miqp(
        threshold=qfit_residue.options.threshold,
        cardinality=qfit_residue.options.cardinality,
    )
    occupancies = np.asarray(qfit_residue._occupancies, dtype=float)
    intercept = float(qfit_residue._intercept)
    selected_mask = occupancies >= 0.01
    selected_indices = np.flatnonzero(selected_mask).astype(int)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "selected.npz",
        candidate_coordinates=candidates,
        candidate_b_factors=b_factors,
        occupancies=occupancies,
        selected_indices=selected_indices,
        selected_mask=selected_mask,
        candidate_atom_names=np.asarray(residue.name.tolist()),
        intercept=np.asarray(intercept),
    )
    result = {
        "status": "complete",
        "site": site_key,
        "start_pdb": str(start_pdb),
        "candidate_count": int(len(candidates)),
        "selection_method": "native qFit _solve_miqp with default BIC threshold loop",
        "qfit_threshold": float(qfit_residue.options.threshold),
        "qfit_cardinality": int(qfit_residue.options.cardinality),
        "selected_indices": selected_indices.tolist(),
        "selected_occupancies": occupancies[selected_mask].tolist(),
        "all_occupancies": occupancies.tolist(),
        "intercept": intercept,
        "candidate_0_is_neutral_start": True,
        "b_factor_provenance": "single-conformer input residue B array; no per-atom B refinement",
        "candidate_rows": candidate_rows,
        "selected_rows": [candidate_rows[int(i)] for i in selected_indices],
        "A_B_separation_A": conventional_rmsd(a, b),
        "candidate_0_b_factors": b_factors[0].tolist(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for site_key, site in SITES.items():
        start = args.starts / "sites" / site_key / "neutral_start_aprime_single_slot.pdb"
        rows.append(run_site(site_key, start, args.output / site_key))
    (args.output / "summary.json").write_text(json.dumps({"status": "complete", "rows": rows}, indent=2))
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
