#!/usr/bin/env python3
"""Trace qFit's actual backbone sampler to measure discarded window motion.

This is deliberately a measurement wrapper: it observes calls made by qFit's
``_sample_backbone`` and never substitutes its own candidate generator.
"""

import argparse
import json
import numpy as np

from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.samplers import BackboneRotator
from qfit.structure import Structure

from run_d6_tier2_realmap import make_map


# Standard protein-restraint targets used by Phenix's protein geometry model.
# qFit itself carries the C--N target/sigma (1.329, 0.014) in rotamers.py.
PEPTIDE_CN_IDEAL_A = 1.329
PEPTIDE_CN_SIGMA_A = 0.014
PEPTIDE_CNCA_IDEAL_DEG = 121.7
PEPTIDE_CNCA_SIGMA_DEG = 2.1


def _local_index(window, residue, atom_name):
    """Return an atom's index within ``window.coor``."""
    global_index = int(residue.select("name", atom_name)[0])
    return int(np.searchsorted(window.selection, global_index))


def _angle_deg(a, b, c):
    """The angle ABC in degrees."""
    u = a - b
    v = c - b
    cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdb", required=True)
    p.add_argument("--mtz", required=True)
    p.add_argument("--chain", required=True)
    p.add_argument("--resnum", type=int, required=True)
    args = p.parse_args()
    structure = Structure.fromfile(args.pdb).extract("altloc", ("", "A"))
    residue = structure[args.chain].conformers[0][(args.resnum, "")]
    xmap, _, _, _ = make_map(args.mtz)
    options = QFitOptions()
    options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(residue, structure, xmap, options)
    index = qfit.segment.find(qfit.residue.id)
    window = qfit.segment[index - 3:index + 4]
    start = window.coor.copy()
    local_by_position = []
    for residue in window.residues:
        local_by_position.append(np.searchsorted(window.selection, residue.selection))
    central_local = local_by_position[3]
    noncentral = np.setdiff1d(np.arange(start.shape[0]), central_local)
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
    distances = [np.linalg.norm(coor - start, axis=1) for coor in final]

    by_position = []
    for offset, indices in zip(range(-3, 4), local_by_position):
        all_distances = np.asarray([distance[indices] for distance in distances])
        by_position.append({
            "position": f"centre{offset:+d}" if offset else "centre",
            "max_displacement_A": float(all_distances.max()),
            "max_rms_displacement_A": float(
                max(np.sqrt(np.mean(d * d)) for d in all_distances)
            ),
        })

    # Identify the exact sample/atom responsible for the headline displacement.
    noncentral_distances = np.asarray([distance[noncentral] for distance in distances])
    call_i, local_noncentral_i = np.unravel_index(
        int(np.argmax(noncentral_distances)), noncentral_distances.shape
    )
    call_i = int(call_i)
    local_noncentral_i = int(local_noncentral_i)
    atom_i = int(noncentral[local_noncentral_i])
    position_i = next(i for i, indices in enumerate(local_by_position) if atom_i in indices)
    residue_i = window.residues[position_i]
    atom_global_i = int(window.selection[atom_i])
    atom_name = str(window.name[np.searchsorted(window.selection, atom_global_i)])

    # qFit only retains the central-residue coordinates in _coor_set.  Measure
    # the outgoing peptide against the (therefore undeformed) next neighbour.
    c_i = _local_index(window, window.residues[3], "C")
    n_next_i = _local_index(window, window.residues[4], "N")
    ca_next_i = _local_index(window, window.residues[4], "CA")

    def peptide_metrics(coor):
        cn = float(np.linalg.norm(coor[c_i] - start[n_next_i]))
        angle = _angle_deg(coor[c_i], start[n_next_i], start[ca_next_i])
        return {
            "C_to_undeformed_next_N_A": cn,
            "C_N_CA_with_undeformed_next_CA_deg": angle,
            "C_N_deviation_from_ideal_A": cn - PEPTIDE_CN_IDEAL_A,
            "C_N_deviation_sigma": (cn - PEPTIDE_CN_IDEAL_A) / PEPTIDE_CN_SIGMA_A,
            "C_N_CA_deviation_from_ideal_deg": angle - PEPTIDE_CNCA_IDEAL_DEG,
            "C_N_CA_deviation_sigma": (
                (angle - PEPTIDE_CNCA_IDEAL_DEG) / PEPTIDE_CNCA_SIGMA_DEG
            ),
        }

    max_motion_metrics = peptide_metrics(final[call_i])
    all_peptide = [peptide_metrics(coor) for coor in final]
    max_cn = max(all_peptide, key=lambda item: abs(item["C_N_deviation_sigma"]))
    max_angle = max(all_peptide, key=lambda item: abs(item["C_N_CA_deviation_sigma"]))
    output = {
        "qfit_candidates_including_input": len(qfit._coor_set),
        "rotator_calls": len(calls),
        "final_solution_calls": len(final),
        "noncentral_max_displacement_A": float(noncentral_distances.max()),
        "noncentral_median_of_max_displacement_A": float(
            np.median([distance[noncentral].max() for distance in distances])
        ),
        "noncentral_max_rms_displacement_A": max(
            float(np.sqrt(np.mean(distance[noncentral] ** 2))) for distance in distances
        ),
        "displacement_by_window_position": by_position,
        "headline_maximum": {
            "qfit_candidate_index_including_input": int(call_i + 2),
            "window_position": f"centre{position_i - 3:+d}",
            "residue_id": str(residue_i.id),
            "atom": atom_name,
            "displacement_A": float(noncentral_distances[call_i, local_noncentral_i]),
            "outgoing_peptide_against_undeformed_neighbour": max_motion_metrics,
        },
        "most_extreme_outgoing_peptide_among_18_candidates": {
            "by_C_N_length": max_cn,
            "by_C_N_CA_angle": max_angle,
        },
        "restraint_reference": {
            "C_N_ideal_A": PEPTIDE_CN_IDEAL_A,
            "C_N_sigma_A": PEPTIDE_CN_SIGMA_A,
            "C_N_CA_ideal_deg": PEPTIDE_CNCA_IDEAL_DEG,
            "C_N_CA_sigma_deg": PEPTIDE_CNCA_SIGMA_DEG,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
