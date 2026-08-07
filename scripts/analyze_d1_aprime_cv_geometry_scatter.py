#!/usr/bin/env python3
"""Quantify fold-to-fold central-backbone scatter for completed A' CV folds."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from run_d1_8d_sequential_poc import atomic_json, rmsd
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential


def positional_sigma(coordinates):
    """RMS 3D sample SD over central atoms, matching a positional sigma unit."""
    variance = np.var(coordinates, axis=0, ddof=1)
    per_atom = np.sqrt(np.sum(variance, axis=1))
    return float(np.sqrt(np.mean(per_atom ** 2))), per_atom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="Completed five-fold CV root; writes geometry_scatter.json there.")
    args = parser.parse_args()
    runner = APrimeSequential(args.output, 1, 1)
    slots = {"slot1": [], "slot2": []}
    for fold in range(5):
        final = np.load(args.output / f"split_{fold}" / "final_slots.npz")
        for slot in slots:
            slots[slot].append(runner.base.central_backbone(final[f"{slot}_window"]))
    splits = blocked_splits(runner.base)
    overlap_fractions = []
    for first, second in itertools.combinations(range(5), 2):
        train_a, train_b = splits[first][0], splits[second][0]
        overlap_fractions.append(float(len(np.intersect1d(train_a, train_b)) / len(train_a)))
    mean_overlap = float(np.mean(overlap_fractions))
    result = {
        "status": "complete",
        "site": "7UTC_A_ARG52",
        "metric": "RMS 3D sample positional SD across central N, CA, C, O coordinates; no superposition because all folds share the deposited-A frame",
        "folds": 5,
        "mean_pairwise_training_overlap_fraction": mean_overlap,
        "pairwise_training_overlap_fractions": overlap_fractions,
        "heuristic_shared-data_reduction_vs_independent": float(np.sqrt(1.0 - mean_overlap)),
        "heuristic_note": "If voxel contributions were independent and estimator correlation equalled the train-set overlap, observed fold scatter would be about sqrt(1-overlap) times an independent-resample SD. Spatial correlation and nonlinear fitting make this a lower-bound heuristic, not a calibrated conversion from the Hessian sigma.",
        "analytic_positional_sigmas_A": {"slot1": 0.398, "slot2": 1.577},
        "slots": {},
    }
    for name, records in slots.items():
        array = np.asarray(records)
        sigma, per_atom = positional_sigma(array)
        pairwise = [float(rmsd(array[i], array[j])) for i, j in itertools.combinations(range(5), 2)]
        result["slots"][name] = {
            "fold_to_fold_positional_sd_A": sigma,
            "per_atom_positional_sd_A": per_atom.tolist(),
            "mean_pairwise_rmsd_A": float(np.mean(pairwise)),
            "pairwise_rmsd_A": pairwise,
            "ratio_to_analytic_sigma": sigma / result["analytic_positional_sigmas_A"][name],
        }
    atomic_json(args.output / "geometry_scatter.json", result)
    print(result)


if __name__ == "__main__":
    main()
