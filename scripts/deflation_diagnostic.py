#!/usr/bin/env python3
"""One-gradient deflation gate for the clean-D1 neutral starts."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import qfit  # noqa: F401
import numpy as np
import torch
from scipy.linalg import null_space

from clean_d1_benchmark import site_key
from qfit.backbone import compute_jacobian
from run_d1_aprime_sequential import APrimeSequential
from run_d1_reachability import dihedrals, wrapped_delta
from run_d1_slot_coordination import (
    FullJointParameterization,
    joint_evaluate,
    joint_jacobian,
)


LAMBDA_AMP = 0.008836284282618265


def measure_site(task: dict[str, object]) -> dict[str, object]:
    site = tuple(task["site"])
    runner = APrimeSequential(
        Path(task["work_root"]), 80, 6, *site,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=task["device"],
        start_pdb=task["start_pdb"], b_factor_mode="single_conformer",
    )
    parameterization = FullJointParameterization(runner.rotator.ndofs)
    parameters = np.zeros(parameterization.reduced_ndofs, dtype=float)
    aa_models = runner.base.model_density_batch(
        np.stack((runner.initial, runner.initial)), slots=np.asarray((0, 1)), b_offset=0.0
    )
    _, _, normalizer = runner.joint_qp_weights(runner.target, aa_models)
    weights = np.asarray((0.5, 0.5), dtype=float)
    state = joint_evaluate(
        runner, parameters, normalizer, np.zeros(12), parameterization,
        fixed_b_offset=0.0, occupancy_weights=weights,
        amplitude_prior_lambda=LAMBDA_AMP,
    )
    jacobian = joint_jacobian(
        runner, parameters, state, normalizer, np.zeros(12), parameterization,
        fixed_b_offset=0.0, amplitude_prior_lambda=LAMBDA_AMP,
    )
    gradient = jacobian.T @ state["residual"]
    g1 = np.asarray(gradient[:runner.rotator.ndofs], dtype=float)
    n = g1 / max(float(np.linalg.norm(g1)), 1e-30)

    # The deposited A->B direction is represented in the same 20-dimensional
    # torsion chart used by the optimizer, measured from the neutral start.
    a_window = runner.base.window_for_deposited_a()
    b_window = runner.base.window_for_deposited_b()
    neutral_phi_psi, neutral_omega = dihedrals(runner.window)
    def torsion_direction(window):
        original = runner.window.coor.copy()
        try:
            runner.window.coor = window.copy()
            phi_psi, omega = dihedrals(runner.window)
        finally:
            runner.window.coor = original
        return np.concatenate((
            wrapped_delta(neutral_phi_psi, phi_psi),
            wrapped_delta(neutral_omega, omega),
        ))
    d_a = torsion_direction(a_window)
    d_b = torsion_direction(b_window)
    d_ab = d_b - d_a
    d_ab_unit = d_ab / max(float(np.linalg.norm(d_ab)), 1e-30)
    d_a_unit = d_a / max(float(np.linalg.norm(d_a)), 1e-30)
    d_b_unit = d_b / max(float(np.linalg.norm(d_b)), 1e-30)

    # Reproduce the fixed axis-2 perturbation and then remove its component
    # along n while retaining the peptide-closure nullspace constraint.
    selection = np.sort(runner.window.select("name", ("N", "CA", "C")))
    closure_basis = null_space(compute_jacobian(runner.window.get_xyz(selection)))
    oxygen_indices = np.asarray([
        int(np.searchsorted(runner.window.selection, int(residue.select("name", "O")[0])))
        for residue in runner.window.residues
    ])
    def oxygen_forward(value):
        coordinates = runner.torch_forward(value.reshape(1, -1))[0]
        return coordinates[oxygen_indices].reshape(-1)
    j_o = torch.autograd.functional.jacobian(
        oxygen_forward, torch.zeros(runner.rotator.ndofs, dtype=torch.float64),
        vectorize=True, strategy="forward-mode",
    ).detach().cpu().numpy()
    _, _, vt = np.linalg.svd(j_o[:, :14] @ closure_basis, full_matrices=False)
    axis2 = np.zeros(runner.rotator.ndofs, dtype=float)
    axis2[:14] = closure_basis @ vt[1]
    axis2 /= max(float(np.linalg.norm(axis2)), 1e-30)
    closure_basis_20 = np.zeros((runner.rotator.ndofs, closure_basis.shape[1]))
    closure_basis_20[:14] = closure_basis
    coeff = closure_basis_20.T @ axis2
    n_in_null = closure_basis_20.T @ n
    coeff_perp = coeff - n_in_null * (np.dot(n_in_null, coeff) / max(float(np.dot(n_in_null, n_in_null)), 1e-30))
    d_perp = closure_basis_20 @ coeff_perp
    d_perp /= max(float(np.linalg.norm(d_perp)), 1e-30)
    rng = np.random.default_rng(20260814)
    random_coefficients = rng.normal(size=(200, closure_basis_20.shape[1]))
    random_directions = random_coefficients @ closure_basis_20.T
    random_directions /= np.maximum(
        np.linalg.norm(random_directions, axis=1, keepdims=True), 1e-30
    )
    random_angles = np.degrees(np.arccos(np.clip(random_directions @ d_ab_unit, -1.0, 1.0)))
    gradient_angle = float(np.degrees(np.arccos(np.clip(np.dot(n, d_ab_unit), -1.0, 1.0))))

    occupancy = np.asarray(runner.base.deposited_occupancies, dtype=float)
    return {
        "site": str(task["site_key"]),
        "gradient_definition": "J^T residual for the full current A' objective at neutral, mirror weights 0.5/0.5, dB=0",
        "gradient_norm_slot1": float(np.linalg.norm(g1)),
        "n_slot1": n.tolist(),
        "deposited_A_to_B_torsion_vector": d_ab.tolist(),
        "deposited_A_to_B_norm": float(np.linalg.norm(d_ab)),
        "A_to_B_component_along_n": float(np.dot(d_ab_unit, n)),
        "A_to_B_angle_to_n_deg": float(np.degrees(np.arccos(np.clip(np.dot(d_ab_unit, n), -1.0, 1.0)))),
        "neutral_to_A_torsion_vector": d_a.tolist(),
        "neutral_to_B_torsion_vector": d_b.tolist(),
        "neutral_to_A_component_along_n": float(np.dot(d_a_unit, n)),
        "neutral_to_A_angle_to_n_deg": float(np.degrees(np.arccos(np.clip(np.dot(d_a_unit, n), -1.0, 1.0)))),
        "neutral_to_B_component_along_n": float(np.dot(d_b_unit, n)),
        "neutral_to_B_angle_to_n_deg": float(np.degrees(np.arccos(np.clip(np.dot(d_b_unit, n), -1.0, 1.0)))),
        "axis2_initial_direction": axis2.tolist(),
        "d_perp_direction": d_perp.tolist(),
        "d_perp_angle_to_A_to_B_deg": float(np.degrees(np.arccos(np.clip(np.dot(d_perp, d_ab_unit), -1.0, 1.0)))),
        "random_nullspace_seed": 20260814,
        "random_nullspace_dimension": int(closure_basis_20.shape[1]),
        "random_nullspace_angle_mean_deg": float(np.mean(random_angles)),
        "random_nullspace_angle_sd_deg": float(np.std(random_angles, ddof=1)),
        "random_nullspace_angle_p05_deg": float(np.percentile(random_angles, 5)),
        "random_nullspace_angle_p95_deg": float(np.percentile(random_angles, 95)),
        "gradient_angle_percentile_in_random_nullspace": float(np.mean(random_angles <= gradient_angle)),
        "deposited_occupancies_A_B": occupancy.tolist(),
        "minor_state": "A" if int(np.argmin(occupancy)) == 0 else "B",
        "closure_null_dimension": int(closure_basis.shape[1]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--starts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    manifest = {site_key(row): row for row in json.loads(args.manifest.read_text())}
    tasks = []
    for key in ("6ZWK_B_PHE47", "8R7O_C_THR1681"):
        row = manifest[key]
        site = (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"]))
        tasks.append({
            "site_key": key, "site": list(site),
            "start_pdb": str(args.starts / "sites" / key / "neutral_start_aprime_single_slot.pdb"),
            "work_root": str(args.output / key), "device": args.device,
        })
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        results = list(pool.map(measure_site, tasks))
    report = {"status": "complete", "device": args.device, "sites": results}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
