#!/usr/bin/env python3
"""Local-Gaussian positional uncertainty for the completed 7UTC A' fit.

The density Hessian is evaluated at the converged two-slot geometry.  It uses
the same unweighted real-space RSS and deposited-A B-factor rendering that the
original A' optimizer used.  Both (interior) QP occupancies are explicit
nuisance parameters, so their correlation with torsions is retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from run_d1_8d_sequential_poc import atomic_csv, atomic_json, rmsd
from run_d1_aprime_representability_gate import Gate
from run_d1_aprime_sequential import APrimeSequential, seam_vector


FINAL = Path("/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_sequential_v3")
SLOT1_SCAN = Path("/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_slot1_to_A_basin_v1")
SLOT2_SCAN = Path("/home/dev/qfit_unet_data/qfit_audit/d1_aprime_7utc_torsion_basin_v1")
FD_STEP_DEG = 0.25


def recover_parameters(runner: APrimeSequential, window_coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    target = window_coordinates[runner.bb_indices]
    result = least_squares(
        lambda q: (runner.forward(q)[runner.bb_indices] - target).ravel(),
        np.zeros(runner.rotator.ndofs), method="lm", max_nfev=5000,
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    fitted = runner.forward(result.x)[runner.bb_indices]
    return result.x, rmsd(fitted, target)


def density_jacobian(runner: APrimeSequential, parameters: np.ndarray, occupancy: float) -> np.ndarray:
    """d(occupancy * rho_calc)/d torsion, in RSS units per degree."""
    columns = []
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += FD_STEP_DEG
        minus[index] -= FD_STEP_DEG
        derivative = (runner.base.model_density(runner.forward(plus)) -
                      runner.base.model_density(runner.forward(minus))) / (2.0 * FD_STEP_DEG)
        columns.append(occupancy * derivative)
    return np.column_stack(columns)


def central_coordinate_jacobian(runner: APrimeSequential, parameters: np.ndarray) -> np.ndarray:
    """d central N/CA/C/O Cartesian coordinates / d torsion (degree)."""
    columns = []
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += FD_STEP_DEG
        minus[index] -= FD_STEP_DEG
        derivative = (runner.base.central_backbone(runner.forward(plus)) -
                      runner.base.central_backbone(runner.forward(minus))) / (2.0 * FD_STEP_DEG)
        columns.append(derivative.reshape(-1))
    return np.column_stack(columns)


def penalty_residual(runner: APrimeSequential, parameters: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    coordinates = runner.forward(parameters)
    g, _, _ = seam_vector(runner.initial_backbone, coordinates[runner.bb_indices])
    _, omega_delta, _, rama_barrier = runner.omega_and_rama(coordinates)
    return np.concatenate((
        np.sqrt(runner.rho / 2.0) * (g + lambdas / runner.rho),
        np.sqrt(runner.rama_weight) * rama_barrier,
        np.sqrt(runner.planar_weight) * omega_delta / runner.omega_scale_deg,
    ))


def penalty_hessian(runner: APrimeSequential, parameters: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    columns = []
    for index in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += FD_STEP_DEG
        minus[index] -= FD_STEP_DEG
        columns.append((penalty_residual(runner, plus, lambdas) -
                        penalty_residual(runner, minus, lambdas)) / (2.0 * FD_STEP_DEG))
    jacobian = np.column_stack(columns)
    return jacobian.T @ jacobian


def inverse_with_report(hessian: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    symmetric = (hessian + hessian.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    maximum = float(eigenvalues[-1])
    minimum = float(eigenvalues[0])
    floor = maximum * 1e-10
    ridge = max(0.0, floor - minimum)
    regularized = symmetric + ridge * np.eye(len(symmetric))
    values = np.linalg.eigvalsh(regularized)
    covariance = np.linalg.inv(regularized)
    return covariance, {
        "eigenvalue_min": minimum,
        "eigenvalue_max": maximum,
        "condition_number_unregularized": None if minimum <= 0 else float(maximum / minimum),
        "unregularized_hessian_singular_or_indefinite": bool(minimum <= 0),
        "ridge_added": float(ridge),
        "ridge_relative_to_lambda_max": float(ridge / maximum if maximum else 0.0),
        "condition_number_regularized": float(values[-1] / values[0]),
    }


def atom_report(coordinate_covariance: np.ndarray, a_to_b: np.ndarray) -> dict[str, object]:
    names = ("N", "CA", "C", "O")
    atoms = []
    for index, name in enumerate(names):
        block = coordinate_covariance[3 * index:3 * index + 3, 3 * index:3 * index + 3]
        values, vectors = np.linalg.eigh((block + block.T) / 2.0)
        order = np.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        displacement = a_to_b[index]
        alignment = (float(abs(np.dot(vectors[:, 0], displacement / np.linalg.norm(displacement))) )
                     if np.linalg.norm(displacement) > 1e-12 else None)
        atoms.append({
            "atom": name,
            "sigma_xyz_A": np.sqrt(np.maximum(np.diag(block), 0.0)).tolist(),
            "positional_sigma_A": float(np.sqrt(max(np.trace(block), 0.0))),
            "principal_sigma_A": np.sqrt(np.maximum(values, 0.0)).tolist(),
            "long_axis_alignment_with_A_to_B": alignment,
        })
    values, vectors = np.linalg.eigh((coordinate_covariance + coordinate_covariance.T) / 2.0)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    direction = a_to_b.reshape(-1)
    alignment = float(abs(np.dot(vectors[:, 0], direction / np.linalg.norm(direction))))
    return {
        "per_atom": atoms,
        "central_backbone_rms_sigma_A": float(np.sqrt(np.trace(coordinate_covariance) / 4.0)),
        "central_collective_principal_sigma_A": np.sqrt(np.maximum(values[:6], 0.0)).tolist(),
        "central_collective_long_axis_alignment_with_A_to_B": alignment,
        "central_collective_anisotropy_lambda1_over_lambda2": float(values[0] / values[1]),
        "central_collective_anisotropy_lambda1_over_median": float(values[0] / np.median(values)),
    }


def blocked_test_indices(runner: APrimeSequential) -> list[np.ndarray]:
    """Recreate the five blocked slabs used by the completed basin scans."""
    n = int(runner.base.mask.sum())
    test_count = round(.2 * n)
    coordinates = np.argwhere(runner.base.mask) * np.asarray(runner.base.qfit.xmap.voxelspacing, float)
    rng = np.random.default_rng(20260805)
    # The original script made these draws for its discarded random-split arm
    # before generating each slab direction.
    for _ in range(5):
        rng.choice(n, size=test_count, replace=False)
    tests = []
    for _ in range(5):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        tests.append(np.sort(np.argsort(coordinates @ direction)[:test_count]))
    return tests


def profiled_hessian(hessian: np.ndarray, torsion_indices: np.ndarray,
                     occupancy_indices: np.ndarray) -> np.ndarray:
    hqq = hessian[np.ix_(torsion_indices, torsion_indices)]
    hqo = hessian[np.ix_(torsion_indices, occupancy_indices)]
    hoo = hessian[np.ix_(occupancy_indices, occupancy_indices)]
    return hqq - hqo @ np.linalg.inv(hoo) @ hqo.T


def endpoint_range(runner: APrimeSequential, parameters: np.ndarray, endpoint: np.ndarray,
                   fractions: list[float]) -> float:
    low, high = min(fractions), max(fractions)
    first = runner.base.central_backbone(runner.forward((1.0 - low) * parameters + low * endpoint))
    second = runner.base.central_backbone(runner.forward((1.0 - high) * parameters + high * endpoint))
    return rmsd(first, second)


def scan_band(runner: APrimeSequential, parameters: np.ndarray, endpoint: np.ndarray,
              joint_jacobian: np.ndarray, torsion_indices: np.ndarray, scan_path: Path) -> dict[str, object]:
    scan = json.loads((scan_path / "result.json").read_text())
    rows = scan["per_fraction"]
    direction = endpoint - parameters
    occupancy_indices = np.array([40, 41])
    per_split_hessians = [profiled_hessian(joint_jacobian[test].T @ joint_jacobian[test], torsion_indices, occupancy_indices)
                          for test in blocked_test_indices(runner)]
    curvatures = np.asarray([direction @ hessian @ direction for hessian in per_split_hessians], float)
    reference = runner.base.central_backbone(runner.forward(parameters))
    values = []
    for row in rows:
        fraction = float(row["fraction"])
        predicted_per_split = fraction ** 2 * curvatures
        predicted_rss_rise = float(np.mean(predicted_per_split))
        threshold = float(row["paired_difference_sd"])
        coordinates = runner.base.central_backbone(runner.forward((1.0 - fraction) * parameters + fraction * endpoint))
        movement = rmsd(coordinates, reference)
        values.append({
            "fraction": fraction,
            "predicted_RSS_rise_per_split": predicted_per_split.tolist(),
            "predicted_RSS_rise_mean": predicted_rss_rise,
            "paired_SD_RSS_threshold": threshold,
            "within_analytic_one_SD_band": bool(predicted_rss_rise <= threshold),
            "central_backbone_movement_from_converged_A": movement,
        })
    accepted = [row for row in values if row["within_analytic_one_SD_band"]]
    return {
        "directional_effective_RSS_curvature_per_heldout_split": curvatures.tolist(),
        "directional_effective_RSS_curvature_mean": float(curvatures.mean()),
        "analytic_one_SD_fractions": [row["fraction"] for row in accepted],
        "analytic_one_SD_range_central_backbone_A": endpoint_range(
            runner, parameters, endpoint, [row["fraction"] for row in accepted]),
        "brute_force_one_SD_fractions": scan["within_one_paired_sd_fractions"],
        "brute_force_one_SD_range_central_backbone_A": endpoint_range(
            runner, parameters, endpoint, scan["within_one_paired_sd_fractions"]),
        "per_fraction": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "progress.json", {"status": "running", "stage": "load"})

    runner = APrimeSequential(args.output, 80, 6)
    final = np.load(FINAL / "final_slots.npz")
    result = json.loads((FINAL / "result.json").read_text())
    slot1, slot2 = final["slot1_window"], final["slot2_window"]
    q1, q1_error = recover_parameters(runner, slot1)
    q2, q2_error = recover_parameters(runner, slot2)
    weights, rss = runner.base.joint_qp(slot1, slot2)
    m1, m2 = runner.base.model_density(slot1), runner.base.model_density(slot2)
    atomic_json(args.output / "progress.json", {"status": "running", "stage": "density_jacobians"})
    jd1, jd2 = density_jacobian(runner, q1, float(weights[0])), density_jacobian(runner, q2, float(weights[1]))
    # Parameter ordering is [slot1 torsions, slot2 torsions, occupancy1, occupancy2].
    jacobian = np.column_stack((jd1, jd2, m1, m2))
    h_density = jacobian.T @ jacobian  # Actual QP objective: unweighted RSS, W=I.
    covariance, conditioning = inverse_with_report(h_density)
    atomic_json(args.output / "progress.json", {"status": "running", "stage": "coordinate_jacobians"})
    jc1, jc2 = central_coordinate_jacobian(runner, q1), central_coordinate_jacobian(runner, q2)
    bfit = np.asarray(Gate("7UTC", "A", 52).solve(np.zeros(20))["parameters_deg"], float)
    a_central, b_central = runner.base.a_backbone, runner.base.b_backbone

    rows, slots = [], {}
    occupancy_indices = np.array([40, 41])
    for number, parameters, endpoint, jc, scan in ((1, q1, np.zeros(20), jc1, SLOT1_SCAN),
                                                    (2, q2, bfit, jc2, SLOT2_SCAN)):
        torsion_indices = np.arange((number - 1) * 20, number * 20)
        # Occupancies are profiled (Schur complement), matching the QP refit in the scans.
        effective = profiled_hessian(h_density, torsion_indices, occupancy_indices)
        # Marginal covariance allows both other-slot geometry and both QP occupancies to vary.
        parameter_covariance = covariance[np.ix_(torsion_indices, torsion_indices)]
        coordinate_covariance = jc @ parameter_covariance @ jc.T
        report = atom_report(coordinate_covariance, b_central - a_central)
        band = scan_band(runner, parameters, endpoint, jacobian, torsion_indices, scan)
        slots[f"slot{number}"] = {
            "occupancy_joint_QP": float(weights[number - 1]),
            "torsion_chart_reconstruction_RMSD_A": q1_error if number == 1 else q2_error,
            "conditioned_density_H_effective_trace": float(np.trace(effective)),
            "coordinate_covariance": report,
            "one_paired_SD_band": band,
        }
        for atom in report["per_atom"]:
            rows.append({"slot": number, **atom})

    atomic_json(args.output / "progress.json", {"status": "running", "stage": "penalty_curvature"})
    penalty = {}
    for number, parameters in ((1, q1), (2, q2)):
        stage = result["slots"][f"slot{number}"]
        normalizer = float(stage["convergence"]["normalizer_initial_rss"])
        occupancy = float(stage["occupancy_single_slot"])
        density = density_jacobian(runner, parameters, occupancy) / np.sqrt(normalizer)
        h_stage_density = density.T @ density
        h_stage_penalty = penalty_hessian(runner, parameters, np.asarray(stage["convergence"]["final_lambdas"], float))
        penalty[f"slot{number}"] = {
            "stage_density_normalizer_RSS": normalizer,
            "density_H_trace": float(np.trace(h_stage_density)),
            "penalty_H_trace": float(np.trace(h_stage_penalty)),
            "density_H_frobenius": float(np.linalg.norm(h_stage_density)),
            "penalty_H_frobenius": float(np.linalg.norm(h_stage_penalty)),
            "penalty_to_density_trace_ratio": float(np.trace(h_stage_penalty) / np.trace(h_stage_density)),
            "penalty_to_density_frobenius_ratio": float(np.linalg.norm(h_stage_penalty) / np.linalg.norm(h_stage_density)),
        }

    occupancy_covariance = covariance[np.ix_(occupancy_indices, occupancy_indices)]
    result_out = {
        "status": "complete",
        "site": "7UTC_A_ARG52",
        "operation": "local Gaussian density-Hessian uncertainty; no coordinate optimisation",
        "density_weighting": {
            "W": "identity: qFit CVXPY QP and the A' density residual use unweighted voxel RSS after the run's MapScaler/residual-map calibration",
            "mask_voxels": int(runner.base.mask.sum()),
            "bfactor_rendering": "original completed A' run: deposited-A B-factor array for both slots",
            "joint_parameter_vector": "20 slot-1 torsions, 20 slot-2 torsions, two interior QP occupancies",
            "joint_QP_occupancies": weights.tolist(),
            "joint_QP_RSS": float(rss),
        },
        "density_hessian_conditioning": conditioning,
        "occupancy_covariance": occupancy_covariance.tolist(),
        "slots": slots,
        "actual_A_prime_stage_penalty_curvature": penalty,
        "interpretation": "Covariance is inv(J^T J + reported ridge) in the unweighted-RSS convention. One-SD directional bands profile both occupancies and compare the quadratic RSS rise against the paired blocked-CV SD at each sampled fraction.",
    }
    atomic_csv(args.output / "per_atom.csv", rows)
    atomic_json(args.output / "result.json", result_out)
    atomic_json(args.output / "progress.json", {"status": "complete", "stage": "complete"})
    print(json.dumps(result_out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
