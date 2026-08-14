#!/usr/bin/env python3
"""Neutral-start per-slot Gauss--Newton and trust-region diagnostic."""

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
from run_d1_aprime_sequential import rmsd


LAMBDA_AMP = 0.008836284282618265
X_SCALE = 10.0
TRUST_RADIUS_SCALED = 1.0  # scipy trf initial radius when x0=0


def unit(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-30)


def angle(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(left), unit(right)), -1.0, 1.0))))


def torsion_direction(runner: APrimeSequential, target_window: np.ndarray) -> np.ndarray:
    phi_psi, omega = dihedrals(runner.window)
    original = runner.window.coor.copy()
    try:
        runner.window.coor = target_window.copy()
        target_phi_psi, target_omega = dihedrals(runner.window)
    finally:
        runner.window.coor = original
    return np.concatenate((
        wrapped_delta(phi_psi, target_phi_psi),
        wrapped_delta(omega, target_omega),
    ))


def closure_basis_20(runner: APrimeSequential) -> np.ndarray:
    selection = np.sort(runner.window.select("name", ("N", "CA", "C")))
    basis = null_space(compute_jacobian(runner.window.get_xyz(selection)))
    output = np.zeros((runner.rotator.ndofs, basis.shape[1]))
    output[:14] = basis
    return output


def measure_site(task: dict[str, object]) -> dict[str, object]:
    site = tuple(task["site"])
    runner = APrimeSequential(
        Path(task["work_root"]), 80, 6, *site,
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device=task["device"],
        start_pdb=task["start_pdb"], b_factor_mode="single_conformer",
    )
    parameterization = FullJointParameterization(runner.rotator.ndofs)
    # The original diagnostic accidentally evaluated the all-zero joint vector.
    # That is the neutral geometry in both slots, not the recorded axis-2
    # initialization.  Load the frozen p2 vector from the neutral-start
    # preflight so this diagnostic probes the state actually passed to GN.
    preflight = json.loads(
        (Path(task["starts"]) / "sites" / str(task["site_key"]) / "preflight.json").read_text()
    )
    p2 = np.asarray(preflight["initialisation"]["p2_parameters_deg"], dtype=float)
    if p2.shape != (runner.rotator.ndofs,):
        raise ValueError(f"expected p2 shape {(runner.rotator.ndofs,)}, got {p2.shape}")
    slot1_parameters = np.zeros(runner.rotator.ndofs, dtype=float)
    parameters = parameterization.pack(slot1_parameters, p2)
    full_torsions = parameterization.expand_numpy(parameters)
    aa_models = runner.base.model_density_batch(
        np.stack((runner.initial, runner.initial)), slots=np.asarray((0, 1)), b_offset=0.0
    )
    _, _, normalizer = runner.joint_qp_weights(runner.target, aa_models)
    state = joint_evaluate(
        runner, parameters, normalizer, np.zeros(12), parameterization,
        fixed_b_offset=0.0, occupancy_weights=np.asarray((0.5, 0.5)),
        amplitude_prior_lambda=LAMBDA_AMP,
    )
    jacobian = joint_jacobian(
        runner, parameters, state, normalizer, np.zeros(12), parameterization,
        fixed_b_offset=0.0, amplitude_prior_lambda=LAMBDA_AMP,
    )
    residual = np.asarray(state["residual"], dtype=float)
    blocks = [jacobian[:, :runner.rotator.ndofs], jacobian[:, runner.rotator.ndofs:2 * runner.rotator.ndofs]]
    raw_steps = [np.linalg.lstsq(block, -residual, rcond=None)[0] for block in blocks]
    raw_scaled_norms = [float(np.linalg.norm(step / X_SCALE)) for step in raw_steps]
    truncation_factors = [min(1.0, TRUST_RADIUS_SCALED / max(value, 1e-30)) for value in raw_scaled_norms]
    applied_steps = [step * factor for step, factor in zip(raw_steps, truncation_factors)]

    d_a = torsion_direction(runner, runner.base.window_for_deposited_a())
    d_b = torsion_direction(runner, runner.base.window_for_deposited_b())
    d_ab = d_b - d_a
    separation_step = applied_steps[1] - applied_steps[0]
    basis = closure_basis_20(runner)
    rng = np.random.default_rng(20260814)
    random_coefficients = rng.normal(size=(200, basis.shape[1]))
    random_directions = random_coefficients @ basis.T
    random_directions /= np.maximum(np.linalg.norm(random_directions, axis=1, keepdims=True), 1e-30)
    random_slot1_to_A = np.degrees(np.arccos(np.clip(random_directions @ unit(d_a), -1.0, 1.0)))
    random_slot2_to_B = np.degrees(np.arccos(np.clip(random_directions @ unit(d_b), -1.0, 1.0)))
    random_pairs = rng.normal(size=(200, 2, basis.shape[1])) @ basis.T
    random_pairs /= np.maximum(np.linalg.norm(random_pairs, axis=2, keepdims=True), 1e-30)
    random_separations = random_pairs[:, 1] - random_pairs[:, 0]
    random_separations /= np.maximum(np.linalg.norm(random_separations, axis=1, keepdims=True), 1e-30)
    random_sep_angles = np.degrees(np.arccos(np.clip(random_separations @ unit(d_ab), -1.0, 1.0)))

    def distribution(values):
        return {
            "mean_deg": float(np.mean(values)),
            "sd_deg": float(np.std(values, ddof=1)),
            "p05_deg": float(np.percentile(values, 5)),
            "p95_deg": float(np.percentile(values, 95)),
        }

    condition_numbers = []
    for block in blocks:
        singular = np.linalg.svd(block, compute_uv=False)
        condition_numbers.append(float((singular[0] / max(singular[-1], 1e-30)) ** 2))
    central_backbone = runner.base.central_backbone
    slot_rmsd = float(rmsd(central_backbone(state["coordinates"][0]),
                           central_backbone(state["coordinates"][1])))
    jacobian_difference = blocks[1] - blocks[0]
    return {
        "site": str(task["site_key"]),
        "gradient_objective": "full A' objective at neutral, mirror weights 0.5/0.5, dB=0",
        "trust_region_scaled_radius": TRUST_RADIUS_SCALED,
        "x_scale": X_SCALE,
        "input_state": {
            "slot1_torsions_deg": full_torsions[0].tolist(),
            "slot2_torsions_deg": full_torsions[1].tolist(),
            "slot2_minus_slot1_torsions_deg": (full_torsions[1] - full_torsions[0]).tolist(),
            "slot1_to_slot2_backbone_rmsd_A": slot_rmsd,
            "rho_slot1_first_10": np.asarray(state["models"][0][:10], dtype=float).tolist(),
            "rho_slot2_first_10": np.asarray(state["models"][1][:10], dtype=float).tolist(),
            "rho_max_abs_difference": float(np.max(np.abs(state["models"][1] - state["models"][0]))),
            "rho_relative_l2_difference": float(
                np.linalg.norm(state["models"][1] - state["models"][0]) /
                max(np.linalg.norm(state["models"][0]), 1e-30)
            ),
            "jacobian_shape": list(jacobian.shape),
            "jacobian_block_max_abs_difference": float(np.max(np.abs(jacobian_difference))),
            "jacobian_block_frobenius_difference": float(np.linalg.norm(jacobian_difference)),
            "jacobian_block_relative_frobenius_difference": float(
                np.linalg.norm(jacobian_difference) / max(np.linalg.norm(blocks[0]), 1e-30)
            ),
        },
        "slot1": {
            "raw_step_norm": float(np.linalg.norm(raw_steps[0])),
            "applied_step_norm": float(np.linalg.norm(applied_steps[0])),
            "raw_scaled_norm": raw_scaled_norms[0],
            "trust_region_truncated": bool(truncation_factors[0] < 1.0),
            "truncation_factor": truncation_factors[0],
            "angle_to_start_to_A_deg": angle(applied_steps[0], d_a),
        },
        "slot2": {
            "raw_step_norm": float(np.linalg.norm(raw_steps[1])),
            "applied_step_norm": float(np.linalg.norm(applied_steps[1])),
            "raw_scaled_norm": raw_scaled_norms[1],
            "trust_region_truncated": bool(truncation_factors[1] < 1.0),
            "truncation_factor": truncation_factors[1],
            "angle_to_start_to_B_deg": angle(applied_steps[1], d_b),
        },
        "separation": {
            "step_difference_norm": float(np.linalg.norm(separation_step)),
            "step_difference_over_slot1": float(np.linalg.norm(separation_step) / max(np.linalg.norm(applied_steps[0]), 1e-30)),
            "step_difference_over_slot2": float(np.linalg.norm(separation_step) / max(np.linalg.norm(applied_steps[1]), 1e-30)),
            "angle_to_A_to_B_deg": angle(separation_step, d_ab),
        },
        "condition_number_JtJ_per_slot": condition_numbers,
        "random_nullspace_dimension": int(basis.shape[1]),
        "random_seed": 20260814,
        "random_slot1_to_A": distribution(random_slot1_to_A),
        "random_slot2_to_B": distribution(random_slot2_to_B),
        "random_separation_to_A_to_B": distribution(random_sep_angles),
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
            "starts": str(args.starts),
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
