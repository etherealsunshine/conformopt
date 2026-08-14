#!/usr/bin/env python3
"""Compare Torch and SciPy on one real A-prime inner objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import (
    FullJointParameterization, _least_squares_with_trust_trace, build_specs,
    joint_evaluate, joint_jacobian, joint_residual_torch,
)
from torch_trf import least_squares as torch_least_squares


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", nargs=3, required=True)
    ap.add_argument("--start", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    pdb_id, chain, resnum = args.site[0], args.site[1], int(args.site[2])
    site = (pdb_id, chain, resnum)
    args.output.mkdir(parents=True, exist_ok=False)
    pre_runner = APrimeSequential(
        args.output / "base_preflight", 40, 1, *site, renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", device=args.device, start_pdb=args.start,
        b_factor_mode="single_conformer",
    )
    train, _, _ = blocked_splits(pre_runner.base)[0]
    runner = APrimeSequential(
        args.output / "base", 40, 1, *site, renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", device=args.device, start_pdb=args.start,
        b_factor_mode="single_conformer", training_indices=train,
    )
    built = build_specs(
        args.output / "seed", args.flip_root, site=site, mask_scope="window",
        rama_floor=0.02, start_pdb=args.start,
        b_factor_mode="single_conformer", device=args.device,
        occupancy_scheme="mirror", mirror_eta=0.001,
        inner_nfev=40, outer_updates=1,
    )
    spec = next(item for item in built if item["label"] == "D_null_axis2_30deg")
    p1 = np.zeros(20, dtype=float)
    p2 = np.asarray(spec["p2"], dtype=float)
    parameterization = FullJointParameterization(20)
    parameters = parameterization.pack(p1, p2)
    aa_models = runner.base.model_density_batch(
        np.stack((runner.initial, runner.initial)), slots=np.array((0, 1)),
    )[:, train]
    _, _, normalizer = runner.joint_qp_weights(runner.target, aa_models)
    normalizer = max(float(normalizer), 1e-12)
    lambdas = np.zeros(12, dtype=float)
    weights = np.asarray((0.5, 0.5), dtype=float)
    state = joint_evaluate(
        runner, parameters, normalizer, lambdas, parameterization,
        fixed_b_offset=0.0, occupancy_weights=weights,
        amplitude_prior_lambda=0.008836284282618265,
    )
    fixed_state = {"weights": weights, "intercept": state["intercept"]}
    prior = np.zeros((2, 20), dtype=float)
    def residual_numpy(value: np.ndarray) -> np.ndarray:
        return joint_evaluate(
            runner, value, normalizer, lambdas, parameterization,
            fixed_b_offset=0.0, occupancy_weights=weights,
            amplitude_prior_lambda=0.008836284282618265,
            amplitude_prior_reference=prior,
            fixed_intercept=state["intercept"],
        )["residual"]

    def jacobian_numpy(value: np.ndarray) -> np.ndarray:
        return joint_jacobian(
            runner, value, fixed_state, normalizer, lambdas, parameterization,
            fixed_b_offset=0.0, amplitude_prior_lambda=0.008836284282618265,
            amplitude_prior_reference=prior,
        )

    x0_torch = torch.as_tensor(parameters, dtype=torch.float64,
                               device=runner.base.torch_device)
    def residual_torch(value: torch.Tensor) -> torch.Tensor:
        return joint_residual_torch(
            runner, value, fixed_state, normalizer, lambdas, parameterization,
            fixed_b_offset=0.0, amplitude_prior_lambda=0.008836284282618265,
            amplitude_prior_reference=prior,
        )

    scipy_trace = []
    scipy_result = _least_squares_with_trust_trace(
        residual_numpy, parameters, jac=jacobian_numpy, method="trf",
        x_scale=10.0, max_nfev=4, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        trace=scipy_trace,
    )
    torch_result = torch_least_squares(
        residual_torch, x0_torch, max_nfev=4, initial_radius=3.0,
        x_scale=10.0,
        ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    torch_x = torch_result.x.detach().cpu().numpy()
    scipy_residual = residual_numpy(scipy_result.x)
    torch_residual = residual_numpy(torch_x)
    payload = {
        "status": "complete", "site": list(site), "fold": 0,
        "device": str(runner.base.torch_device),
        "same_start": True, "same_fixed_weights": weights.tolist(),
        "same_fixed_intercept": float(state["intercept"]),
        "initial_residual_max_abs_difference": float(np.max(np.abs(
            residual_numpy(parameters) - residual_torch(x0_torch).detach().cpu().numpy()
        ))),
        "scipy": {"x": scipy_result.x.tolist(), "cost": float(0.5 * np.dot(scipy_residual, scipy_residual)),
                  "nfev": int(scipy_result.nfev), "optimality": float(scipy_result.optimality),
                  "status": int(scipy_result.status), "message": scipy_result.message,
                  "trust_radius_trace": scipy_trace},
        "torch": {"x": torch_x.tolist(), "cost": float(0.5 * np.dot(torch_residual, torch_residual)),
                  "nfev": int(torch_result.nfev), "optimality": float(torch_result.optimality),
                  "projected_optimality": float(torch_result.projected_optimality),
                  "status": int(torch_result.status), "message": torch_result.message},
        "max_abs_parameter_difference": float(np.max(np.abs(scipy_result.x - torch_x))),
        "max_abs_residual_difference": float(np.max(np.abs(scipy_residual - torch_residual))),
        "cost_difference": float(abs(0.5 * np.dot(scipy_residual, scipy_residual) -
                                      0.5 * np.dot(torch_residual, torch_residual))),
        "torch_trust_radius_trace": torch_result.trust_radius_trace,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: payload[key] for key in (
        "status", "device", "initial_residual_max_abs_difference",
        "max_abs_parameter_difference", "max_abs_residual_difference", "cost_difference",
    )}))


if __name__ == "__main__":
    main()
