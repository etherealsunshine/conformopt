#!/usr/bin/env python3
"""Validate Torch Rama values and derivatives against the existing CCTBX path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import qfit  # noqa: F401  # keep the mixed qFit/CCTBX import order stable
import numpy as np
import torch

import run_d1_slot_coordination as coord
from clean_d1_benchmark import site_key
from run_d1_aprime_sequential import APrimeSequential, rmsd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--flip-root", type=Path, required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--start", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    manifest = json.loads(args.manifest.read_text())
    site = next(item for item in manifest if site_key(item) == args.site)
    specs = coord.build_specs(
        args.output / "specs", args.flip_root,
        site=(site["pdb_id"], site["chain"], int(site["resnum"])),
        mask_scope="window", start_pdb=args.start,
        b_factor_mode="single_conformer", device=args.device,
        occupancy_scheme="mirror", mirror_eta=0.001,
    )
    spec = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
    runner = APrimeSequential(
        args.output, 8, 1, *tuple(spec["site"]), renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", training_indices=None, device=args.device,
        start_pdb=args.start, b_factor_mode="single_conformer",
    )
    runner.rama_floor = 0.02
    p1 = np.zeros(runner.rotator.ndofs)
    p2 = np.asarray(spec["p2"], dtype=float)
    parameterization = coord.FullJointParameterization(runner.rotator.ndofs)
    packed = parameterization.pack(p1, p2)
    coordinates = runner.torch_forward(
        parameterization.expand_torch(torch.as_tensor(packed, dtype=torch.float64,
                                                       device=runner.base.torch_device))
    )

    cctbx_rows = []
    for coordinate in coordinates.detach().cpu().numpy():
        omega, omega_delta, scores, barriers = runner.omega_and_rama(coordinate)
        cctbx_rows.append({
            "omega": omega.tolist(), "omega_delta": omega_delta.tolist(),
            "scores": scores, "barriers": barriers.tolist(),
        })
    _, torch_omega_delta, torch_scores, torch_barriers = runner.torch_omega_and_rama(coordinates)
    torch_rows = []
    for index in range(coordinates.shape[0]):
        torch_rows.append({
            "omega_delta": torch_omega_delta[index].detach().cpu().numpy().tolist(),
            "scores": torch_scores[index],
            "barriers": torch_barriers[index].detach().cpu().numpy().tolist(),
        })

    score_diffs = []
    barrier_diffs = []
    omega_diffs = []
    for cctbx, torch_row in zip(cctbx_rows, torch_rows):
        omega_diffs.extend(np.asarray(cctbx["omega_delta"]) - np.asarray(torch_row["omega_delta"]))
        for cctbx_score, torch_score in zip(cctbx["scores"], torch_row["scores"]):
            if cctbx_score is not None and torch_score is not None:
                score_diffs.append(float(cctbx_score) - float(torch_score))
        barrier_diffs.extend(np.asarray(cctbx["barriers"]) - np.asarray(torch_row["barriers"]))

    def stats(values):
        values = np.asarray(values, dtype=float)
        return {
            "count": int(values.size),
            "max_abs": float(np.max(np.abs(values))) if values.size else 0.0,
            "rms": float(np.sqrt(np.mean(values * values))) if values.size else 0.0,
        }

    def analytic_rama(value):
        slots = parameterization.expand_torch(value)
        return np.sqrt(runner.rama_weight) * runner.torch_rama_barrier(
            runner.torch_forward(slots)
        )[0].reshape(-1)

    value = torch.as_tensor(packed, dtype=torch.float64, device=runner.base.torch_device)
    analytic = torch.autograd.functional.jacobian(
        analytic_rama, value, vectorize=True, strategy="forward-mode"
    ).detach().cpu().numpy()
    fd_step = 0.25
    fd = np.zeros_like(analytic)
    for column in range(value.numel()):
        direction = np.zeros_like(packed)
        direction[column] = fd_step
        fd[:, column] = (
            coord.joint_rama_rows(runner, packed + direction, parameterization, fixed_b_offset=0.0)
            - coord.joint_rama_rows(runner, packed - direction, parameterization, fixed_b_offset=0.0)
        ) / (2.0 * fd_step)
    gradient_delta = analytic - fd
    scale = np.maximum(np.abs(fd), 1e-10)

    report = {
        "site": args.site,
        "device": str(runner.base.torch_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_table_source": "installed mmtbx rama8000_tables.h",
        "neutral_start_slot_rmsd_A": float(rmsd(
            runner.base.central_backbone(coordinates[0].detach().cpu().numpy()),
            runner.base.central_backbone(coordinates[1].detach().cpu().numpy()),
        )),
        "cctbx_rows": cctbx_rows,
        "torch_rows": torch_rows,
        "value_agreement": {
            "score": stats(score_diffs),
            "barrier": stats(barrier_diffs),
            "omega_delta_deg": stats(omega_diffs),
        },
        "gradient_comparison": {
            "finite_difference_step_deg": fd_step,
            "analytic_shape": list(analytic.shape),
            "analytic_vs_fd": stats(gradient_delta.ravel()),
            "relative_max_abs": float(np.max(np.abs(gradient_delta) / scale)),
            "relative_rms": float(np.sqrt(np.mean((gradient_delta / scale) ** 2))),
            "analytic_max_abs": float(np.max(np.abs(analytic))),
            "fd_max_abs": float(np.max(np.abs(fd))),
        },
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
