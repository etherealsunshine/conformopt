#!/usr/bin/env python3
"""One-off timing profile for a single clean-D1 A-prime inner solve."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

# CCTBX's Boost-Python modules and qFit have an import-order interaction in
# the mixed CUDA/CCTBX environment.  Import qFit before the coordination
# module so the GPU timing path uses the same binary stack without a crash.
import qfit  # noqa: F401
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
    ap.add_argument("--inner-nfev", type=int, default=8)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
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
        args.output, args.inner_nfev, 1, *tuple(spec["site"]),
        renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window",
        training_indices=None, device=args.device, start_pdb=args.start,
        b_factor_mode="single_conformer",
    )
    runner.rama_floor = 0.02
    p1 = np.zeros(runner.rotator.ndofs)
    p2 = np.asarray(spec["p2"], dtype=float)
    parameterization = coord.FullJointParameterization(runner.rotator.ndofs)
    initial = runner.torch_forward(
        parameterization.expand_torch(torch.as_tensor(
            parameterization.pack(p1, p2), dtype=torch.float64
        ))
    ).detach().cpu().numpy()
    initial_rmsd = float(rmsd(
        runner.base.central_backbone(initial[0]),
        runner.base.central_backbone(initial[1]),
    ))

    totals = {name: 0.0 for name in (
        "renderer_batch", "renderer_torch", "occupancy", "seam", "omega",
        "rama", "joint_evaluate", "joint_jacobian", "residual_callback",
        "jacobian_callback", "least_squares", "torch_forward",
        "torch_seam", "torch_omega", "omega_and_rama", "rama_rows",
    )}
    counts = {name: 0 for name in totals}

    def timed_method(obj, name, bucket):
        original = getattr(obj, name)

        def wrapped(*pargs, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*pargs, **kwargs)
            finally:
                totals[bucket] += time.perf_counter() - t0
                counts[bucket] += 1
        setattr(obj, name, wrapped)

    timed_method(runner, "torch_forward", "torch_forward")
    timed_method(runner.base, "model_density_batch", "renderer_batch")
    timed_method(runner.base, "model_density_torch", "renderer_torch")
    timed_method(runner, "joint_qp_weights", "occupancy")
    timed_method(runner, "_torch_seam", "torch_seam")
    timed_method(runner, "_torch_omega", "torch_omega")
    timed_method(runner, "omega_and_rama", "omega_and_rama")

    original_seam = coord.seam_vector
    def timed_seam(*pargs, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_seam(*pargs, **kwargs)
        finally:
            totals["seam"] += time.perf_counter() - t0
            counts["seam"] += 1
    coord.seam_vector = timed_seam

    original_rama_rows = coord.joint_rama_rows
    def timed_rama_rows(*pargs, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_rama_rows(*pargs, **kwargs)
        finally:
            totals["rama_rows"] += time.perf_counter() - t0
            counts["rama_rows"] += 1
    coord.joint_rama_rows = timed_rama_rows

    original_eval = coord.joint_evaluate
    def timed_eval(*pargs, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_eval(*pargs, **kwargs)
        finally:
            totals["joint_evaluate"] += time.perf_counter() - t0
            counts["joint_evaluate"] += 1
    coord.joint_evaluate = timed_eval

    original_jac = coord.joint_jacobian
    def timed_jac(*pargs, **kwargs):
        t0 = time.perf_counter()
        try:
            return original_jac(*pargs, **kwargs)
        finally:
            totals["joint_jacobian"] += time.perf_counter() - t0
            counts["joint_jacobian"] += 1
    coord.joint_jacobian = timed_jac

    original_ls = coord.least_squares
    def timed_ls(fun, x0, *pargs, **kwargs):
        def timed_fun(*fargs, **fkwargs):
            t0 = time.perf_counter()
            try:
                return fun(*fargs, **fkwargs)
            finally:
                totals["residual_callback"] += time.perf_counter() - t0
                counts["residual_callback"] += 1

        jac = kwargs.get("jac")
        if jac is not None:
            def timed_jac_callback(*jargs, **jkwargs):
                t0 = time.perf_counter()
                try:
                    return jac(*jargs, **jkwargs)
                finally:
                    totals["jacobian_callback"] += time.perf_counter() - t0
                    counts["jacobian_callback"] += 1
            kwargs["jac"] = timed_jac_callback
        t0 = time.perf_counter()
        try:
            return original_ls(timed_fun, x0, *pargs, **kwargs)
        finally:
            totals["least_squares"] += time.perf_counter() - t0
            counts["least_squares"] += 1
    coord.least_squares = timed_ls

    t0 = time.perf_counter()
    result = coord.joint_run(
        runner, p1, p2, str(spec["label"]), args.output, initial_rmsd,
        per_slot_offsets=None, fixed_b_offset=None,
        occupancy_scheme="mirror", mirror_eta=0.001, mirror_tau=0.0,
        inner_nfev=args.inner_nfev, outer_updates=1,
    )
    total_wall = time.perf_counter() - t0

    # Components are nested inside callbacks.  The callback remainder is the
    # uninstrumented Python/NumPy and conversion work.  The least-squares
    # remainder is SciPy's trust-region/Gauss-Newton driver and linear algebra.
    measured = (totals["renderer_batch"] + totals["renderer_torch"] +
                totals["occupancy"] + totals["seam"] + totals["omega_and_rama"] +
                totals["rama_rows"])
    callback_remainder = max(
        0.0, totals["residual_callback"] + totals["jacobian_callback"] - measured
    )
    solver_remainder = max(
        0.0, totals["least_squares"] - totals["residual_callback"] -
        totals["jacobian_callback"]
    )
    report = {
        "site": args.site, "device": str(runner.base.torch_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "inner_nfev": args.inner_nfev, "outer_updates": 1,
        "total_wall_s": total_wall, "instrumented": totals,
        "counts": counts,
        "derived": {
            "callback_python_numpy_remainder_s": callback_remainder,
            "scipy_gauss_newton_driver_remainder_s": solver_remainder,
            "host_device_transfer_s": 0.0 if str(runner.base.torch_device) == "cpu" else None,
            "autograd_backward_s": 0.0,
            "autodiff_mode": "Torch forward-mode Jacobian; no autograd backward call",
        },
        "result": result,
    }
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
