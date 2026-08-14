#!/usr/bin/env python3
"""One-protein A' versus qFit timing benchmark with a shared Torch renderer.

This is a benchmark harness, not an optimizer change.  It runs both methods on
7UTC A:ARG52, counts density vectors and renderer batches, and evaluates the
final geometries with the same masked Torch RSS/QP metric.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import time
from pathlib import Path
from types import MethodType

import numpy as np

# qFit/CCTBX must be imported before Torch in the combined audit environment.
from qfit.qfit import QFitRotamericResidue
from qfit.solvers import get_qp_solver_class

import torch

from density_denoiser.differentiable_renderer import render_cctbx_density
from run_d1_8d_sequential_poc import SequentialBackbonePOC
from run_d1_aprime_sequential import APrimeSequential


SITE = ("7UTC", "A", 52)
INNER_NFEV = 80
OUTER_UPDATES = 6
QFIT_RENDER_BATCH = 4


def torch_render_batch(base: SequentialBackbonePOC, coordinates, b_factors):
    """Render residue candidates with the same fixed grid and mask as A'."""
    xyz = torch.as_tensor(np.asarray(coordinates), dtype=torch.float64)
    if xyz.ndim == 2:
        xyz = xyz.unsqueeze(0)
    b = torch.as_tensor(np.asarray(b_factors), dtype=torch.float64)
    if b.ndim == 1:
        b = b.unsqueeze(0).expand(xyz.shape[0], -1)
    cell = base._renderer_cell.to(device=xyz.device)  # pylint: disable=protected-access
    shifts = torch.tensor(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=torch.float64,
        device=xyz.device,
    )
    image_shifts = torch.matmul(shifts, cell.T)
    coefficients = base._renderer_coefficients.to(device=xyz.device)  # pylint: disable=protected-access
    coefficients = coefficients.repeat_interleave(27, dim=0)
    rendered = []
    for start in range(0, xyz.shape[0], QFIT_RENDER_BATCH):
        xyz_chunk = xyz[start:start + QFIT_RENDER_BATCH]
        b_chunk = b[start:start + QFIT_RENDER_BATCH]
        fractional = torch.linalg.solve(cell, xyz_chunk.transpose(-1, -2)).transpose(-1, -2)
        fractional = fractional - torch.floor(fractional)
        base_cart = torch.matmul(fractional, cell.T)
        atom_xyz = (base_cart.unsqueeze(-2) + image_shifts.view(1, 1, 27, 3)).reshape(
            xyz_chunk.shape[0], -1, 3
        )
        b_images = b_chunk.repeat_interleave(27, dim=1)
        coeff_chunk = coefficients.unsqueeze(0).expand(xyz_chunk.shape[0], -1, -1, -1)
        with torch.no_grad():
            density = render_cctbx_density(
                base._renderer_grid.to(device=xyz.device),  # pylint: disable=protected-access
                atom_xyz,
                b_images,
                coeff_chunk,
                u_base=base._renderer_u_base,  # pylint: disable=protected-access
                voxel_chunk=4096,
            )
        rendered.append(density.cpu().numpy())
    return np.maximum(np.concatenate(rendered, axis=0), base.qfit.options.bulk_solvent_level)


def qp_quality(base: SequentialBackbonePOC, coordinates, b_factors=None):
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 3 and coordinates.shape[1] == len(base.initial_window):
        coordinates = coordinates[:, base.central_indices]
    if b_factors is None:
        b_factors = np.tile(base.b_factors, (len(coordinates), 1))
    models = torch_render_batch(base, coordinates, b_factors)
    solver = get_qp_solver_class("CVXPYSolver")(base.target, models)
    solver.solve_qp()
    weights = np.asarray(solver.weights, dtype=float)
    residual = base.target - weights @ models
    return {
        "masked_rss": float(np.square(residual).sum()),
        "masked_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "occupancies": weights.tolist(),
        "n_final_models": int(len(models)),
        "metric": "pure masked density RSS; QP occupancy fit 0 <= w and sum(w) <= 1; no seam/Rama/omega terms",
    }


def benchmark_aprime(inner_nfev=INNER_NFEV, outer_updates=OUTER_UPDATES) -> dict[str, object]:
    output = Path(tempfile.mkdtemp(prefix="step3_aprime_"))
    runner = APrimeSequential(
        output,
        inner_nfev,
        outer_updates,
        *SITE,
        renderer_backend="torch",
    )
    print("A_PRIME CONSTRUCTED", flush=True)
    stats = {"density_evaluations": 0, "renderer_batches": 0, "jacobian_builds": 0}
    original_density_torch = runner.base.model_density_torch

    def counted_density_torch(coordinates):
        batch = int(coordinates.shape[0]) if coordinates.ndim == 3 else 1
        stats["density_evaluations"] += batch
        stats["renderer_batches"] += 1
        return original_density_torch(coordinates)

    runner.base.model_density_torch = counted_density_torch
    original_jacobian = runner.residual_jacobian

    def counted_jacobian(*args, **kwargs):
        stats["jacobian_builds"] += 1
        return original_jacobian(*args, **kwargs)

    runner.residual_jacobian = counted_jacobian
    started = time.perf_counter()
    result = runner.run()
    wall = time.perf_counter() - started
    # The run's trajectory records the LM residual calls independently of the
    # finite-difference Jacobian calls.
    lm_evaluations = sum("evaluation" in row for row in runner.trajectory)
    al_updates = sum(row.get("event") == "AL_update" for row in runner.trajectory)
    lm_rows = [row for row in runner.trajectory if row.get("event") == "AL_update"]
    final = np.load(output / "final_slots.npz")
    quality = qp_quality(
        runner.base,
        [final["slot1_window"], final["slot2_window"]],
    )
    return {
        "method": "A_prime",
        "site": "7UTC:A:ARG52",
        "wall_seconds": wall,
        "density_evaluations_optimization": stats["density_evaluations"],
        "renderer_batches_optimization": stats["renderer_batches"],
        "jacobian_builds": stats["jacobian_builds"],
        "lm_residual_evaluations": lm_evaluations,
        "al_outer_updates": al_updates,
        "lm_evaluations_per_outer": lm_evaluations / max(al_updates, 1),
        "lm_statuses": [int(row["lm_status"]) for row in lm_rows],
        "lm_nfev": [int(row["lm_nfev"]) for row in lm_rows],
        "jacobian_columns_per_build": 20,
        "density_batches_per_jacobian": "one Torch autodiff graph; no +/- finite-difference columns",
        "renderer_batching": "Torch batch API; active sequential slot is batch 1 and final two-slot QP render is batch 2",
        "quality": quality,
        "verdict": result["verdict"],
    }


def benchmark_qfit() -> dict[str, object]:
    output = Path(tempfile.mkdtemp(prefix="step3_qfit_"))
    base = SequentialBackbonePOC(
        *SITE,
        output,
        0.25,
        2.0,
        0.0,
        "none",
        "torch",
    )
    qfit = base.qfit
    stats = {
        "density_evaluations": 0,
        "renderer_batches": 0,
        "convert_calls": 0,
        "candidate_pools_scored": [],
        "candidate_rss": [],
        "generation_snapshots": {},
    }

    def torch_convert(self, save_debug_maps_prefix=None):
        del save_debug_maps_prefix
        coordinates = np.asarray(self._coor_set)
        b_factors = np.asarray(self._bs)
        if b_factors.ndim == 1:
            b_factors = np.tile(b_factors, (len(coordinates), 1))
        stats["convert_calls"] += 1
        stats["density_evaluations"] += len(coordinates)
        stats["renderer_batches"] += int(np.ceil(len(coordinates) / QFIT_RENDER_BATCH))
        stats["candidate_pools_scored"].append(len(coordinates))
        self._target = base.target.copy()
        self._models = torch_render_batch(base, coordinates, b_factors)
        target = base.target
        denominator = np.einsum("ij,ij->i", self._models, self._models)
        numerator = self._models @ target
        weights = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0)
        weights = np.clip(weights, 0.0, 1.0)
        stats["candidate_rss"].extend(np.square(target[None, :] - weights[:, None] * self._models).sum(axis=1).tolist())

    qfit._convert = MethodType(torch_convert, qfit)  # pylint: disable=protected-access

    def run_stage(name, function):
        before = len(qfit._coor_set)  # pylint: disable=protected-access
        function()
        after = len(qfit._coor_set)  # pylint: disable=protected-access
        stats["generation_snapshots"][name] = {
            "before": before,
            "after": after,
            "net_retained_additions": max(0, after - before),
        }

    started = time.perf_counter()
    if qfit.options.sample_backbone:
        run_stage("backbone", qfit._sample_backbone)
    if qfit.options.sample_angle:
        run_stage("angle", qfit._sample_angle)
    if qfit.residue.nchi >= 1 and qfit.options.sample_rotamers:
        run_stage("sidechain", qfit._sample_sidechain)

    qfit.residue.active = True
    qfit.residue.update_clash_mask()
    new_coor_set, new_bs = [], []
    for coor, b in zip(qfit._coor_set, qfit._bs):  # pylint: disable=protected-access
        qfit.residue.coor = coor
        qfit.residue.b = b
        if not qfit.is_clashing():
            new_coor_set.append(coor)
            new_bs.append(b)
    qfit._coor_set, qfit._bs = new_coor_set, new_bs  # pylint: disable=protected-access

    qfit._convert()  # pylint: disable=protected-access
    qfit._solve_qp()  # pylint: disable=protected-access
    qfit._update_conformers()  # pylint: disable=protected-access
    qfit.sample_b()
    qfit._convert()  # pylint: disable=protected-access
    qfit._solve_miqp(  # pylint: disable=protected-access
        threshold=qfit.options.threshold,
        cardinality=qfit.options.cardinality,
    )
    qfit._update_conformers()  # pylint: disable=protected-access
    wall = time.perf_counter() - started

    coordinates = np.asarray(qfit._coor_set)  # pylint: disable=protected-access
    b_factors = np.asarray(qfit._bs)  # pylint: disable=protected-access
    quality = qp_quality(base, coordinates, b_factors)
    return {
        "method": "qFit",
        "site": "7UTC:A:ARG52",
        "wall_seconds": wall,
        "density_evaluations": stats["density_evaluations"],
        "renderer_batches": stats["renderer_batches"],
        "convert_calls": stats["convert_calls"],
        "candidates_generated_retained": stats["generation_snapshots"],
        "candidate_pools_scored": stats["candidate_pools_scored"],
        "candidates_scored_total": int(sum(stats["candidate_pools_scored"])),
        "candidate_rss_summary": {
            "definition": "one-candidate bounded NNLS RSS on the same unscaled target and 1,539-voxel mask",
            "n": len(stats["candidate_rss"]),
            "min": float(np.min(stats["candidate_rss"])),
            "median": float(np.median(stats["candidate_rss"])),
            "mean": float(np.mean(stats["candidate_rss"])),
            "max": float(np.max(stats["candidate_rss"])),
        },
        "renderer_batching": "batched over each candidate pool; not one render per candidate",
        "renderer_batch_size": QFIT_RENDER_BATCH,
        "quality": quality,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-nfev", type=int, default=INNER_NFEV)
    parser.add_argument("--outer-updates", type=int, default=OUTER_UPDATES)
    parser.add_argument("--qfit-only", action="store_true")
    parser.add_argument("--aprime-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    # The combined CPU environment otherwise fans out each repeated density
    # render over all host threads, multiplying temporary tensor memory. Both
    # methods use the same deterministic thread setting below.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    a_prime = None
    if not args.qfit_only:
        print("START A_PRIME", flush=True)
        a_prime = benchmark_aprime(args.inner_nfev, args.outer_updates)
        print("DONE A_PRIME", json.dumps(a_prime, sort_keys=True), flush=True)
    qfit = None
    if not args.aprime_only:
        print("START qFit", flush=True)
        qfit = benchmark_qfit()
        print("DONE qFit", json.dumps(qfit, sort_keys=True), flush=True)
    result = {
        "hardware_note": "same running qfit-unet pod; CPU Torch combined qFit/CCTBX environment",
        "renderer": "Torch multi-Gaussian renderer, fixed 1,539-voxel mask, same map/grid/B-factor convention",
        "a_prime": a_prime,
        "qfit": qfit,
        "fairness": "preliminary: same site, hardware, renderer, mask, and masked-RSS metric; A' and qFit retain different search algorithms and stopping/selection logic",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
