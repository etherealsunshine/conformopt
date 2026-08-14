#!/usr/bin/env python3
"""Read-only full-structure ΔB control on MapScaler and residual masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from scipy.optimize import minimize_scalar

from qfit.xtal.scaler import MapScaler
from run_d1_8d_sequential_poc import atomic_json
from run_d1_aprime_sequential import APrimeSequential
from run_d1_supported_panel import site_name


SITES = (
    ("6ZWK", "B", 47),
    ("4HFS", "A", 200),
    ("7ZTL", "A", 257),
    ("8AJK", "A", 240),
    ("8R7O", "C", 1681),
)


def fit_affine(target: np.ndarray, model: np.ndarray) -> dict[str, float]:
    intercept = float(np.mean(target - model))
    residual = target - model - intercept
    return {"intercept": intercept, "rss": float(np.square(residual).sum()),
            "correlation": float(np.corrcoef(target, model)[0, 1])}


def site_control(site: tuple[str, str, int], scratch: Path) -> dict[str, object]:
    runner = APrimeSequential(scratch / site_name(site), 8, 6, *site,
                               renderer_backend="torch", residual_scale_mode="none",
                               map_scaler_structure="full", mask_scope="window", device="cuda")
    base = runner.base
    # MapScaler's own atom-support mask, constructed once from unmodified
    # deposited coordinates.  The same full calculated density is also read on
    # A''s local residual mask to distinguish a global scaling issue from one
    # specific to the seven-residue objective.
    scaler = MapScaler(base.qfit.xmap)
    support_transformer = scaler._get_model_transformer(base.full_structure, transformer="cctbx")  # pylint: disable=protected-access
    support_transformer.mask(0.5 + base.resolution / 3.0)
    support_mask = scaler._model_map.array > 0  # pylint: disable=protected-access
    support_transformer.reset(full=True)
    target_support = base.qfit.xmap.array[support_mask].astype(float, copy=True)
    target_residual = base.qfit.xmap.array[base.mask].astype(float, copy=True)
    min_b = float(base.full_structure.b.min())
    cache: dict[float, dict[str, object]] = {}

    def evaluate(d_b: float) -> dict[str, object]:
        key = round(float(d_b), 8)
        if key not in cache:
            structure = base.full_structure.copy()
            structure.b = np.asarray(structure.b, dtype=float) + key
            local = MapScaler(base.qfit.xmap)
            transformer = local._get_model_transformer(structure, transformer="cctbx")  # pylint: disable=protected-access
            transformer.density()
            rho = local._model_map.array.astype(float, copy=False)  # pylint: disable=protected-access
            cache[key] = {
                "dB_A2": key,
                "MapScaler_support": fit_affine(target_support, rho[support_mask]),
                "A_prime_residual_mask": fit_affine(target_residual, rho[base.mask]),
            }
            transformer.reset(full=True)
        return cache[key]

    lower = -min_b + 1e-6
    solution = minimize_scalar(lambda value: evaluate(value)["MapScaler_support"]["rss"],
                               bounds=(lower, 30.0), method="bounded", options={"xatol": 0.02})
    fitted = evaluate(float(solution.x))
    zero = evaluate(0.0)
    return {
        "site": site_name(site), "resolution_A": float(base.resolution),
        "MapScaler_affine": {"a": float(base.map_scale), "b": float(base.map_offset)},
        "support_voxels": int(support_mask.sum()), "residual_mask_voxels": int(base.mask.sum()),
        "fitted_global_full_structure_dB_A2": float(solution.x),
        "full_structure_occupancy": 1.0,
        "at_dB_0": zero, "at_fitted_dB": fitted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [site_control(site, args.out / "scratch") for site in SITES]
    atomic_json(args.out / "full_structure_dB_control.json", {"status": "complete", "rows": rows})
    print(json.dumps({"status": "complete", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
