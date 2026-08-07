"""Fixed-geometry occupancy selection for A-prime.

The optimizer uses a continuous QP while geometry is moving.  This module is
only for the final, fixed-geometry selection step.  Its default MIQP is the
decoupled A-prime variant of qFit's formulation:

    sum(z) <= cardinality_cap
    t_min * z_i <= w_i <= z_i
    sum(w) <= 1

qFit's native path supplies only the threshold argument, so its threshold
simultaneously imposes the minimum nonzero occupancy and an implicit
cardinality cap.  Passing both arguments to the qFit solver keeps those roles
separate here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


DEFAULT_CARDINALITY_CAP = 4
DEFAULT_MIN_OCCUPANCY = 0.02
LEGACY_CULL_THRESHOLD = 0.09
QFIT_BIC_MIN_OCCUPANCY = 0.002
QFIT_BIC_COMPLEXITY_FACTOR = 0.8
QFIT_THRESHOLDS = (1.0, 0.5, 0.33, 0.25, 0.2)


def _validate_inputs(
    target: np.ndarray, models: np.ndarray, cardinality_cap: int, t_min: float
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=float)
    models = np.asarray(models, dtype=float)
    if target.ndim != 1:
        raise ValueError(f"target must be one-dimensional, got {target.shape}")
    if models.ndim != 2 or models.shape[1] != target.size:
        raise ValueError(
            f"models must have shape (n_models, {target.size}), got {models.shape}"
        )
    if models.shape[0] == 0:
        raise ValueError("at least one candidate model is required")
    if not 1 <= cardinality_cap:
        raise ValueError("cardinality_cap must be positive")
    if not 0.0 < t_min <= 1.0:
        raise ValueError("t_min must be in (0, 1]")
    return target, models


def qfit_bic(
    rss: float,
    n_voxels: int,
    n_atoms: int,
    n_selected_conformers: int,
    *,
    model_params_per_atom: int = 3,
) -> tuple[float, float]:
    """Return qFit's native BIC and parameter count.

    The count uses the number of selected conformers, not the cardinality cap:
    ``k = (3 + sampled-B-factor-parameters) * natoms * nconfs * 0.8``.
    A-prime keeps B factors fixed, hence the default of three parameters per
    atom.  This is used only as a final fixed-geometry model-selection
    criterion.
    """
    if n_voxels <= 0 or n_atoms <= 0 or n_selected_conformers < 0:
        raise ValueError("BIC dimensions must be positive except selected count")
    k = float(model_params_per_atom * n_atoms * n_selected_conformers * QFIT_BIC_COMPLEXITY_FACTOR)
    bic = n_voxels * math.log(max(float(rss) / n_voxels, 1e-30)) + k * math.log(n_voxels)
    return float(bic), k


def legacy_cull(weights: np.ndarray, threshold: float = LEGACY_CULL_THRESHOLD) -> np.ndarray:
    """Apply the historical A-prime hard cull without changing survivors' weights."""
    weights = np.asarray(weights, dtype=float)
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    answer = weights.copy()
    answer[answer < threshold] = 0.0
    return answer


def _solve_qfit_miqp(
    target: np.ndarray, models: np.ndarray, *, cardinality: int | None, threshold: float
) -> tuple[np.ndarray, float]:
    # qFit is a pod runtime dependency.  Keep this import lazy so deterministic
    # tests of the selection bookkeeping do not need the crystallography stack.
    from qfit.solvers import get_miqp_solver_class

    solver = get_miqp_solver_class("CVXPYSolver")(target, models)
    solver.solve_miqp(cardinality=cardinality, threshold=threshold)
    weights = np.asarray(solver.weights, dtype=float)
    if weights.shape != (models.shape[0],) or not np.all(np.isfinite(weights)):
        raise RuntimeError(f"MIQP returned invalid weights with shape {weights.shape}")
    return weights, float(solver.objective_value)


def _candidate_record(
    weights: np.ndarray,
    rss: float,
    *,
    n_voxels: int,
    n_atoms: int,
    cap: int,
    threshold: float,
) -> dict[str, Any]:
    selected = weights >= QFIT_BIC_MIN_OCCUPANCY
    bic, k = qfit_bic(
        rss,
        n_voxels,
        n_atoms,
        int(np.sum(selected)),
    )
    return {
        "cardinality_cap": cap,
        "t_min": threshold,
        "weights": weights.tolist(),
        "rss": float(rss),
        "selected_mask": selected.tolist(),
        "selected_slots": np.flatnonzero(selected).tolist(),
        "n_selected_conformers": int(np.sum(selected)),
        "bic": bic,
        "bic_k": k,
    }


def select_decoupled_miqp(
    target: np.ndarray,
    models: np.ndarray,
    *,
    cardinality_cap: int = DEFAULT_CARDINALITY_CAP,
    t_min: float = DEFAULT_MIN_OCCUPANCY,
    n_atoms: int,
    solve_miqp: Any = _solve_qfit_miqp,
) -> dict[str, Any]:
    """Select a fixed-geometry A-prime ensemble with separated constraints.

    ``cardinality_cap`` is the configurable maximum and is passed directly to
    the MIQP.  It is not chosen by BIC: doing so would reintroduce an
    additional, coupled model-complexity decision.  qFit-style BIC is reported
    for the resulting model and counts selected weights rather than the cap.
    The cap and the actual selected conformers are both returned so they cannot
    be conflated in reports.
    """
    target, models = _validate_inputs(target, models, cardinality_cap, t_min)
    effective_cap = min(cardinality_cap, models.shape[0])
    weights, rss = solve_miqp(
        target, models, cardinality=effective_cap, threshold=t_min
    )
    selected = _candidate_record(
        weights,
        rss,
        n_voxels=target.size,
        n_atoms=n_atoms,
        cap=effective_cap,
        threshold=t_min,
    )
    return {
        "method": "A-prime decoupled MIQP (qFit solver; independent cap and floor)",
        "constraints": [
            "sum(z_i) <= K",
            "t_min * z_i <= w_i <= z_i",
            "sum(w_i) <= 1",
        ],
        "cardinality_cap": int(cardinality_cap),
        "t_min": float(t_min),
        "effective_cardinality_cap": int(effective_cap),
        "weights": selected["weights"],
        "rss": selected["rss"],
        "selected_slots": selected["selected_slots"],
        "n_selected_conformers": selected["n_selected_conformers"],
        "bic": selected["bic"],
        "bic_k": selected["bic_k"],
        "bic_candidates": [selected],
    }


def diagnose_cardinality_caps(
    target: np.ndarray,
    models: np.ndarray,
    *,
    cardinality_caps: tuple[int, ...] = (1, 2, 3, 4),
    t_min: float = DEFAULT_MIN_OCCUPANCY,
    n_atoms: int,
    solve_miqp: Any = _solve_qfit_miqp,
) -> list[dict[str, Any]]:
    """Return fixed-floor MIQP/BIC rows for a requested cap sweep.

    This is a diagnostic for the K rule, not the production selector.  It
    keeps ``t_min`` fixed while varying only the independent cardinality cap,
    and reports qFit-style BIC using the number of actually selected
    conformers.  Caps larger than the candidate set are retained in the
    report but use the corresponding effective cap passed to the solver.
    """
    target, models = _validate_inputs(target, models, 1, t_min)
    if not cardinality_caps:
        raise ValueError("at least one cardinality cap is required")
    if any(cap < 1 for cap in cardinality_caps):
        raise ValueError("cardinality caps must be positive")
    rows = []
    for cap in cardinality_caps:
        effective_cap = min(int(cap), models.shape[0])
        weights, rss = solve_miqp(
            target, models, cardinality=effective_cap, threshold=t_min
        )
        row = _candidate_record(
            weights,
            rss,
            n_voxels=target.size,
            n_atoms=n_atoms,
            cap=int(cap),
            threshold=t_min,
        )
        row["effective_cardinality_cap"] = effective_cap
        rows.append(row)
    return rows


def evaluate_qfit_coupled_thresholds(
    target: np.ndarray,
    models: np.ndarray,
    *,
    n_atoms: int,
    thresholds: tuple[float, ...] = QFIT_THRESHOLDS,
    solve_miqp: Any = _solve_qfit_miqp,
) -> list[dict[str, Any]]:
    """Evaluate qFit's native threshold-only MIQP for comparison."""
    target, models = _validate_inputs(target, models, 1, 0.02)
    records = []
    for threshold in thresholds:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"invalid qFit threshold {threshold}")
        weights, rss = solve_miqp(
            target, models, cardinality=None, threshold=float(threshold)
        )
        records.append(
            _candidate_record(
                weights,
                rss,
                n_voxels=target.size,
                n_atoms=n_atoms,
                cap=models.shape[0],
                threshold=float(threshold),
            )
        )
    return records
