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
from itertools import combinations
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


def solve_affine_qp(
    target: np.ndarray,
    models: np.ndarray,
    *,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    max_total: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """Fit occupancies and a free map intercept at fixed geometry.

    This is the adopted A-prime continuous occupancy solve.  The intercept is
    fitted from the data and is deliberately not included in the geometry
    gradient; callers profile it alongside the occupancies at each objective
    evaluation.  A positive bulk-solvent floor is not applied here.
    """
    target, models = _validate_inputs(target, models, 1, 0.02)
    n_models = models.shape[0]
    lower = np.zeros(n_models) if lower_bounds is None else np.asarray(lower_bounds, dtype=float)
    upper = np.full(n_models, np.inf) if upper_bounds is None else np.asarray(upper_bounds, dtype=float)
    if lower.shape != (n_models,) or upper.shape != (n_models,):
        raise ValueError("occupancy bounds must have one entry per model")
    if np.any(~np.isfinite(lower)) or np.any(lower < 0.0):
        raise ValueError("lower occupancy bounds must be finite and non-negative")
    if np.any(np.isnan(upper)) or np.any(upper < lower):
        raise ValueError("upper occupancy bounds must be at least the lower bounds")
    if max_total < 0.0 or lower.sum() > max_total + 1e-12:
        raise ValueError("occupancy lower bounds exceed the total occupancy budget")

    try:
        # Keep CVXPY lazy: selection bookkeeping can run without qFit's
        # crystallography stack.  Prefer it when the runtime provides it.
        import cvxpy as cp

        weights = cp.Variable(n_models)
        intercept = cp.Variable()
        residual = target - models.T @ weights - intercept
        constraints = [weights >= lower, cp.sum(weights) <= max_total]
        if np.any(np.isfinite(upper)):
            constraints.append(weights <= upper)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(residual)), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True, polish=True)
        if (weights.value is not None and intercept.value is not None and
                problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}):
            answer = np.asarray(weights.value, dtype=float)
            c = float(intercept.value)
            rss = float(np.square(target - models.T @ answer - c).sum())
            return answer, c, rss
    except (ImportError, ModuleNotFoundError):
        pass

    # The pod's current qFit environment does not ship CVXPY.  This convex
    # fallback keeps the adopted continuous solve usable there; MIQP selection
    # remains a separate qFit/CVXPY-dependent final-selection step.
    from scipy.optimize import minimize

    def objective(value):
        return float(np.square(target - models.T @ value[:-1] - value[-1]).sum())

    def gradient(value):
        residual = models.T @ value[:-1] + value[-1] - target
        return 2.0 * np.concatenate((models @ residual, [residual.sum()]))

    finite_upper = np.where(np.isfinite(upper), upper, None)
    bounds = [(float(lo), finite_upper[i]) for i, lo in enumerate(lower)] + [(None, None)]
    initial_weights = lower.copy()
    remaining = max_total - initial_weights.sum()
    for index in range(n_models):
        room = (upper[index] - initial_weights[index]
                if np.isfinite(upper[index]) else remaining)
        addition = min(remaining, max(0.0, room))
        initial_weights[index] += addition
        remaining -= addition
        if remaining <= 1e-12:
            break
    initial = np.concatenate((initial_weights, [float(np.mean(target - models.T @ initial_weights))]))
    result = minimize(
        objective, initial, jac=gradient, method="SLSQP", bounds=bounds,
        constraints={"type": "ineq", "fun": lambda value: max_total - value[:-1].sum(),
                     "jac": lambda value: np.r_[-np.ones(n_models), 0.0]},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"affine occupancy QP failed without CVXPY: {result.message}")
    answer = np.asarray(result.x[:-1], dtype=float)
    c = float(result.x[-1])
    return answer, c, objective(result.x)


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


def _solve_decoupled_affine_miqp(
    target: np.ndarray, models: np.ndarray, *, cardinality: int, threshold: float
) -> tuple[np.ndarray, float, float]:
    """Solve the adopted decoupled MIQP with the same free intercept."""
    # The Zenodo panel has exactly two candidate slots at this point.  SCIP's
    # dense 8k-row MIQP path can spend minutes in presolve for this trivial
    # binary problem, even though there are only four possible subsets.  For
    # small candidate sets, enumerate those subsets exactly and solve the
    # corresponding continuous affine QP.  This preserves every MIQP
    # constraint while avoiding an opaque solver stall; larger candidate sets
    # retain the general SCIP path below.
    n_models = models.shape[0]
    if n_models <= 12:
        best = None
        max_selected = min(int(cardinality), n_models)
        for size in range(max_selected + 1):
            for selected_tuple in combinations(range(n_models), size):
                lower = np.zeros(n_models, dtype=float)
                upper = np.zeros(n_models, dtype=float)
                if selected_tuple:
                    selected = np.asarray(selected_tuple, dtype=int)
                    lower[selected] = float(threshold)
                    upper[selected] = 1.0
                    weights, intercept, rss = solve_affine_qp(
                        target, models, lower_bounds=lower, upper_bounds=upper,
                        max_total=1.0,
                    )
                else:
                    weights = np.zeros(n_models, dtype=float)
                    intercept = float(np.mean(target))
                    rss = float(np.square(target - intercept).sum())
                candidate = (float(rss), np.asarray(weights, dtype=float),
                             float(intercept))
                if best is None or candidate[0] < best[0]:
                    best = candidate
        assert best is not None
        return best[1], best[2], best[0]

    import cvxpy as cp

    weights = cp.Variable(n_models)
    selected = cp.Variable(n_models, boolean=True)
    intercept = cp.Variable()
    constraints = [
        weights >= 0.0,
        weights <= selected,
        weights >= threshold * selected,
        cp.sum(selected) <= cardinality,
        cp.sum(weights) <= 1.0,
    ]
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(target - models.T @ weights - intercept)),
        constraints,
    )
    problem.solve(solver="SCIP")
    if (weights.value is None or intercept.value is None or
            problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}):
        raise RuntimeError(f"decoupled affine MIQP failed: {problem.status}")
    answer = np.asarray(weights.value, dtype=float)
    c = float(intercept.value)
    rss = float(np.square(target - models.T @ answer - c).sum())
    return answer, c, rss


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


def select_decoupled_affine_miqp(
    target: np.ndarray,
    models: np.ndarray,
    *,
    cardinality_cap: int = DEFAULT_CARDINALITY_CAP,
    t_min: float = DEFAULT_MIN_OCCUPANCY,
    n_atoms: int,
) -> dict[str, Any]:
    """Final A-prime MIQP selection with a fitted intercept.

    This is the production counterpart of :func:`select_decoupled_miqp` after
    adopting the floor-off affine occupancy convention.  The qFit-native
    threshold-only rows remain available separately as a comparison.
    """
    target, models = _validate_inputs(target, models, cardinality_cap, t_min)
    effective_cap = min(cardinality_cap, models.shape[0])
    weights, intercept, rss = _solve_decoupled_affine_miqp(
        target, models, cardinality=effective_cap, threshold=t_min
    )
    selected = _candidate_record(
        weights, rss, n_voxels=target.size, n_atoms=n_atoms,
        cap=effective_cap, threshold=t_min,
    )
    return {
        "method": "A-prime decoupled affine MIQP (independent cap, floor, and intercept)",
        "constraints": [
            "sum(z_i) <= K",
            "t_min * z_i <= w_i <= z_i",
            "sum(w_i) <= 1",
            "target ~= sum_i(w_i * rho_i) + c",
        ],
        "cardinality_cap": int(cardinality_cap),
        "t_min": float(t_min),
        "effective_cardinality_cap": int(effective_cap),
        "weights": selected["weights"],
        "intercept": intercept,
        "rss": selected["rss"],
        "selected_slots": selected["selected_slots"],
        "n_selected_conformers": selected["n_selected_conformers"],
        "bic": selected["bic"],
        "bic_k": selected["bic_k"],
        "bic_candidates": [selected],
    }


def diagnose_affine_cardinality_caps(
    target: np.ndarray,
    models: np.ndarray,
    *,
    cardinality_caps: tuple[int, ...] = (1, 2, 3, 4),
    t_min: float = DEFAULT_MIN_OCCUPANCY,
    n_atoms: int,
) -> list[dict[str, Any]]:
    """Report qFit-style BIC for K under the adopted affine objective."""
    target, models = _validate_inputs(target, models, 1, t_min)
    rows = []
    for cap in cardinality_caps:
        if cap < 1:
            raise ValueError("cardinality caps must be positive")
        effective_cap = min(int(cap), models.shape[0])
        weights, intercept, rss = _solve_decoupled_affine_miqp(
            target, models, cardinality=effective_cap, threshold=t_min
        )
        row = _candidate_record(
            weights, rss, n_voxels=target.size, n_atoms=n_atoms,
            cap=int(cap), threshold=t_min,
        )
        row["intercept"] = intercept
        row["effective_cardinality_cap"] = effective_cap
        rows.append(row)
    return rows


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
