#!/usr/bin/env python3
"""One-dimensional occupancy/geometry diagnostic for the D1 starvation loop.

The toy deliberately uses one shared geometry optimizer for all occupancy
schemes.  Only the occupancy block changes:

* ``qp`` solves the exact non-negative, sum-at-most-one quadratic problem;
* ``mirror`` uses multiplicative mirror descent;
* ``mirror_entropy`` adds ``tau * sum(w * log(w))`` to the occupancy loss.

The toy has a known two-Gaussian target and starts both slots near the same
position.  It is an experiment, not a benchmark-selection criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


X = np.linspace(-3.5, 3.5, 701)
SIGMA = 0.28
TRUE_Q = np.asarray([-1.15, 1.15], dtype=float)
TRUE_W = np.asarray([0.60, 0.40], dtype=float)
START_Q = np.asarray([-0.025, 0.025], dtype=float)
START_W = np.asarray([0.50, 0.50], dtype=float)
Q_STEP = 0.001
MAX_STEPS = 5000
CONVERGENCE_WINDOW = 20
CONVERGENCE_TOL = 2e-5


def gaussian(q: float) -> np.ndarray:
    return np.exp(-0.5 * ((X - q) / SIGMA) ** 2)


TARGET = TRUE_W[0] * gaussian(TRUE_Q[0]) + TRUE_W[1] * gaussian(TRUE_Q[1])


def models(q: np.ndarray) -> np.ndarray:
    return np.asarray([gaussian(float(value)) for value in q])


def loss_and_gradients(q: np.ndarray, w: np.ndarray, tau: float = 0.0,
                       occupancy_residual_scale: float = 1.0):
    if occupancy_residual_scale <= 0.0:
        raise ValueError("occupancy_residual_scale must be positive")
    model = models(q)
    prediction = w @ model
    residual = prediction - TARGET
    entropy = float(np.sum(w * np.log(w))) if tau else 0.0
    loss = float(np.dot(residual, residual) + tau * entropy)
    dmodel_dq = model * ((X[None, :] - q[:, None]) / SIGMA**2)
    grad_q = 2.0 * w * np.sum(residual[None, :] * dmodel_dq, axis=1)
    grad_w = 2.0 * occupancy_residual_scale * (model @ residual)
    if tau:
        grad_w = grad_w + tau * (np.log(np.maximum(w, 1e-300)) + 1.0)
    return loss, grad_q, grad_w, model


def _best_on_segment(target: np.ndarray, matrix: np.ndarray, lo: float, hi: float):
    """Minimize ||target - (w0*m0+w1*m1)||² over w0 in [lo,hi]."""
    direction = matrix[0] - matrix[1]
    base = matrix[1]
    denom = float(np.dot(direction, direction))
    value = lo if denom == 0.0 else float(np.dot(target - base, direction) / denom)
    w0 = float(np.clip(value, lo, hi))
    weights = np.asarray([w0, 1.0 - w0])
    rss = float(np.sum((target - weights @ matrix) ** 2))
    return weights, rss


def exact_qp(target: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Exact two-variable QP by enumerating the active-set faces."""
    candidates: list[tuple[float, np.ndarray]] = []
    candidates.extend((float(np.sum((target - matrix[i] * value) ** 2)),
                       np.asarray([value if i == 0 else 0.0,
                                   value if i == 1 else 0.0]))
                      for i in range(2) for value in [
                          max(0.0, min(1.0, float(np.dot(target, matrix[i]) /
                                                   max(np.dot(matrix[i], matrix[i]), 1e-30))))])

    gram = matrix @ matrix.T
    rhs = matrix @ target
    try:
        unconstrained = np.linalg.solve(gram, rhs)
        if np.all(unconstrained >= -1e-10) and unconstrained.sum() <= 1.0 + 1e-10:
            weights = np.maximum(unconstrained, 0.0)
            candidates.append((float(np.sum((target - weights @ matrix) ** 2)), weights))
    except np.linalg.LinAlgError:
        pass

    weights, rss = _best_on_segment(target, matrix, 0.0, 1.0)
    candidates.append((rss, weights))
    candidates.append((float(np.dot(target, target)), np.zeros(2)))
    candidates.sort(key=lambda item: item[0])
    answer = candidates[0][1]
    assert np.all(answer >= -1e-10)
    assert answer.sum() <= 1.0 + 1e-10
    return np.maximum(answer, 0.0)


def normalized_mirror_gradient(grad_w: np.ndarray) -> np.ndarray:
    """Return a scale-free occupancy gradient direction."""
    grad_w = np.asarray(grad_w, dtype=float)
    norm = float(np.linalg.norm(grad_w))
    return grad_w if norm == 0.0 else grad_w / norm


def mirror_update(w: np.ndarray, grad_w: np.ndarray, eta: float) -> np.ndarray:
    if not np.all(w > 0.0):
        raise AssertionError("mirror descent requires strictly positive occupancies")
    log_update = np.log(w) - eta * normalized_mirror_gradient(grad_w)
    log_update -= np.max(log_update)
    answer = np.exp(log_update)
    if answer.sum() > 1.0:
        answer /= answer.sum()
    assert np.all(answer > 0.0), "multiplicative update must never create zero occupancy"
    assert answer.sum() <= 1.0 + 1e-12
    return answer


def run(scheme: str, eta: float, tau: float = 0.0,
        occupancy_residual_scale: float = 1.0) -> dict[str, object]:
    q = START_Q.copy()
    w = START_W.copy()
    trajectory = []
    stable = 0
    for step in range(1, MAX_STEPS + 1):
        model = models(q)
        if scheme == "qp":
            w = exact_qp(TARGET, model)
            loss, grad_q, grad_w, _ = loss_and_gradients(q, w)
        elif scheme in {"mirror", "mirror_entropy"}:
            loss, grad_q, grad_w, model = loss_and_gradients(
                q, w, tau if scheme == "mirror_entropy" else 0.0,
                occupancy_residual_scale=occupancy_residual_scale,
            )
            w = mirror_update(w, grad_w, eta)
            loss, grad_q, _, _ = loss_and_gradients(
                q, w, tau if scheme == "mirror_entropy" else 0.0,
                occupancy_residual_scale=occupancy_residual_scale,
            )
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        q -= Q_STEP * grad_q
        q = np.clip(q, -3.0, 3.0)
        loss, _, _, _ = loss_and_gradients(
            q, w, tau if scheme == "mirror_entropy" else 0.0,
            occupancy_residual_scale=occupancy_residual_scale,
        )
        trajectory.append({"step": step, "q": q.tolist(), "w": w.tolist(), "loss": loss})
        if len(trajectory) >= 2 and abs(trajectory[-1]["loss"] - trajectory[-2]["loss"]) < CONVERGENCE_TOL:
            stable += 1
        else:
            stable = 0
        if stable >= CONVERGENCE_WINDOW:
            break
    order = np.argsort(q)
    q_ordered = q[order]
    w_ordered = w[order]
    return {
        "scheme": scheme,
        "eta": float(eta),
        "tau": float(tau),
        "occupancy_residual_scale": float(occupancy_residual_scale),
        "steps": len(trajectory),
        "initial_q": START_Q.tolist(),
        "initial_w": START_W.tolist(),
        "final_q": q_ordered.tolist(),
        "final_w": w_ordered.tolist(),
        "true_q": TRUE_Q.tolist(),
        "true_w": TRUE_W.tolist(),
        "q_error": float(np.max(np.abs(q_ordered - TRUE_Q))),
        "w_error": float(np.max(np.abs(w_ordered - TRUE_W))),
        "slot_separation": float(abs(q_ordered[1] - q_ordered[0])),
        "truth_separation": float(TRUE_Q[1] - TRUE_Q[0]),
        "strictly_positive_final_weights": bool(np.all(w > 0.0)),
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    rows.append(run("qp", eta=0.0))
    for eta in (0.001, 0.003, 0.01, 0.03, 0.1):
        rows.append(run("mirror", eta=eta))
        for residual_scale in (10.0, 100.0):
            rows.append(run("mirror", eta=eta,
                            occupancy_residual_scale=residual_scale))
        for tau in (0.0001, 0.001, 0.01):
            rows.append(run("mirror_entropy", eta=eta, tau=tau))
    qp = rows[0]
    if not any(row["slot_separation"] > 0.8 * row["truth_separation"] and row["q_error"] < 0.25
               and row["w_error"] < 0.10 for row in rows if row["scheme"] == "mirror"):
        verdict = "toy_mirror_failed"
    else:
        verdict = "toy_mirror_passed"
    payload = {
        "status": "complete", "verdict": verdict,
        "fixed_toy_contract": {
            "true_q": TRUE_Q.tolist(), "true_w": TRUE_W.tolist(),
            "start_q": START_Q.tolist(), "start_w": START_W.tolist(),
            "q_step": Q_STEP, "max_steps": MAX_STEPS,
            "success_diagnostic": "separation >= 80% truth; max position error < 0.25; max occupancy error < 0.10",
            "qp_first_evaluation_weights": qp["trajectory"][0]["w"],
        },
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2))
    (args.output / "progress.json").write_text(json.dumps({"status": "complete", "rows": len(rows)}))
    print(json.dumps({"status": "complete", "verdict": verdict, "rows": len(rows)}))


if __name__ == "__main__":
    main()
