"""Small Torch-native trust-region least-squares kernel.

This is deliberately separate from the benchmark until its numerical behavior
is validated.  The residual and Jacobian algebra stays on ``x.device``;
Python only reads scalar diagnostics needed to accept a step and adapt the
trust radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


TensorResidual = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TorchTRFResult:
    x: torch.Tensor
    cost: float
    nfev: int
    njev: int
    status: int
    message: str
    optimality: float
    projected_optimality: float
    trust_radius_trace: list[dict[str, float | bool]]


def _projected_gradient(g: torch.Tensor, x: torch.Tensor,
                        lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    at_lower = torch.isfinite(lower) & (x <= lower) & (g > 0)
    at_upper = torch.isfinite(upper) & (x >= upper) & (g < 0)
    return torch.where(at_lower | at_upper, torch.zeros_like(g), g)


def _trust_step(g: torch.Tensor, hessian: torch.Tensor,
                radius: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """Compute a Levenberg--Marquardt step constrained by ``||step|| <= radius``."""
    n = g.numel()
    identity = torch.eye(n, dtype=g.dtype, device=g.device)
    scaled_hessian = x_scale[:, None] * hessian * x_scale[None, :]
    scaled_gradient = x_scale * g
    try:
        gauss_newton_scaled = torch.linalg.solve(scaled_hessian, -scaled_gradient)
    except RuntimeError:
        gauss_newton_scaled = torch.linalg.lstsq(scaled_hessian, -scaled_gradient).solution
    if torch.linalg.vector_norm(gauss_newton_scaled) <= radius:
        return x_scale * gauss_newton_scaled

    # Find the smallest damping that puts the normal-equation step on the
    # trust-region boundary.  Bisection is stable for these 20-dimensional
    # blocks and keeps all matrix work on the Torch device.
    lo = torch.zeros((), dtype=g.dtype, device=g.device)
    hi = torch.ones((), dtype=g.dtype, device=g.device)
    while torch.linalg.vector_norm(torch.linalg.solve(hessian + hi * identity, -g)) > radius:
        hi = hi * 2.0
    for _ in range(32):
        mid = (lo + hi) / 2.0
        step_scaled = torch.linalg.solve(scaled_hessian + mid * identity, -scaled_gradient)
        if torch.linalg.vector_norm(step_scaled) > radius:
            lo = mid
        else:
            hi = mid
    return x_scale * torch.linalg.solve(scaled_hessian + hi * identity, -scaled_gradient)


def least_squares(
    residual_fn: TensorResidual,
    x0: torch.Tensor,
    *,
    lower: torch.Tensor | None = None,
    upper: torch.Tensor | None = None,
    max_nfev: int = 40,
    initial_radius: float = 1.0,
    max_radius: float = 1e3,
    x_scale: float | torch.Tensor = 1.0,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
) -> TorchTRFResult:
    """Solve a small bounded nonlinear least-squares problem on Torch.

    Bounds are enforced by projection after each accepted trial.  This is the
    prototype contract; production integration must add the exact reflective
    bound handling used by SciPy before it replaces the current solver.
    """
    if x0.ndim != 1 or not x0.is_floating_point():
        raise ValueError("x0 must be a one-dimensional floating Torch tensor")
    device, dtype = x0.device, x0.dtype
    lower = torch.full_like(x0, -torch.inf) if lower is None else lower.to(device=device, dtype=dtype)
    upper = torch.full_like(x0, torch.inf) if upper is None else upper.to(device=device, dtype=dtype)
    if lower.shape != x0.shape or upper.shape != x0.shape:
        raise ValueError("bounds must have the same shape as x0")
    x = torch.minimum(torch.maximum(x0, lower), upper)
    radius = torch.as_tensor(initial_radius, device=device, dtype=dtype)
    max_radius_t = torch.as_tensor(max_radius, device=device, dtype=dtype)
    scale = torch.as_tensor(x_scale, device=device, dtype=dtype)
    if scale.ndim == 0:
        scale = torch.full_like(x, scale)
    if scale.shape != x.shape or torch.any(scale <= 0):
        raise ValueError("x_scale must be positive and scalar or match x0")
    trace: list[dict[str, float | bool]] = []
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    nfev = 1
    njev = 0
    residual = residual_fn(x)
    cost = 0.5 * torch.dot(residual, residual)
    start_cost = cost
    status = 0
    message = "evaluation cap reached"

    for _ in range(max_nfev - 1):
        x_for_jacobian = x.detach().requires_grad_(True)
        residual = residual_fn(x_for_jacobian)
        jacobian = torch.autograd.functional.jacobian(
            residual_fn, x_for_jacobian, create_graph=False, vectorize=True,
        )
        njev += 1
        cost = 0.5 * torch.dot(residual, residual)
        gradient = jacobian.T @ residual
        projected = _projected_gradient(gradient, x, lower, upper)
        gnorm = torch.linalg.vector_norm(gradient)
        pgnorm = torch.linalg.vector_norm(projected)
        if pgnorm <= gtol:
            status, message = 1, "gtol"
            break

        hessian = jacobian.T @ jacobian
        step = _trust_step(gradient, hessian, radius, scale)
        trial = torch.minimum(torch.maximum(x + step, lower), upper)
        effective_step = trial - x
        trial_residual = residual_fn(trial)
        nfev += 1
        trial_cost = 0.5 * torch.dot(trial_residual, trial_residual)
        actual = cost - trial_cost
        predicted = -(torch.dot(gradient, effective_step) +
                      0.5 * torch.dot(effective_step, hessian @ effective_step))
        ratio = actual / torch.clamp(predicted, min=torch.finfo(dtype).tiny)
        step_norm = torch.linalg.vector_norm(effective_step / scale)
        trace.append({
            "radius_before": float(radius.item()),
            "step_norm": float(step_norm.item()),
            "actual_reduction": float(actual.item()),
            "predicted_reduction": float(predicted.item()),
            "actual_over_predicted": float(ratio.item()),
            "accepted": bool((ratio > 1e-4).item()),
        })
        if ratio > 1e-4:
            previous_cost = cost
            x = trial.detach()
            cost = trial_cost.detach()
            if abs(float((previous_cost - cost).item())) <= ftol * max(1.0, abs(float(previous_cost.item()))):
                status, message = 2, "ftol"
            elif step_norm <= xtol * max(1.0, float(torch.linalg.vector_norm(x).item())):
                status, message = 3, "xtol"
            if ratio > 0.75 and step_norm >= 0.9 * radius:
                radius = torch.minimum(2.0 * radius, max_radius_t)
            elif ratio < 0.25:
                radius = radius * 0.25
            if status:
                break
        else:
            radius = radius * 0.25
        if radius <= torch.finfo(dtype).eps:
            status, message = 4, "trust radius underflow"
            break

    final_residual = residual_fn(x)
    final_jacobian = torch.autograd.functional.jacobian(
        residual_fn, x.detach().requires_grad_(True), create_graph=False, vectorize=True,
    )
    final_gradient = final_jacobian.T @ final_residual
    final_projected = _projected_gradient(final_gradient, x, lower, upper)
    return TorchTRFResult(
        x=x.detach(), cost=float((0.5 * torch.dot(final_residual, final_residual)).item()),
        nfev=nfev, njev=njev, status=status, message=message,
        optimality=float(torch.linalg.vector_norm(final_gradient).item()),
        projected_optimality=float(torch.linalg.vector_norm(final_projected).item()),
        trust_radius_trace=trace,
    )
