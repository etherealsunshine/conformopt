#!/usr/bin/env python3
"""A-prime slot-coordination reachability experiment at 5OHJ A:SER540.

This is an experiment harness.  It does not alter the production optimizer.
All geometry Jacobians use the Torch forward-mode path.  The four published
qFit 2015 peptide-flip centroids are supplied through --flip-root.
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# CCTBX must be loaded before NumPy/SciPy/Torch native libraries in the mixed
# qFit+CUDA environment; importing it later can segfault in boost_python.
from qfit.backbone import compute_jacobian

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import least_squares
from scipy.optimize._lsq import common as scipy_lsq_common
from scipy.optimize._lsq import trf as scipy_trf

import torch
from run_d1_aprime_sequential import APrimeSequential, rmsd, seam_vector
from run_d1_reachability import BACKBONE_NAMES, dihedrals, wrapped_delta
from torch_trf import least_squares as torch_least_squares
from occupancy_selection import (
    DEFAULT_CARDINALITY_CAP,
    DEFAULT_MIN_OCCUPANCY,
    select_decoupled_affine_miqp,
)
from fit_provenance import fit_voxel_provenance
from d1_population_calibrated_weights import D1_RAMA_FLOOR


SITE = ("5OHJ", "A", 540)
INNER_NFEV = 8
OUTER_UPDATES = 6
FLIP_NAMES = ("cent_clust1_3mcwFH_B_100-103_sup.pdb",
              "cent_clust2_2iorFH_A_158-161_sup.pdb",
              "cent_clust3_2g1uFH_A_50-53_sup.pdb",
              "cent_clust4_3g6kFH_F_171-174_sup.pdb")
REF_NAME = "ref_ideal1pep_supto_1byi_11-12.pdb"
REF_KEYS = ((1, "CA"), (1, "C"), (1, "O"), (2, "N"), (2, "CA"))
BACKBONE = ("N", "CA", "C", "O")


class SharedJointParameterization:
    """Pack two 20-parameter slots with shared outer backbone torsions.

    The full per-slot order is qFit's 14 phi/psi values followed by six
    omega values.  The reduced order is shared phi/psi, slot-1 phi/psi,
    slot-2 phi/psi, and shared omega.
    """

    def __init__(self, ndofs: int, per_slot_offsets: tuple[int, ...] = (2, 3, 4)):
        if ndofs != 20:
            raise ValueError(f"expected qFit's 20 torsions per slot, got {ndofs}")
        self.ndofs = ndofs
        self.n_residues = 7
        self.phi_psi_ndofs = 2 * self.n_residues
        self.omega_ndofs = ndofs - self.phi_psi_ndofs
        self.per_slot_offsets = tuple(sorted(set(int(value) for value in per_slot_offsets)))
        if not self.per_slot_offsets or any(
            value < 0 or value >= self.n_residues for value in self.per_slot_offsets
        ):
            raise ValueError("per-slot residue offsets must be nonempty values in [0, 6]")
        self.shared_offsets = tuple(
            value for value in range(self.n_residues)
            if value not in self.per_slot_offsets
        )
        self.shared_phi_indices = np.asarray(
            [2 * value + delta for value in self.shared_offsets for delta in (0, 1)],
            dtype=int,
        )
        self.per_slot_phi_indices = np.asarray(
            [2 * value + delta for value in self.per_slot_offsets for delta in (0, 1)],
            dtype=int,
        )
        self.shared_omega_indices = np.arange(
            self.phi_psi_ndofs, ndofs, dtype=int
        )
        self.shared_omega = True
        self.shared_phi_count = len(self.shared_phi_indices)
        self.per_slot_phi_count = len(self.per_slot_phi_indices)
        self.reduced_ndofs = (
            self.shared_phi_count + 2 * self.per_slot_phi_count + self.omega_ndofs
        )

    def pack(self, slot1: np.ndarray, slot2: np.ndarray) -> np.ndarray:
        slot1 = np.asarray(slot1, dtype=float)
        slot2 = np.asarray(slot2, dtype=float)
        if slot1.shape != (self.ndofs,) or slot2.shape != (self.ndofs,):
            raise ValueError("full slot parameters must each have shape (20,)")
        # Slot 1 is the anchored A-like scaffold; its shared values define the
        # common outer window when an initial slot-2 proposal disagrees there.
        return np.concatenate((
            slot1[self.shared_phi_indices],
            slot1[self.per_slot_phi_indices],
            slot2[self.per_slot_phi_indices],
            slot1[self.shared_omega_indices],
        ))

    def expand_numpy(self, reduced: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced, dtype=float)
        if reduced.shape != (self.reduced_ndofs,):
            raise ValueError(
                f"reduced parameters must have shape ({self.reduced_ndofs},)"
            )
        cursor = 0
        shared = reduced[cursor:cursor + self.shared_phi_count]
        cursor += self.shared_phi_count
        slot1_phi = reduced[cursor:cursor + self.per_slot_phi_count]
        cursor += self.per_slot_phi_count
        slot2_phi = reduced[cursor:cursor + self.per_slot_phi_count]
        cursor += self.per_slot_phi_count
        omega = reduced[cursor:cursor + self.omega_ndofs]
        slots = np.zeros((2, self.ndofs), dtype=float)
        slots[:, self.shared_phi_indices] = shared
        slots[0, self.per_slot_phi_indices] = slot1_phi
        slots[1, self.per_slot_phi_indices] = slot2_phi
        slots[:, self.shared_omega_indices] = omega
        return slots

    def expand_torch(self, reduced: torch.Tensor) -> torch.Tensor:
        if reduced.ndim != 1 or reduced.shape[0] != self.reduced_ndofs:
            raise ValueError(
                f"reduced parameters must have shape ({self.reduced_ndofs},)"
            )
        cursor = 0
        shared = reduced[cursor:cursor + self.shared_phi_count]
        cursor += self.shared_phi_count
        slot1_phi = reduced[cursor:cursor + self.per_slot_phi_count]
        cursor += self.per_slot_phi_count
        slot2_phi = reduced[cursor:cursor + self.per_slot_phi_count]
        cursor += self.per_slot_phi_count
        omega = reduced[cursor:cursor + self.omega_ndofs]
        shared_indices = torch.as_tensor(self.shared_phi_indices, dtype=torch.long, device=reduced.device)
        per_indices = torch.as_tensor(self.per_slot_phi_indices, dtype=torch.long, device=reduced.device)
        omega_indices = torch.as_tensor(self.shared_omega_indices, dtype=torch.long, device=reduced.device)
        slot1 = torch.zeros(self.ndofs, dtype=reduced.dtype, device=reduced.device)
        slot2 = torch.zeros(self.ndofs, dtype=reduced.dtype, device=reduced.device)
        slot1 = slot1.index_copy(0, shared_indices, shared)
        slot2 = slot2.index_copy(0, shared_indices, shared)
        slot1 = slot1.index_copy(0, per_indices, slot1_phi)
        slot2 = slot2.index_copy(0, per_indices, slot2_phi)
        slot1 = slot1.index_copy(0, omega_indices, omega)
        slot2 = slot2.index_copy(0, omega_indices, omega)
        return torch.stack((slot1, slot2))


class FullJointParameterization:
    """The unreduced two-slot chart: 20 torsions per slot, 40 total."""

    reduced_ndofs = 40
    shared_offsets: tuple[int, ...] = ()
    per_slot_offsets = tuple(range(7))
    shared_omega = False

    def __init__(self, ndofs: int):
        if ndofs != 20:
            raise ValueError(f"expected qFit's 20 torsions per slot, got {ndofs}")
        self.ndofs = ndofs

    def pack(self, slot1: np.ndarray, slot2: np.ndarray) -> np.ndarray:
        slot1 = np.asarray(slot1, dtype=float)
        slot2 = np.asarray(slot2, dtype=float)
        if slot1.shape != (self.ndofs,) or slot2.shape != (self.ndofs,):
            raise ValueError("full slot parameters must each have shape (20,)")
        return np.concatenate((slot1, slot2))

    def expand_numpy(self, reduced: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced, dtype=float)
        if reduced.shape != (self.reduced_ndofs,):
            raise ValueError("full joint parameters must have shape (40,)")
        return reduced.reshape(2, self.ndofs)

    def expand_torch(self, reduced: torch.Tensor) -> torch.Tensor:
        if reduced.ndim != 1 or reduced.shape[0] != self.reduced_ndofs:
            raise ValueError("full joint parameters must have shape (40,)")
        return reduced.reshape(2, self.ndofs)


def parse_per_slot_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets:
        raise argparse.ArgumentTypeError("provide at least one per-slot residue offset")
    return offsets


def read_pdb(path: Path) -> dict[tuple[int, str], np.ndarray]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        result[(int(line[22:26]), line[12:16].strip())] = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
        )
    return result


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(0), target.mean(0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    translation = target_mean - source_mean @ rotation
    return rotation, translation


def flip_targets(runner: APrimeSequential, flip_root: Path) -> list[tuple[str, np.ndarray]]:
    """Embed the paper's four centroid geometries in the current A frame.

    The paper's reference is CA1-C1-O1-N2-CA2.  We align that reference to
    residues 2 and 3 of the seven-residue A window (the peptide immediately
    before the central residue), then replace the four-residue backbone span
    with each transformed centroid.  Sidechains and the flanking backbone are
    retained from A; inverse kinematics converts the backbone seed to A-prime
    torsions before optimization.
    """
    ref = read_pdb(flip_root / REF_NAME)
    reference = np.vstack([ref[key] for key in REF_KEYS])
    residues = runner.base.window.residues
    current_keys = ((residues[2], "CA"), (residues[2], "C"), (residues[2], "O"),
                    (residues[3], "N"), (residues[3], "CA"))
    def window_index(residue, name):
        return int(np.searchsorted(runner.base.window.selection,
                                   int(residue.select("name", name)[0])))
    current = np.vstack([
        runner.base.window.coor[window_index(residue, name)]
        for residue, name in current_keys
    ])
    rotation, translation = rigid_fit(reference, current)
    outputs = []
    for index, filename in enumerate(FLIP_NAMES, start=1):
        cluster = read_pdb(flip_root / filename)
        transformed = {key: value @ rotation + translation for key, value in cluster.items()}
        target = runner.initial.copy()
        for offset, residue in enumerate(residues[2:6]):
            source_resnum = sorted({key[0] for key in cluster})[offset]
            for name in BACKBONE:
                target[window_index(residue, name)] = transformed[(source_resnum, name)]
        outputs.append((f"flip_{index}", target))
    return outputs


def inverse_seed(runner: APrimeSequential, target_window: np.ndarray) -> np.ndarray:
    phi_psi, omega = dihedrals(runner.window)
    target_original = runner.window.coor.copy()
    try:
        runner.window.coor = target_window.copy()
        target_phi_psi, target_omega = dihedrals(runner.window)
    finally:
        runner.window.coor = target_original
    seed = np.concatenate((wrapped_delta(phi_psi, target_phi_psi),
                           wrapped_delta(omega, target_omega)))
    # Use the published/deposited dihedral displacement directly as the
    # prospective kinematic seed.  A separate inverse fit would spend
    # hundreds of geometry-only evaluations and would make the initialization
    # protocol itself a hidden optimizer.
    return seed


def joint_rama_rows(runner: APrimeSequential, parameters: np.ndarray,
                    parameterization: SharedJointParameterization,
                    fixed_b_offset: float | None = None) -> np.ndarray:
    rows = []
    coordinates = runner.torch_forward(
        parameterization.expand_torch(torch.as_tensor(
            parameters if fixed_b_offset is not None else parameters[:-1], dtype=torch.float64
        ))
    ).detach().cpu().numpy()
    for coordinate in coordinates:
        rows.append(np.sqrt(runner.rama_weight) * runner.omega_and_rama(coordinate)[3])
    return np.concatenate(rows)


def joint_evaluate(runner: APrimeSequential, parameters: np.ndarray,
                   normalizer: float, lambdas: np.ndarray,
                   parameterization: SharedJointParameterization,
                   fixed_b_offset: float | None = None,
                   slot2_occupancy_floor: float = 0.0,
                   occupancy_weights: np.ndarray | None = None,
                   amplitude_prior_lambda: float = 0.0,
                   amplitude_prior_reference: np.ndarray | None = None,
                   fixed_intercept: float | None = None) -> dict[str, object]:
    """Evaluate the profiled-occupancy joint objective.

    The final scalar is a site-wide B offset.  Occupancies and the affine map
    intercept remain profiled outside the torsion gradient, while the B offset
    is an explicit differentiable density-width parameter.  This is the A''
    post-B-array-fix objective; do not substitute the older all-A-B-factor
    renderer used by the original slot-coordination experiment.
    """
    torsions = parameters if fixed_b_offset is not None else parameters[:-1]
    b_offset = float(fixed_b_offset if fixed_b_offset is not None else parameters[-1])
    coordinates_torch = runner.torch_forward(
        parameterization.expand_torch(torch.as_tensor(torsions, dtype=torch.float64))
    )
    coordinates = coordinates_torch.detach().cpu().numpy()
    models = runner.base.model_density_batch(
        coordinates, slots=np.array((0, 1)), b_offset=b_offset,
    )
    if runner.training_indices is not None:
        models = models[:, runner.training_indices]
    if occupancy_weights is None:
        weights, intercept, rss = runner.joint_qp_weights(
            runner.target, models,
            lower_bounds=(0.0, float(slot2_occupancy_floor)),
        )
    else:
        weights = np.asarray(occupancy_weights, dtype=float)
        if weights.shape != (2,) or not np.all(np.isfinite(weights)):
            raise ValueError("mirror-descent occupancy weights must have shape (2) and be finite")
        if np.any(weights <= 0.0) or weights.sum() > 1.0 + 1e-10:
            raise ValueError("mirror-descent occupancy weights must be positive and sum to at most one")
        intercept = (float(np.mean(runner.target - weights @ models))
                     if fixed_intercept is None else float(fixed_intercept))
        rss = float(np.square(runner.target - weights @ models - intercept).sum())
    residual = [(runner.target - weights @ models - intercept) / np.sqrt(normalizer)]
    seam_rows, rama_rows, planar_rows = [], [], []
    seam_vectors = []
    _, omega_delta_torch, rama_scores, rama_barriers_torch = runner.torch_omega_and_rama(
        coordinates_torch
    )
    omega_delta_numpy = omega_delta_torch.detach().cpu().numpy()
    rama_barriers_numpy = rama_barriers_torch.detach().cpu().numpy()
    for index, coordinate in enumerate(coordinates):
        seam, _, _ = seam_vector(runner.initial_backbone, coordinate[runner.bb_indices])
        seam_vectors.append(seam)
        seam_rows.append(np.sqrt(runner.rho / 2.0) * (seam + lambdas[6 * (len(seam_vectors) - 1):6 * len(seam_vectors)] / runner.rho))
        rama_rows.append(np.sqrt(runner.rama_weight) * rama_barriers_numpy[index])
        planar_rows.append(np.sqrt(runner.planar_weight) * omega_delta_numpy[index] / runner.omega_scale_deg)
    residual.extend([np.concatenate(seam_rows), np.concatenate(rama_rows), np.concatenate(planar_rows)])
    full_torsions = parameterization.expand_numpy(np.asarray(torsions, dtype=float))
    prior_reference = (np.zeros_like(full_torsions) if amplitude_prior_reference is None
                       else np.asarray(amplitude_prior_reference, dtype=float))
    if prior_reference.shape != full_torsions.shape:
        raise ValueError("amplitude prior reference must have shape (2, 20)")
    amplitude_prior_residual = np.sqrt(max(float(amplitude_prior_lambda), 0.0)) * (
        full_torsions - prior_reference
    ).reshape(-1)
    if amplitude_prior_lambda > 0.0:
        residual.append(amplitude_prior_residual)
    residual = np.concatenate(residual)
    return {
        "coordinates": coordinates, "models": models, "weights": np.asarray(weights),
        "intercept": float(intercept), "rss": float(rss),
        "b_offset_A2": b_offset,
        "slot2_occupancy_floor": float(slot2_occupancy_floor),
        "seam_vectors": np.asarray(seam_vectors), "rama_scores": rama_scores,
        "residual": residual,
        "energy": float(np.dot(residual, residual)),
        "amplitude_prior_lambda": float(amplitude_prior_lambda),
        "amplitude_prior_energy": float(np.dot(amplitude_prior_residual, amplitude_prior_residual)),
        "amplitude_prior_norm": float(np.linalg.norm(full_torsions - prior_reference)),
    }


def slot2_geometry_metrics(runner: APrimeSequential, coordinates: np.ndarray) -> dict[str, float]:
    """Return deposited-state distances for trajectory diagnostics only."""
    central = runner.base.central_backbone(coordinates[1])
    return {
        "slot2_rmsd_to_A": float(rmsd(central, runner.a_backbone)),
        "slot2_rmsd_to_B": float(rmsd(central, runner.b_backbone)),
    }


def slot_pair_rmsd(runner: APrimeSequential, coordinates: np.ndarray) -> float:
    """Return the central-backbone RMSD between the two current slots."""
    central = [runner.base.central_backbone(coordinate) for coordinate in coordinates]
    return float(rmsd(central[0], central[1]))


def _scipy_default_initial_trust_radius(x0: np.ndarray, x_scale: float | np.ndarray) -> float:
    """Return SciPy TRF's initial scaled radius for the unbounded path."""
    if isinstance(x_scale, str):
        raise ValueError("trust-radius diagnostics require a numeric x_scale")
    scale_inv = 1.0 / np.asarray(x_scale, dtype=float)
    radius = float(np.linalg.norm(np.asarray(x0, dtype=float) * scale_inv))
    return 1.0 if radius == 0.0 else radius


def _patched_scipy_trf_function(function, initial_radius: float, update):
    """Clone one SciPy TRF entry point with only its initial Delta replaced.

    ``scipy.optimize.least_squares`` has no public initial-trust-radius option.
    The current benchmark deliberately stays on SciPy's TRF implementation,
    so the narrowest compatible hook is to re-execute the installed private
    entry point with its initial-Delta line changed.  All other solver code is
    the installed SciPy implementation, and ``update`` remains the existing
    trace hook.
    """
    source = textwrap.dedent(inspect.getsource(function))
    if function.__name__ == "trf_no_bounds":
        old = "Delta = norm(x0 * scale_inv)\n    if Delta == 0:\n        Delta = 1.0"
    elif function.__name__ == "trf_bounds":
        old = "Delta = norm(x0 * scale_inv / v**0.5)\n    if Delta == 0:\n        Delta = 1.0"
    else:  # pragma: no cover - guarded by the two private SciPy entry points.
        raise ValueError(f"unsupported SciPy TRF function: {function.__name__}")
    new = "Delta = float(_carried_initial_radius)"
    if old not in source:
        raise RuntimeError(
            f"SciPy {function.__name__} source changed; refusing to patch initial trust radius"
        )
    namespace = dict(vars(scipy_trf))
    namespace["_carried_initial_radius"] = float(initial_radius)
    namespace["update_tr_radius"] = update
    exec(compile(source.replace(old, new), f"<{function.__name__}_carried>", "exec"), namespace)
    return namespace[function.__name__]


def _least_squares_with_trust_trace(
    *args, trace: list[dict[str, object]], initial_radius: float | None = None, **kwargs,
):
    """Run SciPy TRF while recording its exact trust-radius updates.

    SciPy keeps ``Delta`` inside its TRF implementation and does not expose it
    through the public result object.  The per-slot mode runs one independent
    TRF solve per slot, so this scoped hook gives each slot its own radius and
    its own actual/predicted reduction ratios without changing SciPy's solver.
    """
    original_common = scipy_lsq_common.update_tr_radius
    original_trf = scipy_trf.update_tr_radius
    original_no_bounds = scipy_trf.trf_no_bounds
    original_bounds = scipy_trf.trf_bounds

    def update(delta, actual_reduction, predicted_reduction, step_norm, bound_hit):
        new_delta, ratio = original_common(
            delta, actual_reduction, predicted_reduction, step_norm, bound_hit
        )
        trace.append({
            "evaluation": len(trace) + 1,
            "radius_before_scaled": float(delta),
            "radius_after_scaled": float(new_delta),
            "radius_before_degrees_at_x_scale_10": float(delta * 10.0),
            "radius_after_degrees_at_x_scale_10": float(new_delta * 10.0),
            "actual_reduction": float(actual_reduction),
            "predicted_reduction": float(predicted_reduction),
            "actual_over_predicted": float(ratio),
            "accepted": bool(actual_reduction > 0.0),
            "step_norm_scaled": float(step_norm),
            "step_norm_degrees_at_x_scale_10": float(step_norm * 10.0),
            "bound_hit": bool(bound_hit),
        })
        return new_delta, ratio

    scipy_lsq_common.update_tr_radius = update
    scipy_trf.update_tr_radius = update
    if initial_radius is not None:
        if initial_radius <= 0.0 or not np.isfinite(initial_radius):
            raise ValueError(f"initial trust radius must be finite and positive, got {initial_radius}")
        scipy_trf.trf_no_bounds = _patched_scipy_trf_function(
            original_no_bounds, initial_radius, update
        )
        scipy_trf.trf_bounds = _patched_scipy_trf_function(
            original_bounds, initial_radius, update
        )
    try:
        return least_squares(*args, **kwargs)
    finally:
        scipy_lsq_common.update_tr_radius = original_common
        scipy_trf.update_tr_radius = original_trf
        scipy_trf.trf_no_bounds = original_no_bounds
        scipy_trf.trf_bounds = original_bounds


def mirror_descent_occupancy_update(target: np.ndarray, models: np.ndarray,
                                   weights: np.ndarray, intercept: float,
                                   eta: float, tau: float = 0.0) -> np.ndarray:
    """Take one strictly-positive multiplicative occupancy step."""
    weights = np.asarray(weights, dtype=float)
    models = np.asarray(models, dtype=float)
    target = np.asarray(target, dtype=float)
    if weights.shape != (2,) or np.any(weights <= 0.0):
        raise ValueError("mirror descent requires two strictly-positive weights")
    if eta <= 0.0 or tau < 0.0:
        raise ValueError("mirror eta must be positive and entropy tau non-negative")
    residual = models.T @ weights + float(intercept) - target
    gradient = 2.0 * (models @ residual)
    if tau:
        gradient = gradient + float(tau) * (np.log(weights) + 1.0)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > 0.0:
        gradient = gradient / gradient_norm
    log_weights = np.log(weights) - float(eta) * gradient
    log_weights -= np.max(log_weights)
    updated = np.exp(log_weights)
    if updated.sum() > 1.0:
        updated /= updated.sum()
    if not np.all(updated > 0.0) or updated.sum() > 1.0 + 1e-12:
        raise AssertionError("multiplicative occupancy update violated positivity or budget")
    return updated


def _forward_mode_jacobian_chunked(function, value: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Build a forward-mode Jacobian in bounded tangent batches.

    ``torch.autograd.functional.jacobian(..., strategy="forward-mode")``
    requires ``vectorize=True``.  A non-vectorized forward-mode call is not a
    lower-memory fallback; it is unsupported by PyTorch.  Chunking the input
    tangent basis keeps the renderer's intermediate tensors proportional to
    ``chunk_size`` while still taking exactly one forward tangent per
    parameter overall.
    """
    if value.ndim != 1 or not value.is_floating_point():
        raise ValueError("chunked forward-mode Jacobian requires a floating-point vector")
    if chunk_size < 1:
        raise ValueError("Jacobian chunk_size must be positive")
    base = value.detach()
    parameter_count = int(base.numel())
    blocks = []
    for start in range(0, parameter_count, chunk_size):
        stop = min(start + chunk_size, parameter_count)
        indices = torch.arange(start, stop, dtype=torch.long, device=base.device)
        tangent_block = base[start:stop].clone().requires_grad_(True)

        def block_function(block: torch.Tensor) -> torch.Tensor:
            current = base.index_copy(0, indices, block)
            return function(current)

        blocks.append(torch.autograd.functional.jacobian(
            block_function, tangent_block, create_graph=False,
            vectorize=True, strategy="forward-mode",
        ))
    return torch.cat(blocks, dim=-1)


def joint_jacobian(runner: APrimeSequential, parameters: np.ndarray,
                   state: dict[str, object], normalizer: float,
                   lambdas: np.ndarray,
                   parameterization: SharedJointParameterization,
                   fixed_b_offset: float | None = None,
                   slot2_occupancy_floor: float = 0.0,
                   amplitude_prior_lambda: float = 0.0,
                   amplitude_prior_reference: np.ndarray | None = None,
                   deflation_mode: str = "none",
                   profile: dict[str, float] | None = None) -> np.ndarray:
    """Return the CPU copy of the Torch-resident objective Jacobian."""
    total_started = time.perf_counter()
    device = runner.base.torch_device
    value = torch.tensor(parameters, dtype=torch.float64, device=device, requires_grad=True)
    # H100 measurement on the frozen 4,655-voxel mask: 40 tangents peak at
    # 5.113 GiB allocated / 5.654 GiB reserved, so the full parameter basis
    # is the production default.  Smaller cards can override this env var.
    chunk_size = int(os.environ.get("D1_JACOBIAN_CHUNK_SIZE", "40"))
    if profile is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    jacobian_torch = _forward_mode_jacobian_chunked(
        lambda current: joint_residual_torch(
            runner, current, state, normalizer, lambdas, parameterization,
            fixed_b_offset, amplitude_prior_lambda, amplitude_prior_reference,
        ), value, chunk_size,
    )
    if profile is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_elapsed = time.perf_counter() - forward_started
    transfer_started = time.perf_counter()
    jacobian = jacobian_torch.detach().cpu().numpy()
    transfer_elapsed = time.perf_counter() - transfer_started
    if deflation_mode == "slot2_gradient":
        if not isinstance(parameterization, FullJointParameterization):
            raise ValueError("slot2 gradient deflation requires the full 40-parameter chart")
        gradient = jacobian.T @ state["residual"]
        g1 = gradient[:runner.rotator.ndofs]
        g1_norm = float(np.linalg.norm(g1))
        if g1_norm > 1e-30:
            normal = g1 / g1_norm
            projection = np.eye(runner.rotator.ndofs) - np.outer(normal, normal)
            jacobian[:, runner.rotator.ndofs:2 * runner.rotator.ndofs] = (
                jacobian[:, runner.rotator.ndofs:2 * runner.rotator.ndofs] @ projection
            )
    elif deflation_mode != "none":
        raise ValueError(f"unknown deflation mode: {deflation_mode}")
    if profile is not None:
        profile["jacobian_calls"] = profile.get("jacobian_calls", 0.0) + 1.0
        profile["jacobian_total_s"] = profile.get("jacobian_total_s", 0.0) + (
            time.perf_counter() - total_started
        )
        profile["jacobian_forward_s"] = profile.get("jacobian_forward_s", 0.0) + forward_elapsed
        profile["jacobian_host_device_s"] = profile.get("jacobian_host_device_s", 0.0) + transfer_elapsed
    return jacobian


def joint_residual_torch(runner: APrimeSequential, parameters: torch.Tensor,
                         state: dict[str, object], normalizer: float,
                         lambdas: np.ndarray,
                         parameterization: SharedJointParameterization,
                         fixed_b_offset: float | None = None,
                         amplitude_prior_lambda: float = 0.0,
                         amplitude_prior_reference: np.ndarray | None = None,
                         fixed_intercept: float | None = None,
                         return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate the A′ residual without a device-to-host round trip.

    The returned tensor is suitable for Torch autodiff and is the canonical
    residual used by the parity harness and the future Torch TRF path.  The
    profiled occupancies/intercept arrive as fixed scalars for one inner solve;
    their outer update remains unchanged.
    """
    if parameters.ndim != 1 or parameters.dtype != torch.float64:
        raise ValueError("parameters must be a float64 Torch vector")
    device = parameters.device
    target = torch.as_tensor(runner.target, dtype=torch.float64, device=device)
    weights = torch.as_tensor(state["weights"], dtype=torch.float64, device=device)
    intercept = (torch.as_tensor(fixed_intercept, dtype=torch.float64, device=device)
                 if fixed_intercept is not None else None)
    initial_bb = runner._torch_initial[runner.bb_indices]
    torsions = parameters if fixed_b_offset is not None else parameters[:-1]
    b_offset = (torch.as_tensor(fixed_b_offset, dtype=torch.float64, device=device)
                if fixed_b_offset is not None else parameters[-1])
    coordinates = runner.torch_forward(parameterization.expand_torch(torsions))
    b_factors = torch.stack((
        torch.as_tensor(runner.base.b_factors_a_model, dtype=torch.float64, device=device),
        torch.as_tensor(runner.base.b_factors_b_model, dtype=torch.float64, device=device),
    )) + b_offset
    models = runner.base.model_density_torch(
        coordinates[:, runner.base.model_atom_indices], b_factors=b_factors,
    )
    if runner.training_indices is not None:
        models = models[:, runner.training_indices]
    if intercept is None:
        intercept = torch.mean(target - (weights[:, None] * models).sum(0))
    density = (target - (weights[:, None] * models).sum(0) - intercept) / np.sqrt(normalizer)
    backbone = coordinates[:, runner.bb_indices]
    seam = np.sqrt(runner.rho / 2.0) * (
        runner._torch_seam(initial_bb, backbone) +
        torch.as_tensor(lambdas[:12], dtype=torch.float64, device=device).reshape(2, 6) / runner.rho
    )
    _, omega_delta, _, rama_barriers = runner.torch_omega_and_rama(coordinates)
    rows = [
        density, seam.reshape(-1),
        (np.sqrt(runner.rama_weight) * rama_barriers).reshape(-1),
        # The omega_delta block is the sole omega/planarity residual.  It is
        # the Torch equivalent of planar_rows in joint_evaluate; do not add a
        # second raw-omega block here.
        (np.sqrt(runner.planar_weight) * omega_delta / runner.omega_scale_deg).reshape(-1),
    ]
    if amplitude_prior_lambda > 0.0:
        reference = (torch.zeros((2, runner.rotator.ndofs), dtype=torch.float64, device=device)
                     if amplitude_prior_reference is None else
                     torch.as_tensor(amplitude_prior_reference, dtype=torch.float64, device=device))
        rows.append(np.sqrt(amplitude_prior_lambda) * (
            parameterization.expand_torch(torsions) - reference
        ).reshape(-1))
    residual = torch.cat(rows)
    if return_aux:
        return residual, {
            "coordinates": coordinates,
            "models": models,
            "intercept": intercept,
        }
    return residual


def projected_gradient_norm(gradient: np.ndarray, value: np.ndarray,
                            lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                            tolerance: float = 1e-8) -> float:
    """Return the raw least-squares gradient after active-bound projection."""
    gradient = np.asarray(gradient, dtype=float).copy()
    value = np.asarray(value, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    at_lower = np.isfinite(lower_bounds) & (value <= lower_bounds + tolerance) & (gradient > 0.0)
    at_upper = np.isfinite(upper_bounds) & (value >= upper_bounds - tolerance) & (gradient < 0.0)
    gradient[at_lower | at_upper] = 0.0
    return float(np.linalg.norm(gradient))


def torch_native_trust_region_inner_solve(
    runner: APrimeSequential,
    parameters: np.ndarray,
    normalizer: float,
    lambdas: np.ndarray,
    parameterization: FullJointParameterization,
    fixed_b_offset: float,
    active_slot2_floor: float,
    occupancy_weights: np.ndarray | None,
    amplitude_prior_lambda: float,
    amplitude_prior_reference: np.ndarray | None,
    inner_nfev: int,
    label: str,
    outer: int,
    trajectory: list[dict[str, object]],
    deflation_mode: str,
) -> tuple[np.ndarray, list[dict[str, object]], int]:
    """Run the two independent slot solves entirely through Torch TRF.

    The benchmark's fixed-dB/per-slot mode has no finite torsion bounds, so
    the Torch kernel's projected-bound handling is inactive here.  Occupancy
    weights are held fixed during the inner solve, while the affine intercept
    is profiled analytically in the Torch residual just as in ``joint_evaluate``.
    """
    if not isinstance(parameterization, FullJointParameterization):
        raise ValueError("Torch per-slot TRF requires the full 40-parameter chart")
    if fixed_b_offset is None:
        raise ValueError("Torch per-slot TRF requires fixed dB")
    if occupancy_weights is None:
        raise ValueError("Torch per-slot TRF currently requires mirror occupancies")

    device = runner.base.torch_device
    current = np.asarray(parameters, dtype=float).copy()
    slot_diagnostics: list[dict[str, object]] = []
    total_nfev = 0
    for slot in range(2):
        indices = np.arange(slot * runner.rotator.ndofs, (slot + 1) * runner.rotator.ndofs)
        indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)
        fixed = current.copy()
        fixed_t = torch.as_tensor(fixed, dtype=torch.float64, device=device)
        local_evaluations = 0
        local_trace: list[dict[str, object]] = []
        fixed_state = {
            "weights": np.asarray(occupancy_weights, dtype=float),
            "intercept": 0.0,
        }

        def full_torch(value: torch.Tensor) -> torch.Tensor:
            return fixed_t.index_copy(0, indices_t, value)

        def residual_core(value: torch.Tensor) -> torch.Tensor:
            return joint_residual_torch(
                runner, full_torch(value), fixed_state, normalizer, lambdas,
                parameterization, fixed_b_offset,
                amplitude_prior_lambda, amplitude_prior_reference,
                fixed_intercept=None,
            )

        def log_evaluation(value: torch.Tensor, residual: torch.Tensor,
                           aux: dict[str, torch.Tensor]) -> None:
            nonlocal local_evaluations
            coordinates = aux["coordinates"].detach().cpu().numpy()
            target_size = len(runner.target)
            density_residual = residual[:target_size] * np.sqrt(normalizer)
            rss = float(torch.dot(density_residual, density_residual).item())
            local_evaluations += 1
            trajectory.append({
                "stage": label, "outer_update": outer, "slot": slot + 1,
                "evaluation": local_evaluations,
                "occupancies": np.asarray(occupancy_weights, dtype=float).tolist(),
                "intercept": float(aux["intercept"].item()), "b_offset_A2": float(fixed_b_offset),
                "slot2_occupancy_floor": active_slot2_floor,
                "rss": rss, "slot_pair_rmsd": slot_pair_rmsd(runner, coordinates),
                **slot2_geometry_metrics(runner, coordinates),
            })

        def residual_function(value: torch.Tensor) -> torch.Tensor:
            # Torch's Jacobian evaluation calls this callback with a tracked
            # input.  Count/log only actual residual evaluations, not those
            # internal autodiff probes.
            if value.requires_grad:
                return residual_core(value)
            residual, aux = joint_residual_torch(
                runner, full_torch(value), fixed_state, normalizer, lambdas,
                parameterization, fixed_b_offset,
                amplitude_prior_lambda, amplitude_prior_reference,
                fixed_intercept=None, return_aux=True,
            )
            log_evaluation(value, residual, aux)
            return residual

        def gradient_at(value: torch.Tensor) -> torch.Tensor:
            value = value.detach().requires_grad_(True)
            residual = residual_core(value)
            # Vectorized reverse-mode Jacobians are fast for the small
            # backbone-only mask, but retain the all-atom residual graph and
            # can exceed H100 memory.  The non-vectorized path computes the
            # same autodiff Jacobian with bounded peak memory.
            jacobian_vectorize = os.environ.get(
                "QFIT_JACOBIAN_VECTORIZE", "1"
            ).lower() not in {"0", "false", "no"}
            jacobian = torch.autograd.functional.jacobian(
                residual_core, value, create_graph=False,
                vectorize=jacobian_vectorize,
            )
            return jacobian.T @ residual

        x0 = torch.as_tensor(current[indices], dtype=torch.float64, device=device)
        start_gradient = gradient_at(x0)
        result = torch_least_squares(
            residual_function, x0, max_nfev=inner_nfev,
            initial_radius=1.0, x_scale=10.0,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        current[indices] = result.x.detach().cpu().numpy()
        end_gradient = gradient_at(result.x)
        total_nfev += int(result.nfev)
        slot_diagnostics.append({
            "slot": slot + 1,
            "gradient_norm_start": float(torch.linalg.vector_norm(start_gradient).item()),
            "gradient_norm_end": float(result.optimality),
            "projected_gradient_norm_start": float(torch.linalg.vector_norm(start_gradient).item()),
            "projected_gradient_norm_end": float(result.projected_optimality),
            "termination_status": int(result.status),
            "termination_message": result.message,
            "nfev": int(result.nfev), "njev": int(result.njev),
            "evaluation_cap": int(inner_nfev),
            "hit_evaluation_cap": bool(result.status == 0 and result.nfev >= inner_nfev),
            "trust_radius_trajectory": result.trust_radius_trace,
            "trust_radius_updates": len(result.trust_radius_trace),
            "solver": "torch_trf",
        })
    return current, slot_diagnostics, total_nfev


def per_slot_trust_region_inner_solve(
    runner: APrimeSequential,
    parameters: np.ndarray,
    normalizer: float,
    lambdas: np.ndarray,
    parameterization: FullJointParameterization,
    fixed_b_offset: float,
    active_slot2_floor: float,
    occupancy_weights: np.ndarray | None,
    amplitude_prior_lambda: float,
    amplitude_prior_reference: np.ndarray | None,
    inner_nfev: int,
    label: str,
    outer: int,
    trajectory: list[dict[str, object]],
    deflation_mode: str,
    torch_native_trf: bool = False,
    profile: dict[str, float] | None = None,
    initial_radii: tuple[float | None, float | None] | None = None,
    carry_trust_radii: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]], int, list[float]]:
    """Do two block-coordinate TRF solves with independent trust radii.

    Each slot is optimized with the other slot held fixed.  The global dB is
    fixed in this mode, which is the prospective benchmark configuration
    requested for this audit.  ``torch_native_trf`` selects the CUDA-resident
    solver; the default remains the legacy SciPy path for reproducibility.
    """
    if not isinstance(parameterization, FullJointParameterization):
        raise ValueError("per-slot trust radii require the full 40-parameter chart")
    if fixed_b_offset is None:
        raise ValueError("per-slot trust radii require fixed dB")
    if torch_native_trf:
        if carry_trust_radii:
            raise ValueError("trust-radius carry-over is currently implemented for SciPy TRF only")
        return torch_native_trust_region_inner_solve(
            runner, parameters, normalizer, lambdas, parameterization,
            fixed_b_offset, active_slot2_floor, occupancy_weights,
            amplitude_prior_lambda, amplitude_prior_reference, inner_nfev,
            label, outer, trajectory, deflation_mode,
        ) + ([1.0, 1.0],)

    current = np.asarray(parameters, dtype=float).copy()
    slot_diagnostics: list[dict[str, object]] = []
    ending_radii: list[float] = []
    total_nfev = 0
    for slot in range(2):
        indices = np.arange(slot * runner.rotator.ndofs, (slot + 1) * runner.rotator.ndofs)
        fixed = current.copy()
        local_evaluations = 0
        local_jacobians = 0
        local_trace: list[dict[str, object]] = []
        carried_radius = (
            None if initial_radii is None else initial_radii[slot]
        )
        start_radius = (
            _scipy_default_initial_trust_radius(fixed[indices], 10.0)
            if carried_radius is None else float(carried_radius)
        )

        def full_value(value: np.ndarray) -> np.ndarray:
            result = fixed.copy()
            result[indices] = value
            return result

        def residual_function(value: np.ndarray) -> np.ndarray:
            nonlocal local_evaluations
            full = full_value(value)
            started = time.perf_counter()
            state = joint_evaluate(
                runner, full, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, occupancy_weights,
                amplitude_prior_lambda, amplitude_prior_reference,
            )
            local_evaluations += 1
            trajectory.append({
                "stage": label, "outer_update": outer,
                "slot": slot + 1, "evaluation": local_evaluations,
                "occupancies": state["weights"].tolist(),
                "intercept": state["intercept"], "b_offset_A2": state["b_offset_A2"],
                "slot2_occupancy_floor": active_slot2_floor,
                "rss": state["rss"], "slot_pair_rmsd": slot_pair_rmsd(runner, state["coordinates"]),
                **slot2_geometry_metrics(runner, state["coordinates"]),
            })
            if profile is not None:
                profile["residual_calls"] = profile.get("residual_calls", 0.0) + 1.0
                profile["residual_eval_s"] = profile.get("residual_eval_s", 0.0) + (
                    time.perf_counter() - started
                )
            return state["residual"]

        def jacobian_function(value: np.ndarray) -> np.ndarray:
            nonlocal local_jacobians
            full = full_value(value)
            state_started = time.perf_counter()
            state = joint_evaluate(
                runner, full, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, occupancy_weights,
                amplitude_prior_lambda, amplitude_prior_reference,
            )
            if profile is not None:
                profile["state_eval_calls"] = profile.get("state_eval_calls", 0.0) + 1.0
                profile["state_eval_s"] = profile.get("state_eval_s", 0.0) + (
                    time.perf_counter() - state_started
                )
            full_jacobian = joint_jacobian(
                runner, full, state, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, amplitude_prior_lambda,
                amplitude_prior_reference, deflation_mode,
                profile=profile,
            )
            local_jacobians += 1
            return full_jacobian[:, indices]

        start_state = joint_evaluate(
            runner, fixed, normalizer, lambdas, parameterization,
            fixed_b_offset, active_slot2_floor, occupancy_weights,
            amplitude_prior_lambda, amplitude_prior_reference,
        )
        start_jacobian = jacobian_function(fixed[indices])
        start_gradient = start_jacobian.T @ start_state["residual"]
        result = _least_squares_with_trust_trace(
            residual_function, fixed[indices], method="trf", jac=jacobian_function,
            x_scale=10.0, max_nfev=inner_nfev,
            ftol=1e-10, xtol=1e-10, gtol=1e-10, trace=local_trace,
            initial_radius=carried_radius,
        )
        current[indices] = result.x
        end_full = current.copy()
        end_state = joint_evaluate(
            runner, end_full, normalizer, lambdas, parameterization,
            fixed_b_offset, active_slot2_floor, occupancy_weights,
            amplitude_prior_lambda, amplitude_prior_reference,
        )
        end_jacobian = jacobian_function(result.x)
        end_gradient = end_jacobian.T @ end_state["residual"]
        end_radius = (
            float(local_trace[-1]["radius_after_scaled"])
            if local_trace else start_radius
        )
        ending_radii.append(end_radius)
        accepted_steps = sum(bool(item["accepted"]) for item in local_trace)
        rejected_steps = len(local_trace) - accepted_steps
        total_nfev += int(result.nfev)
        slot_diagnostics.append({
            "slot": slot + 1,
            "gradient_norm_start": float(np.linalg.norm(start_gradient)),
            "gradient_norm_end": float(np.linalg.norm(end_gradient)),
            "projected_gradient_norm_start": float(np.linalg.norm(start_gradient)),
            "projected_gradient_norm_end": float(np.linalg.norm(end_gradient)),
            "termination_status": int(result.status),
            "termination_message": result.message,
            "nfev": int(result.nfev), "njev": int(result.njev or 0),
            "evaluation_cap": int(inner_nfev),
            "hit_evaluation_cap": bool(result.nfev >= inner_nfev and result.status == 0),
            "trust_radius_carry_enabled": bool(carry_trust_radii),
            "trust_radius_start_scaled": start_radius,
            "trust_radius_end_scaled": end_radius,
            "trust_radius_start_degrees_at_x_scale_10": start_radius * 10.0,
            "trust_radius_end_degrees_at_x_scale_10": end_radius * 10.0,
            "trust_radius_trajectory": local_trace,
            "trust_radius_updates": len(local_trace),
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "acceptance_ratio": accepted_steps / max(len(local_trace), 1),
        })
    return current, slot_diagnostics, total_nfev, ending_radii


def joint_run(runner: APrimeSequential, p1: np.ndarray, p2: np.ndarray,
              label: str, output: Path, initial_pair_rmsd: float,
              per_slot_offsets: tuple[int, ...] | None = None,
              fixed_b_offset: float | None = None,
              slot2_occupancy_floor: float = 0.0,
              slot2_floor_outer_updates: int = 0,
              occupancy_scheme: str = "qp",
              mirror_eta: float = 0.0,
              mirror_tau: float = 0.0,
              amplitude_prior_lambda: float = 0.0,
              amplitude_prior_reference: np.ndarray | None = None,
              inner_nfev: int = INNER_NFEV,
              outer_updates: int = OUTER_UPDATES,
              lambda_relative_tolerance: float | None = None,
              lambda_damping_alpha: float = 1.0,
              lambda_norm_cap: float | None = None,
              deflation_mode: str = "none",
              per_slot_trust_radii: bool = False,
              torch_native_trf: bool = False,
              carry_trust_radii: bool = False) -> dict[str, object]:
    if slot2_occupancy_floor < 0.0 or slot2_occupancy_floor >= 1.0:
        raise ValueError("slot2_occupancy_floor must be in [0, 1)")
    if inner_nfev < 1 or outer_updates < 1:
        raise ValueError("inner_nfev and outer_updates must be positive")
    if lambda_relative_tolerance is not None and not 0.0 < lambda_relative_tolerance < 1.0:
        raise ValueError("lambda_relative_tolerance must be in (0, 1)")
    if not 0.0 < lambda_damping_alpha <= 1.0:
        raise ValueError("lambda_damping_alpha must be in (0, 1]")
    if lambda_norm_cap is not None and lambda_norm_cap <= 0.0:
        raise ValueError("lambda_norm_cap must be positive when provided")
    if slot2_floor_outer_updates < 0 or slot2_floor_outer_updates > outer_updates:
        raise ValueError("slot2_floor_outer_updates must be within the outer-update schedule")
    if occupancy_scheme not in {"qp", "mirror", "mirror_entropy"}:
        raise ValueError("occupancy_scheme must be qp, mirror, or mirror_entropy")
    if occupancy_scheme == "qp" and (mirror_eta != 0.0 or mirror_tau != 0.0):
        raise ValueError("mirror parameters are only valid for mirror occupancy schemes")
    if occupancy_scheme != "qp" and mirror_eta <= 0.0:
        raise ValueError("mirror occupancy schemes require a positive eta")
    if mirror_tau < 0.0:
        raise ValueError("mirror entropy tau must be non-negative")
    if amplitude_prior_lambda < 0.0:
        raise ValueError("amplitude prior lambda must be non-negative")
    if deflation_mode not in {"none", "slot2_gradient"}:
        raise ValueError("deflation_mode must be none or slot2_gradient")
    if per_slot_trust_radii and fixed_b_offset is None:
        raise ValueError("per-slot trust radii are currently benchmarked with dB fixed")
    if per_slot_trust_radii and per_slot_offsets is not None:
        raise ValueError("per-slot trust radii require the full 40-parameter chart")
    output.mkdir(parents=True, exist_ok=True)
    parameterization = (
        FullJointParameterization(runner.rotator.ndofs)
        if per_slot_offsets is None
        else SharedJointParameterization(runner.rotator.ndofs, per_slot_offsets)
    )
    aa_models = runner.base.model_density_batch(
        np.stack((runner.initial, runner.initial)), slots=np.array((0, 1)),
    )
    if runner.training_indices is not None:
        aa_models = aa_models[:, runner.training_indices]
    _, _, normalizer = runner.joint_qp_weights(runner.target, aa_models)
    normalizer = max(float(normalizer), 1e-12)
    parameters = parameterization.pack(p1, p2)
    if fixed_b_offset is None:
        parameters = np.concatenate((parameters, np.array((0.0,))))
    lambdas = np.zeros(12)
    occupancy_weights = (None if occupancy_scheme == "qp"
                         else np.asarray((0.5, 0.5), dtype=float))
    carried_trust_radii: list[float | None] = [None, None]
    trajectory = []
    inner_diagnostics = []
    for outer in range(1, outer_updates + 1):
        outer_started = time.perf_counter()
        profile = {
            "outer_update": float(outer),
            "jacobian_calls": 0.0,
            "jacobian_total_s": 0.0,
            "jacobian_forward_s": 0.0,
            "jacobian_host_device_s": 0.0,
            "residual_calls": 0.0,
            "residual_eval_s": 0.0,
            "state_eval_calls": 0.0,
            "state_eval_s": 0.0,
        }
        evaluations = 0
        active_slot2_floor = (
            float(slot2_occupancy_floor)
            if outer <= slot2_floor_outer_updates else 0.0
        )

        def residual_function(value):
            nonlocal evaluations
            state = joint_evaluate(
                runner, value, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor,
                occupancy_weights,
                amplitude_prior_lambda, amplitude_prior_reference,
            )
            evaluations += 1
            trajectory.append({"stage": label, "outer_update": outer, "evaluation": evaluations,
                               "occupancies": state["weights"].tolist(), "intercept": state["intercept"],
                               "b_offset_A2": state["b_offset_A2"],
                               "slot2_occupancy_floor": active_slot2_floor,
                               "rss": state["rss"],
                               **slot2_geometry_metrics(runner, state["coordinates"])})
            return state["residual"]

        def jacobian_function(value):
            return joint_jacobian(
                runner, value,
                joint_evaluate(
                    runner, value, normalizer, lambdas, parameterization,
                    fixed_b_offset, active_slot2_floor,
                    occupancy_weights,
                    amplitude_prior_lambda, amplitude_prior_reference,
                ),
                normalizer, lambdas, parameterization, fixed_b_offset,
                active_slot2_floor,
                amplitude_prior_lambda, amplitude_prior_reference,
                deflation_mode,
            )

        lower_bounds = np.full(len(parameters), -np.inf)
        upper_bounds = np.full(len(parameters), np.inf)
        if fixed_b_offset is None:
            lower_bounds[-1] = -min(
                float(runner.base.b_factors_a_model.min()),
                float(runner.base.b_factors_b_model.min()),
            ) + 1e-5
            # This upper safety bound matches the previous profile search range.
            # Hitting it is reported as a model-quality failure, never silently
            # interpreted as a physical B-factor estimate.
            upper_bounds[-1] = 200.0
        start_gradient_holder = {}
        lambda_before = lambdas.copy()

        def jacobian_with_start_diagnostic(value):
            jacobian = jacobian_function(value)
            if not start_gradient_holder:
                start_state = joint_evaluate(
                    runner, value, normalizer, lambdas, parameterization,
                    fixed_b_offset, active_slot2_floor, occupancy_weights,
                    amplitude_prior_lambda, amplitude_prior_reference,
                )
                start_gradient_holder["gradient"] = jacobian.T @ start_state["residual"]
            return jacobian

        slot_inner_diagnostics = []
        if per_slot_trust_radii:
            pre_state = joint_evaluate(
                runner, parameters, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, occupancy_weights,
                amplitude_prior_lambda, amplitude_prior_reference,
            )
            pre_jacobian = joint_jacobian(
                runner, parameters, pre_state, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, amplitude_prior_lambda,
                amplitude_prior_reference, deflation_mode, profile=profile,
            )
            start_gradient = pre_jacobian.T @ pre_state["residual"]
            trust_started = time.perf_counter()
            parameters, slot_inner_diagnostics, total_nfev, ending_radii = per_slot_trust_region_inner_solve(
                runner, parameters, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, occupancy_weights,
                amplitude_prior_lambda, amplitude_prior_reference, inner_nfev,
                label, outer, trajectory, deflation_mode, torch_native_trf,
                profile=profile,
                initial_radii=(tuple(carried_trust_radii) if carry_trust_radii else None),
                carry_trust_radii=carry_trust_radii,
            )
            if carry_trust_radii:
                carried_trust_radii = [float(value) for value in ending_radii]
            profile["scipy_trust_region_s"] = time.perf_counter() - trust_started
            result_nfev = total_nfev
            result_njev = sum(int(item["njev"]) for item in slot_inner_diagnostics)
            result_status = max(int(item["termination_status"]) for item in slot_inner_diagnostics)
            result_message = "; ".join(
                f"slot{item['slot']}: {item['termination_message']}"
                for item in slot_inner_diagnostics
            )
            result_active_mask = np.zeros_like(parameters, dtype=int)
            result_optimality = max(
                float(item["projected_gradient_norm_end"])
                for item in slot_inner_diagnostics
            )
        else:
            result = least_squares(
                residual_function, parameters, method="trf", jac=jacobian_with_start_diagnostic,
                x_scale=10.0, bounds=(lower_bounds, upper_bounds), max_nfev=inner_nfev,
                ftol=1e-10, xtol=1e-10, gtol=1e-10,
            )
            parameters = result.x.copy()
            start_gradient = np.asarray(start_gradient_holder["gradient"], dtype=float)
            result_nfev = int(result.nfev)
            result_njev = int(result.njev) if result.njev is not None else 0
            result_status = int(result.status)
            result_message = result.message
            result_active_mask = np.asarray(result.active_mask, dtype=int)
            result_optimality = float(result.optimality)
        state = joint_evaluate(
            runner, parameters, normalizer, lambdas, parameterization,
            fixed_b_offset, active_slot2_floor, occupancy_weights,
            amplitude_prior_lambda, amplitude_prior_reference,
        )
        lambda_increment = lambda_damping_alpha * runner.rho * state["seam_vectors"].reshape(-1)
        lambda_candidate = lambdas + lambda_increment
        lambda_cap_active = False
        if lambda_norm_cap is not None:
            candidate_norm = float(np.linalg.norm(lambda_candidate))
            if candidate_norm > lambda_norm_cap:
                lambda_candidate *= float(lambda_norm_cap) / candidate_norm
                lambda_cap_active = True
        lambdas = lambda_candidate
        if per_slot_trust_radii:
            end_jacobian = joint_jacobian(
                runner, parameters, state, normalizer, lambdas, parameterization,
                fixed_b_offset, active_slot2_floor, amplitude_prior_lambda,
                amplitude_prior_reference, deflation_mode, profile=profile,
            )
            end_gradient = end_jacobian.T @ state["residual"]
        else:
            end_gradient = np.asarray(result.grad, dtype=float)
        lambda_delta = lambdas - lambda_before
        inner_diagnostics.append({
            "outer_update": outer,
            "gradient_norm_start": float(np.linalg.norm(start_gradient)),
            "projected_gradient_norm_start": projected_gradient_norm(
                start_gradient, parameters if False else parameters,
                lower_bounds, upper_bounds,
            ),
            "gradient_norm_end": float(np.linalg.norm(end_gradient)),
            "projected_gradient_norm_end": projected_gradient_norm(
                end_gradient, parameters, lower_bounds, upper_bounds,
            ),
            "scipy_optimality_end": result_optimality,
            "termination_status": result_status,
            "termination_message": result_message,
            "nfev": result_nfev,
            "njev": result_njev,
            "evaluation_cap": int(inner_nfev),
            "hit_evaluation_cap": bool(
                any(item["hit_evaluation_cap"] for item in slot_inner_diagnostics)
                if per_slot_trust_radii else result_nfev >= inner_nfev and result_status == 0
            ),
            "active_mask": result_active_mask.tolist(),
            "per_slot_trust_radii": bool(per_slot_trust_radii),
            "torch_native_trf": bool(torch_native_trf),
            "carry_trust_radii": bool(carry_trust_radii),
            "timing": profile,
            "slot_inner_diagnostics": slot_inner_diagnostics,
            "seam_norm_before_lambda_update": float(np.linalg.norm(state["seam_vectors"])),
            "lambda_norm_before": float(np.linalg.norm(lambda_before)),
            "lambda_norm_after": float(np.linalg.norm(lambdas)),
            "lambda_delta_norm": float(np.linalg.norm(lambda_delta)),
            "lambda_damping_alpha": float(lambda_damping_alpha),
            "lambda_norm_cap": None if lambda_norm_cap is None else float(lambda_norm_cap),
            "lambda_cap_active": bool(lambda_cap_active),
        })
        trajectory.append({"stage": label, "outer_update": outer, "event": "AL_update",
                           "occupancies": state["weights"].tolist(), "intercept": state["intercept"],
                           "b_offset_A2": state["b_offset_A2"],
                           "slot2_occupancy_floor": active_slot2_floor,
                           "rss": state["rss"], **slot2_geometry_metrics(runner, state["coordinates"]),
                           "lm_nfev": result_nfev,
                           "lm_status": result_status, "lm_message": result_message,
                           "inner_gradient_norm_start": inner_diagnostics[-1]["gradient_norm_start"],
                           "inner_gradient_norm_end": inner_diagnostics[-1]["gradient_norm_end"],
                           "inner_projected_gradient_norm_end": inner_diagnostics[-1]["projected_gradient_norm_end"],
                           "lambda_after_update": lambdas.tolist(),
                           "lambda_delta_norm": inner_diagnostics[-1]["lambda_delta_norm"],
                           "lambda_cap_active": inner_diagnostics[-1]["lambda_cap_active"],
                           "slot_pair_rmsd": slot_pair_rmsd(runner, state["coordinates"]),
                           "slot_inner_diagnostics": slot_inner_diagnostics})
        if occupancy_scheme != "qp":
            occupancy_started = time.perf_counter()
            occupancy_weights = mirror_descent_occupancy_update(
                runner.target, state["models"], occupancy_weights,
                state["intercept"], mirror_eta,
                mirror_tau if occupancy_scheme == "mirror_entropy" else 0.0,
            )
            state = joint_evaluate(
                runner, parameters, normalizer, lambdas, parameterization,
                fixed_b_offset, 0.0, occupancy_weights,
                amplitude_prior_lambda, amplitude_prior_reference,
            )
            trajectory.append({"stage": label, "outer_update": outer, "event": "mirror_update",
                               "occupancies": state["weights"].tolist(), "intercept": state["intercept"],
                               "b_offset_A2": state["b_offset_A2"],
                               "slot2_occupancy_floor": 0.0, "rss": state["rss"],
                               **slot2_geometry_metrics(runner, state["coordinates"]),
                               "mirror_eta": float(mirror_eta), "mirror_tau": float(mirror_tau)})
            profile["occupancy_update_s"] = time.perf_counter() - occupancy_started
        else:
            profile["occupancy_update_s"] = 0.0
        profile["outer_wall_s"] = time.perf_counter() - outer_started
        profile["python_orchestration_s"] = max(
            0.0,
            profile["outer_wall_s"] - profile.get("scipy_trust_region_s", 0.0)
            - profile.get("occupancy_update_s", 0.0),
        )
        profile["jacobian_python_s"] = max(
            0.0,
            profile.get("jacobian_total_s", 0.0)
            - profile.get("jacobian_forward_s", 0.0)
            - profile.get("jacobian_host_device_s", 0.0),
        )
        accepted_steps = sum(int(item.get("accepted_steps", 0)) for item in slot_inner_diagnostics)
        rejected_steps = sum(int(item.get("rejected_steps", 0)) for item in slot_inner_diagnostics)
        trust_checkpoint = {
            "status": "running",
            "outer_update": outer,
            "profile": profile,
            "accept_reject": {
                "accepted_steps": accepted_steps,
                "rejected_steps": rejected_steps,
                "total_steps": accepted_steps + rejected_steps,
                "acceptance_ratio": accepted_steps / max(accepted_steps + rejected_steps, 1),
            },
            "trust_radius": {
                "carry_enabled": bool(carry_trust_radii),
                "slots": [
                    {
                        "slot": int(item["slot"]),
                        "start_scaled": float(item["trust_radius_start_scaled"]),
                        "end_scaled": float(item["trust_radius_end_scaled"]),
                        "start_degrees_at_x_scale_10": float(item["trust_radius_start_degrees_at_x_scale_10"]),
                        "end_degrees_at_x_scale_10": float(item["trust_radius_end_degrees_at_x_scale_10"]),
                        "trace": item["trust_radius_trajectory"],
                    }
                    for item in slot_inner_diagnostics
                ],
            },
            "inner_solve": inner_diagnostics[-1],
            "lambda": {
                "norm_before": float(inner_diagnostics[-1]["lambda_norm_before"]),
                "norm_after": float(inner_diagnostics[-1]["lambda_norm_after"]),
                "delta_norm": float(inner_diagnostics[-1]["lambda_delta_norm"]),
            },
        }
        timing_path = output / f"timing_outer_{outer:03d}.json"
        timing_temporary = timing_path.with_suffix(".json.tmp")
        timing_temporary.write_text(json.dumps(profile, indent=2, sort_keys=True))
        timing_temporary.replace(timing_path)
        checkpoint_path = output / f"checkpoint_outer_{outer:03d}.json"
        checkpoint_temporary = checkpoint_path.with_suffix(".json.tmp")
        checkpoint_temporary.write_text(json.dumps(trust_checkpoint, indent=2, sort_keys=True))
        checkpoint_temporary.replace(checkpoint_path)
        with (output / "progress.json").open("w") as handle:
            json.dump({
                "status": "running", "outer_update": outer, "trajectory_rows": len(trajectory),
                "checkpoint": checkpoint_path.name,
                "accept_reject": trust_checkpoint["accept_reject"],
                "trust_radius": trust_checkpoint["trust_radius"],
            }, handle)
        if (lambda_relative_tolerance is not None and outer >= 2
                and inner_diagnostics[-1]["lambda_delta_norm"] <=
                lambda_relative_tolerance * max(inner_diagnostics[-1]["lambda_norm_after"], 1e-12)):
            break

    # The final endpoint and selection are always released from the temporary
    # floor, so a floor-supported trajectory cannot manufacture the verdict.
    final = joint_evaluate(
        runner, parameters, normalizer, lambdas, parameterization,
        fixed_b_offset, 0.0, occupancy_weights,
        amplitude_prior_lambda, amplitude_prior_reference,
    )
    coords = final["coordinates"]
    final_selection = select_decoupled_affine_miqp(
        runner.target, final["models"], cardinality_cap=DEFAULT_CARDINALITY_CAP,
        t_min=DEFAULT_MIN_OCCUPANCY, n_atoms=len(runner.base.a_residue.coor),
    )
    # Preserve the endpoint for read-only validation (blocked RSS and physics).
    # This is diagnostic output only; it does not affect the objective.
    provenance = fit_voxel_provenance(runner.training_indices, len(runner.base.target))
    np.savez_compressed(
        output / "final_slots.npz",
        slot1_window=coords[0],
        slot2_window=coords[1],
        parameters=np.concatenate((parameters, np.array((final["b_offset_A2"],)))) if fixed_b_offset is not None else parameters,
        full_parameters=parameterization.expand_numpy(parameters if fixed_b_offset is not None else parameters[:-1]),
        b_offset_A2=float(final["b_offset_A2"]),
        lambdas=lambdas,
        fit_voxel_indices=(np.asarray(provenance["indices"], dtype=np.int64)
                           if provenance["indices"] is not None else np.empty(0, dtype=np.int64)),
    )
    slot_rmsds = []
    for coordinate in coords:
        central = runner.base.central_backbone(coordinate)
        slot_rmsds.append({"to_A": float(rmsd(central, runner.a_backbone)),
                           "to_B": float(rmsd(central, runner.b_backbone))})
    min_occ = np.min(np.asarray([row["occupancies"] for row in trajectory if "occupancies" in row]), axis=0)
    result_json = {
        "status": "complete", "label": label, "initial_slot_to_slot_rmsd_A": initial_pair_rmsd,
        "final_occupancies": final["weights"].tolist(), "final_intercept": final["intercept"],
        "final_b_offset_A2": final["b_offset_A2"],
        "final_rss": final["rss"], "final_energy": final["energy"],
        "final_selection": final_selection, "slot_rmsds": slot_rmsds,
        "final_seam_vectors": np.asarray(state["seam_vectors"]).tolist(),
        "outer_updates_completed": len(inner_diagnostics),
        "lambda_relative_tolerance": lambda_relative_tolerance,
        "lambda_damping_alpha": float(lambda_damping_alpha),
        "lambda_norm_cap": None if lambda_norm_cap is None else float(lambda_norm_cap),
        "lambda_stop_reached": bool(
            lambda_relative_tolerance is not None and len(inner_diagnostics) >= 2 and
            inner_diagnostics[-1]["lambda_delta_norm"] <=
            lambda_relative_tolerance * max(inner_diagnostics[-1]["lambda_norm_after"], 1e-12)
        ),
        "inner_solve_diagnostics": inner_diagnostics,
        "minimum_occupancy_seen": min_occ.tolist(), "trajectory": trajectory,
        "normalizer_initial_A_A_rss": normalizer,
        "fit_provenance": {
            **{key: value for key, value in provenance.items() if key != "indices"},
            "fit_voxel_indices_artifact": "final_slots.npz:fit_voxel_indices",
        },
            "parameterization": {
            "shared_outer_offsets": list(parameterization.shared_offsets),
            "per_slot_offsets": list(parameterization.per_slot_offsets),
            "shared_omega": bool(parameterization.shared_omega),
            "torsion_parameters": parameterization.reduced_ndofs,
            "global_B_offset_parameters": 1,
            "global_B_offset_fixed_A2": fixed_b_offset,
            "slot2_occupancy_floor": float(slot2_occupancy_floor),
            "slot2_floor_outer_updates": int(slot2_floor_outer_updates),
            "inner_nfev": int(inner_nfev),
            "outer_updates": int(outer_updates),
            "slot2_floor_released_for_outer_updates": list(range(
                slot2_floor_outer_updates + 1, outer_updates + 1
            )),
            "total_parameters": parameterization.reduced_ndofs + 1,
            "occupancies_and_intercept": "profiled outside the torsion/B gradient",
            "occupancy_scheme": occupancy_scheme,
            "mirror_eta": float(mirror_eta),
            "mirror_tau": float(mirror_tau),
            "amplitude_prior_lambda": float(amplitude_prior_lambda),
            "lambda_damping_alpha": float(lambda_damping_alpha),
            "lambda_norm_cap": None if lambda_norm_cap is None else float(lambda_norm_cap),
            "amplitude_prior_reference": "zero neutral-start torsion vector" if amplitude_prior_reference is None else "explicit",
            "amplitude_prior_energy_in_final": float(final["amplitude_prior_energy"]),
            "deflation_mode": deflation_mode,
            "per_slot_trust_radii": bool(per_slot_trust_radii),
            "carry_trust_radii": bool(carry_trust_radii),
            "torch_native_trf": bool(torch_native_trf),
            "trust_radius_definition": (
                (
                    "independent Torch TRF solve per slot; radius and actual/predicted "
                    "reduction ratio are adapted separately"
                    if torch_native_trf else
                    "independent SciPy TRF solve per slot; radius and actual/predicted "
                    "reduction ratio are adapted separately"
                )
                if per_slot_trust_radii else "single joint SciPy TRF solve"
            ),
        },
        "fixed_benchmark_contract": {
            "rama_floor": D1_RAMA_FLOOR,
            "nullspace_label": label if label == "D_null_axis2_30deg" else None,
            "b_factor_mode": runner.base.b_factor_mode,
            "per_slot_b_factor_refinement": False,
        },
    }
    with (output / "result.json").open("w") as handle:
        json.dump(result_json, handle, indent=2)
    with (output / "progress.json").open("w") as handle:
        json.dump({"status": "complete", "trajectory_rows": len(trajectory)}, handle)
    return result_json


def sequential_run(output: Path, site: tuple[str, str, int] = SITE,
                   mask_scope: str = "central", rama_floor: float = D1_RAMA_FLOOR,
                   training_indices=None, start_pdb=None, b_factor_mode=None,
                   device: str = "auto") -> dict[str, object]:
    runner = APrimeSequential(output, INNER_NFEV, OUTER_UPDATES, *site,
                              renderer_backend="torch", residual_scale_mode="none",
                              map_scaler_structure="full", mask_scope=mask_scope,
                              training_indices=training_indices, device=device,
                              start_pdb=start_pdb, b_factor_mode=b_factor_mode)
    runner.rama_floor = D1_RAMA_FLOOR
    result = runner.run()
    values = np.asarray([row["occupancy"] for row in runner.trajectory if "occupancy" in row], dtype=float)
    final_slots = np.load(output / "final_slots.npz")
    initial_rmsd = float(rmsd(runner.base.central_backbone(final_slots["deposited_A_window"]),
                              runner.base.central_backbone(final_slots["slot2_window"])))
    return {"label": "A_sequential", "initial_slot_to_slot_rmsd_A": initial_rmsd,
            "final_occupancies": result["final_joint_occupancies_slot1_slot2"],
            "final_intercept": result["final_joint_qp_intercept"],
            "final_rss": result["final_joint_qp_rss"],
            "slot_rmsds": [{"to_A": result["slots"][name]["rmsd_to_A_A"],
                            "to_B": result["slots"][name]["rmsd_to_B_A"]}
                           for name in ("slot1", "slot2")],
            "minimum_occupancy_seen": [float(np.min(values)), float(np.min(values))],
            "result": result}


def worker(spec: dict[str, object]) -> dict[str, object]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    output = Path(spec["output"])
    if spec["kind"] == "A":
        return sequential_run(output, tuple(spec.get("site", SITE)),
                              spec.get("mask_scope", "central"), spec.get("rama_floor", D1_RAMA_FLOOR),
                              spec.get("training_indices"), spec.get("start_pdb"),
                              spec.get("b_factor_mode"), spec.get("device", "auto"))
    site = tuple(spec.get("site", SITE))
    mask_scope = str(spec.get("mask_scope", "central"))
    inner_nfev = int(spec.get("inner_nfev", INNER_NFEV))
    outer_updates = int(spec.get("outer_updates", OUTER_UPDATES))
    runner = APrimeSequential(output, inner_nfev, outer_updates, *site,
                              renderer_backend="torch", residual_scale_mode="none",
                              map_scaler_structure="full", mask_scope=mask_scope,
                              training_indices=spec.get("training_indices"), device=spec.get("device", "auto"),
                              start_pdb=spec.get("start_pdb"),
                              b_factor_mode=spec.get("b_factor_mode"),
                              density_atom_scope=str(spec.get("density_atom_scope", "backbone")))
    runner.rama_floor = D1_RAMA_FLOOR
    p1 = np.zeros(runner.rotator.ndofs)
    p2 = np.asarray(spec["p2"], dtype=float)
    per_slot_value = spec.get("per_slot_offsets")
    per_slot_offsets = None if per_slot_value is None else tuple(per_slot_value)
    parameterization = (
        FullJointParameterization(runner.rotator.ndofs)
        if per_slot_offsets is None
        else SharedJointParameterization(runner.rotator.ndofs, per_slot_offsets)
    )
    initial = runner.torch_forward(
        parameterization.expand_torch(torch.as_tensor(parameterization.pack(p1, p2), dtype=torch.float64))
    ).detach().cpu().numpy()
    initial_rmsd = float(rmsd(runner.base.central_backbone(initial[0]), runner.base.central_backbone(initial[1])))
    return joint_run(
        runner, p1, p2, str(spec["label"]), output, initial_rmsd,
        per_slot_offsets, fixed_b_offset=spec.get("fixed_b_offset"),
        slot2_occupancy_floor=float(spec.get("slot2_occupancy_floor", 0.0)),
        slot2_floor_outer_updates=int(spec.get("slot2_floor_outer_updates", 0)),
        occupancy_scheme=str(spec.get("occupancy_scheme", "qp")),
        mirror_eta=float(spec.get("mirror_eta", 0.0)),
        mirror_tau=float(spec.get("mirror_tau", 0.0)),
        amplitude_prior_lambda=float(spec.get("amplitude_prior_lambda", 0.0)),
        inner_nfev=inner_nfev,
        outer_updates=outer_updates,
        lambda_relative_tolerance=(None if spec.get("lambda_relative_tolerance") is None
                                   else float(spec["lambda_relative_tolerance"])),
        lambda_damping_alpha=float(spec.get("lambda_damping_alpha", 1.0)),
        lambda_norm_cap=(None if spec.get("lambda_norm_cap") is None
                         else float(spec["lambda_norm_cap"])),
        deflation_mode=str(spec.get("deflation_mode", "none")),
        per_slot_trust_radii=bool(spec.get("per_slot_trust_radii", False)),
        torch_native_trf=bool(spec.get("torch_native_trf", False)),
        carry_trust_radii=bool(spec.get("carry_trust_radii", False)),
    )


def build_specs(output_root: Path, flip_root: Path,
                per_slot_offsets: tuple[int, ...] | None = None,
                site: tuple[str, str, int] = SITE,
                mask_scope: str = "central", rama_floor: float = D1_RAMA_FLOOR,
                start_pdb: str | Path | None = None,
                b_factor_mode: str | None = None,
                density_atom_scope: str = "backbone",
                device: str = "auto",
                slot2_occupancy_floor: float = 0.0,
                slot2_floor_outer_updates: int = 0,
                occupancy_scheme: str = "qp",
                mirror_eta: float = 0.0,
                mirror_tau: float = 0.0,
              amplitude_prior_lambda: float = 0.0,
              lambda_relative_tolerance: float | None = None,
              lambda_damping_alpha: float = 1.0,
              lambda_norm_cap: float | None = None,
              inner_nfev: int = INNER_NFEV,
                outer_updates: int = OUTER_UPDATES,
                per_slot_trust_radii: bool = False,
                torch_native_trf: bool = False,
                carry_trust_radii: bool = False) -> list[dict[str, object]]:
    if abs(float(rama_floor) - D1_RAMA_FLOOR) > 1e-12:
        raise ValueError(
            f"benchmark Rama floor is fixed at calibrated value {D1_RAMA_FLOOR}"
        )
    seed_runner = APrimeSequential(output_root / "seed", INNER_NFEV, OUTER_UPDATES, *site,
                                   renderer_backend="torch", residual_scale_mode="none",
                                   map_scaler_structure="full", mask_scope=mask_scope,
                                   start_pdb=start_pdb, b_factor_mode=b_factor_mode,
                                   density_atom_scope=density_atom_scope,
                                   device=device)
    seed_runner.rama_floor = D1_RAMA_FLOOR
    b_window = seed_runner.base.window_for_deposited_b()
    specs = [{"kind": "A", "label": "A_sequential", "output": str(output_root / "A_sequential"),
              "p2": np.zeros(seed_runner.rotator.ndofs).tolist()}]
    specs.append({"kind": "joint", "label": "B_joint_deposited_B",
                  "output": str(output_root / "B_joint_deposited_B"),
                  "p2": inverse_seed(seed_runner, b_window).tolist()})
    for label, target in flip_targets(seed_runner, flip_root):
        specs.append({"kind": "joint", "label": f"C_{label}",
                      "output": str(output_root / f"C_{label}"),
                      "p2": inverse_seed(seed_runner, target).tolist()})

    selection = np.sort(seed_runner.window.select("name", ("N", "CA", "C")))
    closure_basis = null_space(compute_jacobian(seed_runner.window.get_xyz(selection)))
    oxygen_indices = np.array([
        int(np.searchsorted(seed_runner.window.selection, int(residue.select("name", "O")[0])))
        for residue in seed_runner.window.residues
    ])
    def oxygen_forward(value):
        coordinates = seed_runner.torch_forward(value.reshape(1, -1))[0]
        return coordinates[oxygen_indices].reshape(-1)
    j_o = torch.autograd.functional.jacobian(
        oxygen_forward, torch.zeros(seed_runner.rotator.ndofs, dtype=torch.float64),
        vectorize=True, strategy="forward-mode",
    ).detach().cpu().numpy()
    # compute_jacobian is defined on the 14 phi/psi coordinates; the six
    # internal omega parameters are held at zero for this carbonyl-null seed.
    _, _, vt = np.linalg.svd(j_o[:, :14] @ closure_basis, full_matrices=False)
    for axis in range(min(3, vt.shape[0])):
        phi_psi_direction = closure_basis @ vt[axis]
        phi_psi_direction /= np.linalg.norm(phi_psi_direction)
        direction = np.zeros(seed_runner.rotator.ndofs)
        direction[:14] = phi_psi_direction
        if axis == 1:
            step = 30.0
            specs.append({"kind": "joint", "label": "D_null_axis2_30deg",
                          "output": str(output_root / "D_null_axis2_30deg"),
                          "p2": (step * direction).tolist()})
    if per_slot_offsets is not None:
        for spec in specs:
            spec["per_slot_offsets"] = list(per_slot_offsets)
    for spec in specs:
        spec["site"] = list(site)
        spec["mask_scope"] = mask_scope
        spec["rama_floor"] = float(rama_floor)
        spec["start_pdb"] = None if start_pdb is None else str(start_pdb)
        spec["b_factor_mode"] = b_factor_mode
        spec["density_atom_scope"] = density_atom_scope
        spec["device"] = device
        spec["slot2_occupancy_floor"] = float(slot2_occupancy_floor)
        spec["slot2_floor_outer_updates"] = int(slot2_floor_outer_updates)
        spec["occupancy_scheme"] = occupancy_scheme
        spec["mirror_eta"] = float(mirror_eta)
        spec["mirror_tau"] = float(mirror_tau)
        spec["amplitude_prior_lambda"] = float(amplitude_prior_lambda)
        spec["lambda_relative_tolerance"] = lambda_relative_tolerance
        spec["lambda_damping_alpha"] = float(lambda_damping_alpha)
        spec["lambda_norm_cap"] = None if lambda_norm_cap is None else float(lambda_norm_cap)
        spec["inner_nfev"] = int(inner_nfev)
        spec["outer_updates"] = int(outer_updates)
        spec["per_slot_trust_radii"] = bool(per_slot_trust_radii)
        spec["torch_native_trf"] = bool(torch_native_trf)
        spec["carry_trust_radii"] = bool(carry_trust_radii)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--flip-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--pdb-id", default=SITE[0])
    parser.add_argument("--chain", default=SITE[1])
    parser.add_argument("--resnum", type=int, default=SITE[2])
    parser.add_argument("--mask-scope", choices=("central", "window"), default="central")
    parser.add_argument("--rama-floor", type=float, default=0.02)
    parser.add_argument(
        "--per-slot-offsets", type=parse_per_slot_offsets, default=None,
        help="optional comma-separated window offsets with independent phi/psi per slot; default is full 40-parameter chart",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    site = (args.pdb_id, args.chain, args.resnum)
    specs = build_specs(args.output_root, args.flip_root, args.per_slot_offsets,
                        site, args.mask_scope, args.rama_floor)
    with (args.output_root / "specs.json").open("w") as handle:
        json.dump(specs, handle, indent=2)
    results = []
    # The parent builds Torch nullspace initializations before dispatching.
    # Forking after Torch has initialized its thread pools can leave children
    # permanently blocked on inherited futexes; spawn gives each worker a clean
    # Torch runtime.
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(worker, spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
            with (args.output_root / "progress.json").open("w") as handle:
                json.dump({"status": "running", "completed": len(results), "total": len(specs)}, handle)
    results.sort(key=lambda item: item["label"])
    with (args.output_root / "summary.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    with (args.output_root / "progress.json").open("w") as handle:
        json.dump({"status": "complete", "completed": len(results), "total": len(specs)}, handle)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
