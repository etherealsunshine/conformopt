from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path, PosixPath
from types import SimpleNamespace

import gemmi
import numpy as np
import torch
import torch.nn.functional as F

from experiments.probe4.core import dihedral, torsion_to_coords, wrap_angles

from .clash_environment import (
    OPTIMIZER_PHYSICS_ENVIRONMENT_RULE,
    SoftEnvironmentRecord,
    normalized_altloc,
    partition_soft_environment,
    soft_clash_barrier_penalty,
    soft_clash_penalty,
)
from .data_pipeline import (
    _calculate_fcalc,
    _grid_coordinates,
    _omit_map,
    _sidechain_atoms,
    discover_sites,
    extract_patch,
    normalize_patch,
    synthetic_patch,
)
from .model import ResidualDensityDenoiser
from .residue_geometry import (
    CHI_SPECS,
    canonical_centers_radians,
    canonical_width_degrees,
    reference_permutations,
    symmetry_aware_rmsd,
)


# Dunbrack 2010 backbone-dependent ARG states at the nearest 10-degree grid
# point to 3A1C B/ARG447 (phi=-76.90, psi=-29.88 -> grid -80, -30), sorted
# by probability. Source columns are probability followed by mean chi angles.
DUNBRACK_3A1C_ARG447_TOP10 = (
    (0.151778, (-69.1, 178.4, -179.9, 174.7)),
    (0.079612, (-68.4, 177.7, 66.1, -167.7)),
    (0.073363, (-68.2, 178.8, -67.9, 170.8)),
    (0.072473, (-70.0, -168.5, -64.2, -87.4)),
    (0.069068, (-69.5, -178.3, -176.5, -84.3)),
    (0.057179, (-68.9, -178.4, 178.6, 88.0)),
    (0.041968, (-69.6, 176.9, 63.5, 84.6)),
    (0.036420, (-65.2, -65.8, -175.8, -173.5)),
    (0.032639, (-175.5, 173.7, 178.1, -178.0)),
    (0.026876, (-65.4, -65.5, -66.0, 167.2)),
)

OCCUPANCY_FREEZE_IMPLEMENTATION = "adam_group_lr_zero_warm_state_v1"
INITIALIZATION_MODES = (
    "deposited_a_cloud_60",
    "canonical_stratified_free",
    "canonical_stratified_a_anchor",
    "deposited_a_cloud_120",
)
DEFAULT_INITIALIZATION_MODE = INITIALIZATION_MODES[0]
DENSITY_MASK_MODES = (
    "sphere",
    "reachable_volume",
    "containing_volume",
)
DENSITY_WEIGHT_MODES = ("uniform", "reachable_variance")
RESPAWN_GRAM_CONDITION_THRESHOLD = 100.0
RESPAWN_OCCUPANCY_FLOOR = 1e-6
RESPAWN_IK_STEPS = 50
RESPAWN_IK_LEARNING_RATE = 0.1
RESPAWN_PEAK_REACHED_THRESHOLD = 0.5


def _unique_canonical_centers_radians(
    resname: str, chi_index: int
) -> tuple[float, ...]:
    """Return physically unique marginal centers in [-pi, pi)."""
    values = []
    for center in canonical_centers_radians(resname, chi_index):
        normalized = (float(center) + math.pi) % (2.0 * math.pi) - math.pi
        if not any(
            abs((normalized - existing + math.pi) % (2.0 * math.pi) - math.pi)
            < 1e-7
            for existing in values
        ):
            values.append(normalized)
    return tuple(values)


def _canonical_state_pool(
    resname: str,
    n_chi: int,
    *,
    device: torch.device,
) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    """Build the Cartesian pool of marginal production-center tuples."""
    centers = [
        _unique_canonical_centers_radians(resname, index)
        for index in range(n_chi)
    ]
    state_indices = list(itertools.product(*[
        range(len(values)) for values in centers
    ]))
    state_values = torch.tensor(
        [
            [centers[chi_index][center_index]
             for chi_index, center_index in enumerate(indices)]
            for indices in state_indices
        ],
        dtype=torch.float32,
        device=device,
    )
    return state_indices, state_values


def _select_stratified_canonical_states(
    resname: str,
    n_chi: int,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    anchor_physical_chi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select balanced marginal wells with maximum joint angular separation.

    At every chi, unused wells are selected before any well is reused. Among
    assignments with the same marginal balance, greedy maximin circular
    chi-space distance selects the joint tuple. A seeded random order resolves
    the first free choice and exact ties, so starts remain deterministic but
    explore different stratifications.
    """
    if count <= 0:
        return torch.empty((0, n_chi), dtype=torch.float32, device=device)
    state_indices, state_values = _canonical_state_pool(
        resname, n_chi, device=device
    )
    tie_order = torch.randperm(
        len(state_indices), generator=generator, device=device
    ).detach().cpu().tolist()
    tie_rank = {state_index: rank for rank, state_index in enumerate(tie_order)}
    selected_pool_indices: list[int] = []
    selected_points = (
        [anchor_physical_chi.detach()]
        if anchor_physical_chi is not None else []
    )

    while len(selected_pool_indices) < count:
        candidates = [
            index for index in range(len(state_indices))
            if index not in selected_pool_indices
        ]
        if not candidates:
            # One-chi residues have three physical wells but K=4. Reuse is
            # unavoidable; jitter later prevents exact coordinate degeneracy.
            candidates = list(range(len(state_indices)))

        balanced = candidates
        if selected_pool_indices:
            for chi_index in range(n_chi):
                center_count = len(_unique_canonical_centers_radians(
                    resname, chi_index
                ))
                usage = [0] * center_count
                for selected_index in selected_pool_indices:
                    usage[state_indices[selected_index][chi_index]] += 1
                minimum_usage = min(usage)
                narrowed = [
                    index for index in balanced
                    if usage[state_indices[index][chi_index]] == minimum_usage
                ]
                if narrowed:
                    balanced = narrowed

        if not selected_points:
            chosen = min(balanced, key=tie_rank.__getitem__)
        else:
            reference = torch.stack(selected_points)
            scored = []
            for index in balanced:
                deltas = torch.atan2(
                    torch.sin(state_values[index][None, :] - reference),
                    torch.cos(state_values[index][None, :] - reference),
                )
                minimum_distance = float(
                    torch.linalg.vector_norm(deltas, dim=1).min().cpu()
                )
                scored.append((minimum_distance, -tie_rank[index], index))
            chosen = max(scored)[2]
        selected_pool_indices.append(chosen)
        selected_points.append(state_values[chosen])

    return state_values[selected_pool_indices]


def _initialize_chi_offsets(
    *,
    mode: str,
    resname: str,
    n_chi: int,
    K: int,
    base_physical_chi: torch.Tensor,
    delta_from_physical_chi,
    generator: torch.Generator,
    device: torch.device,
    jitter_degrees: float,
) -> torch.Tensor:
    """Generate deterministic K-slot chi offsets for a production start."""
    if mode == "deposited_a_cloud_60":
        return torch.randn(
            (K, n_chi), generator=generator, device=device
        )
    if mode == "deposited_a_cloud_120":
        return 2.0 * torch.randn(
            (K, n_chi), generator=generator, device=device
        )

    jitter_radians = math.radians(jitter_degrees)
    if mode == "canonical_stratified_free":
        physical = _select_stratified_canonical_states(
            resname,
            n_chi,
            K,
            generator=generator,
            device=device,
        )
    elif mode == "canonical_stratified_a_anchor":
        canonical = _select_stratified_canonical_states(
            resname,
            n_chi,
            K - 1,
            generator=generator,
            device=device,
            anchor_physical_chi=base_physical_chi,
        )
        physical = torch.cat((base_physical_chi[None, :], canonical), dim=0)
    else:
        raise ValueError(f"unsupported initialization mode: {mode}")
    physical = physical + jitter_radians * torch.randn(
        physical.shape, generator=generator, device=device
    )
    return delta_from_physical_chi(physical)


def _production_density_schedule(
    n_chi: int,
    n_steps: int,
    learning_rate: float,
    per_residue_class: bool,
    four_chi_stage_steps: int = 100,
) -> tuple[tuple[float, float, int], ...]:
    """Return (blur FWHM, learning rate, steps) for production stage 1."""
    if per_residue_class and n_chi == 4:
        return (
            (4.0, learning_rate, four_chi_stage_steps),
            (2.0, learning_rate * 0.1, four_chi_stage_steps),
            (0.0, learning_rate * 0.01, four_chi_stage_steps),
        )
    return ((0.0, learning_rate, n_steps),)


def _stage1_adam(
    chi: torch.Tensor,
    occupancy_logits: torch.Tensor,
    *,
    chi_learning_rate: float,
    occupancy_learning_rate: float,
) -> torch.optim.Adam:
    """Build Stage-1 Adam with independently controllable occupancy LR.

    Keeping the logits in Adam while their group LR is zero preserves their
    first- and second-moment estimates during the frozen prefix. Setting both
    learning rates equal is mathematically identical to the historical
    ``Adam([chi, occupancy_logits], lr=...)`` path.
    """
    return torch.optim.Adam(
        [
            {"params": [chi], "lr": chi_learning_rate},
            {"params": [occupancy_logits], "lr": occupancy_learning_rate},
        ],
        lr=chi_learning_rate,
    )


def _set_occupancy_learning_rate(
    optimizer: torch.optim.Adam, learning_rate: float
) -> None:
    if len(optimizer.param_groups) != 2:
        raise ValueError("Stage-1 optimizer must have chi and occupancy groups")
    optimizer.param_groups[1]["lr"] = learning_rate


def _reset_adam_parameter_slice(
    optimizer: torch.optim.Adam,
    parameter: torch.Tensor,
    index: int,
) -> None:
    """Clear accumulated Adam moments for one slot without touching others."""
    state = optimizer.state.get(parameter)
    if not state:
        return
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buffer = state.get(name)
        if isinstance(buffer, torch.Tensor):
            buffer[index].zero_()


def _merge_occupancies_for_respawn(
    logits: torch.Tensor,
    keeper: int,
    freed: int,
    floor: float = RESPAWN_OCCUPANCY_FLOOR,
) -> torch.Tensor:
    """Move a near-duplicate pair's mass to keeper and free a tiny live slot."""
    if keeper == freed:
        raise ValueError("keeper and freed slots must differ")
    occupancies = torch.softmax(logits.detach(), dim=0)
    merged = occupancies[keeper] + occupancies[freed]
    respawn_occupancy = min(float(floor), 0.5 * float(merged))
    updated = occupancies.clone()
    updated[keeper] = merged - respawn_occupancy
    updated[freed] = respawn_occupancy
    updated /= updated.sum()
    with torch.no_grad():
        logits.copy_(torch.log(updated))
    return updated


def _gram_condition(columns: torch.Tensor) -> float:
    gram = columns.T @ columns
    return float(torch.linalg.cond(gram).detach().cpu())


def _respawn_due(cadence: int, step: int, total_steps: int) -> bool:
    """Return whether a non-terminal Stage-1 cadence check is due."""
    return bool(
        cadence > 0
        and step > 0
        and step % cadence == 0
        and step < total_steps
    )


def _linear_chi_noise_sd_degrees(
    initial_sd_degrees: float,
    step: int,
    total_steps: int,
) -> float:
    """Linearly anneal Stage-1 chi noise from its initial SD to exact zero."""
    if initial_sd_degrees < 0:
        raise ValueError("initial chi-noise SD must be non-negative")
    if total_steps <= 0:
        raise ValueError("total Stage-1 steps must be positive")
    if not 1 <= step <= total_steps:
        raise ValueError("noise step must be within Stage 1")
    if initial_sd_degrees == 0.0 or total_steps == 1:
        return 0.0
    return float(
        initial_sd_degrees
        * (1.0 - (step - 1) / (total_steps - 1))
    )


def _apply_stage1_chi_noise_(
    chi: torch.Tensor,
    *,
    initial_sd_degrees: float,
    step: int,
    total_steps: int,
    generator: torch.Generator,
) -> float:
    """Apply chi-only Gaussian noise without consuming RNG on the zero arm."""
    noise_sd_degrees = _linear_chi_noise_sd_degrees(
        initial_sd_degrees, step, total_steps
    )
    if noise_sd_degrees > 0.0:
        with torch.no_grad():
            chi.add_(
                torch.randn(
                    chi.shape,
                    dtype=chi.dtype,
                    device=chi.device,
                    generator=generator,
                )
                * math.radians(noise_sd_degrees)
            )
    return noise_sd_degrees


def _inverse_kinematics_to_peak(
    initial_chi: torch.Tensor,
    peak: torch.Tensor,
    coordinates_from_chi,
    *,
    steps: int = RESPAWN_IK_STEPS,
    learning_rate: float = RESPAWN_IK_LEARNING_RATE,
) -> tuple[torch.Tensor, int, float]:
    """Fit each possible heavy atom to a peak and retain the closest solution."""
    with torch.no_grad():
        atom_count = int(coordinates_from_chi(initial_chi).shape[0])
    candidates = initial_chi.detach().repeat(atom_count, 1)
    candidates.requires_grad_(True)
    optimizer = torch.optim.Adam([candidates], lr=learning_rate)
    atom_indices = torch.arange(atom_count, device=initial_chi.device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        coordinates = torch.stack([
            coordinates_from_chi(wrap_angles(row)) for row in candidates
        ])
        selected = coordinates[atom_indices, atom_indices]
        loss = (selected - peak[None, :]).square().sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            candidates.copy_(wrap_angles(candidates))
    with torch.no_grad():
        coordinates = torch.stack([
            coordinates_from_chi(wrap_angles(row)) for row in candidates
        ])
        distances = torch.linalg.vector_norm(
            coordinates[atom_indices, atom_indices] - peak[None, :],
            dim=1,
        )
        winner = int(torch.argmin(distances))
        return (
            candidates[winner].detach().clone(),
            winner,
            float(distances[winner].cpu()),
        )


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False,
                                     newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _alt_atom_map(residue: gemmi.Residue, alt: str, device: torch.device) -> dict:
    result = {}
    for atom in residue:
        atom_alt = "" if atom.altloc in ("\x00", " ") else atom.altloc
        if atom_alt in ("", alt):
            result[atom.name.strip()] = torch.tensor(
                atom.pos.tolist(), dtype=torch.float32, device=device
            )
    return result


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std().clamp_min(1e-6)


def _reachable_volume_mask(
    grid_coordinates: np.ndarray,
    reachable_coordinates: np.ndarray,
    padding: float,
) -> np.ndarray:
    """Select voxels close to any atom in any enumerated canonical state."""
    if padding <= 0:
        raise ValueError("reachable-volume padding must be positive")
    grid = np.asarray(grid_coordinates, dtype=np.float32).reshape(-1, 3)
    reachable = np.asarray(
        reachable_coordinates, dtype=np.float32
    ).reshape(-1, 3)
    if not len(reachable):
        raise ValueError("reachable coordinate set must not be empty")
    delta = (
        grid[:, None, :] - reachable[None, :, :]
    ).astype(np.float64)
    distance2 = (delta * delta).sum(axis=-1)
    return (distance2.min(axis=1) <= padding ** 2).reshape(
        np.asarray(grid_coordinates).shape[:-1]
    )


def _normalized_variance_weights(
    rendered_states: torch.Tensor,
) -> torch.Tensor:
    """Return nonnegative per-voxel weights with mean exactly one."""
    if rendered_states.ndim != 2 or rendered_states.shape[0] < 2:
        raise ValueError("expected at least two rendered states by voxel")
    variance = rendered_states.var(dim=0, unbiased=False)
    mean = variance.mean()
    if not torch.isfinite(mean) or float(mean.detach().cpu()) <= 0:
        raise ValueError("reachable-state density variance must be positive")
    return variance / mean


def _density_mse(
    rendered: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if rendered.shape != target.shape or rendered.shape != weights.shape:
        raise ValueError("rendered, target, and density weights must align")
    return (weights * (rendered - target).square()).mean()


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _split_matrix(value: str) -> np.ndarray:
    return np.asarray([
        [float(item) for item in row.split(";")]
        for row in value.split("|")
    ], dtype=np.float32)


def _missed_minor_starts(
    ensemble_path: Path,
    site_key: str,
) -> dict[int, dict[str, object]]:
    """Select frozen-metric starts that found only the occupancy-major state."""
    selected: dict[int, dict[str, object]] = {}
    with ensemble_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["site"] != site_key:
                continue
            occupancy_a = float(row["target_A_occupancy"])
            occupancy_b = float(row["target_B_occupancy"])
            if math.isclose(occupancy_a, occupancy_b, abs_tol=1e-6):
                continue
            minor = "A" if occupancy_a < occupancy_b else "B"
            major = "B" if minor == "A" else "A"
            found = {
                "A": _truth(row["found_A_conventional"]),
                "B": _truth(row["found_B_conventional"]),
            }
            if found[major] and not found[minor]:
                selected[int(row["start"])] = {
                    "minor": minor,
                    "major": major,
                    "minor_occupancy": min(occupancy_a, occupancy_b),
                    "major_occupancy": max(occupancy_a, occupancy_b),
                }
    return selected


def _lobe_statistics(
    residual: torch.Tensor,
    selected_grid: torch.Tensor,
    lobe_coordinates: torch.Tensor,
    other_coordinates: torch.Tensor,
    *,
    radius: float,
    voxel_volume: float,
) -> dict[str, float | int]:
    """Summarize a normalized-map residual around a deposited sidechain lobe."""
    lobe_mask = (
        torch.cdist(selected_grid, lobe_coordinates).min(dim=1).values <= radius
    )
    other_mask = (
        torch.cdist(selected_grid, other_coordinates).min(dim=1).values <= radius
    )
    values = residual[lobe_mask]
    if not len(values):
        return {
            "voxels": 0,
            "overlap_fraction": float("nan"),
            "rms": float("nan"),
            "mean_absolute": float("nan"),
            "integrated_positive": 0.0,
            "integrated_signed": 0.0,
        }
    return {
        "voxels": int(lobe_mask.sum().cpu()),
        "overlap_fraction": float(
            (lobe_mask & other_mask).sum().cpu() / lobe_mask.sum().cpu()
        ),
        "rms": float(values.square().mean().sqrt().cpu()),
        "mean_absolute": float(values.abs().mean().cpu()),
        "integrated_positive": float(
            values.clamp_min(0).sum().cpu() * voxel_volume
        ),
        "integrated_signed": float(values.sum().cpu() * voxel_volume),
    }


def _run_residual_minor_probe(
    args: argparse.Namespace,
    sites: list[dict],
    device: torch.device,
) -> None:
    """Fit one fresh slot to the frozen endpoint's unexplained residual."""
    if len(sites) != 1:
        raise ValueError("--residual-minor-probe requires exactly one --site")
    site = sites[0]
    selected = _missed_minor_starts(
        args.residual_probe_ensembles, site["key"]
    )
    if args.residual_probe_start:
        requested_starts = set(args.residual_probe_start)
        selected = {
            start: value for start, value in selected.items()
            if start in requested_starts
        }
        missing_requested = sorted(requested_starts - set(selected))
        if missing_requested:
            raise ValueError(
                "requested starts are not frozen-v3 missed-minor starts: "
                f"{missing_requested}"
            )
    with args.residual_probe_endpoints.open(newline="") as handle:
        endpoint_rows = {
            int(row["start"]): row for row in csv.DictReader(handle)
            if row["site"] == site["key"] and row["target"] == "synthetic"
        }
    missing = sorted(set(selected) - set(endpoint_rows))
    if missing:
        raise RuntimeError(f"missing endpoint rows for starts {missing}")

    output_rows: list[dict] = []
    schedule = _production_density_schedule(
        site["n_chi"],
        args.n_steps,
        args.lr,
        args.per_residue_class_schedule,
        args.four_chi_stage_steps,
    )
    voxel_volume = args.spacing ** 3
    lobe_radius = args.residual_probe_lobe_radius

    for start, classification in sorted(selected.items()):
        endpoint = endpoint_rows[start]
        endpoint_chi = torch.tensor(
            _split_matrix(endpoint["final_chi_radians"]),
            dtype=torch.float32,
            device=device,
        )
        endpoint_occupancies = torch.tensor(
            [float(value) for value in endpoint["occupancies"].split(";")],
            dtype=torch.float32,
            device=device,
        )
        endpoint_logits = torch.log(endpoint_occupancies.clamp_min(1e-8))
        with torch.no_grad():
            endpoint_density, _ = site["render"](
                endpoint_chi, endpoint_logits
            )
            residual_before = (
                site["target_vectors"]["synthetic"] - endpoint_density
            )

        minor = str(classification["minor"])
        major = str(classification["major"])
        minor_coordinates = site[f"kinematic_{minor.lower()}"]
        major_coordinates = site[f"kinematic_{major.lower()}"]
        minor_before = _lobe_statistics(
            residual_before,
            site["selected_grid"],
            minor_coordinates,
            major_coordinates,
            radius=lobe_radius,
            voxel_volume=voxel_volume,
        )
        major_before = _lobe_statistics(
            residual_before,
            site["selected_grid"],
            major_coordinates,
            minor_coordinates,
            radius=lobe_radius,
            voxel_volume=voxel_volume,
        )

        generator = torch.Generator(device=device).manual_seed(args.seed + start)
        initial_chi = torch.randn(
            (1, site["n_chi"]), generator=generator, device=device
        )
        with torch.no_grad():
            initial_coordinates = site["coordinates_from_chi"](
                wrap_angles(initial_chi[0])
            )
            initial_minor_rmsd = float(
                site["rmsd"](initial_coordinates, minor_coordinates).cpu()
            )
            initial_major_rmsd = float(
                site["rmsd"](initial_coordinates, major_coordinates).cpu()
            )

        for occupancy_mode in ("fixed_minor", "free_sigmoid"):
            chi = initial_chi.detach().clone().requires_grad_(True)
            deposited_minor_occupancy = float(
                classification["minor_occupancy"]
            )
            if occupancy_mode == "free_sigmoid":
                initial_probability = torch.tensor(
                    0.25, dtype=torch.float32, device=device
                )
                occupancy_logit = torch.logit(
                    initial_probability
                ).detach().clone().requires_grad_(True)
            else:
                occupancy_logit = None

            def occupancy_tensor() -> torch.Tensor:
                if occupancy_logit is None:
                    return torch.tensor(
                        [deposited_minor_occupancy],
                        dtype=torch.float32,
                        device=device,
                    )
                return torch.sigmoid(occupancy_logit).reshape(1)

            best_stage1_loss = float("inf")
            for blur_fwhm, stage_lr, stage_steps in schedule:
                parameters = [chi]
                if occupancy_logit is not None:
                    parameters.append(occupancy_logit)
                optimizer = torch.optim.Adam(parameters, lr=stage_lr)
                with torch.no_grad():
                    endpoint_blurred, _ = site["render"](
                        endpoint_chi, endpoint_logits, blur_fwhm
                    )
                    target_blurred = site["target_vectors_by_blur"][
                        "synthetic"
                    ][blur_fwhm]
                    residual_blurred = target_blurred - endpoint_blurred
                for _step in range(stage_steps):
                    optimizer.zero_grad(set_to_none=True)
                    single_density, _coordinates = (
                        site["render_single_contribution"](
                            chi[0], blur_fwhm
                        )
                    )
                    density_loss = site["density_loss"](
                        occupancy_tensor()[0] * single_density,
                        residual_blurred,
                    )
                    density_loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        chi.copy_(wrap_angles(chi))
                    best_stage1_loss = min(
                        best_stage1_loss, float(density_loss.detach().cpu())
                    )

            optimizer_parameters = [chi]
            if occupancy_logit is not None:
                optimizer_parameters.append(occupancy_logit)
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=args.lr * args.physics_refinement_lr_scale,
            )
            best_stage2_loss = float("inf")
            for _step in range(args.physics_refinement_steps):
                optimizer.zero_grad(set_to_none=True)
                single_density, coordinates = (
                    site["render_single_contribution"](chi[0])
                )
                occupancy = occupancy_tensor()
                density_loss = site["density_loss"](
                    occupancy[0] * single_density,
                    residual_before,
                )
                vdw, rotamer, symmetry = site["physics_terms"](
                    [coordinates], occupancy
                )
                total_loss = (
                    density_loss
                    + args.lambda_vdw * vdw
                    + args.lambda_rot * rotamer
                    + args.lambda_clash * symmetry
                )
                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    chi.copy_(wrap_angles(chi))
                best_stage2_loss = min(
                    best_stage2_loss, float(total_loss.detach().cpu())
                )

            with torch.no_grad():
                final_density, final_coordinates = (
                    site["render_single_contribution"](chi[0])
                )
                fixed_label_travel_rmsd = float(torch.sqrt(
                    (
                        final_coordinates - initial_coordinates
                    ).square().sum(dim=-1).mean()
                ).cpu())
                symmetry_aware_travel_rmsd = float(
                    site["rmsd"](
                        final_coordinates, initial_coordinates
                    ).cpu()
                )
                chi_travel_degrees = float(
                    torch.linalg.vector_norm(
                        wrap_angles(chi[0] - initial_chi[0])
                    ).cpu() * (180.0 / math.pi)
                )
                final_occupancy = float(occupancy_tensor()[0].cpu())
                residual_after = (
                    residual_before - final_occupancy * final_density
                )
                final_minor_rmsd = float(
                    site["rmsd"](
                        final_coordinates, minor_coordinates
                    ).cpu()
                )
                final_major_rmsd = float(
                    site["rmsd"](
                        final_coordinates, major_coordinates
                    ).cpu()
                )
                final_density_loss = float(site["density_loss"](
                    final_occupancy * final_density,
                    residual_before,
                ).cpu())
                final_vdw, final_rotamer, final_symmetry = (
                    site["physics_terms"](
                        [final_coordinates], occupancy_tensor()
                    )
                )
            minor_after = _lobe_statistics(
                residual_after,
                site["selected_grid"],
                minor_coordinates,
                major_coordinates,
                radius=lobe_radius,
                voxel_volume=voxel_volume,
            )
            output_rows.append({
                "site": site["key"],
                "start": start,
                "minor_state": minor,
                "major_state": major,
                "occupancy_mode": occupancy_mode,
                "initialization": (
                    "production deposited-A-centered N(0,1)-radian slot"
                ),
                "initial_chi_radians": ";".join(
                    f"{value:.9g}" for value in initial_chi[0].cpu().numpy()
                ),
                "final_chi_radians": ";".join(
                    f"{value:.9g}"
                    for value in chi[0].detach().cpu().numpy()
                ),
                "fixed_label_travel_rmsd_A": fixed_label_travel_rmsd,
                "symmetry_aware_travel_rmsd_A": symmetry_aware_travel_rmsd,
                "chi_space_travel_degrees": chi_travel_degrees,
                "initial_rmsd_to_minor": initial_minor_rmsd,
                "initial_rmsd_to_major": initial_major_rmsd,
                "deposited_minor_occupancy": deposited_minor_occupancy,
                "deposited_major_occupancy": classification["major_occupancy"],
                "final_occupancy": final_occupancy,
                "final_rmsd_to_minor": final_minor_rmsd,
                "final_rmsd_to_major": final_major_rmsd,
                "recovered_minor_lt_1A": final_minor_rmsd < 1.0,
                "best_stage1_density_loss": best_stage1_loss,
                "best_stage2_total_loss": best_stage2_loss,
                "final_density_loss": final_density_loss,
                "final_vdw_loss": float(final_vdw.cpu()),
                "final_rotamer_loss": float(final_rotamer.cpu()),
                "final_symmetry_loss": float(final_symmetry.cpu()),
                "minor_lobe_radius_A": lobe_radius,
                **{
                    f"minor_before_{key}": value
                    for key, value in minor_before.items()
                },
                **{
                    f"major_before_{key}": value
                    for key, value in major_before.items()
                },
                **{
                    f"minor_after_{key}": value
                    for key, value in minor_after.items()
                },
            })

    _atomic_csv(args.output / "residual_minor_probe.csv", output_rows)
    _atomic_json(args.output / "residual_minor_probe_summary.json", {
        "status": "complete",
        "diagnostic_only": True,
        "metric_changed": False,
        "site": site["key"],
        "eligible_missed_minor_starts": len(selected),
        "probe_rows": len(output_rows),
        "occupancy_modes": ["fixed_minor", "free_sigmoid"],
        "residual_space": (
            "production z-score-normalized radial vector: "
            "synthetic target minus frozen endpoint render"
        ),
        "single_slot_density": (
            "z-score-normalized single-conformer footprint multiplied by "
            "fixed or sigmoid occupancy"
        ),
        "lobe_mask": (
            f"selected-grid voxels within {lobe_radius:g} A of any deposited "
            "sidechain atom; overlapping A/B mask voxels retained"
        ),
    })
    _atomic_json(args.output / "stage_manifest.json", {
        "status": "complete",
        "diagnostic": "residual_minor_probe_v1",
        "rows": len(output_rows),
    })


def _gaussian_blur_patch(
    values: torch.Tensor, fwhm: float, spacing: float
) -> torch.Tensor:
    """Separable Gaussian blur matching the atom-renderer FWHM convention."""
    if fwhm <= 0:
        return values
    sigma_voxels = fwhm / 2.354820045 / spacing
    radius = max(1, int(math.ceil(3.0 * sigma_voxels)))
    axis = torch.arange(
        -radius, radius + 1, dtype=values.dtype, device=values.device
    )
    kernel = torch.exp(-0.5 * (axis / sigma_voxels).square())
    kernel = kernel / kernel.sum()
    result = values[None, None]
    result = F.conv3d(result, kernel.view(1, 1, -1, 1, 1), padding=(radius, 0, 0))
    result = F.conv3d(result, kernel.view(1, 1, 1, -1, 1), padding=(0, radius, 0))
    result = F.conv3d(result, kernel.view(1, 1, 1, 1, -1), padding=(0, 0, radius))
    return result[0, 0]


def _canonical_centers(resname: str, chi_index: int) -> list[float]:
    """Crystallographic chi centers used by the endpoint physical audit."""
    return list(canonical_centers_radians(resname, chi_index))


def _selected_protein_heavy_atoms(structure: gemmi.Structure) -> list[tuple]:
    """Select protein and water heavy atoms for altloc-aware soft physics."""
    selected = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name not in CHI_SPECS and residue.name not in {
                "ALA", "ASN", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
                "HOH", "WAT", "DOD",
            }:
                continue
            for atom in residue:
                if atom.element.name == "H":
                    continue
                selected.append((chain, residue, atom))
    return selected


def _load_model(checkpoint_path: Path, device: torch.device) -> ResidualDensityDenoiser:
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ResidualDensityDenoiser(checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw-vs-denoised-vs-synthetic K=4 fitting on the five 2O1K sites"
    )
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--pdb", type=Path, default=root / "data" / "2O1K.pdb")
    parser.add_argument("--mtz", type=Path, default=root / "data" / "2O1K.mtz")
    parser.add_argument(
        "--selection", type=Path,
        help="held-out selection.json; use its saved test pairs instead of 2O1K",
    )
    parser.add_argument("--frame", choices=("crystal", "residue"), default="crystal")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-starts", type=int, default=50)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument(
        "--per-residue-class-schedule", action="store_true",
        help=(
            "use the frozen full-resolution --n-steps schedule for 1-3 chi "
            "residues and 4A/2A/full 100-step Adam-reset stages for 4-chi residues"
        ),
    )
    parser.add_argument(
        "--four-chi-stage-steps", type=int, default=100,
        help=(
            "steps at each 4A/2A/full stage for four-chi residues when "
            "--per-residue-class-schedule is enabled (default: 100)"
        ),
    )
    parser.add_argument(
        "--fixed-occupancy-steps", type=int, default=0,
        help=(
            "prefix of Stage 1 that holds occupancies uniform by setting the "
            "occupancy Adam parameter-group LR to zero; gradients and Adam "
            "moments remain live and the LR is restored without an Adam reset"
        ),
    )
    parser.add_argument(
        "--record-stage1-trajectories",
        action="store_true",
        help=(
            "atomically save per-start Stage-1/Stage-2 chi and occupancy traces "
            "for mechanism diagnostics"
        ),
    )
    parser.add_argument(
        "--stage1-chi-noise-initial-degrees",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise SD added to chi parameters after the first "
            "Stage-1 Adam step; linearly annealed to exactly zero at the "
            "final Stage-1 step. Zero is a literal RNG-free no-op."
        ),
    )
    parser.add_argument(
        "--respawn-cadence",
        type=int,
        default=0,
        help=(
            "Stage-1 steps between merge-and-respawn checks; zero is a "
            "literal no-op that preserves the frozen optimizer path"
        ),
    )
    parser.add_argument(
        "--respawn-merge-rmsd",
        type=float,
        default=0.5,
        help=(
            "symmetry-aware conventional RMSD threshold for merging the "
            "closest slot pair at a respawn check"
        ),
    )
    parser.add_argument(
        "--initialization-mode",
        choices=INITIALIZATION_MODES,
        default=DEFAULT_INITIALIZATION_MODE,
        help=(
            "K-slot Stage-1 chi initialization; the default exactly preserves "
            "the historical N(0,1)-radian cloud around deposited A"
        ),
    )
    parser.add_argument(
        "--initialization-jitter-degrees",
        type=float,
        default=12.0,
        help="per-chi Gaussian jitter for canonical-stratified modes",
    )
    parser.add_argument(
        "--initialization-only",
        action="store_true",
        help=(
            "write deterministic per-slot initial chi and deposited-state RMSD "
            "rows without running Stage 1 or Stage 2"
        ),
    )
    parser.add_argument(
        "--residual-minor-probe",
        action="store_true",
        help=(
            "diagnostic-only: fit one fresh slot to target minus a saved "
            "frozen-v3 endpoint for starts that found only the major state"
        ),
    )
    parser.add_argument("--residual-probe-endpoints", type=Path)
    parser.add_argument("--residual-probe-ensembles", type=Path)
    parser.add_argument(
        "--residual-probe-start",
        type=int,
        action="append",
        default=[],
        help="optional frozen-v3 missed-minor start to probe (repeatable)",
    )
    parser.add_argument(
        "--residual-probe-lobe-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument(
        "--seed-deposited-a", action="store_true",
        help="initialize slot 0 exactly at the deposited-A kinematic coordinates",
    )
    parser.add_argument(
        "--sequential-two-stage", action="store_true",
        help=(
            "3A1C diagnostic: 50-start K=1 fit, 50-start K=1 residual fit, "
            "then K=4 joint soft-physics refinement"
        ),
    )
    parser.add_argument(
        "--initialization-tests", action="store_true",
        help=(
            "3A1C-only Dunbrack and transferred-ARG initialization sweeps "
            "with coarse-to-fine density fitting and soft-physics polish"
        ),
    )
    parser.add_argument(
        "--transfer-endpoint-root", type=Path,
        default=(
            root / "five_site_coarse_to_fine_decay_reset"
            / "coarse_to_fine_4A_2A_full_decay"
        ),
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--grid-radius", type=float, default=4.0)
    parser.add_argument(
        "--density-mask-mode",
        choices=DENSITY_MASK_MODES,
        default="sphere",
    )
    parser.add_argument(
        "--reachable-mask-padding",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--density-weight-mode",
        choices=DENSITY_WEIGHT_MODES,
        default="uniform",
    )
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--nontrivial-occupancy", type=float, default=0.05)
    parser.add_argument("--occupancy-tolerance", type=float, default=0.20)
    parser.add_argument("--soft-physics", action="store_true")
    parser.add_argument(
        "--physics-refinement-steps", type=int, default=0,
        help="after density-only optimization, reset Adam and refine with soft physics",
    )
    parser.add_argument("--physics-refinement-lr-scale", type=float, default=0.1)
    parser.add_argument("--lambda-vdw", type=float, default=1.0)
    parser.add_argument("--lambda-rot", type=float, default=0.5)
    parser.add_argument("--lambda-clash", type=float, default=5.0)
    parser.add_argument("--vdw-threshold", type=float, default=3.0)
    parser.add_argument("--clash-threshold", type=float, default=2.5)
    parser.add_argument("--symmetry-hard-threshold", type=float, default=2.0)
    parser.add_argument("--symmetry-barrier-buffer", type=float, default=0.25)
    parser.add_argument(
        "--symmetry-barrier-scale",
        type=float,
        default=0.0,
        help=(
            "raw quartic symmetry-barrier value at the hard threshold; "
            "zero preserves the historical squared hinge"
        ),
    )
    parser.add_argument("--physics-calibration-max-gap", type=float, default=5.0)
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument(
        "--target", action="append", choices=("raw", "denoised", "synthetic"), default=[]
    )
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.physics_refinement_steps < 0:
        raise ValueError("--physics-refinement-steps must be non-negative")
    if args.symmetry_barrier_buffer <= 0:
        raise ValueError("--symmetry-barrier-buffer must be positive")
    if args.symmetry_barrier_scale < 0:
        raise ValueError("--symmetry-barrier-scale must be non-negative")
    if args.four_chi_stage_steps <= 0:
        raise ValueError("--four-chi-stage-steps must be positive")
    if args.respawn_cadence < 0:
        raise ValueError("--respawn-cadence must be non-negative")
    if args.stage1_chi_noise_initial_degrees < 0:
        raise ValueError(
            "--stage1-chi-noise-initial-degrees must be non-negative"
        )
    if args.respawn_merge_rmsd <= 0:
        raise ValueError("--respawn-merge-rmsd must be positive")
    if args.initialization_jitter_degrees < 0:
        raise ValueError("--initialization-jitter-degrees must be non-negative")
    if args.grid_radius <= 0:
        raise ValueError("--grid-radius must be positive")
    if args.reachable_mask_padding <= 0:
        raise ValueError("--reachable-mask-padding must be positive")
    if args.initialization_only and args.calibration_only:
        raise ValueError(
            "--initialization-only and --calibration-only are mutually exclusive"
        )
    if args.residual_minor_probe:
        if args.calibration_only or args.initialization_only:
            raise ValueError(
                "--residual-minor-probe is exclusive with calibration and "
                "initialization-only modes"
            )
        if (
            args.residual_probe_endpoints is None
            or args.residual_probe_ensembles is None
        ):
            raise ValueError(
                "--residual-minor-probe requires endpoint and ensemble tables"
            )
        if args.residual_probe_lobe_radius <= 0:
            raise ValueError("--residual-probe-lobe-radius must be positive")
    if not 0 <= args.fixed_occupancy_steps <= args.n_steps:
        raise ValueError(
            "--fixed-occupancy-steps must be between zero and --n-steps"
        )
    if args.per_residue_class_schedule and args.soft_physics:
        raise ValueError(
            "--per-residue-class-schedule is a density-first schedule; use "
            "--physics-refinement-steps for the physics stage"
        )
    if args.sequential_two_stage:
        if (
            args.fixed_occupancy_steps
            or args.seed_deposited_a
            or args.soft_physics
            or args.initialization_mode != DEFAULT_INITIALIZATION_MODE
        ):
            raise ValueError(
                "--sequential-two-stage owns its fitting and physics schedule"
            )
        if args.K != 4 or args.physics_refinement_steps != 200:
            raise ValueError(
                "--sequential-two-stage requires --K 4 and "
                "--physics-refinement-steps 200"
            )
    if args.initialization_tests:
        if args.sequential_two_stage or args.fixed_occupancy_steps:
            raise ValueError("initialization tests own their optimization schedule")
        if args.initialization_only:
            raise ValueError(
                "--initialization-tests and --initialization-only are exclusive"
            )
        if (
            args.seed_deposited_a
            or args.soft_physics
            or args.initialization_mode != DEFAULT_INITIALIZATION_MODE
        ):
            raise ValueError("do not combine initialization tests with other controls")
        if args.K != 4 or args.n_starts != 50:
            raise ValueError("initialization tests require --K 4 and --n-starts 50")
        if args.physics_refinement_steps != 200:
            raise ValueError(
                "initialization tests require --physics-refinement-steps 200"
            )
    if args.physics_refinement_lr_scale <= 0:
        raise ValueError("--physics-refinement-lr-scale must be positive")
    if args.soft_physics and args.physics_refinement_steps:
        raise ValueError(
            "choose either full-run --soft-physics or staged --physics-refinement-steps"
        )
    if (
        args.seed_deposited_a
        and args.initialization_mode != DEFAULT_INITIALIZATION_MODE
    ):
        raise ValueError(
            "--seed-deposited-a cannot be combined with --initialization-mode"
        )
    physics_enabled = args.soft_physics or args.physics_refinement_steps > 0

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = _load_model(args.checkpoint, device)
    if args.selection:
        selection = json.loads(args.selection.read_text())
        selected_records = selection["sites"]
        if args.site:
            requested = set(args.site)
            selected_records = [record for record in selected_records if record["key"] in requested]
        wanted_sites = {record["key"] for record in selected_records}
        sources = []
        for record in selected_records:
            structure = gemmi.read_structure(record["pdb_path"])
            pair_path = Path(record["pair_path"])
            if args.frame == "residue":
                pair_path = pair_path.parent.parent / "canonical" / "pairs" / pair_path.name
            pair = np.load(pair_path, allow_pickle=False)
            metadata = json.loads(str(pair["metadata"].item()))
            if metadata.get("frame", "crystal") != args.frame:
                raise RuntimeError(
                    f"{record['key']} requested {args.frame} frame but pair metadata says "
                    f"{metadata.get('frame', 'crystal')}"
                )
            site = SimpleNamespace(**{
                "key": record["key"], "center": tuple(metadata["center"]),
                "chain": record["chain"],
                "residue_number": int(record["residue_number"]),
                "insertion_code": record["insertion_code"],
                "residue_name": record["residue_name"], "pdb_id": record["pdb_id"],
                "split": "test", "is_altloc": True,
            })
            sources.append((site, structure, pair, metadata, record["key"]))
    else:
        if args.frame != "crystal":
            raise ValueError("residue frame requires --selection and saved canonical pairs")
        structure = gemmi.read_structure(str(args.pdb))
        mtz = gemmi.read_mtz_file(str(args.mtz))
        calculator = gemmi.StructureFactorCalculatorX(structure.cell)
        miller = mtz.make_miller_array()
        full_fcalc = _calculate_fcalc(calculator, structure[0], miller)
        discovered = discover_sites(structure, "2O1K", "integration", 0, args.seed)
        wanted_sites = set(args.site) if args.site else {
            "A_MET112", "A_ARG129", "B_MET112", "B_ASP114", "B_ARG129"
        }
        sources = [
            (site, structure, None, None, site.key.removeprefix("2O1K_"))
            for site in discovered if site.key.removeprefix("2O1K_") in wanted_sites
        ]
    targets = args.target or ["raw", "denoised", "synthetic"]
    sites = []
    for site, structure, pair, pair_metadata, short_key in sources:
        residue = next(
            residue for chain in structure[0] if chain.name == site.chain
            for residue in chain if residue.seqid.num == site.residue_number
        )
        if residue.name not in CHI_SPECS:
            continue
        map_a = _alt_atom_map(residue, "A", device)
        map_b = _alt_atom_map(residue, "B", device)
        b_atoms = [
            atom for atom in residue
            if atom.altloc == "B"
            and atom.element.name != "H"
            and atom.name.strip() not in {"N", "CA", "C", "O"}
        ]
        names = [atom.name.strip() for atom in b_atoms]
        if not names or any(name not in map_a or name not in map_b for name in names):
            raise RuntimeError(f"incomplete A/B sidechain atoms at {short_key}")
        spec = CHI_SPECS[residue.name]
        chi_a = torch.stack([
            dihedral(*(map_a[name] for name in quartet)) for quartet in spec["dihedrals"]
        ])
        chi_b = torch.stack([
            dihedral(*(map_b[name] for name in quartet)) for quartet in spec["dihedrals"]
        ])
        true_delta = wrap_angles(chi_b - chi_a)
        template = torch.stack([map_a[name] for name in names])
        deposited_b = torch.stack([map_b[name] for name in names])
        fixed_lookup = {name: value for name, value in map_a.items() if name not in names}

        def coordinates_from_chi(
            chi: torch.Tensor, template=template, names=names,
            rotations=tuple(spec["rotations"]), fixed_lookup=fixed_lookup,
        ) -> torch.Tensor:
            return torsion_to_coords(
                template, names, chi, list(rotations), fixed_lookup
            )

        plus = coordinates_from_chi(true_delta)
        minus = coordinates_from_chi(-true_delta)
        if symmetry_aware_rmsd(
            minus, deposited_b, names, residue.name
        ) < symmetry_aware_rmsd(plus, deposited_b, names, residue.name):
            true_delta = -true_delta
        kinematic_a = coordinates_from_chi(torch.zeros(len(spec["rotations"]), device=device)).detach()
        kinematic_b = coordinates_from_chi(true_delta).detach()

        def endpoint_rmsd(
            candidate: torch.Tensor,
            reference: torch.Tensor,
            names=names,
            resname=residue.name,
        ) -> torch.Tensor:
            return symmetry_aware_rmsd(candidate, reference, names, resname)

        def physical_chi(
            candidate: torch.Tensor,
            fixed_lookup=fixed_lookup,
            names=names,
            dihedrals=tuple(spec["dihedrals"]),
        ) -> torch.Tensor:
            lookup = dict(fixed_lookup)
            lookup.update({
                name: candidate[index] for index, name in enumerate(names)
            })
            return torch.stack([
                wrap_angles(
                    dihedral(*(lookup[name] for name in quartet)) - torch.pi
                )
                for quartet in dihedrals
            ])

        base_physical_chi = physical_chi(kinematic_a).detach()
        delta_direction = []
        for chi_index in range(len(spec["rotations"])):
            probe = torch.zeros(len(spec["rotations"]), device=device)
            probe[chi_index] = 0.01
            moved = physical_chi(coordinates_from_chi(probe)).detach()
            direction = torch.sign(wrap_angles(
                moved[chi_index] - base_physical_chi[chi_index]
            ))
            delta_direction.append(float(direction.cpu()) or 1.0)
        delta_direction = torch.tensor(
            delta_direction, dtype=torch.float32, device=device
        )

        def delta_from_physical_chi(
            desired: torch.Tensor,
            base=base_physical_chi,
            direction=delta_direction,
        ) -> torch.Tensor:
            return direction * wrap_angles(desired - base)

        raw_occ_b = np.asarray([atom.occ for atom in b_atoms], dtype=np.float32)
        occ_b = float(np.median(raw_occ_b))
        a_atoms = [atom for atom in residue if atom.altloc == "A" and atom.name.strip() in names]
        occ_a = float(np.median([atom.occ for atom in a_atoms]))
        occ_total = max(occ_a + occ_b, 1e-6)

        patch_center = (
            np.asarray(pair_metadata["patch_center_crystal"], dtype=np.float32)
            if pair_metadata is not None and args.frame == "residue"
            else np.asarray(site.center, dtype=np.float32)
        )
        crystal_to_local = (
            np.asarray(pair_metadata["crystal_to_local"], dtype=np.float32)
            if pair_metadata is not None and args.frame == "residue" else None
        )
        coordinates = _grid_coordinates(
            patch_center, args.patch_size, args.spacing, crystal_to_local
        )
        _reachable_state_indices, reachable_physical_chi = (
            _canonical_state_pool(
                residue.name,
                len(spec["rotations"]),
                device=device,
            )
        )
        with torch.no_grad():
            reachable_delta_chi = delta_from_physical_chi(
                reachable_physical_chi
            )
            reachable_coordinates = torch.stack([
                coordinates_from_chi(wrap_angles(row))
                for row in reachable_delta_chi
            ])
        mask_support_coordinates = reachable_coordinates
        if args.density_mask_mode == "sphere":
            density_mask = (
                np.linalg.norm(
                    coordinates - np.asarray(site.center), axis=-1
                )
                <= args.grid_radius
            )
        else:
            if args.density_mask_mode == "containing_volume":
                mask_support_coordinates = torch.cat((
                    reachable_coordinates,
                    kinematic_a[None, :, :],
                    deposited_b[None, :, :],
                ))
            density_mask = _reachable_volume_mask(
                coordinates,
                mask_support_coordinates.detach().cpu().numpy(),
                args.reachable_mask_padding,
            )
        if not density_mask.any():
            raise RuntimeError(f"empty density mask at {short_key}")
        selected_grid = torch.tensor(
            coordinates[density_mask], dtype=torch.float32, device=device
        )
        if pair is None:
            experimental_grid = _omit_map(
                structure, mtz, site, "omit_mfo_dfc", calculator, miller, full_fcalc
            )
            raw_patch = extract_patch(
                experimental_grid, site.center, args.patch_size, args.spacing
            )
            raw_normalized = normalize_patch(raw_patch)[0]
            synthetic_normalized = normalize_patch(synthetic_patch(
                structure, site, args.patch_size, args.spacing, "sidechain"
            ))[0]
            target_metadata = asdict(site)
        else:
            raw_normalized = np.asarray(pair["input"][0], dtype=np.float32)
            synthetic_normalized = np.asarray(pair["target"][0], dtype=np.float32)
            target_metadata = pair_metadata
        input_tensor = torch.tensor(
            raw_normalized[None, None], dtype=torch.float32, device=device
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            denoised_patch = model(input_tensor)[0, 0].float()
        target_patches = {
            "raw": torch.tensor(raw_normalized, device=device),
            "denoised": denoised_patch,
            "synthetic": torch.tensor(synthetic_normalized, device=device),
        }
        target_vectors = {
            label: _normalize(patch[torch.tensor(density_mask, device=device)])
            for label, patch in target_patches.items()
        }
        blur_levels = (0.0, 2.0, 4.0)
        density_mask_tensor = torch.tensor(density_mask, device=device)
        target_vectors_by_blur = {
            label: {
                blur: _normalize(_gaussian_blur_patch(
                    patch, blur, args.spacing
                )[density_mask_tensor])
                for blur in blur_levels
            }
            for label, patch in target_patches.items()
        }
        np.savez_compressed(
            args.output / f"{short_key}_targets.npz",
            raw=raw_normalized,
            denoised=denoised_patch.cpu().numpy(),
            denoiser_training_target=synthetic_normalized,
            radial_mask=density_mask,
            metadata=np.asarray(json.dumps({**target_metadata, "short_key": short_key})),
        )

        variable_sigma2 = torch.tensor(
            [max(float(atom.b_iso) / (8.0 * math.pi ** 2), 0.04) for atom in b_atoms],
            dtype=torch.float32, device=device,
        )
        variable_weights = torch.tensor(
            [atom.element.atomic_number * atom.occ / max(occ_b, 1e-6) for atom in b_atoms],
            dtype=torch.float32, device=device,
        )
        shared_atoms = [
            atom for atom in _sidechain_atoms(residue)
            if atom.altloc in ("\x00", " ", "") and atom.name.strip() not in names
        ]
        if shared_atoms:
            shared_xyz = torch.tensor(
                [atom.pos.tolist() for atom in shared_atoms], dtype=torch.float32, device=device
            )
            shared_sigma2 = torch.tensor(
                [max(float(atom.b_iso) / (8.0 * math.pi ** 2), 0.04) for atom in shared_atoms],
                dtype=torch.float32, device=device,
            )
            shared_weights = torch.tensor(
                [atom.element.atomic_number * atom.occ for atom in shared_atoms],
                dtype=torch.float32, device=device,
            )
        else:
            shared_xyz = torch.empty((0, 3), device=device)
            shared_sigma2 = torch.empty(0, device=device)
            shared_weights = torch.empty(0, device=device)

        def atom_density(
            xyz: torch.Tensor, sigma2: torch.Tensor, weights: torch.Tensor,
            selected_grid=selected_grid,
        ) -> torch.Tensor:
            if not len(xyz):
                return torch.zeros(len(selected_grid), device=device)
            distance2 = (selected_grid[:, None, :] - xyz[None, :, :]).square().sum(dim=-1)
            normalization = (2.0 * math.pi * sigma2).pow(-1.5)
            return (
                weights[None, :] * normalization[None, :]
                * torch.exp(-distance2 / (2.0 * sigma2[None, :]))
            ).sum(dim=1)

        if args.density_weight_mode == "uniform":
            density_weights = torch.ones(
                len(selected_grid), dtype=torch.float32, device=device
            )
        else:
            with torch.no_grad():
                reachable_rendered_states = torch.stack([
                    atom_density(xyz, variable_sigma2, variable_weights)
                    for xyz in reachable_coordinates
                ])
                density_weights = _normalized_variance_weights(
                    reachable_rendered_states
                )

        def density_loss(
            rendered: torch.Tensor,
            target: torch.Tensor,
            weights=density_weights,
        ) -> torch.Tensor:
            return _density_mse(rendered, target, weights)

        def render(
            all_chi: torch.Tensor, logits: torch.Tensor,
            blur_fwhm: float = 0.0,
            coordinates_from_chi=coordinates_from_chi,
            atom_density=atom_density,
            variable_sigma2=variable_sigma2,
            variable_weights=variable_weights,
            shared_xyz=shared_xyz,
            shared_sigma2=shared_sigma2,
            shared_weights=shared_weights,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            occupancies = torch.softmax(logits, dim=0)
            conformers = [coordinates_from_chi(wrap_angles(row)) for row in all_chi]
            blur_variance = (float(blur_fwhm) / 2.354820045) ** 2
            fixed_density = atom_density(
                shared_xyz, shared_sigma2 + blur_variance, shared_weights
            )
            density = fixed_density + sum(
                occupancies[index] * atom_density(
                    xyz, variable_sigma2 + blur_variance, variable_weights
                )
                for index, xyz in enumerate(conformers)
            )
            return _normalize(density), conformers

        def render_single_contribution(
            chi: torch.Tensor,
            blur_fwhm: float = 0.0,
            coordinates_from_chi=coordinates_from_chi,
            atom_density=atom_density,
            variable_sigma2=variable_sigma2,
            variable_weights=variable_weights,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            xyz = coordinates_from_chi(wrap_angles(chi))
            blur_variance = (float(blur_fwhm) / 2.354820045) ** 2
            density = atom_density(
                xyz, variable_sigma2 + blur_variance, variable_weights
            )
            return _normalize(density), xyz

        def density_column_from_chi(
            chi: torch.Tensor,
            coordinates_from_chi=coordinates_from_chi,
            atom_density=atom_density,
            variable_sigma2=variable_sigma2,
            variable_weights=variable_weights,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return the native unweighted density column used for Gram tests."""
            xyz = coordinates_from_chi(wrap_angles(chi))
            return atom_density(
                xyz, variable_sigma2, variable_weights
            ), xyz

        heavy_atoms = _selected_protein_heavy_atoms(structure)
        ca_position = map_a["CA"].detach().cpu().numpy()
        max_sidechain_radius = max(
            float(torch.linalg.vector_norm(value - map_a["CA"]).detach().cpu())
            for value in map_a.values()
        )
        environment_radius = max_sidechain_radius + max(
            args.vdw_threshold, args.clash_threshold
        ) + 1.0
        direct_context = []
        for context_chain, context_residue, atom in heavy_atoms:
            atom_name = atom.name.strip()
            same_target = (
                context_chain.name == site.chain
                and context_residue.seqid.num == site.residue_number
                and context_residue.seqid.icode == site.insertion_code
            )
            # The moving sidechain replaces every deposited target-sidechain atom.
            if same_target and atom_name in names:
                continue
            if np.linalg.norm(np.asarray(atom.pos.tolist()) - ca_position) <= environment_radius:
                direct_context.append((context_chain, context_residue, atom))

        direct_records = [
            SoftEnvironmentRecord(
                xyz=tuple(atom.pos.tolist()),
                group_key=(
                    f"{context_chain.name}:{context_residue.seqid.num}:"
                    f"{context_residue.seqid.icode}"
                ),
                atom_name=atom.name.strip(),
                altloc=normalized_altloc(atom.altloc),
                occupancy=float(atom.occ),
                is_water=context_residue.name in {"HOH", "WAT", "DOD"},
            )
            for context_chain, context_residue, atom in direct_context
        ]
        (
            direct_environment,
            direct_environment_weights,
            direct_alternate_states,
            invariant_direct_records,
        ) = partition_soft_environment(direct_records, device)
        direct_pair_mask = torch.ones(
            (len(names), len(direct_environment)), dtype=torch.bool, device=device
        )
        for moving_index, moving_name in enumerate(names):
            for environment_index, environment_record in enumerate(
                invariant_direct_records
            ):
                same_target = (
                    environment_record.group_key
                    == (
                        f"{site.chain}:{site.residue_number}:"
                        f"{site.insertion_code}"
                    )
                )
                # CB--CA is the only moving-sidechain/backbone covalent bond.
                if (
                    same_target
                    and moving_name == "CB"
                    and environment_record.atom_name == "CA"
                ):
                    direct_pair_mask[moving_index, environment_index] = False

        cell = structure.cell
        spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
        symmetry_records = []
        for operation_index, operation in enumerate(spacegroup.operations()):
            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    for tz in (-1, 0, 1):
                        if operation_index == 0 and tx == ty == tz == 0:
                            continue
                        for context_chain, context_residue, atom in heavy_atoms:
                            transformed = operation.apply_to_xyz(
                                cell.fractionalize(atom.pos).tolist()
                            )
                            position = cell.orthogonalize(gemmi.Fractional(
                                transformed[0] + tx,
                                transformed[1] + ty,
                                transformed[2] + tz,
                            ))
                            xyz = np.asarray(position.tolist())
                            if np.linalg.norm(xyz - ca_position) <= environment_radius:
                                symmetry_records.append(SoftEnvironmentRecord(
                                    xyz=tuple(xyz.tolist()),
                                    group_key=(
                                        f"sym{operation_index}[{tx},{ty},{tz}]/"
                                        f"{context_chain.name}:"
                                        f"{context_residue.seqid.num}:"
                                        f"{context_residue.seqid.icode}"
                                    ),
                                    atom_name=atom.name.strip(),
                                    altloc=normalized_altloc(atom.altloc),
                                    occupancy=float(atom.occ),
                                    is_water=(
                                        context_residue.name
                                        in {"HOH", "WAT", "DOD"}
                                    ),
                                ))
        (
            symmetry_environment,
            symmetry_environment_weights,
            symmetry_alternate_states,
            _invariant_symmetry_records,
        ) = partition_soft_environment(symmetry_records, device)

        def physics_terms(
            conformers: list[torch.Tensor],
            occupancies: torch.Tensor,
            resname=residue.name,
            direct_environment=direct_environment,
            direct_environment_weights=direct_environment_weights,
            direct_pair_mask=direct_pair_mask,
            direct_alternate_states=direct_alternate_states,
            symmetry_environment=symmetry_environment,
            symmetry_environment_weights=symmetry_environment_weights,
            symmetry_alternate_states=symmetry_alternate_states,
            physical_chi=physical_chi,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            zero = torch.zeros((), dtype=torch.float32, device=device)
            active = torch.nonzero(
                occupancies > args.nontrivial_occupancy, as_tuple=False
            ).flatten()
            if not len(active):
                return zero, zero, zero
            vdw_terms, rotamer_terms, symmetry_terms = [], [], []
            for index_tensor in active:
                candidate = conformers[int(index_tensor)]
                vdw_terms.append(soft_clash_penalty(
                    candidate,
                    direct_environment,
                    direct_environment_weights,
                    direct_alternate_states,
                    args.vdw_threshold,
                    direct_pair_mask,
                ))
                chis = physical_chi(candidate)
                rotamer_terms.append(torch.stack([
                    (
                        1.0 - torch.cos(
                            value - torch.tensor(
                                _canonical_centers(resname, chi_index),
                                dtype=value.dtype, device=device,
                            )
                        )
                    ).min() * (
                        30.0 / canonical_width_degrees(resname, chi_index)
                    ) ** 2
                    for chi_index, value in enumerate(chis)
                ]).sum())
                symmetry_terms.append(soft_clash_barrier_penalty(
                    candidate,
                    symmetry_environment,
                    symmetry_environment_weights,
                    symmetry_alternate_states,
                    args.clash_threshold,
                    args.symmetry_hard_threshold,
                    args.symmetry_barrier_buffer,
                    args.symmetry_barrier_scale,
                ))
            return (
                torch.stack(vdw_terms).sum(),
                torch.stack(rotamer_terms).sum(),
                torch.stack(symmetry_terms).sum(),
            )

        # The stored denoiser target is rendered from deposited coordinates with
        # conformer-specific B factors.  The optimizer uses one differentiable
        # atom model for every moving conformer, so its navigation control must
        # be generated by that exact forward model.  Otherwise a control failure
        # can merely report a renderer mismatch rather than a bad landscape.
        control_chi = torch.stack([
            torch.zeros_like(true_delta), torch.zeros_like(true_delta),
            true_delta, true_delta,
        ])
        control_logits = torch.log(torch.tensor([
            (occ_a / occ_total) / 2.0, (occ_a / occ_total) / 2.0,
            (occ_b / occ_total) / 2.0, (occ_b / occ_total) / 2.0,
        ], dtype=torch.float32, device=device).clamp_min(1e-6))
        with torch.no_grad():
            native_synthetic, _ = render(control_chi, control_logits)
        target_vectors["synthetic"] = native_synthetic.detach()
        with torch.no_grad():
            target_vectors_by_blur["synthetic"] = {
                blur: render(control_chi, control_logits, blur)[0].detach()
                for blur in blur_levels
            }
        np.save(
            args.output / f"{short_key}_optimizer_synthetic_vector.npy",
            native_synthetic.cpu().numpy(),
        )
        np.savez_compressed(
            args.output / f"{short_key}_optimizer_synthetic_by_blur.npz",
            **{
                f"fwhm_{blur:g}A": vector.cpu().numpy()
                for blur, vector in target_vectors_by_blur["synthetic"].items()
            },
        )

        reachable_flat = reachable_coordinates.reshape(-1, 3)
        mask_support_flat = mask_support_coordinates.reshape(-1, 3)

        def position_is_scored(position: torch.Tensor) -> bool:
            if args.density_mask_mode == "sphere":
                return bool(
                    torch.linalg.vector_norm(
                        position
                        - torch.tensor(
                            site.center,
                            dtype=position.dtype,
                            device=position.device,
                        )
                    )
                    <= args.grid_radius
                )
            return bool(
                torch.cdist(
                    position.reshape(1, 3), mask_support_flat
                ).min()
                <= args.reachable_mask_padding
            )

        deposited_a_outside = sum(
            not position_is_scored(position) for position in kinematic_a
        )
        deposited_b_outside = sum(
            not position_is_scored(position) for position in deposited_b
        )
        reachable_outside = sum(
            not position_is_scored(position) for position in reachable_flat
        )

        midpoint_permutations = reference_permutations(names, residue.name)
        midpoint_permutation = min(
            midpoint_permutations,
            key=lambda permutation: float(
                (kinematic_a - kinematic_b[permutation])
                .square().sum(dim=-1).mean().detach().cpu()
            ),
        )
        kinematic_midpoint = (
            kinematic_a + kinematic_b[midpoint_permutation]
        ) / 2.0

        sites.append({
            "key": short_key,
            "resname": residue.name,
            "n_chi": len(spec["rotations"]),
            "true_delta": true_delta,
            "kinematic_a": kinematic_a,
            "kinematic_b": kinematic_b,
            "deposited_b": deposited_b,
            "names": names,
            "rmsd": endpoint_rmsd,
            "target_a": occ_a / occ_total,
            "target_b": occ_b / occ_total,
            "target_vectors": target_vectors,
            "target_vectors_by_blur": target_vectors_by_blur,
            "render": render,
            "render_single_contribution": render_single_contribution,
            "density_column_from_chi": density_column_from_chi,
            "coordinates_from_chi": coordinates_from_chi,
            "physics_terms": physics_terms,
            "physical_chi": physical_chi,
            "delta_from_physical_chi": delta_from_physical_chi,
            "base_physical_chi": base_physical_chi,
            "delta_direction": delta_direction,
            "selected_grid": selected_grid,
            "kinematic_midpoint": kinematic_midpoint,
            "midpoint_permutation": midpoint_permutation,
            "density_loss": density_loss,
            "density_weights": density_weights,
            "density_mask_voxels": int(density_mask.sum()),
            "density_mask_uses_deposited_coordinates": (
                args.density_mask_mode == "containing_volume"
            ),
            "reachable_canonical_states": int(
                reachable_coordinates.shape[0]
            ),
            "reachable_atoms_outside_mask": int(reachable_outside),
            "deposited_A_atoms_outside_mask": int(deposited_a_outside),
            "deposited_B_atoms_outside_mask": int(deposited_b_outside),
            "direct_environment": direct_environment,
            "direct_pair_mask": direct_pair_mask,
            "symmetry_environment": symmetry_environment,
            "symmetry_environment_weights": symmetry_environment_weights,
            "symmetry_alternate_states": symmetry_alternate_states,
        })
    if {site["key"] for site in sites} != wanted_sites:
        raise RuntimeError(
            f"requested {sorted(wanted_sites)}, built {sorted(site['key'] for site in sites)}"
        )

    config = {
        **vars(args),
        "optimizer_physics_environment_rule": (
            OPTIMIZER_PHYSICS_ENVIRONMENT_RULE
        ),
        "occupancy_freeze_implementation": OCCUPANCY_FREEZE_IMPLEMENTATION,
        "pdb": str(args.pdb), "mtz": str(args.mtz),
        "selection": str(args.selection) if args.selection else None,
        "checkpoint": str(args.checkpoint), "output": str(args.output),
        "transfer_endpoint_root": str(args.transfer_endpoint_root),
        "residual_probe_endpoints": (
            str(args.residual_probe_endpoints)
            if args.residual_probe_endpoints else None
        ),
        "residual_probe_ensembles": (
            str(args.residual_probe_ensembles)
            if args.residual_probe_ensembles else None
        ),
        "targets": targets, "sites": sorted(wanted_sites),
        "resolved_density_masks": {
            site["key"]: {
                "mode": args.density_mask_mode,
                "weight_mode": args.density_weight_mode,
                "voxel_count": site["density_mask_voxels"],
                "uses_deposited_coordinates": (
                    site["density_mask_uses_deposited_coordinates"]
                ),
                "reachable_canonical_states": (
                    site["reachable_canonical_states"]
                ),
                "reachable_atoms_outside_mask": (
                    site["reachable_atoms_outside_mask"]
                ),
                "deposited_A_atoms_outside_mask": (
                    site["deposited_A_atoms_outside_mask"]
                ),
                "deposited_B_atoms_outside_mask": (
                    site["deposited_B_atoms_outside_mask"]
                ),
                "density_weight_min": float(
                    site["density_weights"].min().detach().cpu()
                ),
                "density_weight_max": float(
                    site["density_weights"].max().detach().cpu()
                ),
                "density_weight_mean": float(
                    site["density_weights"].mean().detach().cpu()
                ),
            }
            for site in sites
        },
        "resolved_density_schedules": {
            site["key"]: [
                {
                    "blur_fwhm_angstrom": blur_fwhm,
                    "learning_rate": stage_lr,
                    "steps": stage_steps,
                }
                for blur_fwhm, stage_lr, stage_steps in _production_density_schedule(
                    site["n_chi"], args.n_steps, args.lr,
                    args.per_residue_class_schedule,
                    args.four_chi_stage_steps,
                )
            ]
            for site in sites
        },
        "interpretation": (
            "untouched test proteins; held-out denoiser and optimizer generalization"
            if args.selection else "2O1K was in denoiser training; integration upper bound only"
        ),
    }
    _atomic_json(args.output / "run_config.json", config)
    calibration_rows = []
    physics_calibration_failures = []
    zero_logits = torch.zeros(args.K, device=device)
    for site in sites:
        chi_a = torch.zeros(site["n_chi"], device=device)
        chi_b = site["true_delta"]
        ab_chi = torch.stack([chi_a, chi_a, chi_b, chi_b])
        ab_logits = torch.log(torch.tensor([
            site["target_a"] / 2, site["target_a"] / 2,
            site["target_b"] / 2, site["target_b"] / 2,
        ], device=device).clamp_min(1e-6))
        candidates = {
            "A_only": (torch.stack([chi_a] * args.K), zero_logits),
            "B_only": (torch.stack([chi_b] * args.K), zero_logits),
            "A_plus_B": (ab_chi, ab_logits),
        }
        physics_values = {}
        for label, candidate in (
            ("A", site["kinematic_a"]), ("kinematic_B", site["kinematic_b"])
        ):
            vdw, rotamer, symmetry = site["physics_terms"](
                [candidate], torch.ones(1, device=device)
            )
            physics_values[label] = {
                "vdw": float(vdw.detach().cpu()),
                "rotamer": float(rotamer.detach().cpu()),
                "symmetry": float(symmetry.detach().cpu()),
                "soft": float((
                    args.lambda_vdw * vdw
                    + args.lambda_rot * rotamer
                    + args.lambda_clash * symmetry
                ).detach().cpu()),
            }
        physics_gap = (
            physics_values["kinematic_B"]["soft"] - physics_values["A"]["soft"]
        )
        symmetry_probe_loss = float("nan")
        if site["symmetry_environment"].numel():
            translated_probe = (
                site["kinematic_a"]
                + site["symmetry_environment"][0]
                - site["kinematic_a"][0]
            )
            _probe_vdw, _probe_rotamer, probe_symmetry = site["physics_terms"](
                [translated_probe], torch.ones(1, device=device)
            )
            symmetry_probe_loss = float(probe_symmetry.detach().cpu())
        physics_pass = physics_gap <= args.physics_calibration_max_gap
        if physics_enabled and not physics_pass:
            physics_calibration_failures.append({
                "site": site["key"],
                "B_minus_A_soft_physics": physics_gap,
                "A": physics_values["A"],
                "kinematic_B": physics_values["kinematic_B"],
            })
        for target_label in targets:
            values = {}
            with torch.no_grad():
                for candidate, (chis, logits) in candidates.items():
                    rendered, _coordinates = site["render"](chis, logits)
                    values[candidate] = float(site["density_loss"](
                        rendered, site["target_vectors"][target_label]
                    ).cpu())
            calibration_rows.append({
                "site": site["key"], "target": target_label,
                "loss_A_only": values["A_only"], "loss_B_only": values["B_only"],
                "loss_A_plus_B": values["A_plus_B"],
                "A_plus_B_best": values["A_plus_B"] < min(values["A_only"], values["B_only"]),
                "kinematic_to_deposited_B_rmsd": float(site["rmsd"](
                    site["kinematic_b"], site["deposited_b"]
                ).cpu()),
                "physics_A_vdw": physics_values["A"]["vdw"],
                "physics_A_rotamer": physics_values["A"]["rotamer"],
                "physics_A_symmetry": physics_values["A"]["symmetry"],
                "physics_A_soft": physics_values["A"]["soft"],
                "physics_B_vdw": physics_values["kinematic_B"]["vdw"],
                "physics_B_rotamer": physics_values["kinematic_B"]["rotamer"],
                "physics_B_symmetry": physics_values["kinematic_B"]["symmetry"],
                "symmetry_invariant_atoms": int(
                    site["symmetry_environment"].shape[0]
                ),
                "symmetry_alternate_residue_groups": len(
                    site["symmetry_alternate_states"]
                ),
                "symmetry_overlap_probe_loss": symmetry_probe_loss,
                "physics_B_soft": physics_values["kinematic_B"]["soft"],
                "physics_B_minus_A": physics_gap,
                "physics_calibration_pass": physics_pass,
            })
    _atomic_csv(args.output / "calibration.csv", calibration_rows)
    if physics_calibration_failures:
        _atomic_json(args.output / "physics_calibration_failures.json", {
            "status": "failed",
            "maximum_allowed_B_minus_A": args.physics_calibration_max_gap,
            "failures": physics_calibration_failures,
        })
        raise RuntimeError(
            "soft-physics calibration failed: "
            + "; ".join(
                f"{row['site']} B-A={row['B_minus_A_soft_physics']:.4f}"
                for row in physics_calibration_failures
            )
        )
    if args.residual_minor_probe:
        _run_residual_minor_probe(args, sites, device)
        return
    if args.calibration_only:
        _atomic_json(args.output / "stage_manifest.json", {
            "status": "complete", "calibration_only": True,
            "calibration_rows": len(calibration_rows),
        })
        print(json.dumps({"status": "complete", "calibration": calibration_rows}, indent=2))
        return

    if args.initialization_only:
        initialization_rows = []
        for site_index, site in enumerate(sites):
            for start in range(args.n_starts):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_000 * site_index + start
                )
                initial_chi = _initialize_chi_offsets(
                    mode=args.initialization_mode,
                    resname=site["resname"],
                    n_chi=site["n_chi"],
                    K=args.K,
                    base_physical_chi=site["base_physical_chi"],
                    delta_from_physical_chi=site["delta_from_physical_chi"],
                    generator=generator,
                    device=device,
                    jitter_degrees=args.initialization_jitter_degrees,
                )
                with torch.no_grad():
                    initial_coordinates = [
                        site["coordinates_from_chi"](wrap_angles(row))
                        for row in initial_chi
                    ]
                    initial_physical_chi = torch.stack([
                        site["physical_chi"](coordinates)
                        for coordinates in initial_coordinates
                    ])
                for slot, coordinates in enumerate(initial_coordinates):
                    initialization_rows.append({
                        "site": site["key"],
                        "start": start,
                        "slot": slot,
                        "initialization_mode": args.initialization_mode,
                        "base_physical_chi_degrees": ";".join(
                            f"{value:.9g}"
                            for value in np.degrees(
                                site["base_physical_chi"].cpu().numpy()
                            )
                        ),
                        "delta_direction": ";".join(
                            f"{value:.9g}"
                            for value in site["delta_direction"].cpu().numpy()
                        ),
                        "initial_delta_chi_radians": ";".join(
                            f"{value:.9g}"
                            for value in initial_chi[slot].detach().cpu().numpy()
                        ),
                        "initial_physical_chi_degrees": ";".join(
                            f"{value:.9g}"
                            for value in np.degrees(
                                initial_physical_chi[slot].cpu().numpy()
                            )
                        ),
                        "initial_rmsd_to_A": float(site["rmsd"](
                            coordinates, site["kinematic_a"]
                        ).cpu()),
                        "initial_rmsd_to_B": float(site["rmsd"](
                            coordinates, site["kinematic_b"]
                        ).cpu()),
                        "target_A_occupancy": site["target_a"],
                        "target_B_occupancy": site["target_b"],
                    })
        _atomic_csv(
            args.output / "initialization_only.csv", initialization_rows
        )
        _atomic_json(args.output / "stage_manifest.json", {
            "status": "complete",
            "initialization_only": True,
            "rows": len(initialization_rows),
        })
        print(json.dumps({
            "status": "complete",
            "initialization_rows": len(initialization_rows),
        }))
        return

    if args.initialization_tests:
        if targets != ["denoised"] or len(sites) != 1:
            raise ValueError(
                "--initialization-tests requires exactly --target denoised and one site"
            )
        site = sites[0]
        if site["key"] != "3A1C_B_ARG447":
            raise ValueError("initialization tests are gated to 3A1C_B_ARG447")

        schedule = ((4.0, 1.0, 100), (2.0, 0.1, 100), (0.0, 0.01, 100))

        dunbrack_initializations = []
        for rotamer_rank, (probability, chi_degrees) in enumerate(
            DUNBRACK_3A1C_ARG447_TOP10, start=1
        ):
            desired = torch.deg2rad(torch.tensor(
                chi_degrees, dtype=torch.float32, device=device
            ))
            base_delta = site["delta_from_physical_chi"](desired).detach()
            for perturbation in range(5):
                dunbrack_initializations.append({
                    "source": f"dunbrack_rank_{rotamer_rank}",
                    "source_rank": rotamer_rank,
                    "source_probability": probability,
                    "source_endpoint_start": -1,
                    "source_endpoint_loss": float("nan"),
                    "base_physical_chi_degrees": list(chi_degrees),
                    "base_delta": base_delta,
                    "perturbation": perturbation,
                })

        source_structure = gemmi.read_structure(str(root / "data" / "2O1K.pdb"))

        def transferred_physical_chi(
            chain_name: str, endpoint_delta: torch.Tensor
        ) -> torch.Tensor:
            source_residue = next(
                residue
                for chain in source_structure[0] if chain.name == chain_name
                for residue in chain if residue.seqid.num == 129
            )
            source_spec = CHI_SPECS["ARG"]
            source_a = _alt_atom_map(source_residue, "A", device)
            source_b_atoms = [
                atom for atom in source_residue
                if atom.altloc == "B"
                and atom.element.name != "H"
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            source_names = [atom.name.strip() for atom in source_b_atoms]
            source_template = torch.stack([
                source_a[name] for name in source_names
            ])
            source_fixed = {
                name: value for name, value in source_a.items()
                if name not in source_names
            }
            source_coordinates = torsion_to_coords(
                source_template,
                source_names,
                endpoint_delta,
                list(source_spec["rotations"]),
                source_fixed,
            )
            lookup = dict(source_fixed)
            lookup.update({
                name: source_coordinates[index]
                for index, name in enumerate(source_names)
            })
            return torch.stack([
                wrap_angles(dihedral(*(lookup[name] for name in quartet)) - torch.pi)
                for quartet in source_spec["dihedrals"]
            ]).detach()

        transfer_initializations = []
        for source_site, chain_name in (("A_ARG129", "A"), ("B_ARG129", "B")):
            endpoint_path = (
                args.transfer_endpoint_root / f"endpoints_{source_site}.csv"
            )
            with endpoint_path.open(newline="") as handle:
                source_rows = sorted(
                    csv.DictReader(handle),
                    key=lambda row: float(row["final_loss"]),
                )[:5]
            for source_rank, source_row in enumerate(source_rows, start=1):
                endpoint_delta = torch.tensor(
                    [
                        float(value)
                        for value in source_row["final_chi_radians"].split(";")
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                physical = transferred_physical_chi(chain_name, endpoint_delta)
                base_delta = site["delta_from_physical_chi"](physical).detach()
                for perturbation in range(5):
                    transfer_initializations.append({
                        "source": source_site,
                        "source_rank": source_rank,
                        "source_probability": float("nan"),
                        "source_endpoint_start": int(source_row["start"]),
                        "source_endpoint_loss": float(source_row["final_loss"]),
                        "base_physical_chi_degrees": torch.rad2deg(physical)
                        .cpu().tolist(),
                        "base_delta": base_delta,
                        "perturbation": perturbation,
                    })

        if len(dunbrack_initializations) != 50 or len(transfer_initializations) != 50:
            raise RuntimeError("each initialization test must contain exactly 50 starts")

        def audit_initialization_endpoint(
            all_chi: torch.Tensor,
            logits: torch.Tensor,
            best_joint_loss: float,
            density_stage_loss: float,
        ) -> dict:
            with torch.no_grad():
                rendered, coordinates = site["render"](all_chi, logits)
                occupancy_tensor = torch.softmax(logits, dim=0)
                occupancies = occupancy_tensor.cpu().numpy()
                final_density_loss = float(site["density_loss"](
                    rendered, site["target_vectors"]["denoised"]
                ).cpu())
                final_vdw, final_rotamer, final_symmetry = site["physics_terms"](
                    coordinates, occupancy_tensor
                )
                rmsd_a = np.asarray([
                    float(site["rmsd"](xyz, site["kinematic_a"]).cpu())
                    for xyz in coordinates
                ])
                rmsd_b = np.asarray([
                    float(site["rmsd"](xyz, site["kinematic_b"]).cpu())
                    for xyz in coordinates
                ])
                direct_minima, symmetry_minima = [], []
                canonical_flags, rotamer_deviations = [], []
                for xyz in coordinates:
                    if site["direct_environment"].numel():
                        distances = torch.cdist(xyz, site["direct_environment"])
                        direct_minima.append(float(
                            distances.masked_select(site["direct_pair_mask"])
                            .min().cpu()
                        ))
                    else:
                        direct_minima.append(float("nan"))
                    if site["symmetry_environment"].numel():
                        symmetry_minima.append(float(torch.cdist(
                            xyz, site["symmetry_environment"]
                        ).min().cpu()))
                    else:
                        symmetry_minima.append(float("nan"))
                    physical = site["physical_chi"](xyz)
                    deviations = []
                    for chi_index, value in enumerate(physical):
                        centers = torch.tensor(
                            _canonical_centers(site["resname"], chi_index),
                            dtype=value.dtype, device=device,
                        )
                        deviations.append(float(torch.rad2deg(
                            torch.abs(wrap_angles(value - centers)).min()
                        ).cpu()))
                    maximum_deviation = max(deviations)
                    rotamer_deviations.append(maximum_deviation)
                    canonical_flags.append(all(
                        deviation <= canonical_width_degrees(
                            site["resname"], chi_index
                        )
                        for chi_index, deviation in enumerate(deviations)
                    ))

            assignments = []
            for occupancy, distance_a, distance_b in zip(
                occupancies, rmsd_a, rmsd_b
            ):
                if occupancy <= args.nontrivial_occupancy:
                    assignments.append("inactive")
                elif distance_a < 1.0 and distance_a <= distance_b:
                    assignments.append("A")
                elif distance_b < 1.0:
                    assignments.append("B")
                else:
                    assignments.append("other")
            predicted_a = float(sum(
                occupancy for occupancy, label in zip(occupancies, assignments)
                if label == "A"
            ))
            predicted_b = float(sum(
                occupancy for occupancy, label in zip(occupancies, assignments)
                if label == "B"
            ))
            found_a = any(
                occupancy > 0.1 and label == "A"
                for occupancy, label in zip(occupancies, assignments)
            )
            found_b = any(
                occupancy > 0.1 and label == "B"
                for occupancy, label in zip(occupancies, assignments)
            )
            occupancy_accurate = (
                abs(predicted_a - site["target_a"]) <= args.occupancy_tolerance
                and abs(predicted_b - site["target_b"]) <= args.occupancy_tolerance
            )
            active = [
                index for index, occupancy in enumerate(occupancies)
                if occupancy > args.nontrivial_occupancy
            ]
            endpoint_physical_valid = bool(active) and all(
                direct_minima[index] >= 2.0
                and (
                    math.isnan(symmetry_minima[index])
                    or symmetry_minima[index] >= 2.0
                )
                and canonical_flags[index]
                for index in active
            )
            both_valid_occupancies = found_a and found_b and occupancy_accurate
            return {
                "density_stage_loss": density_stage_loss,
                "best_joint_loss": best_joint_loss,
                "final_density_loss": final_density_loss,
                "occupancies": ";".join(f"{value:.9g}" for value in occupancies),
                "rmsd_to_A": ";".join(f"{value:.9g}" for value in rmsd_a),
                "rmsd_to_B": ";".join(f"{value:.9g}" for value in rmsd_b),
                "assignments": ";".join(assignments),
                "predicted_A_occupancy": predicted_a,
                "predicted_B_occupancy": predicted_b,
                "found_A": found_a,
                "found_B": found_b,
                "both_with_valid_occupancies": both_valid_occupancies,
                "occupancy_accurate": occupancy_accurate,
                "direct_min_distances": ";".join(
                    f"{value:.9g}" for value in direct_minima
                ),
                "symmetry_min_distances": ";".join(
                    f"{value:.9g}" for value in symmetry_minima
                ),
                "rotamer_max_deviation_degrees": ";".join(
                    f"{value:.9g}" for value in rotamer_deviations
                ),
                "canonical_flags": ";".join(str(value) for value in canonical_flags),
                "endpoint_physical_valid": endpoint_physical_valid,
                "strict_joint_success": (
                    both_valid_occupancies and endpoint_physical_valid
                ),
                "final_vdw_loss": float(final_vdw.cpu()),
                "final_rotamer_loss": float(final_rotamer.cpu()),
                "final_symmetry_loss": float(final_symmetry.cpu()),
                "final_chi_radians": "|".join(
                    ";".join(f"{value:.9g}" for value in chi)
                    for chi in all_chi.detach().cpu().numpy()
                ),
                "rmsd_definition": "sqrt(mean_atoms(sum_xyz(error^2)))",
            }

        def run_initialization_test(
            test_label: str, initializations: list[dict], seed_offset: int
        ) -> list[dict]:
            result_path = args.output / f"{test_label}_starts.csv"
            result_rows = []
            if result_path.exists() and not args.force:
                with result_path.open(newline="") as handle:
                    result_rows = list(csv.DictReader(handle))
            for start in range(len(result_rows), len(initializations)):
                initialization = initializations[start]
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + seed_offset + start
                )
                all_chi = (
                    initialization["base_delta"][None].repeat(args.K, 1)
                    + 0.1 * torch.randn(
                        (args.K, site["n_chi"]),
                        generator=generator,
                        device=device,
                    )
                ).requires_grad_(True)
                logits = torch.zeros(args.K, device=device, requires_grad=True)
                for blur, learning_rate, steps in schedule:
                    optimizer = torch.optim.Adam(
                        [all_chi, logits], lr=learning_rate
                    )
                    target = site["target_vectors_by_blur"]["denoised"][blur]
                    for _step in range(steps):
                        optimizer.zero_grad(set_to_none=True)
                        density, _coordinates = site["render"](
                            all_chi, logits, blur
                        )
                        loss = site["density_loss"](density, target)
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            all_chi.copy_(wrap_angles(all_chi))
                with torch.no_grad():
                    density_stage_rendered, _ = site["render"](all_chi, logits)
                    density_stage_loss = float(site["density_loss"](
                        density_stage_rendered,
                        site["target_vectors"]["denoised"],
                    ).cpu())

                optimizer = torch.optim.Adam(
                    [all_chi, logits],
                    lr=args.lr * args.physics_refinement_lr_scale,
                )
                best_joint_loss = float("inf")
                best_chi = all_chi.detach().clone()
                best_logits = logits.detach().clone()
                for _step in range(args.physics_refinement_steps):
                    optimizer.zero_grad(set_to_none=True)
                    density, current_coordinates = site["render"](all_chi, logits)
                    density_loss = site["density_loss"](
                        density, site["target_vectors"]["denoised"]
                    )
                    current_occupancies = torch.softmax(logits, dim=0)
                    vdw_loss, rotamer_loss, symmetry_loss = site["physics_terms"](
                        current_coordinates, current_occupancies
                    )
                    loss = (
                        density_loss
                        + args.lambda_vdw * vdw_loss
                        + args.lambda_rot * rotamer_loss
                        + args.lambda_clash * symmetry_loss
                    )
                    current_loss = float(loss.detach().cpu())
                    if current_loss < best_joint_loss:
                        best_joint_loss = current_loss
                        best_chi = all_chi.detach().clone()
                        best_logits = logits.detach().clone()
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        all_chi.copy_(wrap_angles(all_chi))
                with torch.no_grad():
                    all_chi.copy_(best_chi)
                    logits.copy_(best_logits)
                metrics = audit_initialization_endpoint(
                    all_chi, logits, best_joint_loss, density_stage_loss
                )
                result_rows.append({
                    "test": test_label,
                    "start": start,
                    "source": initialization["source"],
                    "source_rank": initialization["source_rank"],
                    "source_probability": initialization["source_probability"],
                    "source_endpoint_start": initialization[
                        "source_endpoint_start"
                    ],
                    "source_endpoint_loss": initialization[
                        "source_endpoint_loss"
                    ],
                    "perturbation": initialization["perturbation"],
                    "base_physical_chi_degrees": ";".join(
                        f"{value:.9g}"
                        for value in initialization["base_physical_chi_degrees"]
                    ),
                    **metrics,
                })
                _atomic_csv(result_path, result_rows)
                _atomic_json(args.output / "stage_manifest.json", {
                    "status": "running",
                    "test": test_label,
                    "completed_starts": start + 1,
                })
                print(json.dumps({
                    "test": test_label, "completed_starts": start + 1
                }), flush=True)
            return result_rows

        dunbrack_rows = run_initialization_test(
            "dunbrack", dunbrack_initializations, 3_000_000
        )
        transfer_rows = run_initialization_test(
            "transfer", transfer_initializations, 4_000_000
        )

        def summarize_initialization(label: str, rows: list[dict]) -> dict:
            truth = lambda row, key: str(row[key]) == "True"
            return {
                "test": label,
                "starts": len(rows),
                "found_A": sum(truth(row, "found_A") for row in rows),
                "found_B": sum(truth(row, "found_B") for row in rows),
                "both_with_valid_occupancies": sum(
                    truth(row, "both_with_valid_occupancies") for row in rows
                ),
                "strict_joint_success": sum(
                    truth(row, "strict_joint_success") for row in rows
                ),
                "endpoint_physical_valid": sum(
                    truth(row, "endpoint_physical_valid") for row in rows
                ),
            }

        comparison = {
            "status": "complete",
            "site": site["key"],
            "dunbrack_source": {
                "library": "Dunbrack 2010 backbone-dependent",
                "target_phi_degrees": -76.89901050950502,
                "target_psi_degrees": -29.883162887937022,
                "queried_grid_phi_degrees": -80,
                "queried_grid_psi_degrees": -30,
                "top_rotamers": [
                    {"probability": probability, "chi_degrees": list(chis)}
                    for probability, chis in DUNBRACK_3A1C_ARG447_TOP10
                ],
            },
            "random_baseline": {
                "starts": 50,
                "strict_joint_success": 0,
            },
            "dunbrack": summarize_initialization("dunbrack", dunbrack_rows),
            "transfer": summarize_initialization("transfer", transfer_rows),
        }
        _atomic_json(args.output / "initialization_comparison.json", comparison)
        _atomic_json(args.output / "stage_manifest.json", {
            "status": "complete",
            "dunbrack_strict_joint_success": comparison["dunbrack"][
                "strict_joint_success"
            ],
            "transfer_strict_joint_success": comparison["transfer"][
                "strict_joint_success"
            ],
        })
        print(json.dumps(comparison, indent=2), flush=True)
        return

    if args.sequential_two_stage:
        if targets != ["denoised"] or len(sites) != 1:
            raise ValueError(
                "--sequential-two-stage requires exactly --target denoised and one site"
            )
        site = sites[0]
        if site["key"] != "3A1C_B_ARG447":
            raise ValueError(
                "the diagnostic gate is restricted to 3A1C_B_ARG447"
            )
        if args.n_starts != 50:
            raise ValueError("the sequential diagnostic requires --n-starts 50")

        schedule = ((4.0, 1.0, 100), (2.0, 0.1, 100), (0.0, 0.01, 100))
        target_by_blur = site["target_vectors_by_blur"]["denoised"]

        def run_single_stage(
            stage_label: str,
            targets_for_stage: dict[float, torch.Tensor],
            maximum_occupancy: float,
            seed_offset: int,
        ) -> tuple[dict, torch.Tensor]:
            path = args.output / f"{stage_label}_starts.csv"
            stage_rows = []
            if path.exists() and not args.force:
                with path.open(newline="") as handle:
                    stage_rows = list(csv.DictReader(handle))
            for start in range(len(stage_rows), args.n_starts):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + seed_offset + start
                )
                chi = torch.randn(
                    site["n_chi"], generator=generator, device=device
                ).requires_grad_(True)
                occupancy_logit = torch.zeros((), device=device, requires_grad=True)
                best_loss = float("inf")
                for blur, learning_rate, steps in schedule:
                    # Reset Adam at every coarse-to-fine transition.
                    optimizer = torch.optim.Adam(
                        [chi, occupancy_logit], lr=learning_rate
                    )
                    target = targets_for_stage[blur]
                    for _step in range(steps):
                        optimizer.zero_grad(set_to_none=True)
                        prediction, _coordinates = site[
                            "render_single_contribution"
                        ](chi, blur)
                        occupancy = maximum_occupancy * torch.sigmoid(
                            occupancy_logit
                        )
                        loss = (occupancy * prediction - target).square().mean()
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            chi.copy_(wrap_angles(chi))
                        best_loss = min(best_loss, float(loss.detach().cpu()))
                with torch.no_grad():
                    prediction, coordinates = site[
                        "render_single_contribution"
                    ](chi, 0.0)
                    occupancy = maximum_occupancy * torch.sigmoid(occupancy_logit)
                    final_loss = float((
                        occupancy * prediction - targets_for_stage[0.0]
                    ).square().mean().cpu())
                    rmsd_a = float(torch.sqrt(
                        (coordinates - site["kinematic_a"]).square()
                        .sum(dim=-1).mean()
                    ).cpu())
                    rmsd_b = float(torch.sqrt(
                        (coordinates - site["kinematic_b"]).square()
                        .sum(dim=-1).mean()
                    ).cpu())
                    chi_values = chi.detach().cpu().numpy().copy()
                stage_rows.append({
                    "stage": stage_label,
                    "start": start,
                    "final_density_loss": final_loss,
                    "best_density_loss": best_loss,
                    "occupancy": float(occupancy.cpu()),
                    "rmsd_to_A": rmsd_a,
                    "rmsd_to_B": rmsd_b,
                    "chi_radians": ";".join(
                        f"{value:.9g}" for value in chi_values
                    ),
                    "schedule": "4A@1.0x100;2A@0.1x100;0A@0.01x100",
                })
                _atomic_csv(path, stage_rows)
                _atomic_json(args.output / "stage_manifest.json", {
                    "status": "running",
                    "stage": stage_label,
                    "completed_starts": start + 1,
                })
                print(json.dumps({
                    "stage": stage_label, "completed_starts": start + 1
                }), flush=True)
            winner = min(
                stage_rows, key=lambda row: float(row["final_density_loss"])
            )
            winner_chi = torch.tensor(
                [float(value) for value in winner["chi_radians"].split(";")],
                dtype=torch.float32, device=device,
            )
            _atomic_json(args.output / f"{stage_label}_winner.json", {
                **winner,
                "winner_selected_by": "lowest final full-resolution density loss",
            })
            return winner, winner_chi

        stage1_winner, locked_chi = run_single_stage(
            "stage1_single", target_by_blur, 1.0, 0
        )
        locked_occupancy = float(stage1_winner["occupancy"])
        residual_by_blur = {}
        with torch.no_grad():
            for blur, target in target_by_blur.items():
                locked_contribution, _ = site["render_single_contribution"](
                    locked_chi, blur
                )
                residual_by_blur[blur] = (
                    target - locked_occupancy * locked_contribution
                ).detach()
        np.savez_compressed(
            args.output / "stage2_residual_density.npz",
            **{
                f"blur_{blur:g}A": values.cpu().numpy()
                for blur, values in residual_by_blur.items()
            },
        )

        stage2_winner, residual_chi = run_single_stage(
            "stage2_residual",
            residual_by_blur,
            max(1.0 - locked_occupancy, 1e-3),
            1_000_000,
        )
        residual_occupancy = float(stage2_winner["occupancy"])

        generator = torch.Generator(device=device).manual_seed(args.seed + 2_000_000)
        empty_chi = torch.randn(
            (2, site["n_chi"]), generator=generator, device=device
        )
        all_chi = torch.cat((
            locked_chi[None], residual_chi[None], empty_chi
        )).requires_grad_(True)
        initial_weights = torch.tensor([
            max(locked_occupancy, 1e-3),
            max(residual_occupancy, 1e-3),
            1e-3,
            1e-3,
        ], dtype=torch.float32, device=device)
        logits = torch.log(initial_weights).requires_grad_(True)
        optimizer = torch.optim.Adam(
            [all_chi, logits],
            lr=args.lr * args.physics_refinement_lr_scale,
        )
        best_joint_loss = float("inf")
        best_joint_chi = all_chi.detach().clone()
        best_joint_logits = logits.detach().clone()
        for step in range(args.physics_refinement_steps):
            optimizer.zero_grad(set_to_none=True)
            density, current_coordinates = site["render"](all_chi, logits)
            density_loss = site["density_loss"](
                density, site["target_vectors"]["denoised"]
            )
            current_occupancies = torch.softmax(logits, dim=0)
            vdw_loss, rotamer_loss, symmetry_loss = site["physics_terms"](
                current_coordinates, current_occupancies
            )
            loss = (
                density_loss
                + args.lambda_vdw * vdw_loss
                + args.lambda_rot * rotamer_loss
                + args.lambda_clash * symmetry_loss
            )
            current_joint_loss = float(loss.detach().cpu())
            if current_joint_loss < best_joint_loss:
                best_joint_loss = current_joint_loss
                best_joint_chi = all_chi.detach().clone()
                best_joint_logits = logits.detach().clone()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                all_chi.copy_(wrap_angles(all_chi))
            if (step + 1) % 25 == 0:
                _atomic_json(args.output / "stage3_progress.json", {
                    "status": "running",
                    "completed_steps": step + 1,
                    "best_joint_loss": best_joint_loss,
                    "occupancies": torch.softmax(logits, dim=0)
                    .detach().cpu().tolist(),
                    "chi_radians": all_chi.detach().cpu().tolist(),
                })

        with torch.no_grad():
            final_candidate_density, final_candidate_coordinates = site["render"](
                all_chi, logits
            )
            final_candidate_occupancies = torch.softmax(logits, dim=0)
            candidate_vdw, candidate_rotamer, candidate_symmetry = site[
                "physics_terms"
            ](final_candidate_coordinates, final_candidate_occupancies)
            candidate_joint_loss = float((
                site["density_loss"](
                    final_candidate_density,
                    site["target_vectors"]["denoised"],
                )
                + args.lambda_vdw * candidate_vdw
                + args.lambda_rot * candidate_rotamer
                + args.lambda_clash * candidate_symmetry
            ).cpu())
            if candidate_joint_loss < best_joint_loss:
                best_joint_loss = candidate_joint_loss
                best_joint_chi = all_chi.detach().clone()
                best_joint_logits = logits.detach().clone()
            all_chi.copy_(best_joint_chi)
            logits.copy_(best_joint_logits)

        with torch.no_grad():
            rendered, coordinates = site["render"](all_chi, logits)
            occupancies = torch.softmax(logits, dim=0).cpu().numpy()
            final_density_loss = float(site["density_loss"](
                rendered, site["target_vectors"]["denoised"]
            ).cpu())
            final_vdw, final_rotamer, final_symmetry = site["physics_terms"](
                coordinates, torch.softmax(logits, dim=0)
            )
            rmsd_a = np.asarray([
                float(site["rmsd"](xyz, site["kinematic_a"]).cpu())
                for xyz in coordinates
            ])
            rmsd_b = np.asarray([
                float(site["rmsd"](xyz, site["kinematic_b"]).cpu())
                for xyz in coordinates
            ])
            direct_minima, symmetry_minima = [], []
            canonical_flags, rotamer_deviations = [], []
            for xyz in coordinates:
                if site["direct_environment"].numel():
                    distances = torch.cdist(xyz, site["direct_environment"])
                    direct_minima.append(float(
                        distances.masked_select(site["direct_pair_mask"]).min().cpu()
                    ))
                else:
                    direct_minima.append(float("nan"))
                if site["symmetry_environment"].numel():
                    symmetry_minima.append(float(torch.cdist(
                        xyz, site["symmetry_environment"]
                    ).min().cpu()))
                else:
                    symmetry_minima.append(float("nan"))
                chis = site["physical_chi"](xyz)
                deviations = []
                for chi_index, value in enumerate(chis):
                    centers = torch.tensor(
                        _canonical_centers(site["resname"], chi_index),
                        dtype=value.dtype, device=device,
                    )
                    deviations.append(float(torch.rad2deg(
                        torch.abs(wrap_angles(value - centers)).min()
                    ).cpu()))
                maximum_deviation = max(deviations)
                rotamer_deviations.append(maximum_deviation)
                canonical_flags.append(all(
                    deviation <= canonical_width_degrees(
                        site["resname"], chi_index
                    )
                    for chi_index, deviation in enumerate(deviations)
                ))

        assignments = []
        for occupancy, distance_a, distance_b in zip(occupancies, rmsd_a, rmsd_b):
            if occupancy <= args.nontrivial_occupancy:
                assignments.append("inactive")
            elif distance_a < 1.0 and distance_a <= distance_b:
                assignments.append("A")
            elif distance_b < 1.0:
                assignments.append("B")
            else:
                assignments.append("other")
        predicted_a = float(sum(
            occupancy for occupancy, label in zip(occupancies, assignments)
            if label == "A"
        ))
        predicted_b = float(sum(
            occupancy for occupancy, label in zip(occupancies, assignments)
            if label == "B"
        ))
        found_a = any(
            occupancy > 0.1 and label == "A"
            for occupancy, label in zip(occupancies, assignments)
        )
        found_b = any(
            occupancy > 0.1 and label == "B"
            for occupancy, label in zip(occupancies, assignments)
        )
        occupancy_accurate = (
            abs(predicted_a - site["target_a"]) <= args.occupancy_tolerance
            and abs(predicted_b - site["target_b"]) <= args.occupancy_tolerance
        )
        active_indices = [
            index for index, occupancy in enumerate(occupancies)
            if occupancy > args.nontrivial_occupancy
        ]
        endpoint_physical_valid = bool(active_indices) and all(
            direct_minima[index] >= 2.0
            and (
                math.isnan(symmetry_minima[index])
                or symmetry_minima[index] >= 2.0
            )
            and canonical_flags[index]
            for index in active_indices
        )
        strict_joint_success = (
            found_a and found_b and occupancy_accurate and endpoint_physical_valid
        )
        result = {
            "status": "complete",
            "site": site["key"],
            "stage1_winner": stage1_winner,
            "stage2_winner": stage2_winner,
            "stage3": {
                "best_joint_loss": best_joint_loss,
                "final_density_loss": final_density_loss,
                "final_vdw_loss": float(final_vdw.cpu()),
                "final_rotamer_loss": float(final_rotamer.cpu()),
                "final_symmetry_loss": float(final_symmetry.cpu()),
                "occupancies": occupancies.tolist(),
                "rmsd_to_A": rmsd_a.tolist(),
                "rmsd_to_B": rmsd_b.tolist(),
                "assignments": assignments,
                "predicted_A_occupancy": predicted_a,
                "predicted_B_occupancy": predicted_b,
                "found_A": found_a,
                "found_B": found_b,
                "occupancy_accurate": occupancy_accurate,
                "direct_min_distances": direct_minima,
                "symmetry_min_distances": symmetry_minima,
                "rotamer_max_deviation_degrees": rotamer_deviations,
                "canonical_flags": canonical_flags,
                "endpoint_physical_valid": endpoint_physical_valid,
                "strict_joint_success": strict_joint_success,
                "rmsd_definition": "sqrt(mean_atoms(sum_xyz(error^2)))",
                "chi_radians": all_chi.detach().cpu().tolist(),
            },
        }
        np.savez_compressed(
            args.output / "stage3_endpoints.npz",
            coordinates=np.stack([
                xyz.detach().cpu().numpy() for xyz in coordinates
            ]),
            chi_radians=all_chi.detach().cpu().numpy(),
            occupancies=occupancies,
            rmsd_to_A=rmsd_a,
            rmsd_to_B=rmsd_b,
        )
        _atomic_json(args.output / "sequential_two_stage_result.json", result)
        _atomic_json(args.output / "stage_manifest.json", {
            "status": "complete",
            "strict_joint_success": strict_joint_success,
        })
        print(json.dumps(result, indent=2), flush=True)
        return

    summaries = []
    for target_label in targets:
        for site_index, site in enumerate(sites):
            result_path = args.output / target_label / f"{site['key']}_starts.csv"
            rows = []
            if result_path.exists() and not args.force:
                with result_path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
            respawn_check_path = (
                args.output / target_label
                / f"{site['key']}_respawn_checks.csv"
            )
            respawn_event_path = (
                args.output / target_label
                / f"{site['key']}_respawn_events.csv"
            )
            respawn_checks = []
            respawn_events = []
            if args.respawn_cadence and respawn_check_path.exists():
                with respawn_check_path.open(newline="") as handle:
                    respawn_checks = list(csv.DictReader(handle))
                respawn_checks = [
                    row for row in respawn_checks
                    if int(row["start"]) < len(rows)
                ]
            if args.respawn_cadence and respawn_event_path.exists():
                with respawn_event_path.open(newline="") as handle:
                    respawn_events = list(csv.DictReader(handle))
                respawn_events = [
                    row for row in respawn_events
                    if int(row["start"]) < len(rows)
                ]
            for start in range(len(rows), args.n_starts):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_000 * site_index + start
                )
                all_chi = _initialize_chi_offsets(
                    mode=args.initialization_mode,
                    resname=site["resname"],
                    n_chi=site["n_chi"],
                    K=args.K,
                    base_physical_chi=site["base_physical_chi"],
                    delta_from_physical_chi=site["delta_from_physical_chi"],
                    generator=generator,
                    device=device,
                    jitter_degrees=args.initialization_jitter_degrees,
                ).requires_grad_(True)
                if args.seed_deposited_a:
                    with torch.no_grad():
                        # coordinates_from_chi uses deposited A as its zero-delta
                        # reference, so this is an exact local-basin stability test.
                        all_chi[0].zero_()
                with torch.no_grad():
                    initial_chi = all_chi.detach().clone()
                    initial_coordinates = [
                        site["coordinates_from_chi"](wrap_angles(row))
                        for row in initial_chi
                    ]
                    initial_physical_chi = torch.stack([
                        site["physical_chi"](coordinates)
                        for coordinates in initial_coordinates
                    ])
                    initial_rmsd_a = np.asarray([
                        float(site["rmsd"](
                            coordinates, site["kinematic_a"]
                        ).cpu())
                        for coordinates in initial_coordinates
                    ])
                    initial_rmsd_b = np.asarray([
                        float(site["rmsd"](
                            coordinates, site["kinematic_b"]
                        ).cpu())
                        for coordinates in initial_coordinates
                    ])
                logits = torch.zeros(
                    args.K, device=device,
                    requires_grad=True,
                )
                best_stage1_loss = float("inf")
                fixed_boundary_density_loss = float("nan")
                unfreeze_pre_density_loss = float("nan")
                unfreeze_post_density_loss = float("nan")
                fixed_boundary_occupancies = np.full(args.K, np.nan)
                fixed_boundary_chi = np.full((args.K, site["n_chi"]), np.nan)
                fixed_boundary_rmsd_a = np.full(args.K, np.nan)
                fixed_boundary_rmsd_b = np.full(args.K, np.nan)
                fixed_boundary_found_a = False
                fixed_boundary_found_b = False
                start_respawn_checks = []
                start_respawn_events = []
                respawned_slots = set()
                if args.record_stage1_trajectories:
                    trajectory_steps = [0]
                    trajectory_phases = [1]
                    trajectory_stage_indices = [0]
                    trajectory_blur_fwhm = [float("nan")]
                    trajectory_chi_learning_rates = [0.0]
                    trajectory_occupancy_learning_rates = [0.0]
                    trajectory_chi_noise_sd_degrees = [0.0]
                    trajectory_chi = [all_chi.detach().clone()]
                    trajectory_occupancies = [
                        torch.softmax(logits, dim=0).detach().clone()
                    ]
                    trajectory_density_loss_pre = []

                density_schedule = _production_density_schedule(
                    site["n_chi"], args.n_steps, args.lr,
                    args.per_residue_class_schedule,
                    args.four_chi_stage_steps,
                )
                total_stage1_steps = sum(
                    stage_steps for _blur, _lr, stage_steps in density_schedule
                )
                if args.fixed_occupancy_steps > total_stage1_steps:
                    raise ValueError(
                        "--fixed-occupancy-steps exceeds the resolved Stage-1 "
                        f"schedule ({args.fixed_occupancy_steps} > "
                        f"{total_stage1_steps})"
                    )

                global_stage1_step = 0
                for stage_index, (
                    blur_fwhm, stage_lr, stage_steps
                ) in enumerate(density_schedule):
                    # The frozen four-chi schedule intentionally resets Adam at
                    # each blur boundary. Within each stage, logits stay in Adam
                    # even while their group LR is zero, so moments remain warm.
                    occupancy_lr = (
                        0.0
                        if global_stage1_step < args.fixed_occupancy_steps
                        else stage_lr
                    )
                    optimizer = _stage1_adam(
                        all_chi,
                        logits,
                        chi_learning_rate=stage_lr,
                        occupancy_learning_rate=occupancy_lr,
                    )
                    if blur_fwhm == 0.0:
                        target_vector = site["target_vectors"][target_label]
                    else:
                        target_vector = site["target_vectors_by_blur"][
                            target_label
                        ][blur_fwhm]

                    for _stage_step in range(stage_steps):
                        optimizer.zero_grad(set_to_none=True)
                        if blur_fwhm == 0.0:
                            density, current_coordinates = site["render"](
                                all_chi, logits
                            )
                        else:
                            density, current_coordinates = site["render"](
                                all_chi, logits, blur_fwhm
                            )
                        density_loss = site["density_loss"](
                            density, target_vector
                        )
                        if (
                            args.fixed_occupancy_steps
                            and global_stage1_step == args.fixed_occupancy_steps
                            and math.isnan(unfreeze_pre_density_loss)
                        ):
                            unfreeze_pre_density_loss = float(
                                density_loss.detach().cpu()
                            )
                        if args.soft_physics:
                            current_occupancies = torch.softmax(logits, dim=0)
                            vdw_loss, rotamer_loss, symmetry_loss = site["physics_terms"](
                                current_coordinates, current_occupancies
                            )
                            loss = (
                                density_loss
                                + args.lambda_vdw * vdw_loss
                                + args.lambda_rot * rotamer_loss
                                + args.lambda_clash * symmetry_loss
                            )
                        else:
                            loss = density_loss
                        loss.backward()
                        optimizer.step()
                        global_stage1_step += 1
                        noise_sd_degrees = _apply_stage1_chi_noise_(
                            all_chi,
                            initial_sd_degrees=(
                                args.stage1_chi_noise_initial_degrees
                            ),
                            step=global_stage1_step,
                            total_steps=total_stage1_steps,
                            generator=generator,
                        )
                        with torch.no_grad():
                            all_chi.copy_(wrap_angles(all_chi))
                        best_stage1_loss = min(
                            best_stage1_loss, float(loss.detach().cpu())
                        )

                        if _respawn_due(
                            args.respawn_cadence,
                            global_stage1_step,
                            total_stage1_steps,
                        ):
                            with torch.no_grad():
                                if blur_fwhm == 0.0:
                                    check_rendered, check_coordinates = site[
                                        "render"
                                    ](all_chi, logits)
                                else:
                                    check_rendered, check_coordinates = site[
                                        "render"
                                    ](all_chi, logits, blur_fwhm)
                                native_columns = torch.stack([
                                    site["density_column_from_chi"](row)[0]
                                    for row in all_chi
                                ], dim=1)
                                current_occupancies = torch.softmax(
                                    logits, dim=0
                                )
                                active_slots = [
                                    index for index in range(args.K)
                                    if float(current_occupancies[index])
                                    > args.nontrivial_occupancy
                                ]
                                pair_diagnostics = []
                                for left, right in itertools.combinations(
                                    active_slots, 2
                                ):
                                    pair_diagnostics.append({
                                        "left": left,
                                        "right": right,
                                        "rmsd": float(site["rmsd"](
                                            check_coordinates[left],
                                            check_coordinates[right],
                                        ).cpu()),
                                        "condition": _gram_condition(
                                            native_columns[:, [left, right]]
                                        ),
                                    })
                                if pair_diagnostics:
                                    closest_pair = min(
                                        pair_diagnostics,
                                        key=lambda item: (
                                            item["rmsd"],
                                            item["left"],
                                            item["right"],
                                        ),
                                    )
                                    gram_pair = max(
                                        pair_diagnostics,
                                        key=lambda item: (
                                            item["condition"],
                                            -item["left"],
                                            -item["right"],
                                        ),
                                    )
                                else:
                                    closest_pair = {
                                        "left": -1,
                                        "right": -1,
                                        "rmsd": float("nan"),
                                        "condition": float("nan"),
                                    }
                                    gram_pair = dict(closest_pair)
                                full_condition = (
                                    _gram_condition(
                                        native_columns[:, active_slots]
                                    )
                                    if len(active_slots) >= 2
                                    else float("nan")
                                )
                                rmsd_triggered = (
                                    bool(pair_diagnostics)
                                    and closest_pair["rmsd"]
                                    < args.respawn_merge_rmsd
                                )
                                check_row = {
                                    "target": target_label,
                                    "site": site["key"],
                                    "start": start,
                                    "step": global_stage1_step,
                                    "stage_index": stage_index,
                                    "blur_fwhm_A": blur_fwhm,
                                    "configured_rmsd_threshold_A": (
                                        args.respawn_merge_rmsd
                                    ),
                                    "active_slot_threshold": (
                                        args.nontrivial_occupancy
                                    ),
                                    "active_slot_count": len(active_slots),
                                    "active_slots": ";".join(
                                        str(slot) for slot in active_slots
                                    ),
                                    "closest_pair_left": closest_pair["left"],
                                    "closest_pair_right": closest_pair["right"],
                                    "closest_pair_rmsd_A": (
                                        closest_pair["rmsd"]
                                    ),
                                    "rmsd_below_0p3": (
                                        closest_pair["rmsd"] < 0.3
                                    ),
                                    "rmsd_below_0p5": (
                                        closest_pair["rmsd"] < 0.5
                                    ),
                                    "rmsd_below_0p8": (
                                        closest_pair["rmsd"] < 0.8
                                    ),
                                    "gram_most_collinear_left": (
                                        gram_pair["left"]
                                    ),
                                    "gram_most_collinear_right": (
                                        gram_pair["right"]
                                    ),
                                    "gram_pair_condition_number": (
                                        gram_pair["condition"]
                                    ),
                                    "gram_pair_rmsd_A": gram_pair["rmsd"],
                                    "gram_full_condition_number": full_condition,
                                    "gram_condition_threshold": (
                                        RESPAWN_GRAM_CONDITION_THRESHOLD
                                    ),
                                    "gram_condition_triggered": (
                                        bool(pair_diagnostics)
                                        and
                                        gram_pair["condition"]
                                        >= RESPAWN_GRAM_CONDITION_THRESHOLD
                                    ),
                                    "rmsd_triggered": rmsd_triggered,
                                    "respawn_triggered": rmsd_triggered,
                                }
                                start_respawn_checks.append(check_row)

                            if rmsd_triggered:
                                left = int(closest_pair["left"])
                                right = int(closest_pair["right"])
                                with torch.no_grad():
                                    occupancies_before = torch.softmax(
                                        logits, dim=0
                                    )
                                    keeper = (
                                        left
                                        if (
                                            float(occupancies_before[left])
                                            > float(occupancies_before[right])
                                            or (
                                                float(occupancies_before[left])
                                                == float(occupancies_before[right])
                                                and left < right
                                            )
                                        )
                                        else right
                                    )
                                    freed = right if keeper == left else left
                                    freed_coordinates = check_coordinates[
                                        freed
                                    ].detach().clone()
                                    freed_pre_rmsd_a = float(site["rmsd"](
                                        freed_coordinates,
                                        site["kinematic_a"],
                                    ).cpu())
                                    freed_pre_rmsd_b = float(site["rmsd"](
                                        freed_coordinates,
                                        site["kinematic_b"],
                                    ).cpu())
                                    freed_pre_rmsd_midpoint = float(
                                        site["rmsd"](
                                            freed_coordinates,
                                            site["kinematic_midpoint"],
                                        ).cpu()
                                    )
                                    density_loss_pre_merge = float(
                                        site["density_loss"](
                                            check_rendered, target_vector
                                        ).cpu()
                                    )
                                    residual = target_vector - check_rendered
                                    peak_index = int(torch.argmax(residual))
                                    peak = site["selected_grid"][
                                        peak_index
                                    ].detach().clone()
                                    peak_magnitude = float(
                                        residual[peak_index].cpu()
                                    )

                                    def peak_to_state(reference):
                                        return float(
                                            torch.linalg.vector_norm(
                                                reference - peak[None, :],
                                                dim=1,
                                            ).min().cpu()
                                        )

                                    peak_to_a = peak_to_state(
                                        site["kinematic_a"]
                                    )
                                    peak_to_b = peak_to_state(
                                        site["kinematic_b"]
                                    )
                                    peak_to_midpoint = peak_to_state(
                                        site["kinematic_midpoint"]
                                    )
                                    merged_pair_occupancy = float(
                                        occupancies_before[keeper]
                                        + occupancies_before[freed]
                                    )
                                    keeper_occupancy_before = float(
                                        occupancies_before[keeper]
                                    )
                                    freed_occupancy_before = float(
                                        occupancies_before[freed]
                                    )

                                _merge_occupancies_for_respawn(
                                    logits, keeper, freed
                                )
                                with torch.no_grad():
                                    if blur_fwhm == 0.0:
                                        merged_rendered, _ = site["render"](
                                            all_chi, logits
                                        )
                                    else:
                                        merged_rendered, _ = site["render"](
                                            all_chi, logits, blur_fwhm
                                        )
                                    density_loss_post_merge = float(
                                        site["density_loss"](
                                            merged_rendered, target_vector
                                        ).cpu()
                                    )

                                (
                                    respawn_chi,
                                    peak_atom_index,
                                    peak_residual_distance,
                                ) = _inverse_kinematics_to_peak(
                                    all_chi[freed],
                                    peak,
                                    site["coordinates_from_chi"],
                                )
                                with torch.no_grad():
                                    all_chi[freed].copy_(respawn_chi)
                                # The zero-cost bookkeeping merge is measured
                                # above.  Give the inserted position exactly the
                                # mass of the slot it replaces, taken back from
                                # the keeper, so its density gradient is live
                                # without introducing an occupancy hyperparameter.
                                _merge_occupancies_for_respawn(
                                    logits,
                                    keeper,
                                    freed,
                                    floor=freed_occupancy_before,
                                )
                                _reset_adam_parameter_slice(
                                    optimizer, all_chi, freed
                                )
                                _reset_adam_parameter_slice(
                                    optimizer, logits, keeper
                                )
                                _reset_adam_parameter_slice(
                                    optimizer, logits, freed
                                )
                                with torch.no_grad():
                                    if blur_fwhm == 0.0:
                                        respawn_rendered, respawn_coordinates = (
                                            site["render"](all_chi, logits)
                                        )
                                    else:
                                        respawn_rendered, respawn_coordinates = (
                                            site["render"](
                                                all_chi, logits, blur_fwhm
                                            )
                                        )
                                    density_loss_post_respawn = float(
                                        site["density_loss"](
                                            respawn_rendered, target_vector
                                        ).cpu()
                                    )
                                    respawn_rmsd_a = float(site["rmsd"](
                                        respawn_coordinates[freed],
                                        site["kinematic_a"],
                                    ).cpu())
                                    respawn_rmsd_b = float(site["rmsd"](
                                        respawn_coordinates[freed],
                                        site["kinematic_b"],
                                    ).cpu())
                                    respawn_rmsd_midpoint = float(
                                        site["rmsd"](
                                            respawn_coordinates[freed],
                                            site["kinematic_midpoint"],
                                        ).cpu()
                                    )
                                respawned_slots.add(freed)
                                start_respawn_events.append({
                                    **check_row,
                                    "merge_trigger": (
                                        f"rmsd_below_"
                                        f"{args.respawn_merge_rmsd:g}_A"
                                    ),
                                    "keeper_slot": keeper,
                                    "freed_slot": freed,
                                    "keeper_occupancy_before": (
                                        keeper_occupancy_before
                                    ),
                                    "freed_occupancy_before": (
                                        freed_occupancy_before
                                    ),
                                    "merged_pair_occupancy": (
                                        merged_pair_occupancy
                                    ),
                                    "respawn_initial_occupancy": (
                                        freed_occupancy_before
                                    ),
                                    "respawn_occupancy_rule": (
                                        "reuse_freed_premerge_occupancy"
                                    ),
                                    "freed_pre_rmsd_to_A_A": (
                                        freed_pre_rmsd_a
                                    ),
                                    "freed_pre_rmsd_to_B_A": (
                                        freed_pre_rmsd_b
                                    ),
                                    "freed_pre_rmsd_to_midpoint_A": (
                                        freed_pre_rmsd_midpoint
                                    ),
                                    "freed_pre_nearest_deposited_rmsd_A": min(
                                        freed_pre_rmsd_a,
                                        freed_pre_rmsd_b,
                                    ),
                                    "merged_away_was_near_deposited": (
                                        min(
                                            freed_pre_rmsd_a,
                                            freed_pre_rmsd_b,
                                        ) < 1.0
                                    ),
                                    "residual_peak_index": peak_index,
                                    "residual_peak_x": float(peak[0].cpu()),
                                    "residual_peak_y": float(peak[1].cpu()),
                                    "residual_peak_z": float(peak[2].cpu()),
                                    "residual_peak_magnitude": peak_magnitude,
                                    "peak_to_A_nearest_atom_distance_A": (
                                        peak_to_a
                                    ),
                                    "peak_to_B_nearest_atom_distance_A": (
                                        peak_to_b
                                    ),
                                    "peak_to_midpoint_nearest_atom_distance_A": (
                                        peak_to_midpoint
                                    ),
                                    "torsion_inversion_method": (
                                        "parallel_per-heavy-atom_local_Adam"
                                    ),
                                    "torsion_inversion_steps": (
                                        RESPAWN_IK_STEPS
                                    ),
                                    "torsion_inversion_learning_rate": (
                                        RESPAWN_IK_LEARNING_RATE
                                    ),
                                    "peak_atom_index": peak_atom_index,
                                    "peak_atom_name": site["names"][
                                        peak_atom_index
                                    ],
                                    "peak_residual_distance_A": (
                                        peak_residual_distance
                                    ),
                                    "peak_reached_within_0p5_A": (
                                        peak_residual_distance
                                        <= RESPAWN_PEAK_REACHED_THRESHOLD
                                    ),
                                    "density_loss_pre_merge": (
                                        density_loss_pre_merge
                                    ),
                                    "density_loss_post_merge": (
                                        density_loss_post_merge
                                    ),
                                    "density_loss_merge_delta": (
                                        density_loss_post_merge
                                        - density_loss_pre_merge
                                    ),
                                    "density_loss_post_respawn": (
                                        density_loss_post_respawn
                                    ),
                                    "respawn_immediate_rmsd_to_A_A": (
                                        respawn_rmsd_a
                                    ),
                                    "respawn_immediate_rmsd_to_B_A": (
                                        respawn_rmsd_b
                                    ),
                                    "respawn_immediate_rmsd_to_midpoint_A": (
                                        respawn_rmsd_midpoint
                                    ),
                                    "adam_state_reset": (
                                        "freed chi exp_avg/exp_avg_sq; "
                                        "freed+keeper logit exp_avg/exp_avg_sq; "
                                        "shared scalar step retained"
                                    ),
                                })

                        if args.record_stage1_trajectories:
                            trajectory_density_loss_pre.append(
                                density_loss.detach().clone()
                            )
                            trajectory_steps.append(global_stage1_step)
                            trajectory_phases.append(1)
                            trajectory_stage_indices.append(stage_index)
                            trajectory_blur_fwhm.append(blur_fwhm)
                            trajectory_chi_learning_rates.append(stage_lr)
                            trajectory_occupancy_learning_rates.append(
                                float(optimizer.param_groups[1]["lr"])
                            )
                            trajectory_chi_noise_sd_degrees.append(
                                noise_sd_degrees
                            )
                            trajectory_chi.append(all_chi.detach().clone())
                            trajectory_occupancies.append(
                                torch.softmax(logits, dim=0).detach().clone()
                            )

                        if (
                            args.fixed_occupancy_steps
                            and global_stage1_step == args.fixed_occupancy_steps
                        ):
                            # Preserve the historical boundary diagnostics,
                            # evaluated against the target used by the last
                            # frozen update.
                            with torch.no_grad():
                                if blur_fwhm == 0.0:
                                    fixed_rendered, fixed_coordinates = site["render"](
                                        all_chi, logits
                                    )
                                else:
                                    fixed_rendered, fixed_coordinates = site["render"](
                                        all_chi, logits, blur_fwhm
                                    )
                                fixed_boundary_density_loss = float(
                                    site["density_loss"](
                                        fixed_rendered, target_vector
                                    ).cpu()
                                )
                                fixed_boundary_occupancies = (
                                    torch.softmax(logits, dim=0)
                                    .cpu().numpy().copy()
                                )
                                fixed_boundary_chi = (
                                    all_chi.detach().cpu().numpy().copy()
                                )
                                fixed_boundary_rmsd_a = np.asarray([
                                    float(site["rmsd"](
                                        xyz, site["kinematic_a"]
                                    ).cpu())
                                    for xyz in fixed_coordinates
                                ])
                                fixed_boundary_rmsd_b = np.asarray([
                                    float(site["rmsd"](
                                        xyz, site["kinematic_b"]
                                    ).cpu())
                                    for xyz in fixed_coordinates
                                ])
                                fixed_boundary_found_a = bool(
                                    (fixed_boundary_rmsd_a < 1.0).any()
                                )
                                fixed_boundary_found_b = bool(
                                    (fixed_boundary_rmsd_b < 1.0).any()
                                )
                            _set_occupancy_learning_rate(optimizer, stage_lr)

                        if (
                            args.fixed_occupancy_steps
                            and global_stage1_step
                            == args.fixed_occupancy_steps + 1
                            and math.isnan(unfreeze_post_density_loss)
                        ):
                            with torch.no_grad():
                                if blur_fwhm == 0.0:
                                    live_rendered, _live_coordinates = site["render"](
                                        all_chi, logits
                                    )
                                else:
                                    live_rendered, _live_coordinates = site["render"](
                                        all_chi, logits, blur_fwhm
                                    )
                                unfreeze_post_density_loss = float(
                                    site["density_loss"](
                                        live_rendered, target_vector
                                    ).cpu()
                                )

                released_occupancy_steps = (
                    total_stage1_steps - args.fixed_occupancy_steps
                )
                with torch.no_grad():
                    stage1_rendered, stage1_coordinates = site["render"](all_chi, logits)
                    stage1_density_loss = float(site["density_loss"](
                        stage1_rendered,
                        site["target_vectors"][target_label],
                    ).cpu())
                    stage1_chi = all_chi.detach().cpu().numpy().copy()
                    stage1_occupancies = torch.softmax(logits, dim=0).cpu().numpy().copy()
                    stage1_seeded_a_rmsd = (
                        float(torch.sqrt(
                            (stage1_coordinates[0] - site["kinematic_a"])
                            .square().sum(dim=-1).mean()
                        ).cpu())
                        if args.seed_deposited_a else float("nan")
                    )

                best_refinement_loss = float("nan")
                if args.physics_refinement_steps:
                    # Stage 2 intentionally resets Adam so no high-LR density-stage
                    # momentum leaks into the low-LR physical refinement.
                    optimizer = torch.optim.Adam(
                        [all_chi, logits],
                        lr=args.lr * args.physics_refinement_lr_scale,
                    )
                    best_refinement_loss = float("inf")
                    for refinement_step in range(args.physics_refinement_steps):
                        optimizer.zero_grad(set_to_none=True)
                        density, current_coordinates = site["render"](all_chi, logits)
                        density_loss = site["density_loss"](
                            density, site["target_vectors"][target_label]
                        )
                        current_occupancies = torch.softmax(logits, dim=0)
                        vdw_loss, rotamer_loss, symmetry_loss = site["physics_terms"](
                            current_coordinates, current_occupancies
                        )
                        loss = (
                            density_loss
                            + args.lambda_vdw * vdw_loss
                            + args.lambda_rot * rotamer_loss
                            + args.lambda_clash * symmetry_loss
                        )
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            all_chi.copy_(wrap_angles(all_chi))
                        best_refinement_loss = min(
                            best_refinement_loss, float(loss.detach().cpu())
                        )
                        if args.record_stage1_trajectories:
                            trajectory_density_loss_pre.append(
                                density_loss.detach().clone()
                            )
                            trajectory_steps.append(
                                total_stage1_steps + refinement_step + 1
                            )
                            trajectory_phases.append(2)
                            trajectory_stage_indices.append(-1)
                            trajectory_blur_fwhm.append(0.0)
                            refinement_lr = (
                                args.lr * args.physics_refinement_lr_scale
                            )
                            trajectory_chi_learning_rates.append(refinement_lr)
                            trajectory_occupancy_learning_rates.append(
                                refinement_lr
                            )
                            trajectory_chi_noise_sd_degrees.append(0.0)
                            trajectory_chi.append(all_chi.detach().clone())
                            trajectory_occupancies.append(
                                torch.softmax(logits, dim=0).detach().clone()
                            )
                with torch.no_grad():
                    rendered, coordinates = site["render"](all_chi, logits)
                    final_density_loss_tensor = site["density_loss"](
                        rendered, site["target_vectors"][target_label]
                    )
                    occupancy_tensor = torch.softmax(logits, dim=0)
                    final_vdw, final_rotamer, final_symmetry = site["physics_terms"](
                        coordinates, occupancy_tensor
                    )
                    final_total_loss_tensor = final_density_loss_tensor
                    if physics_enabled:
                        final_total_loss_tensor = (
                            final_total_loss_tensor
                            + args.lambda_vdw * final_vdw
                            + args.lambda_rot * final_rotamer
                            + args.lambda_clash * final_symmetry
                        )
                    final_loss = float(final_total_loss_tensor.cpu())
                    final_density_loss = float(final_density_loss_tensor.cpu())
                    occupancies = occupancy_tensor.cpu().numpy()
                    rmsd_a = np.asarray([
                        float(site["rmsd"](xyz, site["kinematic_a"]).cpu())
                        for xyz in coordinates
                    ])
                    rmsd_b = np.asarray([
                        float(site["rmsd"](xyz, site["kinematic_b"]).cpu())
                        for xyz in coordinates
                    ])
                    direct_minima, symmetry_minima = [], []
                    canonical_flags, rotamer_deviations = [], []
                    for xyz in coordinates:
                        if site["direct_environment"].numel():
                            distances = torch.cdist(xyz, site["direct_environment"])
                            direct_minima.append(float(
                                distances.masked_select(site["direct_pair_mask"]).min().cpu()
                            ))
                        else:
                            direct_minima.append(float("nan"))
                        if site["symmetry_environment"].numel():
                            symmetry_minima.append(float(torch.cdist(
                                xyz, site["symmetry_environment"]
                            ).min().cpu()))
                        else:
                            symmetry_minima.append(float("nan"))
                        chis = site["physical_chi"](xyz)
                        deviations = []
                        for chi_index, value in enumerate(chis):
                            centers = torch.tensor(
                                _canonical_centers(site["resname"], chi_index),
                                dtype=value.dtype, device=device,
                            )
                            deviation = torch.abs(wrap_angles(value - centers)).min()
                            deviations.append(float(torch.rad2deg(deviation).cpu()))
                        maximum_deviation = max(deviations)
                        rotamer_deviations.append(maximum_deviation)
                        canonical_flags.append(all(
                            deviation <= canonical_width_degrees(
                                site["resname"], chi_index
                            )
                            for chi_index, deviation in enumerate(deviations)
                        ))
                if args.record_stage1_trajectories:
                    _atomic_npz(
                        args.output / "trajectories"
                        / f"{target_label}_start_{start:03d}.npz",
                        step=np.asarray(trajectory_steps, dtype=np.int32),
                        phase=np.asarray(trajectory_phases, dtype=np.int8),
                        stage_index=np.asarray(
                            trajectory_stage_indices, dtype=np.int8
                        ),
                        blur_fwhm_angstrom=np.asarray(
                            trajectory_blur_fwhm, dtype=np.float32
                        ),
                        chi_learning_rate=np.asarray(
                            trajectory_chi_learning_rates, dtype=np.float32
                        ),
                        occupancy_learning_rate=np.asarray(
                            trajectory_occupancy_learning_rates,
                            dtype=np.float32,
                        ),
                        chi_noise_sd_degrees=np.asarray(
                            trajectory_chi_noise_sd_degrees,
                            dtype=np.float32,
                        ),
                        chi_radians=torch.stack(trajectory_chi)
                        .cpu().numpy(),
                        base_physical_chi_radians=site[
                            "base_physical_chi"
                        ].detach().cpu().numpy(),
                        delta_direction=site["delta_direction"]
                        .detach().cpu().numpy(),
                        occupancies=torch.stack(trajectory_occupancies)
                        .cpu().numpy(),
                        density_loss_pre=torch.stack(
                            trajectory_density_loss_pre
                        ).cpu().numpy(),
                    )
                assignments = []
                for occupancy, distance_a, distance_b in zip(occupancies, rmsd_a, rmsd_b):
                    if occupancy <= args.nontrivial_occupancy:
                        assignments.append("inactive")
                    elif distance_a < 1.0 and distance_a <= distance_b:
                        assignments.append("A")
                    elif distance_b < 1.0:
                        assignments.append("B")
                    else:
                        assignments.append("other")
                predicted_a = float(sum(
                    occupancy for occupancy, label in zip(occupancies, assignments)
                    if label == "A"
                ))
                predicted_b = float(sum(
                    occupancy for occupancy, label in zip(occupancies, assignments)
                    if label == "B"
                ))
                found_a = any(
                    occupancy > 0.1 and label == "A"
                    for occupancy, label in zip(occupancies, assignments)
                )
                found_b = any(
                    occupancy > 0.1 and label == "B"
                    for occupancy, label in zip(occupancies, assignments)
                )
                occupancy_accurate = (
                    abs(predicted_a - site["target_a"]) <= args.occupancy_tolerance
                    and abs(predicted_b - site["target_b"]) <= args.occupancy_tolerance
                )
                active_indices = [
                    index for index, occupancy in enumerate(occupancies)
                    if occupancy > args.nontrivial_occupancy
                ]
                endpoint_physical_valid = bool(active_indices) and all(
                    direct_minima[index] >= 2.0
                    and (
                        math.isnan(symmetry_minima[index])
                        or symmetry_minima[index] >= 2.0
                    )
                    and canonical_flags[index]
                    for index in active_indices
                )
                conventional_recovery = found_a and found_b and occupancy_accurate
                reported_best_loss = (
                    best_refinement_loss
                    if args.physics_refinement_steps else best_stage1_loss
                )
                with torch.no_grad():
                    final_physical_chi = torch.stack([
                        site["physical_chi"](coordinates)
                        for coordinates in coordinates
                    ])
                    chi_space_net_distance = torch.linalg.vector_norm(
                        torch.atan2(
                            torch.sin(final_physical_chi - initial_physical_chi),
                            torch.cos(final_physical_chi - initial_physical_chi),
                        ),
                        dim=1,
                    ) * (180.0 / math.pi)
                if site["target_a"] < site["target_b"]:
                    minor_label = "A"
                    initial_minor_rmsd = initial_rmsd_a
                    final_minor_rmsd = np.asarray(rmsd_a)
                elif site["target_b"] < site["target_a"]:
                    minor_label = "B"
                    initial_minor_rmsd = initial_rmsd_b
                    final_minor_rmsd = np.asarray(rmsd_b)
                else:
                    minor_label = "equal"
                    initial_minor_rmsd = np.full(args.K, np.nan)
                    final_minor_rmsd = np.full(args.K, np.nan)
                ended_nearer_minor = final_minor_rmsd < initial_minor_rmsd
                if found_a and not found_b:
                    endpoint_recovered_state = "A"
                    endpoint_missed_state = "B"
                elif found_b and not found_a:
                    endpoint_recovered_state = "B"
                    endpoint_missed_state = "A"
                else:
                    endpoint_recovered_state = ""
                    endpoint_missed_state = ""
                for event in start_respawn_events:
                    slot = int(event["freed_slot"])
                    endpoint_nearest = min(rmsd_a[slot], rmsd_b[slot])
                    replaced_nearest = float(
                        event["freed_pre_nearest_deposited_rmsd_A"]
                    )
                    peak_to_a = float(
                        event["peak_to_A_nearest_atom_distance_A"]
                    )
                    peak_to_b = float(
                        event["peak_to_B_nearest_atom_distance_A"]
                    )
                    peak_to_midpoint = float(
                        event[
                            "peak_to_midpoint_nearest_atom_distance_A"
                        ]
                    )
                    event.update({
                        "endpoint_found_A": found_a,
                        "endpoint_found_B": found_b,
                        "endpoint_recovered_state": endpoint_recovered_state,
                        "endpoint_missed_state": endpoint_missed_state,
                        "endpoint_slot_occupancy": occupancies[slot],
                        "endpoint_slot_rmsd_to_A_A": rmsd_a[slot],
                        "endpoint_slot_rmsd_to_B_A": rmsd_b[slot],
                        "endpoint_slot_nearest_deposited_rmsd_A": (
                            endpoint_nearest
                        ),
                        "endpoint_minus_replaced_nearest_rmsd_A": (
                            endpoint_nearest - replaced_nearest
                        ),
                        "endpoint_worse_than_replaced": (
                            endpoint_nearest > replaced_nearest
                        ),
                        "endpoint_slot_within_1A_of_deposited": (
                            endpoint_nearest < 1.0
                        ),
                        "endpoint_slot_survived_above_0p10": (
                            occupancies[slot] > 0.10
                        ),
                        "endpoint_slot_direct_min_distance_A": (
                            direct_minima[slot]
                        ),
                        "endpoint_slot_symmetry_min_distance_A": (
                            symmetry_minima[slot]
                        ),
                        "endpoint_slot_rotamer_max_deviation_degrees": (
                            rotamer_deviations[slot]
                        ),
                        "endpoint_slot_canonical": canonical_flags[slot],
                        "peak_midpoint_is_closest": (
                            peak_to_midpoint
                            < min(peak_to_a, peak_to_b)
                        ),
                        "peak_in_midpoint_region_within_1A": (
                            peak_to_midpoint < 1.0
                            and peak_to_midpoint
                            < min(peak_to_a, peak_to_b)
                        ),
                        "peak_to_missed_nearest_atom_distance_A": (
                            peak_to_a
                            if endpoint_missed_state == "A"
                            else peak_to_b
                            if endpoint_missed_state == "B"
                            else float("nan")
                        ),
                    })
                rows.append({
                    "target": target_label, "site": site["key"], "start": start,
                    "initialization_mode": args.initialization_mode,
                    "initialization_jitter_degrees": (
                        args.initialization_jitter_degrees
                    ),
                    "initial_delta_chi_radians": "|".join(
                        ";".join(f"{value:.8g}" for value in chi)
                        for chi in initial_chi.cpu().numpy()
                    ),
                    "initial_physical_chi_degrees": "|".join(
                        ";".join(
                            f"{value:.8g}"
                            for value in np.degrees(chi)
                        )
                        for chi in initial_physical_chi.cpu().numpy()
                    ),
                    "initial_rmsd_to_A": ";".join(
                        f"{value:.8g}" for value in initial_rmsd_a
                    ),
                    "initial_rmsd_to_B": ";".join(
                        f"{value:.8g}" for value in initial_rmsd_b
                    ),
                    "final_loss": final_loss, "best_loss": reported_best_loss,
                    "best_stage1_loss": best_stage1_loss,
                    "best_refinement_loss": best_refinement_loss,
                    "density_schedule": "|".join(
                        f"{blur_fwhm:g}A:{stage_steps}:lr{stage_lr:g}"
                        for blur_fwhm, stage_lr, stage_steps in density_schedule
                    ),
                    "fixed_occupancy_steps": args.fixed_occupancy_steps,
                    "released_occupancy_steps": released_occupancy_steps,
                    "occupancy_freeze_implementation": (
                        OCCUPANCY_FREEZE_IMPLEMENTATION
                    ),
                    "fixed_boundary_density_loss": fixed_boundary_density_loss,
                    "unfreeze_pre_density_loss": unfreeze_pre_density_loss,
                    "unfreeze_post_density_loss": unfreeze_post_density_loss,
                    "unfreeze_density_loss_delta": (
                        unfreeze_post_density_loss - unfreeze_pre_density_loss
                    ),
                    "fixed_boundary_occupancies": ";".join(
                        f"{value:.8g}" for value in fixed_boundary_occupancies
                    ),
                    "fixed_boundary_chi_radians": "|".join(
                        ";".join(f"{value:.8g}" for value in chi)
                        for chi in fixed_boundary_chi
                    ),
                    "fixed_boundary_rmsd_to_A": ";".join(
                        f"{value:.8g}" for value in fixed_boundary_rmsd_a
                    ),
                    "fixed_boundary_rmsd_to_B": ";".join(
                        f"{value:.8g}" for value in fixed_boundary_rmsd_b
                    ),
                    "fixed_boundary_found_A": fixed_boundary_found_a,
                    "fixed_boundary_found_B": fixed_boundary_found_b,
                    "fixed_boundary_both_found": (
                        fixed_boundary_found_a and fixed_boundary_found_b
                    ),
                    "seeded_A_slot": 0 if args.seed_deposited_a else -1,
                    "fixed_boundary_seeded_A_rmsd": (
                        fixed_boundary_rmsd_a[0]
                        if args.seed_deposited_a and args.fixed_occupancy_steps
                        else float("nan")
                    ),
                    "stage1_seeded_A_rmsd": stage1_seeded_a_rmsd,
                    "final_seeded_A_rmsd": (
                        rmsd_a[0] if args.seed_deposited_a else float("nan")
                    ),
                    "final_seeded_A_occupancy": (
                        occupancies[0] if args.seed_deposited_a else float("nan")
                    ),
                    "stage1_density_loss": stage1_density_loss,
                    "stage1_occupancies": ";".join(
                        f"{value:.8g}" for value in stage1_occupancies
                    ),
                    "stage1_chi_radians": "|".join(
                        ";".join(f"{value:.8g}" for value in chi)
                        for chi in stage1_chi
                    ),
                    "final_density_loss": final_density_loss,
                    "final_vdw_loss": float(final_vdw.cpu()),
                    "final_rotamer_loss": float(final_rotamer.cpu()),
                    "final_symmetry_loss": float(final_symmetry.cpu()),
                    "occupancies": ";".join(f"{value:.8g}" for value in occupancies),
                    "rmsd_to_A": ";".join(f"{value:.8g}" for value in rmsd_a),
                    "rmsd_to_B": ";".join(f"{value:.8g}" for value in rmsd_b),
                    "rmsd_definition": "sqrt(mean_atoms(sum_xyz(error^2)))",
                    "assignments": ";".join(assignments),
                    "target_A_occupancy": site["target_a"],
                    "target_B_occupancy": site["target_b"],
                    "predicted_A_occupancy": predicted_a,
                    "predicted_B_occupancy": predicted_b,
                    "found_A": found_a, "found_B": found_b,
                    "occupancy_accurate": occupancy_accurate,
                    "ensemble_success": conventional_recovery,
                    "direct_min_distances": ";".join(
                        f"{value:.8g}" for value in direct_minima
                    ),
                    "symmetry_min_distances": ";".join(
                        f"{value:.8g}" for value in symmetry_minima
                    ),
                    "rotamer_max_deviation_degrees": ";".join(
                        f"{value:.8g}" for value in rotamer_deviations
                    ),
                    "canonical_flags": ";".join(str(value) for value in canonical_flags),
                    "endpoint_physical_valid": endpoint_physical_valid,
                    "joint_success_without_tmol": conventional_recovery and endpoint_physical_valid,
                    "active_conformers": int((occupancies > args.nontrivial_occupancy).sum()),
                    "final_chi_radians": "|".join(
                        ";".join(f"{value:.8g}" for value in chi)
                        for chi in all_chi.detach().cpu().numpy()
                    ),
                    "final_physical_chi_degrees": "|".join(
                        ";".join(
                            f"{value:.8g}"
                            for value in np.degrees(chi)
                        )
                        for chi in final_physical_chi.cpu().numpy()
                    ),
                    "chi_space_net_distance_degrees": ";".join(
                        f"{value:.8g}"
                        for value in chi_space_net_distance.cpu().numpy()
                    ),
                    "minor_deposited_state": minor_label,
                    "ended_nearer_minor_than_initial": ";".join(
                        str(bool(value)) for value in ended_nearer_minor
                    ),
                    "any_slot_ended_nearer_minor_than_initial": bool(
                        ended_nearer_minor.any()
                    ),
                    "stage1_chi_noise_initial_sd_degrees": (
                        args.stage1_chi_noise_initial_degrees
                    ),
                    "stage1_chi_noise_schedule": (
                        "linear_first_step_initial_to_final_step_zero"
                    ),
                    "respawn_cadence": args.respawn_cadence,
                    "respawn_merge_rmsd_A": args.respawn_merge_rmsd,
                    "respawn_check_count": len(start_respawn_checks),
                    "respawn_event_count": len(start_respawn_events),
                    "respawned_unique_slot_count": len(respawned_slots),
                    "respawned_slots": ";".join(
                        str(slot) for slot in sorted(respawned_slots)
                    ),
                    "respawned_endpoint_slots_above_0p10": sum(
                        occupancies[slot] > 0.10
                        for slot in respawned_slots
                    ),
                })
                _atomic_csv(result_path, rows)
                if args.respawn_cadence:
                    respawn_checks.extend(start_respawn_checks)
                    respawn_events.extend(start_respawn_events)
                    _atomic_csv(respawn_check_path, respawn_checks)
                    if respawn_events:
                        _atomic_csv(respawn_event_path, respawn_events)
                _atomic_json(args.output / "stage_manifest.json", {
                    "status": "running", "target": target_label, "site": site["key"],
                    "completed_starts": start + 1,
                })
                print(json.dumps({
                    "target": target_label, "site": site["key"],
                    "completed_starts": start + 1,
                }), flush=True)
            both = [
                row for row in rows
                if str(row["found_A"]) == "True" and str(row["found_B"]) == "True"
            ]
            successes = [row for row in rows if str(row["ensemble_success"]) == "True"]
            physical = [
                row for row in rows if str(row.get("endpoint_physical_valid")) == "True"
            ]
            joint = [
                row for row in rows if str(row.get("joint_success_without_tmol")) == "True"
            ]
            fixed_a = [
                row for row in rows
                if str(row.get("fixed_boundary_found_A")) == "True"
            ]
            fixed_b = [
                row for row in rows
                if str(row.get("fixed_boundary_found_B")) == "True"
            ]
            fixed_both = [
                row for row in rows
                if str(row.get("fixed_boundary_both_found")) == "True"
            ]
            seeded_fixed_stable = [
                row for row in rows
                if float(row.get("fixed_boundary_seeded_A_rmsd", "nan")) < 1.0
            ]
            seeded_stage1_stable = [
                row for row in rows
                if float(row.get("stage1_seeded_A_rmsd", "nan")) < 1.0
            ]
            seeded_final_stable = [
                row for row in rows
                if float(row.get("final_seeded_A_rmsd", "nan")) < 1.0
            ]
            summaries.append({
                "target": target_label, "site": site["key"], "starts": len(rows),
                "fixed_boundary_found_A": len(fixed_a),
                "fixed_boundary_found_B": len(fixed_b),
                "fixed_boundary_both_found": len(fixed_both),
                "seeded_A_stable_at_fixed_boundary": len(seeded_fixed_stable),
                "seeded_A_stable_after_density_stage": len(seeded_stage1_stable),
                "seeded_A_stable_final": len(seeded_final_stable),
                "both_found": len(both), "ensemble_success": len(successes),
                "endpoint_physical_valid": len(physical),
                "joint_success_without_tmol": len(joint),
                "mean_predicted_A": float(np.mean([
                    float(row["predicted_A_occupancy"]) for row in rows
                ])),
                "mean_predicted_B": float(np.mean([
                    float(row["predicted_B_occupancy"]) for row in rows
                ])),
                "mean_final_loss": float(np.mean([
                    float(row["final_loss"]) for row in rows
                ])),
            })
            _atomic_csv(args.output / "aggregate_summary.csv", summaries)
    _atomic_json(args.output / "stage_manifest.json", {
        "status": "complete", "summary_rows": len(summaries),
    })
    print(json.dumps({"status": "complete", "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
