from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch


OPTIMIZER_PHYSICS_ENVIRONMENT_RULE = (
    "2026-07-24-altloc-minstate-water-minstate-v2"
)


@dataclass(frozen=True)
class EnvironmentAtom:
    xyz: tuple[float, float, float]
    label: str
    residue_group: str
    altloc: str
    occupancy: float
    is_water: bool


@dataclass(frozen=True)
class SoftEnvironmentRecord:
    xyz: tuple[float, float, float]
    group_key: str
    atom_name: str
    altloc: str
    occupancy: float
    is_water: bool


def normalized_altloc(value: str) -> str:
    return "" if value in ("", "\x00", " ") else value


def partition_soft_environment(
    records: list[SoftEnvironmentRecord],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[list[torch.Tensor]],
    list[SoftEnvironmentRecord],
]:
    """Partition a soft environment into invariant and alternate states.

    Blank protein atoms are invariant. Alternate protein residues use the
    minimum-penalty deposited state. Unlabeled waters remain invariant and are
    occupancy-weighted. Labeled waters use the same state selection, with an
    explicit absent state when their summed state occupancy is below one.
    """
    invariant: list[SoftEnvironmentRecord] = []
    grouped: dict[str, dict[str, list[SoftEnvironmentRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    water_state_occupancies: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        altloc = normalized_altloc(record.altloc)
        if altloc:
            grouped[record.group_key][altloc].append(record)
            if record.is_water:
                water_state_occupancies[record.group_key][altloc] = max(
                    record.occupancy,
                    water_state_occupancies[record.group_key].get(altloc, 0.0),
                )
        else:
            invariant.append(record)

    invariant_xyz = (
        torch.tensor(
            [record.xyz for record in invariant],
            dtype=torch.float32,
            device=device,
        )
        if invariant
        else torch.empty((0, 3), dtype=torch.float32, device=device)
    )
    invariant_weights = (
        torch.tensor(
            [
                record.occupancy if record.is_water else 1.0
                for record in invariant
            ],
            dtype=torch.float32,
            device=device,
        )
        if invariant
        else torch.empty((0,), dtype=torch.float32, device=device)
    )

    alternate_states: list[list[torch.Tensor]] = []
    for group_key, states in grouped.items():
        tensors = [
            torch.tensor(
                [record.xyz for record in state],
                dtype=torch.float32,
                device=device,
            )
            for state in states.values()
        ]
        water_occupancies = water_state_occupancies.get(group_key)
        if (
            water_occupancies
            and sum(water_occupancies.values()) < 1.0 - 1e-6
        ):
            tensors.append(torch.empty((0, 3), dtype=torch.float32, device=device))
        alternate_states.append(tensors)
    return invariant_xyz, invariant_weights, alternate_states, invariant


def compatible_spatial_metrics(
    candidate_xyz: np.ndarray,
    atoms: list[EnvironmentAtom],
    clash_cutoff: float,
    candidate_altloc: str | None,
) -> dict[str, float | str | bool]:
    """Evaluate hard clashes with per-residue min-over-altloc selection.

    Blank atoms are invariant. For alternate protein residues, the environment
    state with the largest clearance is selected independently per residue.
    Labeled waters are included only when they match an assigned candidate;
    unlabeled waters remain present with an occupancy-scaled hard cutoff.
    """
    selected: list[EnvironmentAtom] = []
    alternate_groups: dict[str, dict[str, list[EnvironmentAtom]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for atom in atoms:
        if atom.is_water:
            if atom.altloc and atom.altloc != candidate_altloc:
                continue
            selected.append(atom)
        elif atom.altloc:
            alternate_groups[atom.residue_group][atom.altloc].append(atom)
        else:
            selected.append(atom)

    for states in alternate_groups.values():
        best_state = max(
            states.values(),
            key=lambda state: _minimum_clearance(
                candidate_xyz, state, clash_cutoff
            ),
        )
        selected.extend(best_state)

    if not selected:
        return {
            "minimum_distance": float("nan"),
            "closest_atom": "",
            "minimum_clearance": float("inf"),
            "no_clash": True,
        }

    coordinates = np.asarray([atom.xyz for atom in selected])
    distances = np.linalg.norm(
        candidate_xyz[:, None, :] - coordinates[None, :, :], axis=-1
    )
    thresholds = np.asarray([
        clash_cutoff * atom.occupancy if atom.is_water else clash_cutoff
        for atom in selected
    ])
    clearances = distances - thresholds[None, :]
    flat_index = int(np.argmin(clearances))
    moving_index, environment_index = np.unravel_index(
        flat_index, clearances.shape
    )
    return {
        "minimum_distance": float(distances[moving_index, environment_index]),
        "closest_atom": selected[environment_index].label,
        "minimum_clearance": float(clearances[moving_index, environment_index]),
        "no_clash": bool(clearances[moving_index, environment_index] >= 0.0),
    }


def soft_clash_penalty(
    candidate_xyz: torch.Tensor,
    invariant_xyz: torch.Tensor,
    invariant_weights: torch.Tensor,
    alternate_states: list[list[torch.Tensor]],
    threshold: float,
    invariant_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft counterpart: weighted invariant penalty plus min per alt group."""
    zero = torch.zeros((), dtype=candidate_xyz.dtype, device=candidate_xyz.device)
    if invariant_xyz.numel():
        overlap = torch.clamp(
            threshold - torch.cdist(candidate_xyz, invariant_xyz), min=0.0
        ).square()
        if invariant_mask is not None:
            overlap = overlap.masked_fill(~invariant_mask, 0.0)
        total = (overlap * invariant_weights[None, :]).sum()
    else:
        total = zero
    for states in alternate_states:
        state_penalties = []
        for state in states:
            if state.numel():
                state_penalties.append(
                    torch.clamp(
                        threshold - torch.cdist(candidate_xyz, state), min=0.0
                    ).square().sum()
                )
            else:
                state_penalties.append(zero)
        if state_penalties:
            total = total + torch.stack(state_penalties).min()
    return total


def soft_clash_barrier_penalty(
    candidate_xyz: torch.Tensor,
    invariant_xyz: torch.Tensor,
    invariant_weights: torch.Tensor,
    alternate_states: list[list[torch.Tensor]],
    soft_threshold: float,
    hard_threshold: float,
    barrier_buffer: float,
    barrier_scale: float,
) -> torch.Tensor:
    """Squared overlap plus a quartic shoulder centered on the hard gate."""
    if barrier_buffer <= 0:
        raise ValueError("barrier_buffer must be positive")
    if barrier_scale < 0:
        raise ValueError("barrier_scale must be non-negative")

    def overlap_penalty(
        environment: torch.Tensor, weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        distances = torch.cdist(candidate_xyz, environment)
        base = torch.clamp(soft_threshold - distances, min=0.0).square()
        barrier = barrier_scale * (
            torch.clamp(
                hard_threshold + barrier_buffer - distances, min=0.0
            ) / barrier_buffer
        ).pow(4)
        combined = base + barrier
        return (
            (combined * weights[None, :]).sum()
            if weights is not None
            else combined.sum()
        )

    zero = torch.zeros((), dtype=candidate_xyz.dtype, device=candidate_xyz.device)
    total = (
        overlap_penalty(invariant_xyz, invariant_weights)
        if invariant_xyz.numel()
        else zero
    )
    for states in alternate_states:
        penalties = [
            overlap_penalty(state) if state.numel() else zero
            for state in states
        ]
        if penalties:
            total = total + torch.stack(penalties).min()
    return total


def _minimum_clearance(
    candidate_xyz: np.ndarray,
    atoms: list[EnvironmentAtom],
    clash_cutoff: float,
) -> float:
    coordinates = np.asarray([atom.xyz for atom in atoms])
    distances = np.linalg.norm(
        candidate_xyz[:, None, :] - coordinates[None, :, :], axis=-1
    )
    return float((distances - clash_cutoff).min())
