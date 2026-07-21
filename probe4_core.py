"""Reusable, testable pieces of Probe 4's learned torsion-space energy."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


AMINO_ACIDS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)


class LearnedEnergy(nn.Module):
    """Overparameterized scalar energy conditioned on density, torsions, and type."""

    def __init__(
        self,
        density_feat_dim: int = 512,
        max_chi: int = 4,
        hidden: int = 512,
        n_layers: int = 6,
    ) -> None:
        super().__init__()
        input_dim = density_feat_dim + max_chi * 2 + len(AMINO_ACIDS)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend((nn.Linear(hidden, hidden), nn.GELU()))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        self.max_chi = max_chi

    def forward(
        self,
        density_features: torch.Tensor,
        chi_sin_cos: torch.Tensor,
        residue_onehot: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat((density_features, chi_sin_cos, residue_onehot), dim=-1)).squeeze(-1)


def residue_onehot(name: str, *, device: torch.device | str) -> torch.Tensor:
    out = torch.zeros(len(AMINO_ACIDS), device=device)
    out[AMINO_ACIDS.index(name)] = 1.0
    return out


def chi_features(chi: torch.Tensor, n_chi: int, max_chi: int = 4) -> torch.Tensor:
    """Return fixed-width interleaved sin/cos features plus a zero padding mask."""
    padded = torch.zeros(max_chi, device=chi.device, dtype=chi.dtype)
    padded[:n_chi] = chi[:n_chi]
    features = torch.stack((torch.sin(padded), torch.cos(padded)), dim=-1)
    # A padded chi must not look like a real zero-radian torsion.
    features[n_chi:] = 0
    return features.flatten()


def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def rotate_points(
    points: torch.Tensor,
    origin: torch.Tensor,
    axis_endpoint: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Differentiable Rodrigues rotation of points around an oriented bond."""
    axis = axis_endpoint - origin
    axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
    vectors = points - origin
    expanded = axis.expand_as(vectors)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return (
        origin
        + vectors * cosine
        + torch.cross(expanded, vectors, dim=-1) * sine
        + expanded * (vectors @ axis).unsqueeze(-1) * (1 - cosine)
    )


def torsion_to_coords(
    template: torch.Tensor,
    atom_names: Iterable[str],
    chi: torch.Tensor,
    rotations: list[tuple[str, str, tuple[str, ...]]],
    fixed_atoms: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply sidechain chi deltas to a deposited conformer template."""
    names = list(atom_names)
    name_to_index = {name: index for index, name in enumerate(names)}
    coords = template.clone()
    for index, (origin_name, endpoint_name, downstream_names) in enumerate(rotations):
        lookup = {name: coords[i] for i, name in enumerate(names)}
        origin = lookup.get(origin_name, fixed_atoms.get(origin_name))
        endpoint = lookup.get(endpoint_name, fixed_atoms.get(endpoint_name))
        if origin is None or endpoint is None:
            raise KeyError(f"missing rotation-axis atom {origin_name}-{endpoint_name}")
        downstream = torch.tensor(
            [name_to_index[name] for name in downstream_names], device=coords.device
        )
        coords[downstream] = rotate_points(coords[downstream], origin, endpoint, chi[index])
    return coords


def first_order_refine(
    model: nn.Module,
    density_features: torch.Tensor,
    residue_features: torch.Tensor,
    initial_chi: torch.Tensor,
    n_chi: int,
    steps: int,
    alpha: float,
    *,
    training: bool,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Refine chi while truncating state-to-state Hessian paths.

    The final update intentionally remains attached to model parameters. Detaching
    it too (as in a literal reading of common FOMAML pseudocode) would make the
    reciprocal-space endpoint loss independent of the learned energy.
    """
    chi = initial_chi
    energies: list[torch.Tensor] = []
    for step in range(steps):
        chi = chi.detach().requires_grad_(True)
        energy = model(density_features, chi_features(chi, n_chi), residue_features)
        gradient = torch.autograd.grad(energy, chi, create_graph=training)[0]
        chi = chi - alpha * gradient
        energies.append(energy)
        if step + 1 < steps:
            chi = chi.detach()
    return wrap_angles(chi), energies


def circular_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return wrap_angles(predicted - target).abs()


def dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Signed dihedral angle in radians."""
    b0 = b - a
    b1 = c - b
    b2 = d - c
    b1 = b1 / torch.linalg.vector_norm(b1).clamp_min(1e-12)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return torch.atan2(torch.cross(b1, v, dim=0) @ w, v @ w)


def angular_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(wrap_angles(a - b).square())).detach().cpu())


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid_offsets(size: int = 8, spacing: float = 1.0) -> torch.Tensor:
    axis = (torch.arange(size, dtype=torch.float32) - (size - 1) / 2) * spacing
    return torch.cartesian_prod(axis, axis, axis)


def normalized_rfactor(fcalc: torch.Tensor, fobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """R factor after a least-squares amplitude scale."""
    calc = fcalc.abs()[mask]
    obs = fobs[mask]
    scale = (calc @ obs) / (calc.square().sum().clamp_min(1e-12))
    return (scale * calc - obs).abs().sum() / obs.abs().sum().clamp_min(1e-12)


def assert_outer_gradient(model: nn.Module) -> None:
    total = sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not math.isfinite(total) or total == 0:
        raise RuntimeError("endpoint loss produced no gradient for LearnedEnergy")
