from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LandscapeDataset(Dataset):
    """Compact atom labels paired with the original, unaugmented density patch."""

    def __init__(self, cache: Path, split: str):
        root = cache / split
        self.index = json.loads((root / "index.json").read_text())
        self.positions = np.load(root / "positions.npy", mmap_mode="r")
        self.sigma2 = np.load(root / "sigma2.npy", mmap_mode="r")
        self.weights = np.load(root / "weights.npy", mmap_mode="r")
        self.atom_mask = np.load(root / "atom_mask.npy", mmap_mode="r")
        expected = len(self.index["keys"])
        if any(len(array) != expected for array in (
            self.positions, self.sigma2, self.weights, self.atom_mask
        )):
            raise ValueError(f"inconsistent landscape cache at {root}")

    def __len__(self) -> int:
        return len(self.index["keys"])

    def __getitem__(self, index: int) -> dict:
        with np.load(self.index["pair_paths"][index]) as archive:
            input_patch = torch.from_numpy(archive["input"].astype(np.float32))
            target_patch = torch.from_numpy(archive["target"].astype(np.float32))
        return {
            "input": input_patch,
            "target": target_patch,
            "positions": torch.from_numpy(np.array(self.positions[index], dtype=np.float32)),
            "sigma2": torch.from_numpy(np.array(self.sigma2[index], dtype=np.float32)),
            "weights": torch.from_numpy(np.array(self.weights[index], dtype=np.float32)),
            "atom_mask": torch.from_numpy(np.array(self.atom_mask[index], dtype=bool)),
            "key": self.index["keys"][index],
            "residue_name": self.index["residue_names"][index],
        }


def patch_grid(size: int, spacing: float, *, device: torch.device,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
    axis = (torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0) * spacing
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)


def radial_mask(size: int, spacing: float, radius: float, *,
                device: torch.device) -> torch.Tensor:
    return patch_grid(size, spacing, device=device).square().sum(dim=-1) <= radius**2


def render_candidates(positions: torch.Tensor, sigma2: torch.Tensor,
                      weights: torch.Tensor, atom_mask: torch.Tensor,
                      *, size: int = 32, spacing: float = 0.5,
                      voxel_chunk: int = 4096) -> torch.Tensor:
    """Render normalized candidate side-chain densities.

    Inputs have shapes BxCxSx3 and BxCxS. Positions are relative to the
    crystal-frame patch center. The chunking bounds memory while preserving
    gradients with respect to any future learnable coordinates.
    """
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [batch, candidates, atoms, 3]")
    grid = patch_grid(size, spacing, device=positions.device, dtype=torch.float32)
    positions = positions.float()
    sigma2 = sigma2.float().clamp_min(0.04)
    weights = weights.float() * atom_mask.float()
    normalization = (2.0 * math.pi * sigma2).pow(-1.5)
    rendered = []
    for start in range(0, len(grid), voxel_chunk):
        points = grid[start:start + voxel_chunk]
        distance2 = (
            points[None, None, :, None, :] - positions[:, :, None, :, :]
        ).square().sum(dim=-1)
        density = (
            weights[:, :, None, :] * normalization[:, :, None, :]
            * torch.exp(-distance2 / (2.0 * sigma2[:, :, None, :]))
        ).sum(dim=-1)
        rendered.append(density)
    density = torch.cat(rendered, dim=2)
    mean = density.mean(dim=2, keepdim=True)
    standard_deviation = density.std(dim=2, correction=0, keepdim=True).clamp_min(1e-6)
    return ((density - mean) / standard_deviation).reshape(
        positions.shape[0], positions.shape[1], size, size, size
    )


def candidate_energies(density: torch.Tensor, candidates: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """MSE energy of each candidate inside the optimizer's local sphere."""
    if density.ndim != 5 or density.shape[1] != 1:
        raise ValueError("density must have shape [batch, 1, size, size, size]")
    flat_density = density[:, 0].flatten(1)[:, None, :]
    flat_candidates = candidates.flatten(2)
    return (flat_density[:, :, mask] - flat_candidates[:, :, mask]).square().mean(dim=2)


def landscape_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_oracle_gap: float = 1e-4,
    margin_fraction: float = 0.5,
    ranking_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match oracle candidate-energy gaps and enforce the native basin ordering."""
    predicted_energy = candidate_energies(prediction.float(), candidates, mask)
    with torch.no_grad():
        oracle_energy = candidate_energies(target.float(), candidates, mask)
        oracle_gap = oracle_energy[:, 1:] - oracle_energy[:, :1]
        valid = oracle_gap > minimum_oracle_gap
    predicted_gap = predicted_energy[:, 1:] - predicted_energy[:, :1]
    if valid.any():
        gap_loss = F.smooth_l1_loss(
            predicted_gap[valid], oracle_gap[valid], beta=0.1
        )
        ranking_loss = torch.relu(
            margin_fraction * oracle_gap[valid] - predicted_gap[valid]
        ).mean()
        loss = gap_loss + ranking_weight * ranking_loss
    else:
        gap_loss = predicted_energy.sum() * 0.0
        ranking_loss = predicted_energy.sum() * 0.0
        loss = gap_loss
    native_top1 = (
        predicted_energy[:, 0] <= predicted_energy[:, 1:].min(dim=1).values
    ).float().mean()
    oracle_native_top1 = (
        oracle_energy[:, 0] <= oracle_energy[:, 1:].min(dim=1).values
    ).float().mean()
    metrics = {
        "loss": loss.detach(),
        "gap_loss": gap_loss.detach(),
        "ranking_loss": ranking_loss.detach(),
        "native_top1": native_top1.detach(),
        "oracle_native_top1": oracle_native_top1.detach(),
        "mean_predicted_native_margin": (
            predicted_energy[:, 1:].min(dim=1).values - predicted_energy[:, 0]
        ).mean().detach(),
        "mean_oracle_native_margin": (
            oracle_energy[:, 1:].min(dim=1).values - oracle_energy[:, 0]
        ).mean().detach(),
        "valid_gap_fraction": valid.float().mean().detach(),
        "native_render_mse": (
            candidates[:, 0] - target[:, 0].float()
        ).square().flatten(1).mean(dim=1).mean().detach(),
    }
    return loss, metrics
