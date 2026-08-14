"""Differentiable Ramachandran lookup matching mmtbx's rama_eval table."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


_TABLE_NAMES = {
    "general": "general",
    "glycine": "glycine",
    "cis-proline": "cis_pro",
    "trans-proline": "trans_pro",
    "pre-proline": "pre_pro",
    "isoleucine or valine": "ile_val",
}
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@lru_cache(maxsize=4)
def load_rama_tables(header_path: str | None = None) -> dict[str, np.ndarray]:
    """Load the exact 180x180 tables used by the installed mmtbx evaluator."""
    if header_path is None:
        import mmtbx
        header = Path(mmtbx.__file__).resolve().parent / "validation" / "ramachandran" / "rama8000_tables.h"
    else:
        header = Path(header_path)
    text = header.read_text()
    tables: dict[str, np.ndarray] = {}
    for key in _TABLE_NAMES.values():
        match = re.search(
            rf"const double linear_table_{re.escape(key)}\[\] = \{{(.*?)\}};",
            text, re.DOTALL,
        )
        if match is None:
            raise FileNotFoundError(f"missing Ramachandran table {key} in {header}")
        values = np.asarray([float(value) for value in _NUMBER.findall(match.group(1))], dtype=np.float64)
        if values.size != 180 * 180:
            raise ValueError(f"table {key} has {values.size} values, expected 32400")
        tables[key] = values.reshape(180, 180)
    return tables


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _bin_number(value: torch.Tensor) -> torch.Tensor:
    integer = torch.div((value + 179.0).to(torch.int64), 2, rounding_mode="trunc")
    integer = torch.where(integer > 179, integer - 180, integer)
    return torch.where(integer < 0, integer + 180, integer)


class TorchRamaEvaluator:
    """Piecewise-bilinear Torch evaluator with the mmtbx bin convention."""

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tables = {
            key: torch.as_tensor(value, dtype=dtype, device=self.device)
            for key, value in load_rama_tables().items()
        }

    def score(self, category: str, phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        key = _TABLE_NAMES[category]
        phi = _wrap_angle(phi)
        psi = _wrap_angle(psi)
        # Bin locations are piecewise-constant metadata.  Detach only the
        # indices; the fractional coordinates retain their gradients.
        phi_bin_value = torch.floor(phi.detach())
        phi_bin_value = phi_bin_value - (
            torch.remainder(phi_bin_value.to(torch.int64), 2) == 0
        ).to(phi.dtype)
        phi_high_value = torch.ceil(phi.detach())
        phi_high_value = phi_high_value + (
            torch.remainder(phi_high_value.to(torch.int64), 2) == 0
        ).to(phi.dtype)
        phi_high_value = torch.where(
            phi_high_value == phi_bin_value, phi_high_value + 2.0, phi_high_value
        )
        psi_bin_value = torch.floor(psi.detach())
        psi_bin_value = psi_bin_value - (
            torch.remainder(psi_bin_value.to(torch.int64), 2) == 0
        ).to(psi.dtype)
        psi_high_value = torch.ceil(psi.detach())
        psi_high_value = psi_high_value + (
            torch.remainder(psi_high_value.to(torch.int64), 2) == 0
        ).to(psi.dtype)
        psi_high_value = torch.where(
            psi_high_value == psi_bin_value, psi_high_value + 2.0, psi_high_value
        )
        phi0, phi1 = _bin_number(phi_bin_value), _bin_number(phi_high_value)
        psi0, psi1 = _bin_number(psi_bin_value), _bin_number(psi_high_value)
        tx = (phi - phi_bin_value) / (phi_high_value - phi_bin_value)
        ty = (psi - psi_bin_value) / (psi_high_value - psi_bin_value)
        table = self.tables[key]
        v00 = table[phi0, psi0]
        v11 = table[phi1, psi1]
        v01 = table[phi0, psi1]
        v10 = table[phi1, psi0]
        return ((1.0 - tx) * (1.0 - ty) * v00 + tx * ty * v11
                + (1.0 - tx) * ty * v01 + tx * (1.0 - ty) * v10)

    def barrier(self, category: str, phi: torch.Tensor, psi: torch.Tensor,
                floor: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.score(category, phi, psi)
        return score, torch.relu(torch.log(float(floor) / torch.clamp(score, min=1e-12)))


def torch_dihedral(a: torch.Tensor, b: torch.Tensor,
                   c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Return the signed dihedral angle in degrees, matching qFit's convention."""
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / torch.linalg.vector_norm(b1, dim=-1, keepdim=True)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.rad2deg(torch.atan2(y, x))
