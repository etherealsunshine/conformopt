"""Differentiable backbone kinematics and CCTBX-compatible model density.

This module is deliberately independent of the production optimizers.  The
renderer uses the six-term ``n_gaussian`` scattering representation used by
CCTBX's ``sampled_model_density``.  For an atom and Gaussian term ``g`` the
real-space kernel is

    a_g * (4*pi / beta_g)**1.5 * exp(-4*pi**2*r**2 / beta_g)

where ``beta_g = b_g + B_iso + 8*pi**2*u_base``.  ``u_base`` is supplied by the
caller because it is a map/resolution setting, not an atom setting.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def _normalize(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return vector / vector.square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()


def nerf_place(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    bond_length: torch.Tensor | float,
    bond_angle: torch.Tensor | float,
    dihedral: torch.Tensor | float,
) -> torch.Tensor:
    """Place the next atom with standard NeRF forward kinematics.

    ``a``, ``b`` and ``c`` are the preceding three Cartesian positions.  The
    bond angle is the angle between ``b-c`` and the new ``d-c`` bond; all
    angles are in radians.  Leading dimensions are broadcast, so a batch of
    candidate geometries can be built in one call.
    """

    a, b, c = torch.broadcast_tensors(a, b, c)
    length = torch.as_tensor(bond_length, dtype=c.dtype, device=c.device)
    angle = torch.as_tensor(bond_angle, dtype=c.dtype, device=c.device)
    torsion = torch.as_tensor(dihedral, dtype=c.dtype, device=c.device)
    bc = _normalize(b - c)
    normal = _normalize(torch.cross(b - a, bc, dim=-1))
    in_plane = torch.cross(normal, bc, dim=-1)
    direction = (
        torch.cos(angle)[..., None] * bc
        + torch.sin(angle)[..., None]
        * (
            torch.cos(torsion)[..., None] * in_plane
            + torch.sin(torsion)[..., None] * normal
        )
    )
    return c + length[..., None] * direction


def nerf_chain(
    initial_xyz: torch.Tensor,
    bond_lengths: torch.Tensor,
    bond_angles: torch.Tensor,
    torsions: torch.Tensor,
) -> torch.Tensor:
    """Build batched Cartesian coordinates from a linear NeRF topology.

    Args:
        initial_xyz: ``[..., 3, 3]`` coordinates for the first three atoms.
        bond_lengths: ``[..., n-3]`` lengths for atoms 4 through n.
        bond_angles: ``[..., n-3]`` internal angles in radians.
        torsions: ``[..., n-3]`` dihedrals in radians.

    The returned tensor has shape ``[..., n, 3]``.  The loop is over topology,
    not candidates; all numerical operations remain Torch operations and are
    differentiable with respect to the torsions and geometry inputs.
    """

    if initial_xyz.shape[-2:] != (3, 3):
        raise ValueError("initial_xyz must have shape [..., 3, 3]")
    if bond_lengths.shape != bond_angles.shape or bond_lengths.shape != torsions.shape:
        raise ValueError("bond_lengths, bond_angles, and torsions must have the same shape")
    if bond_lengths.shape[:-1] != initial_xyz.shape[:-2]:
        raise ValueError("batch dimensions do not agree")

    coordinates = [initial_xyz[..., i, :] for i in range(3)]
    for index in range(bond_lengths.shape[-1]):
        coordinates.append(
            nerf_place(
                coordinates[-3],
                coordinates[-2],
                coordinates[-1],
                bond_lengths[..., index],
                bond_angles[..., index],
                torsions[..., index],
            )
        )
    return torch.stack(coordinates, dim=-2)


# CCTBX n_gaussian coefficients (a, b) for common protein atoms.  Keeping
# this small table here makes the Torch renderer usable without importing
# CCTBX; callers may pass their own coefficient tensor for other atom types.
CCTBX_N_GAUSSIAN = {
    "H": ((-1.09389877635, 1.7298482421), (0.767521006066, 2.01966794458),
          (0.442917711399, 1.47691208171), (0.426815007605, 9.30887772639),
          (0.350065008233, 20.9666815726), (0.106474638445, 44.6312548375)),
    "C": ((2.18188567686, 13.4533708328), (1.77612377639, 32.5790123523),
          (1.08772011297, 0.747293264573), (0.641460989931, 0.251251498175),
          (0.207885994451, 80.9799313275), (0.105219184507, 0.0587297979816)),
    "N": ((2.77545321643, 15.0644760293), (1.37595750403, 7.17746883597),
          (1.06289560478, 0.527446769306), (1.03805703625, 37.9622771317),
          (0.625821830249, 0.187618748878), (0.120841771171, 0.0471843880812)),
    "O": ((2.91262062551, 14.4846217841), (2.58808134565, 6.03817696541),
          (0.98056775412, 0.422545666842), (0.696629143599, 0.154463824621),
          (0.685079864349, 35.5389178226), (0.136773884437, 0.0384134680409)),
    "S": ((6.19471472659, 1.53178413602), (5.16819319919, 22.1507217003),
          (1.61773530525, 55.6787080933), (1.3650575984, 0.703821007249),
          (1.34512095473, 0.0683366362597), (0.308863001721, 0.0118868832735)),
    "SE": ((17.2768180281, 2.33102560067), (9.16660654752, 0.1603154705),
           (4.50821648644, 43.2977383756), (4.00155978598, 14.2929240931),
           (-2.80109328916, 0.110854365863), (1.83632028986, 0.0139037684536)),
    "P": ((4.58130864999, 2.17601600204), (4.57605457281, 28.9931294943),
          (3.08152339098, 1.11652407588), (1.12796052602, 0.0956320460069),
          (1.05066132618, 81.0370412982), (0.585707561577, 0.0257396812627)),
    "F": ((3.45429724648, 10.9155367588), (2.86373677364, 4.46289905515),
          (0.872647810615, 27.6707970367), (0.789193654941, 0.153245585458),
          (0.786475326738, 0.361687002613), (0.233254739503, 0.0418247586665)),
}


def coefficients_for_elements(
    elements: list[str],
    *,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return an ``[atoms, 6, 2]`` n_gaussian ``(a, b)`` tensor."""

    unknown = sorted(set(elements) - set(CCTBX_N_GAUSSIAN))
    if unknown:
        raise KeyError(f"no CCTBX n_gaussian coefficients for {unknown}")
    return torch.tensor([CCTBX_N_GAUSSIAN[element] for element in elements], dtype=dtype, device=device)


def render_cctbx_density(
    grid_xyz: torch.Tensor,
    atom_xyz: torch.Tensor,
    b_factors: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    u_base: float | torch.Tensor = 0.0,
    atom_weights: Optional[torch.Tensor] = None,
    atom_mask: Optional[torch.Tensor] = None,
    grid_mask: Optional[torch.Tensor] = None,
    voxel_atom_mask: Optional[torch.Tensor] = None,
    voxel_chunk: int = 4096,
    # The differentiable default must evaluate the same smooth function that
    # autodiff differentiates.  Pass a negative inverse step explicitly only
    # when reproducing CCTBX's quantized exponent table is required.
    exp_table_one_over_step_size: float = 0.0,
    straight_through_exp_table: bool = True,
) -> torch.Tensor:
    """Render CCTBX ``n_gaussian`` density on an explicit Cartesian grid.

    ``grid_xyz`` is ``[voxels, 3]`` or ``[batch, voxels, 3]``.  ``atom_xyz``
    and ``b_factors`` are ``[atoms, 3]``/``[batch, atoms, 3]`` and
    ``[atoms]``/``[batch, atoms]``.  ``coefficients`` is ``[atoms, 6, 2]`` or
    ``[batch, atoms, 6, 2]``.  Optional masks are applied only to the output;
    the returned tensor retains the full grid shape, which makes it safe to
    compare the exact same CCTBX mask.
    """

    if grid_xyz.ndim == 2:
        grid_xyz = grid_xyz.unsqueeze(0)
    if atom_xyz.ndim == 2:
        atom_xyz = atom_xyz.unsqueeze(0)
    if b_factors.ndim == 1:
        b_factors = b_factors.unsqueeze(0)
    if coefficients.ndim == 3:
        coefficients = coefficients.unsqueeze(0)
    batch = max(grid_xyz.shape[0], atom_xyz.shape[0], b_factors.shape[0], coefficients.shape[0])
    grid_xyz = grid_xyz.expand(batch, -1, -1)
    atom_xyz = atom_xyz.expand(batch, -1, -1)
    b_factors = b_factors.expand(batch, -1)
    coefficients = coefficients.expand(batch, -1, -1, -1)
    atoms = atom_xyz.shape[1]
    if coefficients.shape[1] != atoms or coefficients.shape[-1] != 2:
        raise ValueError("coefficients must have shape [batch, atoms, gaussians, 2]")

    a = coefficients[..., 0]
    beta = coefficients[..., 1] + b_factors[..., None]
    ub = torch.as_tensor(u_base, dtype=grid_xyz.dtype, device=grid_xyz.device)
    beta = beta + (8.0 * math.pi**2) * ub
    if torch.any(beta <= 0):
        raise ValueError("CCTBX Gaussian widths must be positive")
    scale = a * (4.0 * math.pi / beta).pow(1.5)
    if atom_weights is not None:
        weights = atom_weights
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        scale = scale * weights.to(dtype=scale.dtype, device=scale.device)[..., None]
    if atom_mask is not None:
        mask = atom_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        scale = scale * mask.to(dtype=scale.dtype, device=scale.device)[..., None]

    outputs = []
    for start in range(0, grid_xyz.shape[1], voxel_chunk):
        points = grid_xyz[:, start:start + voxel_chunk]
        distance2 = (points[:, :, None, :] - atom_xyz[:, None, :, :]).square().sum(dim=-1)
        exponent = -4.0 * math.pi**2 * distance2[..., None] / beta[:, None, :, :]
        if exp_table_one_over_step_size == 0:
            exponential = torch.exp(exponent)
        else:
            # CCTBX's default exponent table rounds the positive value
            # (-bs_real * d^2) to a 0.01 grid before evaluating exp().
            inverse_step = torch.as_tensor(
                exp_table_one_over_step_size,
                dtype=exponent.dtype,
                device=exponent.device,
            )
            table_index = torch.floor((-exponent * (-inverse_step)) + 0.5)
            exponential = torch.exp(table_index / inverse_step)
            if straight_through_exp_table:
                # Preserve CCTBX's quantized forward value while retaining
                # the derivative of the underlying smooth Gaussian kernel.
                smooth = torch.exp(exponent)
                exponential = exponential + smooth - smooth.detach()
        if voxel_atom_mask is not None:
            contribution_mask = voxel_atom_mask
            if contribution_mask.ndim == 2:
                contribution_mask = contribution_mask.unsqueeze(0)
            exponential = exponential * contribution_mask[:, start:start + voxel_chunk, :, None].to(
                dtype=exponential.dtype,
                device=exponential.device,
            )
        outputs.append((scale[:, None, :, :] * exponential).sum(dim=(-1, -2)))
    density = torch.cat(outputs, dim=1)
    if grid_mask is not None:
        mask = grid_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        density = density * mask.to(dtype=density.dtype, device=density.device)
    return density
