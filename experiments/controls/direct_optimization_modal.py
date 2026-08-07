"""Direct chi optimization control for the Probe 4c.1 complex-SF target.

Run detached so the experiment survives client disconnects::

    UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run --detach direct_optimization_modal.py

There is no neural network and no training in this control. Results are written
to a named Modal Volume after the LR sweep, after every 10 starts, and after
every completed site/configuration.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("probe4c1-direct-optimization-control")
RESULTS_VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install(
        "SFcalculator-torch==0.3.3",
        "gemmi==0.6.7",
        "matplotlib==3.9.4",
        "numpy==1.26.4",
    )
    .add_local_file(ROOT / "probe4_core.py", remote_path="/root/probe4_core.py", copy=True)
    .add_local_dir(ROOT / "data", remote_path="/data", copy=True)
)


@APP.function(image=IMAGE, gpu="L4", timeout=86_400, volumes={"/outputs": RESULTS_VOLUME})
def run_control(config: dict) -> dict:
    import csv
    import json
    import math
    import os
    import random
    import time

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import gemmi
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from SFC_Torch import SFcalculator

    from probe4_core import dihedral, seed_everything, torsion_to_coords, wrap_angles

    device = torch.device("cuda")
    out = Path("/outputs") / config["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "stage_manifest.json"
    seed_everything(config["seed"])

    def atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)

    def commit() -> None:
        RESULTS_VOLUME.commit()

    def load_manifest() -> dict:
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"protein": "2O1K", "created_at": time.time(), "stages": {}}

    def mark(stage: str, status: str, **details) -> None:
        value = load_manifest()
        value["stages"][stage] = {"status": status, "updated_at": time.time(), **details}
        atomic_json(manifest_path, value)
        commit()

    def done(stage: str) -> bool:
        return (
            not config["force"]
            and load_manifest().get("stages", {}).get(stage, {}).get("status") == "complete"
        )

    chi_specs = {
        "ARG": {
            "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"),
                          ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
            "rotations": (("CA", "CB", ("CG", "CD", "NE", "CZ", "NH1", "NH2")),
                          ("CB", "CG", ("CD", "NE", "CZ", "NH1", "NH2")),
                          ("CG", "CD", ("NE", "CZ", "NH1", "NH2")),
                          ("CD", "NE", ("CZ", "NH1", "NH2"))),
        },
        "MET": {
            "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"),
                          ("CB", "CG", "SD", "CE")),
            "rotations": (("CA", "CB", ("CG", "SD", "CE")),
                          ("CB", "CG", ("SD", "CE")),
                          ("CG", "SD", ("CE",))),
        },
        "ASP": {
            "dihedrals": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
            "rotations": (("CA", "CB", ("CG", "OD1", "OD2")),
                          ("CB", "CG", ("OD1", "OD2"))),
        },
    }

    pdb_path, mtz_path = "/data/2O1K.pdb", "/data/2O1K.mtz"
    calc = SFcalculator(pdb_path, mtz_path, device=device)
    fobs = calc.Fo.detach()
    deposited_fcalc = calc.calc_fprotein(Return=True).detach()
    base_positions = calc.atom_pos_orth.detach().clone()
    base_occupancies = calc.atom_occ.detach().clone()
    hkl = torch.as_tensor(calc.HKL_array, dtype=torch.float32, device=device)
    orth_to_frac = calc.orth2frac_tensor.detach()
    valid = torch.isfinite(fobs) & (fobs > 0)
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    test_mask = valid & (
        torch.rand(fobs.shape, generator=generator, device=device) < config["reflection_holdout"]
    )
    train_mask = valid & ~test_mask
    if not test_mask.any():
        test_mask[torch.where(valid)[0][0]] = True
        train_mask = valid & ~test_mask

    structure = gemmi.read_structure(pdb_path)
    atoms_with_context = [
        (chain, residue, atom)
        for chain in structure[0]
        for residue in chain
        for atom in residue
    ]
    if len(atoms_with_context) != base_positions.shape[0]:
        raise RuntimeError("gemmi and SFcalculator disagree on atom ordering")
    base_bfactors = torch.tensor(
        [atom.b_iso for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32,
        device=device,
    )
    base_it92_a = torch.tensor(
        [atom.element.it92.a for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32,
        device=device,
    )
    base_it92_b = torch.tensor(
        [atom.element.it92.b for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32,
        device=device,
    )
    base_it92_c = torch.tensor(
        [atom.element.it92.c for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32,
        device=device,
    )
    base_atomic_numbers = torch.tensor(
        [atom.element.atomic_number for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32,
        device=device,
    )

    def alt_atom_map(residue, alt: str) -> dict:
        result = {}
        for atom in residue:
            atom_alt = atom.altloc if atom.altloc not in ("\x00", " ") else ""
            if atom_alt in ("", alt):
                result[atom.name.strip()] = torch.tensor(
                    atom.pos.tolist(), dtype=torch.float32, device=device
                )
        return result

    sites = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name not in chi_specs:
                continue
            alts = {atom.altloc for atom in residue if atom.altloc not in ("\x00", " ", "")}
            if not {"A", "B"}.issubset(alts):
                continue
            map_a, map_b = alt_atom_map(residue, "A"), alt_atom_map(residue, "B")
            indices = [
                i for i, (candidate_chain, candidate_residue, atom) in enumerate(atoms_with_context)
                if candidate_chain.name == chain.name
                and candidate_residue.seqid.num == residue.seqid.num
                and atom.altloc == "B"
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [atoms_with_context[i][2].name.strip() for i in indices]
            if not indices or any(name not in map_a or name not in map_b for name in names):
                continue
            spec = chi_specs[residue.name]
            chi_a = torch.stack([
                dihedral(*(map_a[name] for name in quartet)) for quartet in spec["dihedrals"]
            ])
            chi_b = torch.stack([
                dihedral(*(map_b[name] for name in quartet)) for quartet in spec["dihedrals"]
            ])
            delta = wrap_angles(chi_b - chi_a)
            template = torch.stack([map_a[name] for name in names])
            deposited_b = torch.stack([map_b[name] for name in names])
            fixed = {name: value for name, value in map_a.items() if name not in names}
            plus = torsion_to_coords(template, names, delta, list(spec["rotations"]), fixed)
            minus = torsion_to_coords(template, names, -delta, list(spec["rotations"]), fixed)
            if torch.mean((minus - deposited_b).square()) < torch.mean((plus - deposited_b).square()):
                delta = -delta
            site = {
                "key": f"{chain.name}_{residue.name}{residue.seqid.num}",
                "n_chi": len(spec["rotations"]),
                "rotations": list(spec["rotations"]),
                "names": names,
                "indices": torch.tensor(indices, dtype=torch.long, device=device),
                "template": template,
                "deposited_b": deposited_b,
                "fixed": fixed,
                "true_delta": delta,
            }
            sites.append(site)
    if len(sites) != 5:
        raise RuntimeError(f"expected five sites, found {[site['key'] for site in sites]}")
    site_by_key = {site["key"]: site for site in sites}

    def coords_from_chi(site: dict, chi: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            site["template"], site["names"], chi, site["rotations"], site["fixed"]
        )

    # Reproduce Probe 4c.1 exactly: fixed contribution and target are computed
    # using the same full SFcalculator path and kinematic B coordinates.
    for site in sites:
        without_sidechain = base_occupancies.clone()
        without_sidechain[site["indices"]] = 0.0
        site["f_fixed"] = calc.calc_fprotein(
            base_positions, atoms_occ_tensor=without_sidechain, Return=True
        ).detach()
        site["kinematic_b"] = coords_from_chi(site, site["true_delta"]).detach()
        kinematic_positions = base_positions.index_copy(
            0, site["indices"], site["kinematic_b"]
        )
        kinematic_fcalc = calc.calc_fprotein(
            kinematic_positions, atoms_occ_tensor=base_occupancies, Return=True
        ).detach()
        site["complex_target"] = (kinematic_fcalc - site["f_fixed"]).detach()
        site["normalizer"] = (
            site["complex_target"][train_mask].abs().square().mean().clamp_min(1e-12)
        )

    def atom_density(
        atom_coordinates: torch.Tensor,
        bfactors: torch.Tensor,
        occupancies: torch.Tensor,
        grid_coordinates: torch.Tensor,
        blur_fwhm: float = 0.0,
    ) -> torch.Tensor:
        """Differentiable isotropic Gaussian density from the Parseval prompt."""
        sigma2 = (bfactors / (8.0 * math.pi ** 2)).clamp_min(1e-4)
        # Convolution of two Gaussians adds their variances.  Treat the user-facing
        # blur values as map-resolution-like FWHM values in Angstroms.
        if blur_fwhm > 0:
            sigma2 = sigma2 + (blur_fwhm / 2.354820045) ** 2
        differences = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
        distance2 = differences.square().sum(dim=-1)
        normalization = (2.0 * math.pi * sigma2).pow(-1.5)
        return (
            occupancies[None, :]
            * normalization[None, :]
            * torch.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(dim=1)

    def xray_atom_density(
        atom_coordinates: torch.Tensor,
        bfactors: torch.Tensor,
        occupancies: torch.Tensor,
        it92_a: torch.Tensor,
        it92_b: torch.Tensor,
        it92_c: torch.Tensor,
        grid_coordinates: torch.Tensor,
        blur_fwhm: float = 0.0,
    ) -> torch.Tensor:
        """Differentiable IT92 X-ray density as a five-Gaussian mixture."""
        weights = torch.cat((it92_a, it92_c[:, None]), dim=1)
        reciprocal_widths = torch.cat(
            (it92_b, torch.zeros_like(it92_c[:, None])), dim=1
        )
        sigma2 = (bfactors[:, None] + reciprocal_widths) / (8.0 * math.pi ** 2)
        if blur_fwhm > 0:
            sigma2 = sigma2 + (blur_fwhm / 2.354820045) ** 2
        sigma2 = sigma2.clamp_min(1e-4)
        differences = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
        distance2 = differences.square().sum(dim=-1)
        normalization = (2.0 * math.pi * sigma2).pow(-1.5)
        components = (
            occupancies[None, :, None]
            * weights[None, :, :]
            * normalization[None, :, :]
            * torch.exp(-distance2[:, :, None] / (2.0 * sigma2[None, :, :]))
        )
        return components.sum(dim=(1, 2))

    if config.get("target_basis", "complex_sf") == "realspace_gaussian":
        radius, spacing = 4.0, 0.5
        axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
        local_offsets = torch.cartesian_prod(axis, axis, axis)
        local_offsets = local_offsets[
            torch.linalg.vector_norm(local_offsets, dim=1) <= radius
        ]
        for site in sites:
            center = torch.cat((site["template"], site["kinematic_b"]), dim=0).mean(dim=0)
            grid_coordinates = center.unsqueeze(0) + local_offsets
            fixed_occupancies = base_occupancies.clone()
            fixed_occupancies[site["indices"]] = 0.0
            # The fixed density cancels algebraically in the loss, but it is
            # rendered and stored explicitly to test the exact requested path.
            with torch.no_grad():
                rho_fixed = atom_density(
                    base_positions, base_bfactors, fixed_occupancies, grid_coordinates
                )
                rho_kinematic_b_sc = atom_density(
                    site["kinematic_b"], base_bfactors[site["indices"]],
                    base_occupancies[site["indices"]], grid_coordinates,
                )
            site["grid_coordinates"] = grid_coordinates
            site["rho_fixed"] = rho_fixed.detach()
            site["rho_target"] = (rho_fixed + rho_kinematic_b_sc).detach()
            site["rho_sidechain_target_by_blur"] = {}
            blur_levels = {0.0, *map(float, config.get("blur_schedule_fwhm", []))}
            for blur_fwhm in blur_levels:
                site["rho_sidechain_target_by_blur"][blur_fwhm] = atom_density(
                    site["kinematic_b"], base_bfactors[site["indices"]],
                    base_occupancies[site["indices"]], grid_coordinates,
                    blur_fwhm=blur_fwhm,
                ).detach()

    if config.get("target_basis") == "experimental_omit_realspace":
        radius, spacing = 4.0, 0.5
        axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
        local_offsets = torch.cartesian_prod(axis, axis, axis)
        local_offsets = local_offsets[
            torch.linalg.vector_norm(local_offsets, dim=1) <= radius
        ]
        requested_keys = set(config.get("site_keys") or site_by_key)
        reciprocal_vectors = hkl @ orth_to_frac
        reciprocal_norm2 = reciprocal_vectors.square().sum(dim=1)
        blur_levels = {0.0, *map(float, config.get("blur_schedule_fwhm", []))}
        for site in sites:
            if site["key"] not in requested_keys:
                continue
            center = torch.cat((site["template"], site["kinematic_b"]), dim=0).mean(dim=0)
            grid_coordinates = center.unsqueeze(0) + local_offsets
            fractional = grid_coordinates @ orth_to_frac.T

            # Sidechain-omit experimental map: phases come from the model with
            # the movable B-altloc sidechain removed, while amplitudes are Fo.
            fixed_amplitudes = site["f_fixed"].abs()
            scale = (
                fixed_amplitudes[valid] @ fobs[valid]
            ) / fixed_amplitudes[valid].square().sum().clamp_min(1e-12)
            phase_unit = site["f_fixed"] / fixed_amplitudes.clamp_min(1e-8)
            two_fo_fc = (2.0 * fobs - scale * fixed_amplitudes) * phase_unit
            experimental_residual = (two_fo_fc - scale * site["f_fixed"]).detach()

            site["experimental_scale"] = scale.detach()
            site["experimental_grid_coordinates"] = grid_coordinates
            site["experimental_target_by_blur"] = {}
            site["experimental_blur_factor"] = {}
            site["experimental_normalizer_by_blur"] = {}
            site["experimental_atom_scale_by_blur"] = {}
            site["experimental_diagnostics_by_blur"] = {}
            for blur_fwhm in blur_levels:
                sigma = blur_fwhm / 2.354820045
                blur_factor = torch.exp(
                    -2.0 * math.pi ** 2 * sigma ** 2 * reciprocal_norm2[valid]
                )
                target_chunks, model_chunks = [], []
                target_valid = experimental_residual[valid] * blur_factor
                model_valid = scale * site["complex_target"][valid] * blur_factor
                for start in range(0, len(fractional), 256):
                    selected = fractional[start:start + 256]
                    kernel = torch.exp(
                        -1j * 2.0 * torch.pi * (hkl[valid] @ selected.T)
                    )
                    target_chunks.append(
                        2.0 * torch.real(target_valid @ kernel) / int(valid.sum())
                    )
                    model_chunks.append(
                        2.0 * torch.real(model_valid @ kernel) / int(valid.sum())
                    )
                target_density = torch.cat(target_chunks)
                model_reference_density = torch.cat(model_chunks)
                analytic_reference_density = xray_atom_density(
                    site["kinematic_b"], base_bfactors[site["indices"]],
                    base_occupancies[site["indices"]],
                    base_it92_a[site["indices"]], base_it92_b[site["indices"]],
                    base_it92_c[site["indices"]],
                    grid_coordinates,
                    blur_fwhm=blur_fwhm,
                )
                atom_scale = (
                    analytic_reference_density @ model_reference_density
                ) / analytic_reference_density.square().sum().clamp_min(1e-12)
                centered_target = target_density - target_density.mean()
                centered_model = model_reference_density - model_reference_density.mean()
                centered_analytic = analytic_reference_density - analytic_reference_density.mean()
                target_model_correlation = (
                    centered_target @ centered_model
                ) / torch.sqrt(
                    centered_target.square().sum().clamp_min(1e-12)
                    * centered_model.square().sum().clamp_min(1e-12)
                )
                analytic_model_correlation = (
                    centered_analytic @ centered_model
                ) / torch.sqrt(
                    centered_analytic.square().sum().clamp_min(1e-12)
                    * centered_model.square().sum().clamp_min(1e-12)
                )
                site["experimental_blur_factor"][blur_fwhm] = blur_factor.detach()
                site["experimental_target_by_blur"][blur_fwhm] = target_density.detach()
                site["experimental_atom_scale_by_blur"][blur_fwhm] = atom_scale.detach()
                site["experimental_diagnostics_by_blur"][blur_fwhm] = {
                    "atom_scale": float(atom_scale.cpu()),
                    "target_model_correlation": float(target_model_correlation.cpu()),
                    "analytic_model_correlation": float(analytic_model_correlation.cpu()),
                }
                site["experimental_normalizer_by_blur"][blur_fwhm] = (
                    target_density.square().mean().clamp_min(1e-12).detach()
                )

    if config.get("target_basis") == "experimental_omit_sampling":
        radius, spacing = 6.0, 0.5
        axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
        local_offsets = torch.cartesian_prod(axis, axis, axis)
        requested_keys = set(config.get("site_keys") or site_by_key)
        reciprocal_vectors = hkl @ orth_to_frac
        reciprocal_norm2 = reciprocal_vectors.square().sum(dim=1)
        blur_levels = {0.0, *map(float, config.get("blur_schedule_fwhm", []))}
        for site in sites:
            if site["key"] not in requested_keys:
                continue
            center = torch.cat((site["template"], site["kinematic_b"]), dim=0).mean(dim=0)
            grid_coordinates = center.unsqueeze(0) + local_offsets
            fractional = grid_coordinates @ orth_to_frac.T

            fixed_amplitudes = site["f_fixed"].abs()
            scale = (
                fixed_amplitudes[valid] @ fobs[valid]
            ) / fixed_amplitudes[valid].square().sum().clamp_min(1e-12)
            phase_unit = site["f_fixed"] / fixed_amplitudes.clamp_min(1e-8)
            two_fo_fc = (2.0 * fobs - scale * fixed_amplitudes) * phase_unit
            experimental_residual = (two_fo_fc - scale * site["f_fixed"]).detach()

            site["sampling_center"] = center
            site["sampling_radius"] = radius
            site["sampling_spacing"] = spacing
            site["sampling_grid_size"] = len(axis)
            site["sampling_map_by_blur"] = {}
            for blur_fwhm in blur_levels:
                sigma = blur_fwhm / 2.354820045
                blur_factor = torch.exp(
                    -2.0 * math.pi ** 2 * sigma ** 2 * reciprocal_norm2[valid]
                )
                coefficients = experimental_residual[valid] * blur_factor
                density_chunks = []
                for start in range(0, len(fractional), 256):
                    selected = fractional[start:start + 256]
                    kernel = torch.exp(
                        -1j * 2.0 * torch.pi * (hkl[valid] @ selected.T)
                    )
                    density_chunks.append(
                        2.0 * torch.real(coefficients @ kernel) / int(valid.sum())
                    )
                density = torch.cat(density_chunks)
                density = (density - density.mean()) / density.std().clamp_min(1e-8)
                site["sampling_map_by_blur"][blur_fwhm] = density.reshape(
                    len(axis), len(axis), len(axis)
                ).detach()

    def complex_sf_loss(site: dict, chi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = coords_from_chi(site, chi)
        positions = base_positions.index_copy(0, site["indices"], coordinates)
        fcalc = calc.calc_fprotein(
            positions, atoms_occ_tensor=base_occupancies, Return=True
        )
        difference = (fcalc - site["f_fixed"])[train_mask] - site["complex_target"][train_mask]
        loss = difference.abs().square().mean() / site["normalizer"]
        return loss, coordinates

    def realspace_loss(
        site: dict, chi: torch.Tensor, blur_fwhm: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = coords_from_chi(site, chi)
        rho_sidechain = atom_density(
            coordinates,
            base_bfactors[site["indices"]],
            base_occupancies[site["indices"]],
            site["grid_coordinates"],
            blur_fwhm=blur_fwhm,
        )
        rho_target = site["rho_sidechain_target_by_blur"][float(blur_fwhm)]
        return (rho_sidechain - rho_target).square().sum(), coordinates

    def experimental_omit_realspace_loss(
        site: dict, chi: torch.Tensor, blur_fwhm: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = coords_from_chi(site, chi)
        candidate_density = xray_atom_density(
            coordinates,
            base_bfactors[site["indices"]],
            base_occupancies[site["indices"]],
            base_it92_a[site["indices"]], base_it92_b[site["indices"]],
            base_it92_c[site["indices"]],
            site["experimental_grid_coordinates"],
            blur_fwhm=blur_fwhm,
        ) * site["experimental_atom_scale_by_blur"][float(blur_fwhm)]
        target_density = site["experimental_target_by_blur"][float(blur_fwhm)]
        loss = (candidate_density - target_density).square().mean()
        loss = loss / site["experimental_normalizer_by_blur"][float(blur_fwhm)]
        return loss, coordinates

    def trilinear_sample(
        grid: torch.Tensor, coordinates: torch.Tensor, site: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin = site["sampling_center"] - site["sampling_radius"]
        continuous = (coordinates - origin) / site["sampling_spacing"]
        maximum = site["sampling_grid_size"] - 1.0001
        below = torch.relu(-continuous)
        above = torch.relu(continuous - maximum)
        outside_penalty = (below.square() + above.square()).sum(dim=1)
        clamped = continuous.clamp(0.0, maximum)
        lower = torch.floor(clamped).long()
        upper = lower + 1
        fraction = clamped - lower.to(clamped.dtype)

        values = torch.zeros(len(coordinates), dtype=grid.dtype, device=grid.device)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    index = torch.stack((
                        lower[:, 0] if dx == 0 else upper[:, 0],
                        lower[:, 1] if dy == 0 else upper[:, 1],
                        lower[:, 2] if dz == 0 else upper[:, 2],
                    ), dim=1)
                    weight = (
                        (1.0 - fraction[:, 0] if dx == 0 else fraction[:, 0])
                        * (1.0 - fraction[:, 1] if dy == 0 else fraction[:, 1])
                        * (1.0 - fraction[:, 2] if dz == 0 else fraction[:, 2])
                    )
                    values = values + weight * grid[
                        index[:, 0], index[:, 1], index[:, 2]
                    ]
        return values, outside_penalty

    def experimental_omit_sampling_loss(
        site: dict, chi: torch.Tensor, blur_fwhm: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates = coords_from_chi(site, chi)
        values, outside_penalty = trilinear_sample(
            site["sampling_map_by_blur"][float(blur_fwhm)], coordinates, site
        )
        weights = (
            base_occupancies[site["indices"]] * base_atomic_numbers[site["indices"]]
        )
        score = (weights * values).sum() / weights.sum().clamp_min(1e-8)
        loss = -score + 2.0 * outside_penalty.mean()
        return loss, coordinates

    def loss_and_coordinates(
        site: dict, chi: torch.Tensor, blur_fwhm: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if config.get("target_basis", "complex_sf") == "realspace_gaussian":
            return realspace_loss(site, chi, blur_fwhm=blur_fwhm)
        if config.get("target_basis") == "experimental_omit_realspace":
            return experimental_omit_realspace_loss(site, chi, blur_fwhm=blur_fwhm)
        if config.get("target_basis") == "experimental_omit_sampling":
            return experimental_omit_sampling_loss(site, chi, blur_fwhm=blur_fwhm)
        return complex_sf_loss(site, chi)

    # Hard identity gate before optimization.
    assertion_rows = []
    calibration_failures = []
    assertion_sites = sites
    if config.get("target_basis") in {
        "experimental_omit_realspace", "experimental_omit_sampling"
    }:
        assertion_sites = [
            site_by_key[key] for key in (config.get("site_keys") or site_by_key)
        ]
    for site in assertion_sites:
        with torch.no_grad():
            loss_a, _ = loss_and_coordinates(
                site, torch.zeros(site["n_chi"], device=device)
            )
            loss_b, _ = loss_and_coordinates(site, site["true_delta"])
        assertion_row = {
            "site": site["key"],
            "loss_A": float(loss_a.cpu()),
            "loss_kinematic_B": float(loss_b.cpu()),
            "kinematic_to_deposited_B_rmsd": float(torch.sqrt(torch.mean(
                (site["kinematic_b"] - site["deposited_b"]).square()
            )).cpu()),
        }
        if config.get("target_basis") == "experimental_omit_realspace":
            assertion_row.update(site["experimental_diagnostics_by_blur"][0.0])
            if (
                assertion_row["analytic_model_correlation"]
                < config.get("experimental_min_calibration_correlation", 0.85)
            ):
                calibration_failures.append(
                    f"{site['key']}={assertion_row['analytic_model_correlation']:.4f}"
                )
        assertion_rows.append(assertion_row)
        if not (math.isfinite(float(loss_a.cpu())) and math.isfinite(float(loss_b.cpu()))):
            raise RuntimeError(f"non-finite target loss at {site['key']}")
        if (
            config.get("target_basis") not in {
                "experimental_omit_realspace", "experimental_omit_sampling"
            }
            and (float(loss_b.cpu()) >= 1e-6 or float(loss_b.cpu()) >= float(loss_a.cpu()))
        ):
            raise RuntimeError(f"kinematic target identity gate failed at {site['key']}")
    with (out / "pre_optimization_assertions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=assertion_rows[0].keys())
        writer.writeheader()
        writer.writerows(assertion_rows)
    commit()
    if calibration_failures:
        raise RuntimeError(
            "experimental candidate-renderer calibration below threshold: "
            + ", ".join(calibration_failures)
        )
    if config.get("calibration_only"):
        mark("pipeline", "complete", calibration_only=True)
        return load_manifest()

    def optimize(
        site: dict,
        *,
        n_starts: int,
        n_steps: int,
        lr: float,
        seed_offset: int,
        init: str = "random",
        noise: float = 0.0,
        blur_schedule: list[tuple] | None = None,
        reset_optimizer_at_blur_transition: bool = False,
        progress_callback=None,
    ) -> tuple[list[dict], dict[str, np.ndarray]]:
        endpoints = []
        all_chi, all_loss, all_rmsd_a, all_rmsd_b, all_blur = [], [], [], [], []
        if blur_schedule:
            step_plan = []
            for stage_index, stage in enumerate(blur_schedule):
                blur, steps = stage[:2]
                stage_lr = stage[2] if len(stage) == 3 else lr
                step_plan.extend(
                    (float(blur), float(stage_lr), offset == 0 and stage_index > 0)
                    for offset in range(int(steps))
                )
        else:
            step_plan = [(0.0, lr, False)] * n_steps
        n_steps = len(step_plan)
        for start in range(n_starts):
            start_generator = torch.Generator(device=device).manual_seed(
                config["seed"] + seed_offset + start
            )
            if init == "A":
                initial = 0.1 * torch.randn(
                    site["n_chi"], generator=start_generator, device=device
                )
            else:
                initial = torch.randn(
                    site["n_chi"], generator=start_generator, device=device
                )
            chi = initial.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([chi], lr=lr)
            chi_history, loss_history, rmsd_a_history, rmsd_b_history = [], [], [], []
            blur_history = []
            first_below = None

            def record(
                value: torch.Tensor, coordinates: torch.Tensor, blur_fwhm: float
            ) -> None:
                nonlocal first_below
                with torch.no_grad():
                    rmsd_a = torch.sqrt(torch.mean((coordinates - site["template"]).square()))
                    rmsd_b = torch.sqrt(torch.mean((coordinates - site["kinematic_b"]).square()))
                    chi_history.append(wrap_angles(chi).detach().cpu().numpy())
                    loss_history.append(float(value.cpu()))
                    rmsd_a_history.append(float(rmsd_a.cpu()))
                    rmsd_b_history.append(float(rmsd_b.cpu()))
                    blur_history.append(float(blur_fwhm))
                    if first_below is None and rmsd_b_history[-1] < 0.50:
                        first_below = len(loss_history) - 1

            for _step, (blur_fwhm, stage_lr, is_transition) in enumerate(step_plan):
                if is_transition and reset_optimizer_at_blur_transition:
                    optimizer = torch.optim.Adam([chi], lr=stage_lr)
                else:
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = stage_lr
                optimizer.zero_grad(set_to_none=True)
                loss, coordinates = loss_and_coordinates(
                    site, wrap_angles(chi), blur_fwhm=blur_fwhm
                )
                record(loss, coordinates, blur_fwhm)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if noise > 0:
                        chi.add_(noise * torch.randn(
                            chi.shape, generator=start_generator, device=device
                        ))
                    chi.copy_(wrap_angles(chi))
            # Record the state after the final Adam/Langevin update. This gives
            # n_steps + 1 trajectory points using only n_steps + 1 SF calls,
            # rather than evaluating SFcalculator twice per update.
            with torch.no_grad():
                final_loss, final_coordinates = loss_and_coordinates(site, wrap_angles(chi))
            record(final_loss, final_coordinates, 0.0)

            endpoints.append({
                "site": site["key"],
                "start": start,
                "n_steps": n_steps,
                "lr": lr,
                "init": init,
                "langevin_noise": noise,
                "blur_schedule_fwhm_steps": (
                    ";".join(
                        f"{stage[0]:g}:{stage[1]}:lr{stage[2]:g}"
                        if len(stage) == 3 else f"{stage[0]:g}:{stage[1]}"
                        for stage in blur_schedule
                    )
                    if blur_schedule else ""
                ),
                "reset_optimizer_at_blur_transition": reset_optimizer_at_blur_transition,
                "final_loss": loss_history[-1],
                "best_loss": min(loss_history),
                "final_rmsd_to_A": rmsd_a_history[-1],
                "final_rmsd_to_B": rmsd_b_history[-1],
                "best_rmsd_to_B": min(rmsd_b_history),
                "first_step_below_0_50": first_below if first_below is not None else "",
                "final_chi_radians": ";".join(f"{x:.8g}" for x in chi_history[-1]),
            })
            all_chi.append(np.stack(chi_history))
            all_loss.append(np.asarray(loss_history, dtype=np.float32))
            all_rmsd_a.append(np.asarray(rmsd_a_history, dtype=np.float32))
            all_rmsd_b.append(np.asarray(rmsd_b_history, dtype=np.float32))
            all_blur.append(np.asarray(blur_history, dtype=np.float32))
            if progress_callback is not None and ((start + 1) % 10 == 0 or start + 1 == n_starts):
                progress_callback(start + 1, endpoints)
        return endpoints, {
            "chi": np.stack(all_chi),
            "loss": np.stack(all_loss),
            "rmsd_to_A": np.stack(all_rmsd_a),
            "rmsd_to_B": np.stack(all_rmsd_b),
            "blur_fwhm": np.stack(all_blur),
        }

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # LR sweep on A_MET112, as specified.
    sweep_dir = out / "lr_sweep"
    sweep_dir.mkdir(exist_ok=True)
    if config.get("chosen_lr") is not None:
        chosen_lr = float(config["chosen_lr"])
    elif not done("lr_sweep"):
        sweep_rows = []
        for lr_index, lr in enumerate(config["lr_sweep"]):
            endpoints, _ = optimize(
                site_by_key["A_MET112"], n_starts=10, n_steps=200, lr=lr,
                seed_offset=10_000 * (lr_index + 1),
            )
            sweep_rows.append({
                "lr": lr,
                "mean_final_loss": float(np.mean([row["final_loss"] for row in endpoints])),
                "mean_final_rmsd_to_B": float(np.mean([
                    row["final_rmsd_to_B"] for row in endpoints
                ])),
                "hits_rmsd_B_lt_0_50": sum(
                    row["final_rmsd_to_B"] < 0.50 for row in endpoints
                ),
                "hits_rmsd_B_lt_0_75": sum(
                    row["final_rmsd_to_B"] < 0.75 for row in endpoints
                ),
            })
            write_csv(sweep_dir / "lr_sweep_results.csv", sweep_rows)
            commit()
        if config.get("target_basis", "complex_sf") == "realspace_gaussian":
            selection_criterion = "most_lt_0.50_then_lowest_mean_rmsd_then_loss"
            chosen_lr = min(
                sweep_rows,
                key=lambda row: (
                    -row["hits_rmsd_B_lt_0_50"],
                    row["mean_final_rmsd_to_B"],
                    row["mean_final_loss"],
                ),
            )["lr"]
            write_csv(out / "lr_sweep_results.csv", sweep_rows)
        else:
            selection_criterion = "lowest_mean_final_loss"
            chosen_lr = min(
                sweep_rows, key=lambda row: (
                    row["mean_final_loss"], row["mean_final_rmsd_to_B"]
                )
            )["lr"]
        atomic_json(sweep_dir / "selection.json", {
            "criterion": selection_criterion, "chosen_lr": chosen_lr
        })
        mark("lr_sweep", "complete", chosen_lr=chosen_lr)
    else:
        chosen_lr = json.loads((sweep_dir / "selection.json").read_text())["chosen_lr"]

    if config.get("coarse_to_fine"):
        variants = [{
            "name": "coarse_to_fine_4A_2A_full_decay",
            "n_steps": sum(config["blur_schedule_steps"]),
            "lr": chosen_lr,
            "init": "random",
            "noise": 0.0,
            "blur_schedule": list(zip(
                config["blur_schedule_fwhm"], config["blur_schedule_steps"],
                config["blur_schedule_lrs"],
            )),
            "reset_optimizer_at_blur_transition": True,
        }]
    else:
        variants = [
            {"name": "vanilla_200steps", "n_steps": 200, "lr": chosen_lr, "init": "random", "noise": 0.0},
            {"name": "thorough_1000steps", "n_steps": 1000, "lr": 0.001, "init": "random", "noise": 0.0},
            {"name": "from_A_init", "n_steps": 500, "lr": chosen_lr, "init": "A", "noise": 0.0},
        ]
        for noise in config["langevin_noises"]:
            variants.append({
                "name": f"langevin_500steps/noise_{noise:g}", "n_steps": 500,
                "lr": chosen_lr, "init": "random", "noise": noise,
            })
        if config.get("target_basis", "complex_sf") == "realspace_gaussian":
            variants = variants[:2]
    if config.get("selected_variant"):
        variants = [
            variant for variant in variants
            if variant["name"] == config["selected_variant"]
        ]
        if not variants:
            raise ValueError(f"unknown selected variant {config['selected_variant']}")

    site_keys = config.get("site_keys") or [
        "A_MET112", "A_ARG129", "B_MET112", "B_ASP114", "B_ARG129"
    ]
    for variant_index, variant in enumerate(variants):
        directory = out / variant["name"]
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "run_config.json", {
            **variant, "n_starts": config["n_starts"], "target": config["target_basis"],
            "reflection_split_seed": config["seed"],
        })
        all_endpoints = []
        trajectory_by_site = {}
        for site_index, site_key in enumerate(site_keys):
            stage = f"{variant['name']}::{site_key}"
            endpoint_path = directory / f"endpoints_{site_key}.csv"
            trajectory_path = directory / f"trajectories_{site_key}.npz"
            if done(stage) and endpoint_path.exists() and trajectory_path.exists():
                with endpoint_path.open() as handle:
                    endpoints = list(csv.DictReader(handle))
                trajectory_by_site[site_key] = dict(np.load(trajectory_path))
                all_endpoints.extend(endpoints)
                continue

            def checkpoint(completed_starts: int, partial: list[dict]) -> None:
                write_csv(endpoint_path, partial)
                mark(stage, "running", completed_starts=completed_starts)

            endpoints, trajectories = optimize(
                site_by_key[site_key],
                n_starts=config["n_starts"], n_steps=variant["n_steps"], lr=variant["lr"],
                init=variant["init"], noise=variant["noise"],
                blur_schedule=variant.get("blur_schedule"),
                reset_optimizer_at_blur_transition=variant.get(
                    "reset_optimizer_at_blur_transition", False
                ),
                seed_offset=100_000 * (variant_index + 1) + 1_000 * (site_index + 1),
                progress_callback=checkpoint,
            )
            write_csv(endpoint_path, endpoints)
            np.savez_compressed(trajectory_path, **trajectories)
            all_endpoints.extend(endpoints)
            trajectory_by_site[site_key] = trajectories
            mark(stage, "complete", completed_starts=len(endpoints))

        # Normalize values loaded from CSV on resume.
        def number(row: dict, key: str) -> float:
            return float(row[key])

        recovery_rows = []
        for site_key in site_keys:
            selected = [row for row in all_endpoints if row["site"] == site_key]
            first_steps = [
                number(row, "first_step_below_0_50") for row in selected
                if str(row["first_step_below_0_50"]) != ""
            ]
            recovery_rows.append({
                "site": site_key,
                "split": "held_out" if site_key in {"B_ASP114", "B_ARG129"} else "train",
                "hits_rmsd_B_lt_0_50": sum(number(row, "final_rmsd_to_B") < 0.50 for row in selected),
                "hits_rmsd_B_lt_0_75": sum(number(row, "final_rmsd_to_B") < 0.75 for row in selected),
                "mean_final_rmsd_to_B": float(np.mean([number(row, "final_rmsd_to_B") for row in selected])),
                "mean_final_loss": float(np.mean([number(row, "final_loss") for row in selected])),
                "best_final_loss": min(number(row, "final_loss") for row in selected),
                "mean_steps_to_first_lt_0_50": float(np.mean(first_steps)) if first_steps else "",
                "std_steps_to_first_lt_0_50": float(np.std(first_steps)) if first_steps else "",
            })
        write_csv(directory / "recovery_table.csv", recovery_rows)
        write_csv(directory / "all_endpoints.csv", all_endpoints)

        # One RMSD trajectory plot per site.
        for site_key in site_keys:
            trajectories = trajectory_by_site[site_key]
            plt.figure(figsize=(8, 5))
            plt.plot(trajectories["rmsd_to_B"].T, alpha=0.28, linewidth=0.7)
            plt.axhline(0.50, color="black", linestyle="--", linewidth=1)
            plt.xlabel("Optimization step")
            plt.ylabel("RMSD to kinematic B (A)")
            plt.title(f"{variant['name']} — {site_key}")
            plt.tight_layout()
            plt.savefig(directory / f"trajectory_rmsd_to_B_{site_key}.png", dpi=180)
            plt.close()

        # Loss trajectories for all sites.
        figure, axes = plt.subplots(len(site_keys), 1, figsize=(9, 13), sharex=True)
        axes = np.atleast_1d(axes)
        for axis, site_key in zip(axes, site_keys):
            axis.plot(trajectory_by_site[site_key]["loss"].T, alpha=0.25, linewidth=0.6)
            axis.set_yscale("symlog", linthresh=1e-8)
            axis.set_ylabel(site_key)
        axes[-1].set_xlabel("Optimization step")
        figure.suptitle(f"Complex-SF loss — {variant['name']}")
        figure.tight_layout()
        figure.savefig(directory / "trajectory_loss_all_sites.png", dpi=180)
        plt.close(figure)

        # Final-state loss/RMSD scatter.
        plt.figure(figsize=(8, 5))
        for site_key in site_keys:
            selected = [row for row in all_endpoints if row["site"] == site_key]
            plt.scatter(
                [number(row, "final_loss") for row in selected],
                [number(row, "final_rmsd_to_B") for row in selected],
                s=16, alpha=0.65, label=site_key,
            )
        plt.xscale("symlog", linthresh=1e-10)
        plt.axhline(0.50, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Final normalized complex-SF loss")
        plt.ylabel("Final RMSD to kinematic B (A)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(directory / "rmsd_vs_loss_scatter.png", dpi=180)
        plt.close()

        # A_ARG129 chi1/chi2 paths, when that site is part of this shard.
        if "A_ARG129" in trajectory_by_site:
            arg_chi = trajectory_by_site["A_ARG129"]["chi"]
            plt.figure(figsize=(7, 6))
            colors = np.linspace(0, 1, arg_chi.shape[1])
            for start in range(arg_chi.shape[0]):
                plt.scatter(
                    np.degrees(arg_chi[start, :, 0]), np.degrees(arg_chi[start, :, 1]),
                    c=colors, cmap="viridis", s=3, alpha=0.45,
                )
            true_delta = site_by_key["A_ARG129"]["true_delta"].detach().cpu().numpy()
            plt.scatter([0], [0], marker="X", s=130, color="red", label="Deposited A")
            plt.scatter(
                [math.degrees(float(true_delta[0]))], [math.degrees(float(true_delta[1]))],
                marker="*", s=180, color="gold", edgecolor="black", label="Kinematic B",
            )
            plt.xlabel("chi1 delta (degrees)")
            plt.ylabel("chi2 delta (degrees)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(directory / "chi_trajectory_A_ARG129.png", dpi=180)
            plt.close()
        mark(f"{variant['name']}::summary", "complete")

    # Matched basis comparison: hold chi3/chi4 (when present) at kinematic B and
    # scan chi1/chi2 on exactly the same grid for every requested site.
    if (
        config.get("target_basis", "complex_sf") == "realspace_gaussian"
        and not config.get("skip_landscape", False)
        and not done("landscape_comparison")
    ):
        landscape_directory = out / "landscape_comparison"
        landscape_directory.mkdir(exist_ok=True)
        axis = torch.linspace(-math.pi, math.pi, config["landscape_size"], device=device)
        extent = [-180, 180, -180, 180]

        def robust_unit(values: np.ndarray) -> np.ndarray:
            logged = np.log10(values + 1e-8)
            low, high = np.nanpercentile(logged, [1.0, 99.0])
            return np.clip((logged - low) / max(high - low, 1e-8), 0.0, 1.0)

        landscape_values = {}
        for site_key in site_keys:
            site = site_by_key[site_key]
            site_directory = landscape_directory / site_key
            site_directory.mkdir(exist_ok=True)
            complex_values = np.full((len(axis), len(axis)), np.nan, dtype=np.float32)
            realspace_values = np.full_like(complex_values, np.nan)
            for i, chi1 in enumerate(axis):
                for j, chi2 in enumerate(axis):
                    chi = site["true_delta"].clone()
                    chi[0], chi[1] = chi1, chi2
                    with torch.no_grad():
                        complex_value, _ = complex_sf_loss(site, chi)
                        realspace_value, _ = realspace_loss(site, chi)
                    complex_values[i, j] = float(complex_value.cpu())
                    realspace_values[i, j] = float(realspace_value.cpu())
                if (i + 1) % 5 == 0 or i + 1 == len(axis):
                    np.savez_compressed(
                        site_directory / "landscape_partial.npz",
                        chi_degrees=np.degrees(axis.detach().cpu().numpy()),
                        complex_sf=complex_values,
                        realspace=realspace_values,
                        completed_rows=i + 1,
                    )
                    mark(
                        f"landscape_comparison::{site_key}", "running",
                        completed_rows=i + 1, total_rows=len(axis),
                    )
            np.savez_compressed(
                site_directory / "landscape_values.npz",
                chi_degrees=np.degrees(axis.detach().cpu().numpy()),
                complex_sf=complex_values,
                realspace=realspace_values,
            )
            landscape_values[site_key] = (complex_values, realspace_values)

            # A surface view matching the original diagnostic figure, but with
            # both bases robustly normalized to the same 0-1 shape scale.
            degrees = np.degrees(axis.detach().cpu().numpy())
            grid_chi1, grid_chi2 = np.meshgrid(degrees, degrees, indexing="ij")
            surface_figure = plt.figure(figsize=(13, 5.5))
            for panel, values, title in (
                (1, robust_unit(complex_values), "Complex-SF landscape"),
                (2, robust_unit(realspace_values), "Real-space landscape"),
            ):
                surface_axis = surface_figure.add_subplot(1, 2, panel, projection="3d")
                surface_axis.plot_surface(
                    grid_chi1, grid_chi2, values, cmap="viridis",
                    linewidth=0, antialiased=True,
                )
                surface_axis.set_xlabel("chi1 delta (degrees)")
                surface_axis.set_ylabel("chi2 delta (degrees)")
                surface_axis.set_zlabel("normalized log-loss shape")
                surface_axis.set_zlim(0, 1)
                surface_axis.set_title(title)
            surface_figure.suptitle(site_key)
            surface_figure.tight_layout()
            surface_figure.savefig(site_directory / "paired_surfaces.png", dpi=180)
            plt.close(surface_figure)
            mark(f"landscape_comparison::{site_key}", "complete", total_rows=len(axis))

        # Raw log-loss panels retain absolute dynamic range; matched normalized
        # panels reveal differences in basin topology independently of scale.
        figure, axes = plt.subplots(
            len(site_keys), 3, figsize=(13.5, 4.0 * len(site_keys)),
            sharex=True, sharey=True, squeeze=False,
        )
        for row, site_key in enumerate(site_keys):
            site = site_by_key[site_key]
            complex_values, realspace_values = landscape_values[site_key]
            complex_unit = robust_unit(complex_values)
            realspace_unit = robust_unit(realspace_values)
            panels = (
                (complex_unit, "Complex-SF (shape)"),
                (realspace_unit, "Real space (shape)"),
                (realspace_unit - complex_unit, "Real − SF shape"),
            )
            for column, (values, title) in enumerate(panels):
                image = axes[row, column].imshow(
                    values.T, origin="lower", extent=extent, aspect="auto",
                    cmap="coolwarm" if column == 2 else "viridis",
                    vmin=-1 if column == 2 else 0,
                    vmax=1,
                )
                axes[row, column].scatter([0], [0], marker="x", s=55, color="white")
                axes[row, column].scatter(
                    [math.degrees(float(site["true_delta"][0]))],
                    [math.degrees(float(site["true_delta"][1]))],
                    marker="*", s=110, color="gold", edgecolor="black",
                )
                if row == 0:
                    axes[row, column].set_title(title)
                if row == len(site_keys) - 1:
                    axes[row, column].set_xlabel("chi1 delta (degrees)")
                if column == 0:
                    axes[row, column].set_ylabel(f"{site_key}\nchi2 delta (degrees)")
                figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.03)
        figure.tight_layout()
        figure.savefig(landscape_directory / "five_site_shape_comparison.png", dpi=200)
        plt.close(figure)
        mark("landscape_comparison", "complete", sites=site_keys)

    # Assemble the requested key comparison. For Langevin, keep all noise levels.
    comparison_rows = []
    for variant in variants:
        with (out / variant["name"] / "recovery_table.csv").open() as handle:
            for row in csv.DictReader(handle):
                comparison_rows.append({"variant": variant["name"], **row})
    write_csv(out / "comparison_table.csv", comparison_rows)

    lines = [
        "# Direct Optimization Control",
        "",
        f"LR sweep selected `{chosen_lr}` by lowest mean final complex-SF loss.",
        "",
        "| Variant | Site | <0.50 A | <0.75 A | Mean RMSD-B | Mean final loss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['variant']} | {row['site']} | {row['hits_rmsd_B_lt_0_50']}/{config['n_starts']} | "
            f"{row['hits_rmsd_B_lt_0_75']}/{config['n_starts']} | {float(row['mean_final_rmsd_to_B']):.4f} | "
            f"{float(row['mean_final_loss']):.6g} |"
        )
    lines.extend([
        "", "## MLP comparison", "",
        "| Site | Direct vanilla | Direct thorough | MLP 20 inference steps | MLP training inner steps |",
        "|---|---:|---:|---:|---:|",
    ])
    vanilla = {row["site"]: row for row in comparison_rows if row["variant"] == "vanilla_200steps"}
    thorough = {row["site"]: row for row in comparison_rows if row["variant"] == "thorough_1000steps"}
    if vanilla and thorough:
        mlp = {"A_MET112": 0, "A_ARG129": 0, "B_MET112": 50}
        for site_key in ("A_MET112", "A_ARG129", "B_MET112"):
            lines.append(
                f"| {site_key} | {vanilla[site_key]['hits_rmsd_B_lt_0_50']}/50 | "
                f"{thorough[site_key]['hits_rmsd_B_lt_0_50']}/50 | {mlp[site_key]}/50 | 3 |"
            )
    (out / "comparison_table.md").write_text("\n".join(lines) + "\n")
    mark("pipeline", "complete", chosen_lr=chosen_lr)
    return load_manifest()


@APP.local_entrypoint()
def main(
    run_name: str = "direct_optimization_control",
    force: bool = False,
    chosen_lr: float | None = None,
    selected_variant: str = "",
    target_basis: str = "complex_sf",
    n_starts: int = 50,
    site_keys: str = "",
    skip_landscape: bool = False,
    coarse_to_fine: bool = False,
    landscape_size: int = 100,
    calibration_only: bool = False,
):
    config = {
        "run_name": run_name,
        "force": force,
        "seed": 41,
        "reflection_holdout": 0.05,
        "n_starts": n_starts,
        "lr_sweep": [0.001, 0.003, 0.01, 0.03, 0.1],
        "langevin_noises": [0.01, 0.05, 0.1],
        "chosen_lr": chosen_lr,
        "selected_variant": selected_variant or None,
        "target_basis": target_basis,
        "landscape_size": landscape_size,
        "site_keys": [key.strip() for key in site_keys.split(",") if key.strip()] or None,
        "skip_landscape": skip_landscape,
        "coarse_to_fine": coarse_to_fine,
        "blur_schedule_fwhm": [4.0, 2.0, 0.0],
        "blur_schedule_steps": [100, 100, 100],
        "blur_schedule_lrs": [1.0, 0.1, 0.01],
        "calibration_only": calibration_only,
        "experimental_min_calibration_correlation": 0.85,
    }
    call = run_control.spawn(config)
    print({"status": "submitted", "function_call_id": call.object_id, "run_name": run_name})
