"""K=4 ensemble fitting against the experimental 2mFo-DFc map for 2O1K.

The experimental cross-term is evaluated by differentiable trilinear map
sampling.  A Gaussian model self-overlap term supplies the quadratic part of
least-squares map fitting; without it, the objective is linear in occupancy and
cannot distinguish an A+B ensemble from duplicated copies of one conformer.

Run detached::

    UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run --detach \
        experimental_multi_conformer_modal.py

Results are committed after calibration, every five starts, every completed
site/condition, and final aggregation.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("experimental-multi-conformer-2o1k")
RESULTS_VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.40/"
    "tmol-0.1.40%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl"
)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .pip_install(
        "SFcalculator-torch==0.3.3",
        "gemmi==0.6.7",
        "matplotlib==3.9.4",
        "numpy==1.26.4",
    )
    .run_commands(f"pip install 'tmol @ {TMOL_WHEEL}'")
    .add_local_file(ROOT / "probe4_core.py", remote_path="/root/probe4_core.py", copy=True)
    .add_local_file(ROOT / "data" / "2O1K.pdb", remote_path="/data/2O1K.pdb", copy=True)
    .add_local_file(ROOT / "data" / "2O1K.mtz", remote_path="/data/2O1K.mtz", copy=True)
    .add_local_file(
        ROOT / "multi_conformer_multi_protein" / "2O1K" / "multi_conformer_summary.csv",
        remote_path="/reference/synthetic_summary.csv",
        copy=True,
    )
    .add_local_dir(
        ROOT / "probe4b_results" / "endpoint_audit" / "visualization",
        remote_path="/audit",
        copy=True,
    )
)


@APP.function(image=IMAGE, gpu="L4", timeout=86_400, volumes={"/outputs": RESULTS_VOLUME})
def run_experiment(config: dict) -> dict:
    import csv
    import json
    import math
    import os
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
    per_site = out / "per_site"
    figures = out / "figures"
    overlays = figures / "ensemble_overlays"
    for directory in (out, per_site, figures, overlays):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "stage_manifest.json"
    seed_everything(config["seed"])

    def atomic_json(path: Path, value) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)

    def commit() -> None:
        RESULTS_VOLUME.commit()

    def manifest() -> dict:
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"protein": "2O1K", "created_at": time.time(), "stages": {}}

    def mark(stage: str, status: str, **details) -> None:
        value = manifest()
        value["stages"][stage] = {"status": status, "updated_at": time.time(), **details}
        atomic_json(manifest_path, value)
        commit()

    def done(stage: str) -> bool:
        return (
            not config["force"]
            and manifest().get("stages", {}).get(stage, {}).get("status") == "complete"
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
                          ("CB", "CG", ("SD", "CE")), ("CG", "SD", ("CE",))),
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
    valid = torch.isfinite(fobs) & (fobs > 0) & torch.isfinite(deposited_fcalc.abs())
    scale = (
        deposited_fcalc.abs()[valid] @ fobs[valid]
    ) / deposited_fcalc.abs()[valid].square().sum().clamp_min(1e-12)
    phase = deposited_fcalc / deposited_fcalc.abs().clamp_min(1e-8)
    standard_map_coefficients = (
        (2.0 * fobs - scale * deposited_fcalc.abs()) * phase
    ).detach()
    reciprocal_vectors = hkl @ orth_to_frac
    reciprocal_norm2 = reciprocal_vectors.square().sum(dim=1)

    structure = gemmi.read_structure(pdb_path)
    atoms_with_context = [
        (chain, residue, atom)
        for chain in structure[0]
        for residue in chain
        for atom in residue
    ]
    if len(atoms_with_context) != len(base_positions):
        raise RuntimeError("gemmi and SFcalculator disagree on atom ordering")
    base_bfactors = torch.tensor(
        [atom.b_iso for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32, device=device,
    )
    base_atomic_numbers = torch.tensor(
        [atom.element.atomic_number for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32, device=device,
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
            idx = torch.tensor(indices, dtype=torch.long, device=device)
            omit_indices = [
                i for i, (candidate_chain, candidate_residue, atom) in enumerate(atoms_with_context)
                if candidate_chain.name == chain.name
                and candidate_residue.seqid.num == residue.seqid.num
                and atom.altloc in {"A", "B"}
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            raw_occ = torch.tensor(
                [atoms_with_context[i][2].occ for i in indices], dtype=torch.float32, device=device
            )
            occ_b = float(raw_occ.median().cpu())
            a_atoms = [atom for atom in residue if atom.altloc == "A" and atom.name.strip() in names]
            occ_a = float(np.median([atom.occ for atom in a_atoms]))
            occ_total = max(occ_a + occ_b, 1e-6)
            direct_indices = [
                i for i, (candidate_chain, candidate_residue, _atom) in enumerate(atoms_with_context)
                if not (
                    candidate_chain.name == chain.name
                    and candidate_residue.seqid.num == residue.seqid.num
                )
            ]
            sites.append({
                "key": f"{chain.name}_{residue.name}{residue.seqid.num}",
                "chain": chain.name,
                "number": residue.seqid.num,
                "resname": residue.name,
                "n_chi": len(spec["rotations"]),
                "rotations": list(spec["rotations"]),
                "names": names,
                "indices": idx,
                "omit_indices": torch.tensor(omit_indices, dtype=torch.long, device=device),
                "template": template,
                "deposited_b": deposited_b,
                "fixed": fixed,
                "true_delta": delta,
                "bfactors": base_bfactors[idx],
                "atom_weights": base_atomic_numbers[idx] * (raw_occ / max(occ_b, 1e-6)),
                "target_occ_a": occ_a / occ_total,
                "target_occ_b": occ_b / occ_total,
                "direct_environment": base_positions.index_select(
                    0, torch.tensor(direct_indices, dtype=torch.long, device=device)
                ).detach(),
            })
    if len(sites) != 5:
        raise RuntimeError(f"expected five sites, found {[site['key'] for site in sites]}")

    def coords_from_chi(site: dict, chi: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            site["template"], site["names"], chi, site["rotations"], site["fixed"]
        )

    # Build the requested experimental coefficients.  Omit variants remove
    # BOTH A and B sidechains; removing only the B-altloc would leave A phase
    # bias in place and is not a valid sidechain omit map.
    for site_index, site in enumerate(sites):
        variant = config["map_variant"]
        if variant == "standard_2mfo_dfc":
            site["map_coefficients"] = standard_map_coefficients
            continue
        omitted_occupancies = base_occupancies.clone()
        omitted_occupancies[site["omit_indices"]] = 0.0

        def coefficients_from_model(model_fcalc: torch.Tensor, difference: bool):
            amplitudes = model_fcalc.abs()
            model_scale = (
                amplitudes[valid] @ fobs[valid]
            ) / amplitudes[valid].square().sum().clamp_min(1e-12)
            model_phase = model_fcalc / amplitudes.clamp_min(1e-8)
            multiplier = 1.0 if difference else 2.0
            return (
                (multiplier * fobs - model_scale * amplitudes) * model_phase
            ).detach()

        if variant in {"omit_2mfo_dfc", "omit_mfo_dfc"}:
            omitted_fcalc = calc.calc_fprotein(
                base_positions, atoms_occ_tensor=omitted_occupancies, Return=True
            ).detach()
            site["map_coefficients"] = coefficients_from_model(
                omitted_fcalc, difference=variant == "omit_mfo_dfc"
            )
        elif variant == "averaged_kick_omit_2mfo_dfc":
            generator = torch.Generator(device=device).manual_seed(
                config["seed"] + 1_000_000 + site_index
            )
            kicked_coefficients = []
            for _ in range(config["kick_models"]):
                kicked_positions = base_positions + config["kick_sigma_angstrom"] * torch.randn(
                    base_positions.shape, generator=generator, device=device
                )
                kicked_fcalc = calc.calc_fprotein(
                    kicked_positions, atoms_occ_tensor=omitted_occupancies, Return=True
                ).detach()
                kicked_coefficients.append(
                    coefficients_from_model(kicked_fcalc, difference=False)
                )
            site["map_coefficients"] = torch.stack(kicked_coefficients).mean(dim=0)
        elif variant == "external_polder_2mfo_dfc":
            raise RuntimeError(
                "external_polder_2mfo_dfc requires supplied Phenix Polder map coefficients; "
                "Gemmi/SFcalculator do not implement the local bulk-solvent exclusion mask"
            )
        else:
            raise ValueError(f"unknown map variant: {variant}")

    # Build local maps directly from the selected coefficients.
    radius, spacing = config["map_radius"], config["map_spacing"]
    axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
    offsets = torch.cartesian_prod(axis, axis, axis)
    blur_levels = (0.0, 2.0, 4.0)
    for site in sites:
        site["kinematic_a"] = coords_from_chi(
            site, torch.zeros(site["n_chi"], device=device)
        ).detach()
        site["kinematic_b"] = coords_from_chi(site, site["true_delta"]).detach()
        center = torch.cat((site["kinematic_a"], site["kinematic_b"]), dim=0).mean(dim=0)
        points = center.unsqueeze(0) + offsets
        fractional = points @ orth_to_frac.T
        site["sampling_center"] = center
        site["sampling_radius"] = radius
        site["sampling_spacing"] = spacing
        site["sampling_grid_size"] = len(axis)
        site["map_by_blur"] = {}
        for blur in blur_levels:
            sigma = blur / 2.354820045
            blur_factor = torch.exp(-2.0 * math.pi ** 2 * sigma ** 2 * reciprocal_norm2[valid])
            coefficients = site["map_coefficients"][valid] * blur_factor
            chunks = []
            for start in range(0, len(fractional), 256):
                selected = fractional[start:start + 256]
                kernel = torch.exp(-1j * 2.0 * torch.pi * (hkl[valid] @ selected.T))
                chunks.append(2.0 * torch.real(coefficients @ kernel) / int(valid.sum()))
            density = torch.cat(chunks)
            density = (density - density.mean()) / density.std().clamp_min(1e-8)
            site["map_by_blur"][blur] = density.reshape(len(axis), len(axis), len(axis)).detach()

    # Symmetry contacts for both optimization regularization and the endpoint audit.
    cell = structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
    all_atoms = [atom for chain in structure[0] for residue in chain for atom in residue]
    for site in sites:
        symmetry_xyz = []
        center_np = site["sampling_center"].cpu().numpy()
        for operation_index, operation in enumerate(spacegroup.operations()):
            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    for tz in (-1, 0, 1):
                        if operation_index == 0 and tx == ty == tz == 0:
                            continue
                        for atom in all_atoms:
                            transformed = operation.apply_to_xyz(cell.fractionalize(atom.pos).tolist())
                            position = cell.orthogonalize(gemmi.Fractional(
                                transformed[0] + tx, transformed[1] + ty, transformed[2] + tz
                            ))
                            xyz = np.asarray(position.tolist())
                            if np.linalg.norm(xyz - center_np) <= 8.0:
                                symmetry_xyz.append(xyz)
        site["symmetry_environment"] = torch.tensor(
            np.asarray(symmetry_xyz), dtype=torch.float32, device=device
        ) if symmetry_xyz else torch.empty((0, 3), dtype=torch.float32, device=device)

    def trilinear_sample(grid: torch.Tensor, coordinates: torch.Tensor, site: dict):
        origin = site["sampling_center"] - site["sampling_radius"]
        continuous = (coordinates - origin) / site["sampling_spacing"]
        maximum = site["sampling_grid_size"] - 1.0001
        below, above = torch.relu(-continuous), torch.relu(continuous - maximum)
        outside = (below.square() + above.square()).sum(dim=1)
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
                    values = values + weight * grid[index[:, 0], index[:, 1], index[:, 2]]
        return values, outside

    def canonical_centers(resname: str, chi_index: int) -> list[float]:
        if resname == "ARG" and chi_index == 3:
            return [0.0, math.pi, -math.pi]
        if resname == "ASP" and chi_index == 1:
            return [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi]
        if resname == "MET" and chi_index == 2:
            return [-math.pi / 2, math.pi / 2, math.pi, -math.pi]
        return [-math.pi / 3, math.pi / 3, math.pi, -math.pi]

    def physical_chi(site: dict, coordinates: torch.Tensor) -> torch.Tensor:
        lookup = dict(site["fixed"])
        lookup.update({name: coordinates[i] for i, name in enumerate(site["names"])})
        return torch.stack([
            wrap_angles(dihedral(*(lookup[name] for name in quartet)) - torch.pi)
            for quartet in chi_specs[site["resname"]]["dihedrals"]
        ])

    def rotamer_penalty(site: dict, coordinates: torch.Tensor) -> torch.Tensor:
        chis = physical_chi(site, coordinates)
        terms = []
        for index, value in enumerate(chis):
            centers = torch.tensor(
                canonical_centers(site["resname"], index), dtype=value.dtype, device=device
            )
            terms.append((1.0 - torch.cos(value - centers)).min())
        return torch.stack(terms).mean()

    def score_ensemble(site: dict, all_chi: torch.Tensor, occupancy_logits: torch.Tensor,
                       blur: float, physics_scale: float):
        occupancies = torch.softmax(occupancy_logits, dim=0)
        coordinates = [coords_from_chi(site, wrap_angles(row)) for row in all_chi]
        sampled, outside_terms = [], []
        for xyz in coordinates:
            values, outside = trilinear_sample(site["map_by_blur"][blur], xyz, site)
            sampled.append(values)
            outside_terms.append(outside.mean())
        sampled_tensor = torch.stack(sampled)
        atom_weights = site["atom_weights"] / site["atom_weights"].sum().clamp_min(1e-8)
        linear_score = (
            occupancies[:, None] * atom_weights[None, :] * sampled_tensor
        ).sum()

        # Analytic ||rho_model||^2.  Combined with the sampled cross-term and
        # analytically optimized global map scale, this is a renderer-free
        # least-squares coverage objective rather than a linear peak picker.
        flat_xyz = torch.cat(coordinates, dim=0)
        flat_weights = (occupancies[:, None] * atom_weights[None, :]).reshape(-1)
        atom_sigma2 = site["bfactors"] / (8.0 * math.pi ** 2)
        atom_sigma2 = atom_sigma2 + (blur / 2.354820045) ** 2
        flat_sigma2 = atom_sigma2.repeat(config["K"]).clamp_min(0.04)
        pair_sigma2 = flat_sigma2[:, None] + flat_sigma2[None, :]
        distance2 = torch.cdist(flat_xyz, flat_xyz).square()
        overlap = (2.0 * math.pi * pair_sigma2).pow(-1.5) * torch.exp(
            -distance2 / (2.0 * pair_sigma2)
        )
        self_overlap = (
            flat_weights[:, None] * flat_weights[None, :] * overlap
        ).sum().clamp_min(1e-8)
        density_loss = -linear_score.clamp_min(0.0).square() / self_overlap

        if physics_scale > 0:
            vdw, rot, sym = [], [], []
            for xyz in coordinates:
                direct_distance = torch.cdist(xyz, site["direct_environment"])
                vdw.append(
                    torch.relu(config["vdw_threshold"] - direct_distance).square().sum()
                    / len(xyz)
                )
                rot.append(rotamer_penalty(site, xyz))
                if site["symmetry_environment"].numel():
                    sym_distance = torch.cdist(xyz, site["symmetry_environment"])
                    sym.append(
                        torch.relu(config["clash_threshold"] - sym_distance).square().sum()
                        / len(xyz)
                    )
                else:
                    sym.append(torch.zeros((), device=device))
            vdw_term = (occupancies * torch.stack(vdw)).sum()
            rot_term = (occupancies * torch.stack(rot)).sum()
            sym_term = (occupancies * torch.stack(sym)).sum()
        else:
            vdw_term = torch.zeros((), device=device)
            rot_term = torch.zeros((), device=device)
            sym_term = torch.zeros((), device=device)
        physics = (
            config["lambda_vdw"] * vdw_term
            + config["lambda_rot"] * rot_term
            + config["lambda_clash"] * sym_term
        )
        outside = torch.stack(outside_terms).mean()
        loss = density_loss + physics_scale * physics + config["outside_weight"] * outside
        return loss, {
            "density_loss": density_loss,
            "linear_score": linear_score,
            "self_overlap": self_overlap,
            "physics": physics,
            "vdw": vdw_term,
            "rot": rot_term,
            "sym": sym_term,
            "outside": outside,
            "occupancies": occupancies,
            "coordinates": coordinates,
        }

    # Calibration demonstrates both map signal and ensemble identifiability.
    calibration_rows = []
    calibrated_sites = []
    zero_logits = torch.zeros(config["K"], device=device)
    for site_index, site in enumerate(sites):
        occ_a, occ_b = site["target_occ_a"], site["target_occ_b"]
        ab_logits = torch.log(torch.tensor(
            [occ_a / 2, occ_a / 2, occ_b / 2, occ_b / 2], device=device
        ).clamp_min(1e-6))
        chi_a = torch.zeros(site["n_chi"], device=device)
        chi_b = site["true_delta"]
        ensembles = {
            "A_only": torch.stack([chi_a] * config["K"]),
            "B_only": torch.stack([chi_b] * config["K"]),
            "A_plus_B": torch.stack([chi_a, chi_a, chi_b, chi_b]),
        }
        values = {}
        with torch.no_grad():
            for label, chis in ensembles.items():
                logits = ab_logits if label == "A_plus_B" else zero_logits
                loss, parts = score_ensemble(site, chis, logits, 0.0, 0.0)
                values[label] = {
                    "loss": float(loss.cpu()),
                    "linear": float(parts["linear_score"].cpu()),
                    "self_overlap": float(parts["self_overlap"].cpu()),
                }
            random_scores = []
            generator = torch.Generator(device=device).manual_seed(config["seed"] + site_index)
            for _ in range(config["calibration_random_ensembles"]):
                chis = torch.randn(
                    (config["K"], site["n_chi"]), generator=generator, device=device
                )
                random_scores.append(float(score_ensemble(
                    site, chis, zero_logits, 0.0, 0.0
                )[1]["linear_score"].cpu()))
        linear_identity_error = abs(
            values["A_plus_B"]["linear"]
            - (occ_a * values["A_only"]["linear"] + occ_b * values["B_only"]["linear"])
        )
        identifiable = (
            values["A_plus_B"]["loss"] < values["A_only"]["loss"]
            and values["A_plus_B"]["loss"] < values["B_only"]["loss"]
        )
        signal = min(values["A_only"]["linear"], values["B_only"]["linear"]) > float(
            np.median(random_scores)
        )
        if identifiable and signal:
            calibrated_sites.append(site["key"])
        calibration_rows.append({
            "map_variant": config["map_variant"],
            "site": site["key"],
            "target_A": occ_a,
            "target_B": occ_b,
            "linear_A": values["A_only"]["linear"],
            "linear_B": values["B_only"]["linear"],
            "linear_AB": values["A_plus_B"]["linear"],
            "linear_identity_error": linear_identity_error,
            "random_linear_median": float(np.median(random_scores)),
            "coverage_loss_A": values["A_only"]["loss"],
            "coverage_loss_B": values["B_only"]["loss"],
            "coverage_loss_AB": values["A_plus_B"]["loss"],
            "both_states_have_signal": signal,
            "ensemble_identifiable": identifiable,
            "calibration_pass": identifiable and signal,
            "kinematic_to_deposited_B_rmsd": float(torch.sqrt(torch.mean(
                (site["kinematic_b"] - site["deposited_b"]).square()
            )).cpu()),
        })
    write_csv(out / "calibration.csv", calibration_rows)
    atomic_json(out / "run_config.json", config)
    mark("calibration", "complete", calibrated_sites=calibrated_sites)
    if len(calibrated_sites) < config["minimum_calibrated_sites"]:
        mark("pipeline", "calibration_blocked", calibrated_sites=calibrated_sites)
        return manifest()
    if config["calibration_only"]:
        mark("pipeline", "complete", calibration_only=True, calibrated_sites=calibrated_sites)
        return manifest()

    def evaluate(occupancies, rmsd_a, rmsd_b, site):
        assignments = []
        for occupancy, distance_a, distance_b in zip(occupancies, rmsd_a, rmsd_b):
            if occupancy <= config["nontrivial_occupancy"]:
                assignments.append("inactive")
            elif distance_a < 1.0 and distance_a <= distance_b:
                assignments.append("A")
            elif distance_b < 1.0:
                assignments.append("B")
            else:
                assignments.append("other")
        predicted_a = float(sum(
            occupancy for occupancy, label in zip(occupancies, assignments) if label == "A"
        ))
        predicted_b = float(sum(
            occupancy for occupancy, label in zip(occupancies, assignments) if label == "B"
        ))
        found_a = any(
            occupancy > 0.1 and label == "A" for occupancy, label in zip(occupancies, assignments)
        )
        found_b = any(
            occupancy > 0.1 and label == "B" for occupancy, label in zip(occupancies, assignments)
        )
        accurate = (
            abs(predicted_a - site["target_occ_a"]) <= config["occupancy_tolerance"]
            and abs(predicted_b - site["target_occ_b"]) <= config["occupancy_tolerance"]
        )
        return assignments, predicted_a, predicted_b, found_a, found_b, accurate

    def endpoint_audit(site, occupancies, coordinates, assignments):
        active = [i for i, value in enumerate(occupancies) if value > config["nontrivial_occupancy"]]
        direct_min, symmetry_min, canonical = [], [], []
        deviations = []
        for index in active:
            xyz = coordinates[index]
            direct_min.append(float(torch.cdist(xyz, site["direct_environment"]).min().cpu()))
            if site["symmetry_environment"].numel():
                symmetry_min.append(float(torch.cdist(xyz, site["symmetry_environment"]).min().cpu()))
            chis = physical_chi(site, xyz)
            conformer_deviations = []
            for chi_index, value in enumerate(chis):
                centers = torch.tensor(
                    canonical_centers(site["resname"], chi_index), device=device
                )
                conformer_deviations.append(float(torch.rad2deg(
                    torch.acos(torch.cos(value - centers).max().clamp(-1.0, 1.0))
                ).cpu()))
            deviations.append(max(conformer_deviations))
            canonical.append(max(conformer_deviations) <= 40.0)
        return {
            "active_direct_min_distance": min(direct_min) if direct_min else float("nan"),
            "active_symmetry_min_distance": min(symmetry_min) if symmetry_min else float("nan"),
            "active_all_rotamer_canonical": all(canonical) if canonical else False,
            "active_max_rotamer_deviation_deg": max(deviations) if deviations else float("nan"),
            "active_assignment_labels": ";".join(assignments[i] for i in active),
        }

    def write_overlay(site, row, path):
        occupancies = [float(value) for value in row["occupancies"].split(";")]
        chi_rows = [[float(value) for value in values.split(";")]
                    for values in row["final_chi_radians"].split("|")]
        lines = [f"REMARK experimental K=4 ensemble for {site['key']}"]
        serial = 1
        for k, (occupancy, chi_values) in enumerate(zip(occupancies, chi_rows)):
            xyz = coords_from_chi(
                site, torch.tensor(chi_values, dtype=torch.float32, device=device)
            ).detach().cpu().numpy()
            for atom_index, (name, position) in enumerate(zip(site["names"], xyz)):
                element = "S" if name.startswith("S") else name[0]
                lines.append(
                    f"ATOM  {serial:5d} {name:>4s}{chr(65+k)}{site['resname']:>3s} {site['chain']}"
                    f"{site['number']:4d}    {position[0]:8.3f}{position[1]:8.3f}{position[2]:8.3f}"
                    f"{occupancy:6.2f}{float(site['bfactors'][atom_index]):6.2f}          {element:>2s}"
                )
                serial += 1
        path.write_text("\n".join(lines + ["TER", "END"]) + "\n")

    conditions = (("density_only", 0.0), ("soft_physics", config["lambda_physics"]))
    if config["condition"] != "all":
        conditions = tuple(value for value in conditions if value[0] == config["condition"])
        if not conditions:
            raise ValueError(f"unknown condition filter: {config['condition']}")
    optimization_sites = sites
    if config["site_key"]:
        optimization_sites = [site for site in sites if site["key"] == config["site_key"]]
        if not optimization_sites:
            raise ValueError(f"unknown site filter: {config['site_key']}")
    optimization_site_keys = {site["key"] for site in optimization_sites}
    all_summaries, all_endpoints = [], []
    for condition, physics_scale in conditions:
        for site_index, site in enumerate(sites):
            if site["key"] not in optimization_site_keys:
                continue
            stage = f"{condition}::{site['key']}"
            site_dir = per_site / site["key"]
            site_dir.mkdir(parents=True, exist_ok=True)
            result_path = site_dir / f"{condition}_recovery.csv"
            trajectory_path = site_dir / f"{condition}_trajectories.npz"
            if done(stage) and result_path.exists():
                rows = list(csv.DictReader(result_path.open()))
            else:
                partial_path = site_dir / f"{condition}_partial.csv"
                partial_trajectory_path = site_dir / f"{condition}_partial_trajectories.npz"
                rows, trajectories = [], []
                if (
                    not config["force"]
                    and partial_path.exists()
                    and partial_trajectory_path.exists()
                ):
                    rows = list(csv.DictReader(partial_path.open()))
                    saved = np.load(partial_trajectory_path)
                    trajectories = [
                        {key: saved[key][index] for key in saved.files}
                        for index in range(len(rows))
                    ]
                for start in range(len(rows), config["n_starts"]):
                    generator = torch.Generator(device=device).manual_seed(
                        config["seed"] + 100_000 * site_index + 10_000_000 * int(physics_scale > 0) + start
                    )
                    all_chi = torch.randn(
                        (config["K"], site["n_chi"]), generator=generator, device=device
                    ).requires_grad_(True)
                    logits = torch.zeros(config["K"], device=device, requires_grad=True)
                    if site["n_chi"] >= 4:
                        schedule = ((4.0, 1.0, 100), (2.0, 0.1, 100), (0.0, 0.01, 100))
                    else:
                        schedule = ((0.0, 1.0, 300),)
                    paths = {key: [] for key in (
                        "loss", "density_loss", "linear_score", "self_overlap", "physics",
                        "occupancies", "rmsd_to_A", "rmsd_to_B", "blur"
                    )}
                    for blur, learning_rate, steps in schedule:
                        optimizer = torch.optim.Adam([all_chi, logits], lr=learning_rate)
                        for _ in range(steps):
                            optimizer.zero_grad(set_to_none=True)
                            loss, parts = score_ensemble(
                                site, all_chi, logits, blur, physics_scale
                            )
                            loss.backward()
                            optimizer.step()
                            with torch.no_grad():
                                all_chi.copy_(wrap_angles(all_chi))
                                _loss, current = score_ensemble(
                                    site, all_chi, logits, blur, physics_scale
                                )
                                paths["loss"].append(float(_loss.cpu()))
                                for key in ("density_loss", "linear_score", "self_overlap", "physics"):
                                    paths[key].append(float(current[key].cpu()))
                                occ = current["occupancies"].cpu().numpy()
                                paths["occupancies"].append(occ)
                                paths["rmsd_to_A"].append([
                                    float(torch.sqrt(torch.mean((xyz - site["kinematic_a"]).square())).cpu())
                                    for xyz in current["coordinates"]
                                ])
                                paths["rmsd_to_B"].append([
                                    float(torch.sqrt(torch.mean((xyz - site["kinematic_b"]).square())).cpu())
                                    for xyz in current["coordinates"]
                                ])
                                paths["blur"].append(blur)
                    final_occ = np.asarray(paths["occupancies"][-1])
                    final_a = np.asarray(paths["rmsd_to_A"][-1])
                    final_b = np.asarray(paths["rmsd_to_B"][-1])
                    final_coords = score_ensemble(site, all_chi, logits, 0.0, physics_scale)[1]["coordinates"]
                    assignment, predicted_a, predicted_b, found_a, found_b, accurate = evaluate(
                        final_occ, final_a, final_b, site
                    )
                    audit = endpoint_audit(site, final_occ, final_coords, assignment)
                    row = {
                        "condition": condition, "site": site["key"], "start": start,
                        "final_loss": paths["loss"][-1], "best_loss": min(paths["loss"]),
                        "occupancies": ";".join(f"{x:.8g}" for x in final_occ),
                        "rmsd_to_A": ";".join(f"{x:.8g}" for x in final_a),
                        "rmsd_to_B": ";".join(f"{x:.8g}" for x in final_b),
                        "assignments": ";".join(assignment),
                        "target_A_occupancy": site["target_occ_a"],
                        "target_B_occupancy": site["target_occ_b"],
                        "predicted_A_occupancy": predicted_a,
                        "predicted_B_occupancy": predicted_b,
                        "found_A": found_a, "found_B": found_b,
                        "occupancy_accurate": accurate,
                        "ensemble_success": found_a and found_b and accurate,
                        "active_conformers": int((final_occ > config["nontrivial_occupancy"]).sum()),
                        **audit,
                        "final_chi_radians": "|".join(
                            ";".join(f"{value:.8g}" for value in values)
                            for values in all_chi.detach().cpu().numpy()
                        ),
                    }
                    rows.append(row)
                    trajectories.append({key: np.asarray(value, dtype=np.float32)
                                         for key, value in paths.items()})
                    if (start + 1) % config["checkpoint_starts"] == 0:
                        write_csv(partial_path, rows)
                        np.savez_compressed(
                            partial_trajectory_path,
                            **{key: np.stack([trajectory[key] for trajectory in trajectories])
                               for key in trajectories[0]},
                        )
                        mark(stage, "running", completed_starts=start + 1)
                write_csv(result_path, rows)
                np.savez_compressed(
                    trajectory_path,
                    **{key: np.stack([trajectory[key] for trajectory in trajectories])
                       for key in trajectories[0]},
                )
                mark(stage, "complete", completed_starts=len(rows))
            normalized = [{key: value for key, value in row.items()} for row in rows]
            successes = [row for row in normalized if str(row["ensemble_success"]) == "True"]
            both = [row for row in normalized
                    if str(row["found_A"]) == "True" and str(row["found_B"]) == "True"]
            summary = {
                "condition": condition, "site": site["key"], "starts": len(rows),
                "both_found": len(both), "ensemble_success": len(successes),
                "mean_predicted_A": float(np.mean([float(row["predicted_A_occupancy"]) for row in rows])),
                "mean_predicted_B": float(np.mean([float(row["predicted_B_occupancy"]) for row in rows])),
                "mean_active_conformers": float(np.mean([float(row["active_conformers"]) for row in rows])),
                "sub2A_direct_clash_endpoints": sum(float(row["active_direct_min_distance"]) < 2.0 for row in rows),
                "sub2A_symmetry_clash_endpoints": sum(
                    math.isfinite(float(row["active_symmetry_min_distance"]))
                    and float(row["active_symmetry_min_distance"]) < 2.0 for row in rows
                ),
                "all_active_rotamers_canonical": sum(
                    str(row["active_all_rotamer_canonical"]) == "True" for row in rows
                ),
            }
            all_summaries.append(summary)
            all_endpoints.extend(rows)
            best_pool = successes or both or rows
            best = min(best_pool, key=lambda row: float(row["final_loss"]))
            write_overlay(site, best, overlays / f"{condition}_{site['key']}_best.pdb")
            write_csv(out / "aggregate_summary.csv", all_summaries)
            commit()

    # Optimization shards intentionally stop here.  Their collision-free
    # endpoint/trajectory artifacts are merged locally before a single tmol
    # audit and final report, avoiding duplicated scoring and incomplete plots.
    if config["shard_mode"]:
        mark(
            "pipeline", "complete", shard_mode=True,
            site_keys=[site["key"] for site in optimization_sites],
            conditions=[condition for condition, _scale in conditions],
            calibrated_sites=calibrated_sites,
        )
        return manifest()

    synthetic = {row["site"]: row for row in csv.DictReader(
        Path("/reference/synthetic_summary.csv").open()
    )}
    comparison = []
    for row in all_summaries:
        reference = synthetic[row["site"]]
        comparison.append({
            "condition": row["condition"], "site": row["site"],
            "synthetic_both_found": reference["both_found"],
            "experimental_both_found": row["both_found"],
            "synthetic_ensemble_success": reference["ensemble_success"],
            "experimental_ensemble_success": row["ensemble_success"],
            "synthetic_mean_A": reference["mean_predicted_A_occupancy"],
            "experimental_mean_A": row["mean_predicted_A"],
            "synthetic_mean_B": reference["mean_predicted_B_occupancy"],
            "experimental_mean_B": row["mean_predicted_B"],
        })
    write_csv(out / "synthetic_vs_experimental_comparison.csv", comparison)

    # Independent reparse-based tmol audit.  Reparsing each heavy-atom
    # conformer rebuilds hydrogens at the candidate coordinates and avoids the
    # orphaned-hydrogen artifact identified in the Probe 4b audit.
    if config["run_tmol_audit"]:
        import tmol

        audit_rows = []
        for site in sites:
            base_path = Path(f"/audit/base_chain_{site['chain']}.pdb")
            base_lines = base_path.read_text().splitlines()
            base_pose = tmol.pose_stack_from_pdb(str(base_path), device=device)
            scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(
                base_pose
            )
            candidate_path = Path(f"/tmp/{site['key']}_tmol_candidate.pdb")
            energy_cache = {}

            def tmol_energy(coordinates) -> float:
                cache_key = tuple(
                    round(float(value), 3) for atom in coordinates for value in atom
                )
                if cache_key in energy_cache:
                    return energy_cache[cache_key]
                replacements = dict(zip(site["names"], coordinates))
                candidate_lines = []
                for line in base_lines:
                    if (
                        line.startswith("ATOM")
                        and line[21].strip() == site["chain"]
                        and int(line[22:26]) == site["number"]
                        and line[12:16].strip() in replacements
                    ):
                        x, y, z = replacements[line[12:16].strip()]
                        line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
                    candidate_lines.append(line)
                candidate_path.write_text("\n".join(candidate_lines) + "\n")
                pose = tmol.pose_stack_from_pdb(str(candidate_path), device=device)
                value = float(scorer(pose.coords).sum().detach().cpu())
                energy_cache[cache_key] = value
                return value

            energy_a = tmol_energy(site["kinematic_a"].cpu().numpy())
            energy_b = tmol_energy(site["kinematic_b"].cpu().numpy())
            site_rows = [row for row in all_endpoints if row["site"] == site["key"]]
            for endpoint_index, row in enumerate(site_rows):
                occupancies = [float(value) for value in row["occupancies"].split(";")]
                chi_rows = [[float(value) for value in values.split(";")]
                            for values in row["final_chi_radians"].split("|")]
                for slot, (occupancy, chi_values) in enumerate(zip(occupancies, chi_rows)):
                    if occupancy <= config["nontrivial_occupancy"]:
                        continue
                    coordinates = coords_from_chi(
                        site, torch.tensor(chi_values, dtype=torch.float32, device=device)
                    ).detach().cpu().numpy()
                    audit_rows.append({
                        "condition": row["condition"], "site": site["key"],
                        "start": row["start"], "slot": slot, "occupancy": occupancy,
                        "tmol_energy": tmol_energy(coordinates),
                        "tmol_A": energy_a, "tmol_B": energy_b,
                    })
                if (endpoint_index + 1) % config["tmol_checkpoint_endpoints"] == 0:
                    write_csv(out / "physical_audit_tmol.csv", audit_rows)
                    mark(
                        f"tmol::{site['key']}", "running",
                        completed_endpoints=endpoint_index + 1,
                    )
            write_csv(per_site / site["key"] / "physical_audit_tmol.csv", [
                row for row in audit_rows if row["site"] == site["key"]
            ])
            mark(f"tmol::{site['key']}", "complete", completed_endpoints=len(site_rows))
        write_csv(out / "physical_audit_tmol.csv", audit_rows)
        atomic_json(out / "tmol_audit_manifest.json", {
            "status": "complete", "tmol_version": tmol.__version__,
            "conformers_scored": len(audit_rows),
        })
        commit()

    plt.figure(figsize=(7, 6))
    for condition, marker in (("density_only", "o"), ("soft_physics", "s")):
        selected = [row for row in all_summaries if row["condition"] == condition]
        true_values, predicted = [], []
        for row in selected:
            site = next(value for value in sites if value["key"] == row["site"])
            true_values.extend([site["target_occ_a"], site["target_occ_b"]])
            predicted.extend([row["mean_predicted_A"], row["mean_predicted_B"]])
        plt.scatter(true_values, predicted, label=condition, marker=marker, s=60)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("deposited occupancy")
    plt.ylabel("mean predicted occupancy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "occupancy_scatter.png", dpi=180)
    plt.close()

    labels = [site["key"] for site in sites]
    x = np.arange(len(labels))
    width = 0.25
    synthetic_rates = [int(synthetic[label]["both_found"]) for label in labels]
    density_rates = [next(row["both_found"] for row in all_summaries
                          if row["site"] == label and row["condition"] == "density_only")
                     for label in labels]
    physics_rates = [next(row["both_found"] for row in all_summaries
                          if row["site"] == label and row["condition"] == "soft_physics")
                     for label in labels]
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, synthetic_rates, width, label="synthetic")
    plt.bar(x, density_rates, width, label="experimental")
    plt.bar(x + width, physics_rates, width, label="experimental + physics")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("starts finding both A and B (of 50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "recovery_comparison.png", dpi=180)
    plt.close()
    commit()
    mark("pipeline", "complete", calibrated_sites=calibrated_sites)
    return manifest()


@APP.local_entrypoint()
def main(
    run_name: str = "experimental_multi_conformer_2o1k",
    n_starts: int = 50,
    map_variant: str = "standard_2mfo_dfc",
    site_key: str = "",
    condition: str = "all",
    shard_mode: bool = False,
    calibration_only: bool = False,
    force: bool = False,
):
    config = {
        "run_name": run_name,
        "n_starts": n_starts,
        "map_variant": map_variant,
        "site_key": site_key,
        "condition": condition,
        "shard_mode": shard_mode,
        "K": 4,
        "seed": 41,
        "map_radius": 6.0,
        "map_spacing": 0.5,
        "nontrivial_occupancy": 0.05,
        "occupancy_tolerance": 0.20,
        "checkpoint_starts": 5,
        "calibration_random_ensembles": 16,
        "minimum_calibrated_sites": 2,
        "kick_models": 20,
        "kick_sigma_angstrom": 0.25,
        "lambda_physics": 0.01,
        "lambda_vdw": 1.0,
        "lambda_rot": 0.5,
        "lambda_clash": 5.0,
        "vdw_threshold": 2.0,
        "clash_threshold": 2.0,
        "outside_weight": 2.0,
        "run_tmol_audit": True,
        "tmol_checkpoint_endpoints": 10,
        "calibration_only": calibration_only,
        "force": force,
    }
    call = run_experiment.spawn(config)
    print({"status": "submitted", "function_call_id": call.object_id, "run_name": run_name})
