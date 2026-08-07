"""Probe 4: learn a navigable sidechain energy for 2O1K on a Modal GPU.

Run the complete pipeline detached so it survives terminal/client disconnects:

    UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run --detach probe4_modal.py

Every stage and every periodic training checkpoint is committed to the named
Modal Volume. Re-running the same command resumes the checkpoint and skips
completed evaluation stages unless ``--force`` is supplied.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).parent
APP = modal.App("probe4-learned-energy-2o1k")
RESULTS_VOLUME = modal.Volume.from_name("qfit-probe4-results", create_if_missing=True)
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.40/"
    "tmol-0.1.40%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl"
)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .run_commands(f"pip install 'tmol @ {TMOL_WHEEL}'")
    .pip_install(
        "SFcalculator-torch==0.3.3",
        "gemmi==0.6.7",
        "matplotlib==3.9.4",
        "numpy==1.26.4",
    )
    .add_local_file(ROOT / "probe4_core.py", remote_path="/root/probe4_core.py", copy=True)
    .add_local_dir(ROOT / "data", remote_path="/data", copy=True)
    .add_local_dir(
        ROOT / "probe4b_results" / "endpoint_audit" / "visualization",
        remote_path="/audit_visualization",
        copy=True,
    )
)


@APP.function(
    image=IMAGE,
    gpu="L4",
    timeout=86_400,
    volumes={"/outputs": RESULTS_VOLUME},
)
def pipeline(config: dict) -> dict:
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
    import tmol
    from SFC_Torch import SFcalculator

    from probe4_core import (
        LearnedEnergy,
        angular_rmsd,
        assert_outer_gradient,
        chi_features,
        circular_error,
        dihedral,
        first_order_refine,
        grid_offsets,
        normalized_rfactor,
        residue_onehot,
        seed_everything,
        torsion_to_coords,
        rotate_points,
        wrap_angles,
    )

    device = torch.device("cuda")
    out = Path("/outputs") / config["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "model_checkpoint.pt"
    manifest_path = out / "stage_manifest.json"
    pdb_path, mtz_path = "/data/2O1K.pdb", "/data/2O1K.mtz"

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

    def atomic_json(path: Path, value) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temp, path)

    def atomic_torch_save(path: Path, value) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(value, temp)
        os.replace(temp, path)

    def commit() -> None:
        RESULTS_VOLUME.commit()

    def manifest() -> dict:
        if manifest_path.exists():
            return json.loads(manifest_path.read_text())
        return {"protein": "2O1K", "stages": {}, "created_at": time.time()}

    def stage_mark(name: str, status: str, **details) -> None:
        value = manifest()
        value["stages"][name] = {"status": status, "updated_at": time.time(), **details}
        atomic_json(manifest_path, value)
        commit()

    def stage_done(name: str) -> bool:
        return (not config["force"] and manifest().get("stages", {}).get(name, {}).get("status") == "complete")

    seed_everything(config["seed"])
    calc = SFcalculator(pdb_path, mtz_path, device=device)
    fobs = calc.Fo.detach()
    deposited_fcalc = calc.calc_fprotein(Return=True).detach()
    base_positions = calc.atom_pos_orth.detach()
    # SFcalculator updates its internal atom tensors when overrides are passed.
    # Keep an immutable copy so a fixed-atom precomputation cannot silently
    # zero the sidechain in later candidate calculations.
    base_occupancies = calc.atom_occ.detach().clone()
    hkl = torch.as_tensor(calc.HKL_array, dtype=torch.float32, device=device)
    orth_to_frac = calc.orth2frac_tensor.detach()

    valid = torch.isfinite(fobs) & (fobs > 0)
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    random_values = torch.rand(fobs.shape, generator=generator, device=device)
    test_mask = valid & (random_values < config["reflection_holdout"])
    train_mask = valid & ~test_mask
    if test_mask.sum() == 0:
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
        raise RuntimeError("gemmi and SFcalculator disagree on PDB atom ordering")
    base_bfactors = torch.tensor(
        [atom.b_iso for _chain, _residue, atom in atoms_with_context],
        dtype=torch.float32,
        device=device,
    )

    def alt_atom_map(residue, alt: str) -> dict:
        result = {}
        for atom in residue:
            atom_alt = atom.altloc if atom.altloc not in ("\x00", " ") else ""
            if atom_alt in ("", alt):
                result[atom.name.strip()] = torch.tensor(atom.pos.tolist(), dtype=torch.float32, device=device)
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
            b_indices = [
                index for index, (candidate_chain, candidate_residue, atom) in enumerate(atoms_with_context)
                if candidate_chain.name == chain.name
                and candidate_residue.seqid.num == residue.seqid.num
                and atom.altloc == "B"
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [atoms_with_context[index][2].name.strip() for index in b_indices]
            if not b_indices or any(name not in map_a or name not in map_b for name in names):
                continue
            spec = chi_specs[residue.name]
            chi_a = torch.stack([dihedral(*(map_a[name] for name in names4)) for names4 in spec["dihedrals"]])
            chi_b = torch.stack([dihedral(*(map_b[name] for name in names4)) for names4 in spec["dihedrals"]])
            delta = wrap_angles(chi_b - chi_a)
            template = torch.stack([map_a[name] for name in names])
            target_b = torch.stack([map_b[name] for name in names])
            fixed = {name: value for name, value in map_a.items() if name not in names}
            plus = torsion_to_coords(template, names, delta, list(spec["rotations"]), fixed)
            minus = torsion_to_coords(template, names, -delta, list(spec["rotations"]), fixed)
            if torch.mean((minus - target_b).square()) < torch.mean((plus - target_b).square()):
                delta = -delta
            sites.append({
                "key": f"{chain.name}_{residue.name}{residue.seqid.num}",
                "chain": chain.name,
                "number": residue.seqid.num,
                "resname": residue.name,
                "n_chi": len(spec["rotations"]),
                "rotations": list(spec["rotations"]),
                "names": names,
                "indices": torch.tensor(b_indices, dtype=torch.long, device=device),
                "template": template,
                "target_b": target_b,
                "fixed": fixed,
                "true_delta": delta,
                "center": map_a["CA"],
            })
    if len(sites) != 5:
        raise RuntimeError(f"expected five 2O1K altloc sites, found {[site['key'] for site in sites]}")

    def coords_from_chi(site: dict, chi: torch.Tensor) -> torch.Tensor:
        return torsion_to_coords(
            site["template"], site["names"], chi, site["rotations"], site["fixed"]
        )

    offsets = grid_offsets(config["density_grid_size"], config["density_spacing"]).to(device)

    def density_patch(center: torch.Tensor, complex_coefficients: torch.Tensor) -> torch.Tensor:
        points = center.unsqueeze(0) + offsets
        fractional = points @ orth_to_frac.T
        phase = 2 * torch.pi * (hkl @ fractional.T)
        # ASU-only coefficients give a proportional local map; normalization
        # removes the omitted Friedel/symmetry multiplicative scale.
        density = 2 * torch.real(
            (complex_coefficients.unsqueeze(1) * torch.exp(-1j * phase)).sum(dim=0)
        ) / complex_coefficients.numel()
        density = (density - density.mean()) / density.std().clamp_min(1e-6)
        return density.float().detach()

    calc_amplitudes = deposited_fcalc.abs()
    sf_scale = (calc_amplitudes[valid] @ fobs[valid]) / calc_amplitudes[valid].square().sum().clamp_min(1e-12)
    phase_unit = deposited_fcalc / calc_amplitudes.clamp_min(1e-8)
    two_fo_fc = (2 * fobs - sf_scale * calc_amplitudes) * phase_unit
    for site in sites:
        site["density"] = density_patch(site["center"], two_fo_fc)
        site["onehot"] = residue_onehot(site["resname"], device=device)

    active_keys = set(config.get("active_sites") or [site["key"] for site in sites])
    sites = [site for site in sites if site["key"] in active_keys]
    if {site["key"] for site in sites} != active_keys:
        raise RuntimeError(f"invalid active sites {sorted(active_keys)}")
    held_out_keys = set(config["held_out_sites"])
    requested_training_keys = set(config.get("training_sites") or [])
    train_sites = [
        site for site in sites
        if site["key"] in requested_training_keys
        or (not requested_training_keys and site["key"] not in held_out_keys)
    ]
    held_out_sites = [site for site in sites if site["key"] in held_out_keys]
    if requested_training_keys and {site["key"] for site in train_sites} != requested_training_keys:
        raise RuntimeError(f"invalid training sites {sorted(requested_training_keys)}")
    if not train_sites or len(held_out_sites) != len(held_out_keys):
        raise RuntimeError(f"invalid held-out sites {sorted(held_out_keys)}")

    # Loss-specific constants are computed once.  Experiments B and C remove
    # the fixed-atom contribution before training so the moving sidechain is no
    # longer a ~0.1% perturbation of the target.
    obs_to_calc_scale = (
        calc_amplitudes[valid] @ fobs[valid]
    ) / fobs[valid].square().sum().clamp_min(1e-12)
    map_coefficients_calc_scale = (
        2 * obs_to_calc_scale * fobs - calc_amplitudes
    ) * phase_unit
    if config["loss_mode"] in {
        "localized_sf", "realspace_local", "complex_target", "kinematic_complex_target",
        "realspace_kinematic",
    }:
        for site in train_sites:
            without_sidechain_occ = base_occupancies.clone()
            without_sidechain_occ[site["indices"]] = 0.0
            f_fixed = calc.calc_fprotein(
                base_positions, atoms_occ_tensor=without_sidechain_occ, Return=True
            ).detach()
            site["f_fixed"] = f_fixed
            deposited_sidechain = (deposited_fcalc - f_fixed).detach()
            if config["loss_mode"] in {"kinematic_complex_target", "realspace_kinematic"}:
                kinematic_positions = base_positions.index_copy(
                    0, site["indices"], coords_from_chi(site, site["true_delta"])
                )
                kinematic_fcalc = calc.calc_fprotein(
                    kinematic_positions, atoms_occ_tensor=base_occupancies, Return=True
                ).detach()
                site["complex_target"] = (kinematic_fcalc - f_fixed).detach()
            else:
                site["complex_target"] = deposited_sidechain
            site["sf_local_normalizer"] = (
                deposited_sidechain[train_mask].abs().square().mean().clamp_min(1e-12)
            )
            fixed_phase = f_fixed / f_fixed.abs().clamp_min(1e-8)
            site["f_residual"] = (
                obs_to_calc_scale * fobs * fixed_phase - f_fixed
            ).detach()
            if config["loss_mode"] == "realspace_local":
                radius, spacing = config["local_density_radius"], config["local_density_spacing"]
                axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
                local_offsets = torch.cartesian_prod(axis, axis, axis)
                local_offsets = local_offsets[torch.linalg.vector_norm(local_offsets, dim=1) <= radius]
                center = torch.cat((site["template"], site["target_b"]), dim=0).mean(dim=0)
                points = center.unsqueeze(0) + local_offsets
                fractional = points @ orth_to_frac.T
                phase = 2 * torch.pi * (hkl[train_mask] @ fractional.T)
                kernel = torch.exp(-1j * phase)
                target_coefficients = map_coefficients_calc_scale[train_mask] - f_fixed[train_mask]
                target_density = 2 * torch.real(target_coefficients @ kernel) / int(train_mask.sum())
                deposited_sidechain_density = 2 * torch.real(
                    deposited_sidechain[train_mask] @ kernel
                ) / int(train_mask.sum())
                site["local_kernel"] = kernel
                site["target_local_density"] = target_density.detach()
                site["local_density_normalizer"] = (
                    deposited_sidechain_density.square().mean().clamp_min(1e-12)
                )
                site["local_grid_points"] = int(points.shape[0])

    def atom_density(
        atom_coordinates: torch.Tensor,
        bfactors: torch.Tensor,
        occupancies: torch.Tensor,
        grid_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        sigma2 = (bfactors / (8.0 * math.pi ** 2)).clamp_min(1e-4)
        differences = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
        distance2 = differences.square().sum(dim=-1)
        normalization = (2.0 * math.pi * sigma2).pow(-1.5)
        return (
            occupancies[None, :]
            * normalization[None, :]
            * torch.exp(-distance2 / (2.0 * sigma2[None, :]))
        ).sum(dim=1)

    if config["loss_mode"] == "realspace_kinematic":
        radius, spacing = config["local_density_radius"], config["local_density_spacing"]
        axis = torch.arange(-radius, radius + spacing / 2, spacing, device=device)
        local_offsets = torch.cartesian_prod(axis, axis, axis)
        local_offsets = local_offsets[
            torch.linalg.vector_norm(local_offsets, dim=1) <= radius
        ]
        for site in train_sites:
            kinematic_b = coords_from_chi(site, site["true_delta"]).detach()
            center = torch.cat((site["template"], kinematic_b), dim=0).mean(dim=0)
            grid_coordinates = center.unsqueeze(0) + local_offsets
            fixed_occupancies = base_occupancies.clone()
            fixed_occupancies[site["indices"]] = 0.0
            with torch.no_grad():
                rho_fixed = atom_density(
                    base_positions, base_bfactors, fixed_occupancies, grid_coordinates
                )
                rho_target = rho_fixed + atom_density(
                    kinematic_b,
                    base_bfactors[site["indices"]],
                    base_occupancies[site["indices"]],
                    grid_coordinates,
                )
            site["gaussian_grid"] = grid_coordinates
            site["gaussian_rho_fixed"] = rho_fixed.detach()
            site["gaussian_rho_target"] = rho_target.detach()

    # Physics regularization is evaluated only at the learned endpoint.  tmol
    # hydrogens are rotated with their parent heavy atoms; replacing heavy atoms
    # in an already hydrogenated pose would create spurious bond/clash energies.
    physics_enabled = any(
        config[name] > 0
        for name in ("lambda_tmol", "lambda_vdw", "lambda_rot", "lambda_clash")
    )
    if physics_enabled:
        tmol_by_chain = {}
        for chain_name in {site["chain"] for site in train_sites}:
            pose = tmol.pose_stack_from_pdb(
                f"/audit_visualization/base_chain_{chain_name}.pdb", device=device
            )
            scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(pose)
            tmol_by_chain[chain_name] = (pose, scorer)

        cell = structure.cell
        spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
        all_structure_atoms = [atom for chain in structure[0] for residue in chain for atom in residue]
        for site in train_sites:
            pose, scorer = tmol_by_chain[site["chain"]]
            residue_index = site["number"] - 108
            block_type = int(pose.block_type_ind[0, residue_index])
            residue_type = pose.packed_block_types.active_block_types[block_type]
            n_atoms = int(pose.packed_block_types.n_atoms[block_type])
            atom_names = [str(residue_type.atom_name(i)).strip() for i in range(n_atoms)]
            offset = int(pose.block_coord_offset[0, residue_index])
            local_xyz = pose.coords[0, offset : offset + n_atoms].detach()
            heavy_local = [i for i, name in enumerate(atom_names) if not name.startswith("H")]
            hydrogen_parent = {}
            for i, name in enumerate(atom_names):
                if name.startswith("H"):
                    distances = torch.linalg.vector_norm(
                        local_xyz[heavy_local] - local_xyz[i], dim=1
                    )
                    hydrogen_parent[i] = heavy_local[int(torch.argmin(distances))]
            rotations = []
            for origin_name, endpoint_name, downstream_names in site["rotations"]:
                downstream_local = [atom_names.index(name) for name in downstream_names]
                downstream_local += [
                    h for h, parent in hydrogen_parent.items() if parent in downstream_local
                ]
                rotations.append({
                    "origin": offset + atom_names.index(origin_name),
                    "endpoint": offset + atom_names.index(endpoint_name),
                    "downstream": torch.tensor(
                        [offset + i for i in downstream_local], dtype=torch.long, device=device
                    ),
                })
            site["tmol_context"] = {
                "base": pose.coords.detach(), "scorer": scorer, "rotations": rotations
            }

            symmetry_xyz = []
            center_np = site["center"].detach().cpu().numpy()
            for operation_index, operation in enumerate(spacegroup.operations()):
                for tx in (-1, 0, 1):
                    for ty in (-1, 0, 1):
                        for tz in (-1, 0, 1):
                            if operation_index == 0 and tx == ty == tz == 0:
                                continue
                            for atom in all_structure_atoms:
                                fractional = cell.fractionalize(atom.pos)
                                transformed = operation.apply_to_xyz(fractional.tolist())
                                transformed = gemmi.Fractional(
                                    transformed[0] + tx, transformed[1] + ty, transformed[2] + tz
                                )
                                orthogonal = cell.orthogonalize(transformed)
                                xyz = np.asarray(orthogonal.tolist())
                                if np.linalg.norm(xyz - center_np) <= 10.0:
                                    symmetry_xyz.append(xyz)
            site["symmetry_xyz"] = torch.tensor(
                np.asarray(symmetry_xyz), dtype=torch.float32, device=device
            )
            direct_indices = [
                index for index, (candidate_chain, candidate_residue, _) in enumerate(atoms_with_context)
                if not (
                    candidate_chain.name == site["chain"]
                    and candidate_residue.seqid.num == site["number"]
                )
            ]
            site["direct_environment_xyz"] = base_positions.index_select(
                0, torch.tensor(direct_indices, dtype=torch.long, device=device)
            ).detach()

    metadata = {
        "protein": "2O1K",
        "sites": [
            {
                "key": site["key"], "residue_type": site["resname"], "n_chi": site["n_chi"],
                "atom_names": site["names"], "true_delta_chi_rad": site["true_delta"].cpu().tolist(),
                "template_to_B_rmsd": float(torch.sqrt(torch.mean((site["template"] - site["target_b"]).square())).cpu()),
                "kinematic_true_to_B_rmsd": float(torch.sqrt(torch.mean((coords_from_chi(site, site["true_delta"]) - site["target_b"]).square())).cpu()),
                "split": "held_out" if site["key"] in held_out_keys else "train",
            }
            for site in sites
        ],
        "train_reflections": int(train_mask.sum().cpu()),
        "held_out_reflections": int(test_mask.sum().cpu()),
        "config": config,
    }
    atomic_json(out / "run_config.json", metadata)
    np.save(out / "reflection_train_mask.npy", train_mask.cpu().numpy())
    np.save(out / "reflection_test_mask.npy", test_mask.cpu().numpy())
    commit()

    model = LearnedEnergy(
        density_feat_dim=offsets.shape[0], hidden=config["hidden"], n_layers=config["layers"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["outer_lr"])
    start_step = 0
    if checkpoint_path.exists() and not config["force"]:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        torch.set_rng_state(saved["torch_rng_cpu"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved["torch_rng_cuda"]])
        random.setstate(saved["python_rng"])

    log_path = out / "training_log.csv"
    if config["force"] or not log_path.exists():
        energy_columns = ",".join(f"energy_{index}" for index in range(config["inner_steps"]))
        log_path.write_text(
            f"step,site,loss,loss_cryst,loss_tmol,loss_vdw,loss_rot,loss_clash,loss_mode,"
            f"chi_error_rad,chi_move_rad,{energy_columns}\n"
        )

    def physical_chi(site: dict, candidate: torch.Tensor) -> torch.Tensor:
        lookup = dict(site["fixed"])
        lookup.update({name: candidate[index] for index, name in enumerate(site["names"])})
        spec = chi_specs[site["resname"]]
        # probe4_core's signed torsion convention is offset by pi from the
        # crystallographic chi convention used by rotamer libraries.
        return torch.stack([
            wrap_angles(dihedral(*(lookup[name] for name in quartet)) - torch.pi)
            for quartet in spec["dihedrals"]
        ])

    def canonical_centers(resname: str, chi_index: int) -> list[float]:
        if resname == "ARG" and chi_index == 3:
            return [0.0, math.pi, -math.pi]
        if resname == "ASP" and chi_index == 1:
            return [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi]
        if resname == "MET" and chi_index == 2:
            return [-math.pi / 2, math.pi / 2, math.pi, -math.pi]
        return [-math.pi / 3, math.pi / 3, math.pi, -math.pi]

    def rotamer_prior(site: dict, candidate: torch.Tensor) -> torch.Tensor:
        chis = physical_chi(site, candidate)
        terms = []
        for index, value in enumerate(chis):
            centers = torch.tensor(
                canonical_centers(site["resname"], index), device=device, dtype=value.dtype
            )
            terms.append((1.0 - torch.cos(value - centers)).min())
        return torch.stack(terms).sum()

    def differentiable_tmol(site: dict, chi: torch.Tensor) -> torch.Tensor:
        context = site["tmol_context"]
        coordinates = context["base"]
        for index, rotation in enumerate(context["rotations"]):
            selected = coordinates[0].index_select(0, rotation["downstream"])
            moved = rotate_points(
                selected,
                coordinates[0, rotation["origin"]],
                coordinates[0, rotation["endpoint"]],
                chi[index],
            )
            coordinates = coordinates.index_copy(1, rotation["downstream"], moved.unsqueeze(0))
        return context["scorer"](coordinates).sum()

    def crystallographic_loss(site: dict, chi: torch.Tensor) -> torch.Tensor:
        candidate = coords_from_chi(site, chi)
        final_positions = base_positions.index_copy(0, site["indices"], candidate)
        fcalc = calc.calc_fprotein(
            final_positions, atoms_occ_tensor=base_occupancies, Return=True
        )
        if config["loss_mode"] in {"complex_target", "kinematic_complex_target"}:
            sidechain = fcalc - site["f_fixed"]
            difference = sidechain[train_mask] - site["complex_target"][train_mask]
            normalizer = site["complex_target"][train_mask].abs().square().mean().clamp_min(1e-12)
            return difference.abs().square().mean() / normalizer
        if config["loss_mode"] == "synthetic_fobs":
            residual = (fcalc.abs() - calc_amplitudes) / calc_amplitudes.clamp_min(1.0)
            return residual[train_mask].square().mean()
        if config["loss_mode"] == "localized_sf":
            sidechain = fcalc - site["f_fixed"]
            difference = sidechain[train_mask] - site["f_residual"][train_mask]
            return difference.abs().square().mean() / site["sf_local_normalizer"]
        if config["loss_mode"] == "realspace_local":
            sidechain = fcalc[train_mask] - site["f_fixed"][train_mask]
            rho_sidechain = 2 * torch.real(
                sidechain @ site["local_kernel"]
            ) / int(train_mask.sum())
            return (
                (rho_sidechain - site["target_local_density"]).square().mean()
                / site["local_density_normalizer"]
            )
        if config["loss_mode"] == "realspace_kinematic":
            rho_sidechain = atom_density(
                candidate,
                base_bfactors[site["indices"]],
                base_occupancies[site["indices"]],
                site["gaussian_grid"],
            )
            rho_total = site["gaussian_rho_fixed"] + rho_sidechain
            return (rho_total - site["gaussian_rho_target"]).square().sum()
        residual = (fcalc.abs() - fobs) / fobs.clamp_min(1.0)
        return residual[train_mask].square().mean()

    def endpoint_loss(site: dict, chi: torch.Tensor) -> tuple[torch.Tensor, dict]:
        candidate = coords_from_chi(site, chi)
        cryst = crystallographic_loss(site, chi)
        zero = torch.zeros((), device=device)
        tmol_term = differentiable_tmol(site, chi) if config["lambda_tmol"] > 0 else zero
        if config["lambda_vdw"] > 0:
            direct_distances = torch.cdist(
                candidate.unsqueeze(0), site["direct_environment_xyz"].unsqueeze(0)
            ).squeeze(0)
            vdw_term = torch.clamp(
                config["vdw_threshold"] - direct_distances, min=0.0
            ).square().sum()
        else:
            vdw_term = zero
        rot_term = rotamer_prior(site, candidate) if config["lambda_rot"] > 0 else zero
        if config["lambda_clash"] > 0 and site["symmetry_xyz"].numel():
            distances = torch.cdist(candidate.unsqueeze(0), site["symmetry_xyz"].unsqueeze(0)).squeeze(0)
            clash_term = torch.clamp(config["clash_threshold"] - distances, min=0.0).square().sum()
        else:
            clash_term = zero
        total = (
            cryst
            + config["lambda_tmol"] * tmol_term
            + config["lambda_vdw"] * vdw_term
            + config["lambda_rot"] * rot_term
            + config["lambda_clash"] * clash_term
        )
        return total, {
            "cryst": cryst, "tmol": tmol_term, "vdw": vdw_term,
            "rot": rot_term, "clash": clash_term
        }

    # Non-negotiable pre-training gates for Probe 4c.1 and 4c.2.
    if start_step == 0 or config["force"]:
        if config["loss_mode"] in {"kinematic_complex_target", "realspace_kinematic"}:
            assertion_lines = [
                f"Pre-training reachable-target assertions: {config['loss_mode']}"
            ]
            assertion_failures = []
            with torch.no_grad():
                for site in train_sites:
                    chi_a = torch.zeros(site["n_chi"], device=device)
                    loss_a = crystallographic_loss(site, chi_a)
                    loss_b = crystallographic_loss(site, site["true_delta"])
                    residual = float(torch.sqrt(torch.mean(
                        (coords_from_chi(site, site["true_delta"]) - site["target_b"]).square()
                    )).cpu())
                    assertion_lines.append(
                        f"{site['key']}: kinematic_residual_A={residual:.6f}; "
                        f"loss_kinematic_B={float(loss_b.cpu()):.12g}; "
                        f"loss_A={float(loss_a.cpu()):.12g}; PASS"
                    )
                    if float(loss_b.cpu()) >= 1e-6:
                        assertion_failures.append(
                            f"kinematic target is not exact at {site['key']}"
                        )
                    if float(loss_b.cpu()) >= float(loss_a.cpu()):
                        assertion_failures.append(f"kinematic B does not beat A at {site['key']}")
            assertion_lines.append(
                "ALL ASSERTIONS PASSED" if not assertion_failures
                else "ASSERTION FAILURES: " + "; ".join(assertion_failures)
            )
            (out / "pre_training_assertions.txt").write_text("\n".join(assertion_lines) + "\n")
            commit()
            if assertion_failures:
                raise RuntimeError("; ".join(assertion_failures))

        if config["lambda_vdw"] > 0:
            diagnostic_lines = [
                "Probe 4c.2 soft-physics calibration",
                "columns: site candidate cryst vdw sym rot soft total full_tmol_reference",
            ]
            calibration_failures = []
            with torch.no_grad():
                for site in train_sites:
                    values = {}
                    for label, chi in {
                        "A": torch.zeros(site["n_chi"], device=device),
                        "kinematic_B": site["true_delta"],
                    }.items():
                        total, components = endpoint_loss(site, chi)
                        soft = (
                            config["lambda_vdw"] * components["vdw"]
                            + config["lambda_rot"] * components["rot"]
                            + config["lambda_clash"] * components["clash"]
                        )
                        full_tmol_reference = differentiable_tmol(site, chi)
                        values[label] = {
                            "total": float(total.cpu()), "soft": float(soft.cpu()),
                            "cryst": float(components["cryst"].cpu()),
                            "vdw": float(components["vdw"].cpu()),
                            "sym": float(components["clash"].cpu()),
                            "rot": float(components["rot"].cpu()),
                            "tmol": float(full_tmol_reference.cpu()),
                        }
                        value = values[label]
                        diagnostic_lines.append(
                            f"{site['key']} {label} {value['cryst']:.8g} {value['vdw']:.8g} "
                            f"{value['sym']:.8g} {value['rot']:.8g} {value['soft']:.8g} "
                            f"{value['total']:.8g} {value['tmol']:.8g}"
                        )
                    if values["kinematic_B"]["soft"] > values["A"]["soft"] + 5.0:
                        calibration_failures.append(
                            f"soft physics penalizes kinematic B at {site['key']}"
                        )
                    if values["kinematic_B"]["total"] >= values["A"]["total"]:
                        calibration_failures.append(
                            f"composite loss does not prefer kinematic B at {site['key']}"
                        )
            diagnostic_lines.append(
                "ALL CALIBRATION GATES PASSED" if not calibration_failures
                else "CALIBRATION FAILURES: " + "; ".join(calibration_failures)
            )
            (out / "physics_calibration_diagnostic.txt").write_text(
                "\n".join(diagnostic_lines) + "\n"
            )
            commit()
            if calibration_failures:
                raise RuntimeError("; ".join(calibration_failures))

    def save_checkpoint(step: int) -> None:
        atomic_torch_save(checkpoint_path, {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "torch_rng_cpu": torch.get_rng_state(),
            "torch_rng_cuda": torch.cuda.get_rng_state_all(),
            "python_rng": random.getstate(),
        })
        commit()

    if start_step < config["steps"]:
        stage_mark("training", "running", resumed_from=start_step)
        model.train()
        rows = []
        for step in range(start_step, config["steps"]):
            site = random.choice(train_sites)
            initial = torch.randn(site["n_chi"], device=device) * config["initial_chi_sigma"]
            final_chi, energies = first_order_refine(
                model, site["density"], site["onehot"], initial, site["n_chi"],
                config["inner_steps"], config["inner_alpha"], training=True,
            )
            if config["loss_mode"] == "chi_supervised":
                loss = circular_error(final_chi, site["true_delta"]).square().mean()
                components = {
                    "cryst": loss, "tmol": torch.zeros((), device=device),
                    "vdw": torch.zeros((), device=device), "rot": torch.zeros((), device=device),
                    "clash": torch.zeros((), device=device),
                }
            else:
                loss, components = endpoint_loss(site, final_chi)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert_outer_gradient(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            rows.append([
                step + 1, site["key"], float(loss.detach().cpu()),
                float(components["cryst"].detach().cpu()),
                float(components["tmol"].detach().cpu()),
                float(components["vdw"].detach().cpu()),
                float(components["rot"].detach().cpu()),
                float(components["clash"].detach().cpu()),
                config["loss_mode"],
                angular_rmsd(final_chi, site["true_delta"]),
                angular_rmsd(final_chi, initial),
                *[float(value.detach().cpu()) for value in energies],
            ])
            if (step + 1) % config["checkpoint_every"] == 0 or step + 1 == config["steps"]:
                with log_path.open("a", newline="") as handle:
                    csv.writer(handle).writerows(rows)
                rows.clear()
                save_checkpoint(step + 1)
                stage_mark("training", "running", completed_steps=step + 1, latest_loss=float(loss.detach().cpu()))
                print(f"training {step + 1}/{config['steps']} loss={float(loss.detach()):.6g}")
        stage_mark("training", "complete", completed_steps=config["steps"])
    elif not stage_done("training"):
        stage_mark("training", "complete", completed_steps=start_step)

    # Persist the two diagnostics that distinguish a useful crystallographic
    # loss from Probe 4's false positive: endpoint loss and chi error.
    training_rows = list(csv.DictReader(log_path.open()))
    if training_rows:
        loss_values = np.asarray([float(row["loss"]) for row in training_rows])
        chi_values = np.asarray([float(row["chi_error_rad"]) for row in training_rows])
        window = min(100, len(training_rows))
        weights = np.ones(window) / window
        loss_smooth = np.convolve(loss_values, weights, mode="valid")
        chi_smooth = np.convolve(chi_values, weights, mode="valid")
        steps_smooth = np.arange(window, len(training_rows) + 1)
        figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        axes[0].plot(steps_smooth, loss_smooth); axes[0].set_ylabel("loss (100-step mean)")
        axes[1].plot(steps_smooth, chi_smooth); axes[1].set_ylabel("chi error (rad)")
        axes[1].set_xlabel("training step"); figure.tight_layout()
        figure.savefig(out / "training_curves.png", dpi=180); plt.close(figure)
        commit()

    model.eval()

    def infer(site: dict, density: torch.Tensor, starts: int, steps: int) -> list[dict]:
        results = []
        for start in range(starts):
            initial = torch.randn(site["n_chi"], device=device) * config["initial_chi_sigma"]
            chi = initial
            energies, rmsd_b_path, rmsd_a_path = [], [], []
            for trajectory_step in range(steps + 1):
                xyz_step = coords_from_chi(site, wrap_angles(chi)).detach()
                rmsd_a_path.append(float(torch.sqrt(torch.mean((xyz_step - site["template"]).square())).cpu()))
                rmsd_b_path.append(float(torch.sqrt(torch.mean((xyz_step - site["target_b"]).square())).cpu()))
                if trajectory_step == steps:
                    break
                chi = chi.detach().requires_grad_(True)
                energy = model(density, chi_features(chi, site["n_chi"]), site["onehot"])
                gradient = torch.autograd.grad(energy, chi)[0]
                chi = wrap_angles(chi - config["inner_alpha"] * gradient)
                energies.append(float(energy.detach().cpu()))
            final_chi = wrap_angles(chi).detach()
            xyz = coords_from_chi(site, final_chi).detach()
            rmsd_a = float(torch.sqrt(torch.mean((xyz - site["template"]).square())).cpu())
            rmsd_b = float(torch.sqrt(torch.mean((xyz - site["target_b"]).square())).cpu())
            with torch.no_grad():
                score = model(density, chi_features(final_chi, site["n_chi"]), site["onehot"])
            results.append({
                "site": site["key"], "start": start, "rmsd_to_A": rmsd_a, "rmsd_to_B": rmsd_b,
                "endpoint": "B" if rmsd_b < rmsd_a else "A", "energy": float(score.cpu()),
                "initial_chi": initial.cpu().tolist(), "final_chi": final_chi.cpu().tolist(),
                "inner_energies": energies,
                "rmsd_to_A_by_step": rmsd_a_path,
                "rmsd_to_B_by_step": rmsd_b_path,
            })
        return results

    if not stage_done("altloc_test"):
        stage_mark("altloc_test", "running")
        directory = out / "altloc_test"
        directory.mkdir(exist_ok=True)
        records = [record for site in sites for record in infer(site, site["density"], config["eval_starts"], config["eval_steps"])]
        atomic_json(directory / "trajectories.json", records)
        recovery_rows = []
        for site in sites:
            selected = [record for record in records if record["site"] == site["key"]]
            recovery_rows.append({
                "site": site["key"],
                "split": "held_out" if site["key"] in held_out_keys else "train",
                "hits_rmsd_B_lt_0.50": sum(record["rmsd_to_B"] < 0.50 for record in selected),
                "hits_rmsd_B_lt_0.75": sum(record["rmsd_to_B"] < 0.75 for record in selected),
                "mean_rmsd_to_B": float(np.mean([record["rmsd_to_B"] for record in selected])),
                "mean_rmsd_to_A": float(np.mean([record["rmsd_to_A"] for record in selected])),
            })
        with (out / "altloc_recovery.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=recovery_rows[0].keys())
            writer.writeheader(); writer.writerows(recovery_rows)
        rmsd_a = np.asarray([[record["rmsd_to_A"] for record in records if record["site"] == site["key"]] for site in sites])
        rmsd_b = np.asarray([[record["rmsd_to_B"] for record in records if record["site"] == site["key"]] for site in sites])
        np.save(directory / "rmsd_to_A.npy", rmsd_a)
        np.save(directory / "rmsd_to_B.npy", rmsd_b)
        plt.figure(figsize=(6, 5))
        for site in sites:
            selected = [record for record in records if record["site"] == site["key"]]
            plt.scatter([x["rmsd_to_A"] for x in selected], [x["rmsd_to_B"] for x in selected], s=18, label=site["key"], alpha=.75)
        limit = max(rmsd_a.max(), rmsd_b.max()) * 1.05
        plt.plot([0, limit], [0, limit], "k--", linewidth=.8)
        plt.xlabel("endpoint RMSD to altloc A (Å)"); plt.ylabel("endpoint RMSD to altloc B (Å)")
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(directory / "rmsd_scatter.png", dpi=180); plt.close()
        figure, axes = plt.subplots(len(sites), 1, figsize=(8, 12), sharex=True)
        for axis_plot, site in zip(axes, sites):
            selected = [record for record in records if record["site"] == site["key"]]
            for record in selected:
                axis_plot.plot(range(config["eval_steps"] + 1), record["rmsd_to_B_by_step"], alpha=.22, linewidth=.8)
            axis_plot.axhline(.5, color="black", linestyle="--", linewidth=.8)
            axis_plot.set_ylabel(f"{site['key']}\nRMSD-B")
        axes[-1].set_xlabel("inference step")
        figure.tight_layout(); figure.savefig(out / "trajectory_rmsd_to_B.png", dpi=180); plt.close(figure)
        summary = {site["key"]: sum(x["endpoint"] == "B" for x in records if x["site"] == site["key"]) for site in sites}
        absolute_050 = {site["key"]: sum(x["rmsd_to_B"] < 0.50 for x in records if x["site"] == site["key"]) for site in sites}
        absolute_075 = {site["key"]: sum(x["rmsd_to_B"] < 0.75 for x in records if x["site"] == site["key"]) for site in sites}
        atomic_json(directory / "summary.json", {
            "hits_B_relative": summary,
            "hits_B_rmsd_lt_0.50A": absolute_050,
            "hits_B_rmsd_lt_0.75A": absolute_075,
            "probe2_A_ARG129_baseline_hits_B": 3,
        })
        commit()
        stage_mark("altloc_test", "complete", hits_B_relative=summary, hits_B_rmsd_lt_0_50A=absolute_050, hits_B_rmsd_lt_0_75A=absolute_075)

    if not stage_done("oracle_analysis"):
        stage_mark("oracle_analysis", "running")
        records = json.loads((out / "altloc_test" / "trajectories.json").read_text())
        oracle_rows = []
        with torch.no_grad():
            for site in train_sites:
                site_records = [record for record in records if record["site"] == site["key"]]
                learned_record = min(site_records, key=lambda record: record["energy"])
                candidates = {
                    "A": torch.zeros(site["n_chi"], device=device),
                    "B": site["true_delta"],
                    "learned": torch.tensor(learned_record["final_chi"], device=device),
                }
                for label, chi in candidates.items():
                    total, components = endpoint_loss(site, chi)
                    oracle_rows.append({
                        "site": site["key"], "candidate": label,
                        "loss_total": float(total.cpu()),
                        "loss_crystallographic": float(components["cryst"].cpu()),
                        "tmol_energy": float(components["tmol"].cpu()),
                        "vdw_penalty": float(components["vdw"].cpu()),
                        "rotamer_penalty": float(components["rot"].cpu()),
                        "symmetry_clash_penalty": float(components["clash"].cpu()),
                    })
        with (out / "oracle_analysis.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=oracle_rows[0].keys())
            writer.writeheader(); writer.writerows(oracle_rows)
        commit(); stage_mark("oracle_analysis", "complete")

    if not stage_done("energy_landscape"):
        stage_mark("energy_landscape", "running")
        directory = out / "energy_landscape"; directory.mkdir(exist_ok=True)
        site = next(site for site in train_sites if site["n_chi"] >= 2)
        axis = torch.linspace(-math.pi, math.pi, config["landscape_size"], device=device)
        values = torch.empty((axis.numel(), axis.numel()), device=device)
        with torch.no_grad():
            for i, chi1 in enumerate(axis):
                chis = torch.zeros((axis.numel(), site["n_chi"]), device=device)
                chis[:, 0], chis[:, 1] = chi1, axis
                density = site["density"].expand(axis.numel(), -1)
                onehot = site["onehot"].expand(axis.numel(), -1)
                features = torch.stack([chi_features(chi, site["n_chi"]) for chi in chis])
                values[i] = model(density, features, onehot)
        np.save(directory / "chi1_chi2_energy.npy", values.cpu().numpy())
        plt.figure(figsize=(6, 5)); plt.imshow(values.cpu().numpy().T, origin="lower", extent=(-math.pi, math.pi, -math.pi, math.pi), aspect="auto", cmap="viridis")
        plt.colorbar(label="learned energy"); plt.scatter([0, float(site["true_delta"][0].cpu())], [0, float(site["true_delta"][1].cpu())], c=["white", "red"], edgecolors="black", label="A / B")
        plt.xlabel("χ1 delta (rad)"); plt.ylabel("χ2 delta (rad)"); plt.legend(); plt.tight_layout()
        plt.savefig(directory / "landscape_plot.png", dpi=180)
        plt.savefig(out / "energy_landscape.png", dpi=180); plt.close()
        commit(); stage_mark("energy_landscape", "complete", site=site["key"])

    if not stage_done("rfree_test"):
        stage_mark("rfree_test", "running")
        directory = out / "rfree_test"; directory.mkdir(exist_ok=True)
        ensemble_positions = base_positions.clone()
        selected = {}
        for site in sites:
            candidates = infer(site, site["density"], config["eval_starts"], config["eval_steps"])
            best = min(candidates, key=lambda value: value["energy"])
            selected[site["key"]] = best
            chi = torch.tensor(best["final_chi"], device=device)
            ensemble_positions = ensemble_positions.index_copy(0, site["indices"], coords_from_chi(site, chi))
        ensemble_fcalc = calc.calc_fprotein(
            ensemble_positions, atoms_occ_tensor=base_occupancies, Return=True
        ).detach()
        r_ebt = float(normalized_rfactor(ensemble_fcalc, fobs, test_mask).cpu())
        r_deposited = float(normalized_rfactor(deposited_fcalc, fobs, test_mask).cpu())
        (directory / "rfree_ebt_ensemble.txt").write_text(f"{r_ebt:.8f}\n")
        (directory / "rfree_deposited.txt").write_text(f"{r_deposited:.8f}\n")
        with (directory / "reflection_comparison.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(("h", "k", "l", "f_obs", "f_calc_ebt", "f_calc_deposited"))
            for h, obs, ebt, dep in zip(hkl[test_mask].cpu().tolist(), fobs[test_mask].cpu().tolist(), ensemble_fcalc.abs()[test_mask].cpu().tolist(), deposited_fcalc.abs()[test_mask].cpu().tolist()):
                writer.writerow((*[int(x) for x in h], obs, ebt, dep))
        atomic_json(directory / "selected_conformers.json", selected)
        commit(); stage_mark("rfree_test", "complete", rfree_ebt=r_ebt, rfree_deposited=r_deposited)

    if not stage_done("perturbation_test"):
        stage_mark("perturbation_test", "running")
        directory = out / "perturbation_test"; directory.mkdir(exist_ok=True)
        site = next(site for site in train_sites if site["n_chi"] >= 2)
        f_a = calc.calc_fprotein(
            base_positions.index_copy(0, site["indices"], site["template"]),
            atoms_occ_tensor=base_occupancies, Return=True,
        ).detach()
        f_b = calc.calc_fprotein(
            base_positions.index_copy(0, site["indices"], site["target_b"]),
            atoms_occ_tensor=base_occupancies, Return=True,
        ).detach()
        mixed_complex = .7 * f_a + .3 * f_b
        mixed_amplitude = .7 * f_a.abs() + .3 * f_b.abs()
        mixed_coefficients = mixed_amplitude * mixed_complex / mixed_complex.abs().clamp_min(1e-8)
        modified_density = density_patch(site["center"], mixed_coefficients)
        controls = {
            "50_50": infer(site, site["density"], config["eval_starts"], config["eval_steps"]),
            "70_30": infer(site, modified_density, config["eval_starts"], config["eval_steps"]),
        }
        fractions = {}
        for label, values in controls.items():
            counts = {endpoint: sum(value["endpoint"] == endpoint for value in values) for endpoint in ("A", "B")}
            fractions[label] = {endpoint: counts[endpoint] / len(values) for endpoint in counts}
            with (directory / f"occupancy_{label}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(("condition", "endpoint", "count", "fraction"))
                for endpoint in ("A", "B"): writer.writerow((label, endpoint, counts[endpoint], fractions[label][endpoint]))
        x = np.arange(2); width = .35
        plt.figure(figsize=(5, 4)); plt.bar(x - width / 2, [fractions[k]["A"] for k in controls], width, label="A"); plt.bar(x + width / 2, [fractions[k]["B"] for k in controls], width, label="B")
        plt.xticks(x, controls.keys()); plt.ylim(0, 1); plt.ylabel("endpoint fraction"); plt.legend(); plt.tight_layout(); plt.savefig(directory / "occupancy_response.png", dpi=180); plt.close()
        commit(); stage_mark("perturbation_test", "complete", fractions=fractions)

    if not stage_done("generalization_test"):
        stage_mark("generalization_test", "running")
        directory = out / "generalization_test"; directory.mkdir(exist_ok=True)
        all_results = []
        for site in sites:
            values = infer(site, site["density"], config["eval_starts"], config["eval_steps"])
            all_results.append({
                "site": site["key"], "split": "held_out" if site["key"] in held_out_keys else "train",
                "hits_B": sum(value["endpoint"] == "B" for value in values),
                "mean_rmsd_to_B": float(np.mean([value["rmsd_to_B"] for value in values])),
            })
        with (directory / "held_out_residues.csv").open("w", newline="") as handle:
            writer = csv.writer(handle); writer.writerow(("site",)); writer.writerows((site["key"],) for site in held_out_sites)
        with (directory / "held_out_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_results[0].keys()); writer.writeheader(); writer.writerows(all_results)
        labels = [row["site"] for row in all_results]; colors = ["#e45756" if row["split"] == "held_out" else "#4c78a8" for row in all_results]
        plt.figure(figsize=(7, 4)); plt.bar(labels, [row["hits_B"] for row in all_results], color=colors); plt.axhline(3, color="black", linestyle="--", linewidth=.8, label="Probe 2 baseline"); plt.ylabel(f"B-like endpoints / {config['eval_starts']}"); plt.xticks(rotation=30); plt.legend(); plt.tight_layout(); plt.savefig(directory / "train_vs_heldout.png", dpi=180); plt.close()
        commit(); stage_mark("generalization_test", "complete", results=all_results)

    stage_mark("pipeline", "complete")
    return manifest()


@APP.local_entrypoint()
def main(
    run_name: str = "probe4_2O1K",
    steps: int = 10_000,
    checkpoint_every: int = 100,
    inner_steps: int = 3,
    inner_alpha: float = 0.1,
    outer_lr: float = 1e-4,
    eval_starts: int = 50,
    eval_steps: int = 20,
    hidden: int = 512,
    layers: int = 6,
    loss_mode: str = "fobs",
    lambda_tmol: float = 0.0,
    lambda_vdw: float = 0.0,
    lambda_rot: float = 0.0,
    lambda_clash: float = 0.0,
    active_sites: str = "",
    training_sites: str = "",
    held_out_sites: str = "B_ASP114,B_ARG129",
    force: bool = False,
):
    """Submit with ``modal run --detach``; the named Volume makes it resumable."""
    valid_loss_modes = {
        "fobs", "chi_supervised", "synthetic_fobs", "localized_sf",
        "realspace_local", "complex_target", "kinematic_complex_target",
        "realspace_kinematic",
    }
    if loss_mode not in valid_loss_modes:
        raise ValueError(f"loss_mode must be one of {sorted(valid_loss_modes)}")
    config = {
        "run_name": run_name,
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "inner_steps": inner_steps,
        "inner_alpha": inner_alpha,
        "outer_lr": outer_lr,
        "eval_starts": eval_starts,
        "eval_steps": eval_steps,
        "hidden": hidden,
        "layers": layers,
        "loss_mode": loss_mode,
        "lambda_tmol": lambda_tmol,
        "lambda_vdw": lambda_vdw,
        "lambda_rot": lambda_rot,
        "lambda_clash": lambda_clash,
        "clash_threshold": 2.5,
        "vdw_threshold": 3.0,
        "force": force,
        "seed": 41,
        "initial_chi_sigma": 1.0,
        "reflection_holdout": 0.05,
        "active_sites": [key.strip() for key in active_sites.split(",") if key.strip()] or None,
        "training_sites": [key.strip() for key in training_sites.split(",") if key.strip()] or None,
        "held_out_sites": [key.strip() for key in held_out_sites.split(",") if key.strip()],
        "density_grid_size": 8,
        "density_spacing": 1.0,
        "landscape_size": 100,
        "local_density_radius": 4.0,
        "local_density_spacing": 0.5,
        "grad_clip": 10.0,
    }
    # Fire-and-forget is intentionally used in addition to `modal run --detach`.
    # A blocking `.remote()` call remains associated with the local entrypoint
    # and can receive a cancellation signal when that client disconnects.
    call = pipeline.spawn(config)
    print({"status": "submitted", "function_call_id": call.object_id, "run_name": run_name})
