from __future__ import annotations

import argparse
import csv
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

from probe4_core import dihedral, torsion_to_coords, wrap_angles

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


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


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
    """Select blank atoms plus one highest-coverage altloc per protein residue."""
    selected = []
    for chain in structure[0]:
        for residue in chain:
            if residue.name not in CHI_SPECS and residue.name not in {
                "ALA", "ASN", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                "LEU", "LYS", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
            }:
                continue
            blank = {}
            alternate = {}
            for atom in residue:
                if atom.element.name == "H":
                    continue
                name = atom.name.strip()
                alt = "" if atom.altloc in ("\x00", " ") else atom.altloc
                if not alt:
                    blank[name] = atom
                else:
                    alternate.setdefault(alt, {})[name] = atom
            if alternate:
                _alt, atoms = max(
                    alternate.items(),
                    key=lambda item: (
                        len(item[1]), sum(float(atom.occ) for atom in item[1].values())
                    ),
                )
                chosen = {**blank, **atoms}
            else:
                chosen = blank
            selected.extend((chain, residue, atom) for atom in chosen.values())
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
        "--fixed-occupancy-steps", type=int, default=0,
        help=(
            "prefix of --n-steps that optimizes chi angles with uniform "
            "occupancies; Adam is reset when occupancies are released"
        ),
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
    if not 0 <= args.fixed_occupancy_steps <= args.n_steps:
        raise ValueError(
            "--fixed-occupancy-steps must be between zero and --n-steps"
        )
    if args.sequential_two_stage:
        if args.fixed_occupancy_steps or args.seed_deposited_a or args.soft_physics:
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
        if args.seed_deposited_a or args.soft_physics:
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
        radial_mask = np.linalg.norm(coordinates - np.asarray(site.center), axis=-1) <= args.grid_radius
        selected_grid = torch.tensor(coordinates[radial_mask], dtype=torch.float32, device=device)
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
            label: _normalize(patch[torch.tensor(radial_mask, device=device)])
            for label, patch in target_patches.items()
        }
        blur_levels = (0.0, 2.0, 4.0)
        radial_mask_tensor = torch.tensor(radial_mask, device=device)
        target_vectors_by_blur = {
            label: {
                blur: _normalize(_gaussian_blur_patch(
                    patch, blur, args.spacing
                )[radial_mask_tensor])
                for blur in blur_levels
            }
            for label, patch in target_patches.items()
        }
        np.savez_compressed(
            args.output / f"{short_key}_targets.npz",
            raw=raw_normalized,
            denoised=denoised_patch.cpu().numpy(),
            denoiser_training_target=synthetic_normalized,
            radial_mask=radial_mask,
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
        direct_environment = torch.tensor(
            [atom.pos.tolist() for _, _, atom in direct_context],
            dtype=torch.float32, device=device,
        )
        direct_pair_mask = torch.ones(
            (len(names), len(direct_context)), dtype=torch.bool, device=device
        )
        for moving_index, moving_name in enumerate(names):
            for environment_index, (context_chain, context_residue, atom) in enumerate(
                direct_context
            ):
                same_target = (
                    context_chain.name == site.chain
                    and context_residue.seqid.num == site.residue_number
                    and context_residue.seqid.icode == site.insertion_code
                )
                # CB--CA is the only moving-sidechain/backbone covalent bond.
                if same_target and moving_name == "CB" and atom.name.strip() == "CA":
                    direct_pair_mask[moving_index, environment_index] = False

        cell = structure.cell
        spacegroup = gemmi.find_spacegroup_by_name(structure.spacegroup_hm)
        symmetry_positions = []
        for operation_index, operation in enumerate(spacegroup.operations()):
            for tx in (-1, 0, 1):
                for ty in (-1, 0, 1):
                    for tz in (-1, 0, 1):
                        if operation_index == 0 and tx == ty == tz == 0:
                            continue
                        for _context_chain, _context_residue, atom in heavy_atoms:
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
                                symmetry_positions.append(xyz)
        symmetry_environment = torch.tensor(
            np.asarray(symmetry_positions), dtype=torch.float32, device=device
        ) if symmetry_positions else torch.empty((0, 3), dtype=torch.float32, device=device)

        def physical_chi(
            candidate: torch.Tensor,
            fixed_lookup=fixed_lookup,
            names=names,
            dihedrals=tuple(spec["dihedrals"]),
        ) -> torch.Tensor:
            lookup = dict(fixed_lookup)
            lookup.update({name: candidate[index] for index, name in enumerate(names)})
            return torch.stack([
                wrap_angles(dihedral(*(lookup[name] for name in quartet)) - torch.pi)
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

        def physics_terms(
            conformers: list[torch.Tensor],
            occupancies: torch.Tensor,
            resname=residue.name,
            direct_environment=direct_environment,
            direct_pair_mask=direct_pair_mask,
            symmetry_environment=symmetry_environment,
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
                if direct_environment.numel():
                    distances = torch.cdist(candidate, direct_environment)
                    penalties = torch.clamp(
                        args.vdw_threshold - distances, min=0.0
                    ).square()
                    vdw_terms.append(penalties.masked_select(direct_pair_mask).sum())
                else:
                    vdw_terms.append(zero)
                chis = physical_chi(candidate)
                rotamer_terms.append(torch.stack([
                    (
                        1.0 - torch.cos(
                            value - torch.tensor(
                                _canonical_centers(resname, chi_index),
                                dtype=value.dtype, device=device,
                            )
                        )
                    ).min()
                    for chi_index, value in enumerate(chis)
                ]).sum())
                if symmetry_environment.numel():
                    symmetry_terms.append(torch.clamp(
                        args.clash_threshold - torch.cdist(
                            candidate, symmetry_environment
                        ), min=0.0
                    ).square().sum())
                else:
                    symmetry_terms.append(zero)
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
        np.save(
            args.output / f"{short_key}_optimizer_synthetic_vector.npy",
            native_synthetic.cpu().numpy(),
        )

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
            "physics_terms": physics_terms,
            "physical_chi": physical_chi,
            "delta_from_physical_chi": delta_from_physical_chi,
            "base_physical_chi": base_physical_chi,
            "direct_environment": direct_environment,
            "direct_pair_mask": direct_pair_mask,
            "symmetry_environment": symmetry_environment,
        })
    if {site["key"] for site in sites} != wanted_sites:
        raise RuntimeError(
            f"requested {sorted(wanted_sites)}, built {sorted(site['key'] for site in sites)}"
        )

    config = {
        **vars(args),
        "pdb": str(args.pdb), "mtz": str(args.mtz),
        "selection": str(args.selection) if args.selection else None,
        "checkpoint": str(args.checkpoint), "output": str(args.output),
        "transfer_endpoint_root": str(args.transfer_endpoint_root),
        "targets": targets, "sites": sorted(wanted_sites),
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
                    values[candidate] = float(
                        (rendered - site["target_vectors"][target_label]).square().mean().cpu()
                    )
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
    if args.calibration_only:
        _atomic_json(args.output / "stage_manifest.json", {
            "status": "complete", "calibration_only": True,
            "calibration_rows": len(calibration_rows),
        })
        print(json.dumps({"status": "complete", "calibration": calibration_rows}, indent=2))
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
                final_density_loss = float((
                    rendered - site["target_vectors"]["denoised"]
                ).square().mean().cpu())
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
                    canonical_flags.append(maximum_deviation <= 30.0)

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
                        loss = (density - target).square().mean()
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            all_chi.copy_(wrap_angles(all_chi))
                with torch.no_grad():
                    density_stage_rendered, _ = site["render"](all_chi, logits)
                    density_stage_loss = float((
                        density_stage_rendered
                        - site["target_vectors"]["denoised"]
                    ).square().mean().cpu())

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
                    density_loss = (
                        density - site["target_vectors"]["denoised"]
                    ).square().mean()
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
            density_loss = (
                density - site["target_vectors"]["denoised"]
            ).square().mean()
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
                (
                    final_candidate_density
                    - site["target_vectors"]["denoised"]
                ).square().mean()
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
            final_density_loss = float((
                rendered - site["target_vectors"]["denoised"]
            ).square().mean().cpu())
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
                canonical_flags.append(maximum_deviation <= 30.0)

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
            for start in range(len(rows), args.n_starts):
                generator = torch.Generator(device=device).manual_seed(
                    args.seed + 100_000 * site_index + start
                )
                all_chi = torch.randn(
                    (args.K, site["n_chi"]), generator=generator, device=device
                ).requires_grad_(True)
                if args.seed_deposited_a:
                    with torch.no_grad():
                        # coordinates_from_chi uses deposited A as its zero-delta
                        # reference, so this is an exact local-basin stability test.
                        all_chi[0].zero_()
                logits = torch.zeros(
                    args.K, device=device,
                    requires_grad=args.fixed_occupancy_steps == 0,
                )
                best_stage1_loss = float("inf")
                fixed_boundary_density_loss = float("nan")
                fixed_boundary_occupancies = np.full(args.K, np.nan)
                fixed_boundary_chi = np.full((args.K, site["n_chi"]), np.nan)
                fixed_boundary_rmsd_a = np.full(args.K, np.nan)
                fixed_boundary_rmsd_b = np.full(args.K, np.nan)
                fixed_boundary_found_a = False
                fixed_boundary_found_b = False

                if args.fixed_occupancy_steps:
                    # Equal weights prevent a slot from starving its own coordinate
                    # gradient before it has had a chance to discover a second basin.
                    optimizer = torch.optim.Adam([all_chi], lr=args.lr)
                else:
                    optimizer = torch.optim.Adam([all_chi, logits], lr=args.lr)
                for _step in range(args.fixed_occupancy_steps):
                    optimizer.zero_grad(set_to_none=True)
                    density, current_coordinates = site["render"](all_chi, logits)
                    density_loss = (
                        density - site["target_vectors"][target_label]
                    ).square().mean()
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
                    with torch.no_grad():
                        all_chi.copy_(wrap_angles(all_chi))
                    best_stage1_loss = min(
                        best_stage1_loss, float(loss.detach().cpu())
                    )
                if args.fixed_occupancy_steps:
                    with torch.no_grad():
                        fixed_rendered, fixed_coordinates = site["render"](
                            all_chi, logits
                        )
                        fixed_boundary_density_loss = float((
                            fixed_rendered - site["target_vectors"][target_label]
                        ).square().mean().cpu())
                        fixed_boundary_occupancies = (
                            torch.softmax(logits, dim=0).cpu().numpy().copy()
                        )
                        fixed_boundary_chi = (
                            all_chi.detach().cpu().numpy().copy()
                        )
                        fixed_boundary_rmsd_a = np.asarray([
                            float(site["rmsd"](xyz, site["kinematic_a"]).cpu())
                            for xyz in fixed_coordinates
                        ])
                        fixed_boundary_rmsd_b = np.asarray([
                            float(site["rmsd"](xyz, site["kinematic_b"]).cpu())
                            for xyz in fixed_coordinates
                        ])
                        fixed_boundary_found_a = bool(
                            (fixed_boundary_rmsd_a < 1.0).any()
                        )
                        fixed_boundary_found_b = bool(
                            (fixed_boundary_rmsd_b < 1.0).any()
                        )

                    # Release occupancies with a clean optimizer state. The total
                    # density-stage budget remains exactly --n-steps.
                    logits.requires_grad_(True)
                    optimizer = torch.optim.Adam([all_chi, logits], lr=args.lr)

                released_occupancy_steps = (
                    args.n_steps - args.fixed_occupancy_steps
                )
                for _step in range(released_occupancy_steps):
                    optimizer.zero_grad(set_to_none=True)
                    density, current_coordinates = site["render"](all_chi, logits)
                    density_loss = (
                        density - site["target_vectors"][target_label]
                    ).square().mean()
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
                    with torch.no_grad():
                        all_chi.copy_(wrap_angles(all_chi))
                    best_stage1_loss = min(
                        best_stage1_loss, float(loss.detach().cpu())
                    )
                with torch.no_grad():
                    stage1_rendered, stage1_coordinates = site["render"](all_chi, logits)
                    stage1_density_loss = float((
                        stage1_rendered - site["target_vectors"][target_label]
                    ).square().mean().cpu())
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
                    for _step in range(args.physics_refinement_steps):
                        optimizer.zero_grad(set_to_none=True)
                        density, current_coordinates = site["render"](all_chi, logits)
                        density_loss = (
                            density - site["target_vectors"][target_label]
                        ).square().mean()
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
                with torch.no_grad():
                    rendered, coordinates = site["render"](all_chi, logits)
                    final_density_loss_tensor = (
                        rendered - site["target_vectors"][target_label]
                    ).square().mean()
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
                        canonical_flags.append(maximum_deviation <= 30.0)
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
                rows.append({
                    "target": target_label, "site": site["key"], "start": start,
                    "final_loss": final_loss, "best_loss": reported_best_loss,
                    "best_stage1_loss": best_stage1_loss,
                    "best_refinement_loss": best_refinement_loss,
                    "fixed_occupancy_steps": args.fixed_occupancy_steps,
                    "released_occupancy_steps": released_occupancy_steps,
                    "fixed_boundary_density_loss": fixed_boundary_density_loss,
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
                })
                _atomic_csv(result_path, rows)
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
