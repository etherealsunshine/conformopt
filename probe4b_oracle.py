#!/usr/bin/env python3
"""Score the A and representable-B endpoints under each Probe 4b loss."""

from __future__ import annotations

import json
from pathlib import Path

import gemmi
import torch
from SFC_Torch import SFcalculator

from probe4_core import dihedral, torsion_to_coords, wrap_angles


ROOT = Path(__file__).parent
SPECS = {
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
}


def main() -> None:
    device = torch.device("cpu")
    calc = SFcalculator(str(ROOT / "data/2O1K.pdb"), str(ROOT / "data/2O1K.mtz"), device=device)
    fobs = calc.Fo.detach()
    base = calc.atom_pos_orth.detach()
    occupancy = calc.atom_occ.detach().clone()
    deposited = calc.calc_fprotein(base, atoms_occ_tensor=occupancy, Return=True).detach()
    hkl = torch.as_tensor(calc.HKL_array, dtype=torch.float32)
    orth_to_frac = calc.orth2frac_tensor.detach()
    valid = torch.isfinite(fobs) & (fobs > 0)
    deposited_amp = deposited.abs()
    phase = deposited / deposited_amp.clamp_min(1e-8)
    obs_to_calc = (deposited_amp[valid] @ fobs[valid]) / fobs[valid].square().sum()
    map_coefficients = (2 * obs_to_calc * fobs - deposited_amp) * phase
    trajectory_paths = {
        "synthetic_fobs": ROOT / "probe4b_results/experiment_A_synthetic_fobs/altloc_test/trajectories.json",
        "localized_sf": ROOT / "probe4b_results/experiment_B_localized_sf/altloc_test/trajectories.json",
        "realspace_local": ROOT / "probe4b_results/experiment_C_realspace_local/altloc_test/trajectories.json",
    }
    learned_chi = {}
    for mode, path in trajectory_paths.items():
        records = json.loads(path.read_text())
        learned_chi[mode] = {
            site: torch.tensor(next(row["final_chi"] for row in records if row["site"] == site))
            for site in {row["site"] for row in records}
        }

    structure = gemmi.read_structure(str(ROOT / "data/2O1K.pdb"))
    atoms = [(chain, residue, atom) for chain in structure[0] for residue in chain for atom in residue]
    results = []
    for chain in structure[0]:
        for residue in chain:
            key = f"{chain.name}_{residue.name}{residue.seqid.num}"
            if key not in {"A_MET112", "A_ARG129", "B_MET112"}:
                continue
            spec = SPECS[residue.name]

            def atom_map(alt: str) -> dict[str, torch.Tensor]:
                out = {}
                for atom in residue:
                    atom_alt = atom.altloc if atom.altloc not in ("\x00", " ") else ""
                    if atom_alt in ("", alt):
                        out[atom.name.strip()] = torch.tensor(atom.pos.tolist())
                return out

            map_a, map_b = atom_map("A"), atom_map("B")
            indices = [
                index for index, (candidate_chain, candidate_residue, atom) in enumerate(atoms)
                if candidate_chain.name == chain.name
                and candidate_residue.seqid.num == residue.seqid.num
                and atom.altloc == "B"
                and atom.name.strip() not in {"N", "CA", "C", "O"}
            ]
            names = [atoms[index][2].name.strip() for index in indices]
            index_tensor = torch.tensor(indices)
            template = torch.stack([map_a[name] for name in names])
            target_b = torch.stack([map_b[name] for name in names])
            fixed_atoms = {name: value for name, value in map_a.items() if name not in names}
            chi_a = torch.stack([dihedral(*(map_a[name] for name in quartet)) for quartet in spec["dihedrals"]])
            chi_b = torch.stack([dihedral(*(map_b[name] for name in quartet)) for quartet in spec["dihedrals"]])
            delta = wrap_angles(chi_b - chi_a)

            def coords(angles: torch.Tensor) -> torch.Tensor:
                return torsion_to_coords(template, names, angles, list(spec["rotations"]), fixed_atoms)

            plus, minus = coords(delta), coords(-delta)
            if torch.mean((minus - target_b).square()) < torch.mean((plus - target_b).square()):
                delta = -delta
            candidate_a, candidate_b = coords(torch.zeros_like(delta)), coords(delta)

            without_occ = occupancy.clone()
            without_occ[index_tensor] = 0
            fixed_sf = calc.calc_fprotein(base, atoms_occ_tensor=without_occ, Return=True).detach()

            def fcalc(candidate: torch.Tensor) -> torch.Tensor:
                positions = base.index_copy(0, index_tensor, candidate)
                return calc.calc_fprotein(positions, atoms_occ_tensor=occupancy, Return=True).detach()

            sf_a, sf_b = fcalc(candidate_a), fcalc(candidate_b)
            synthetic = {}
            for label, value in (("A", sf_a), ("B", sf_b)):
                residual = (value.abs() - deposited_amp) / deposited_amp.clamp_min(1)
                synthetic[label] = float(residual[valid].square().mean())
            learned_synthetic_sf = fcalc(coords(learned_chi["synthetic_fobs"][key]))
            learned_synthetic_residual = (
                learned_synthetic_sf.abs() - deposited_amp
            ) / deposited_amp.clamp_min(1)
            synthetic["learned"] = float(learned_synthetic_residual[valid].square().mean())

            fixed_phase = fixed_sf / fixed_sf.abs().clamp_min(1e-8)
            target_residual = obs_to_calc * fobs * fixed_phase - fixed_sf
            side_normalizer = (deposited[valid] - fixed_sf[valid]).abs().square().mean().clamp_min(1e-12)
            localized_sf = {
                label: float(((value[valid] - fixed_sf[valid] - target_residual[valid]).abs().square().mean() / side_normalizer))
                for label, value in (("A", sf_a), ("B", sf_b))
            }
            learned_localized_sf = fcalc(coords(learned_chi["localized_sf"][key]))
            localized_sf["learned"] = float(
                (
                    learned_localized_sf[valid]
                    - fixed_sf[valid]
                    - target_residual[valid]
                ).abs().square().mean() / side_normalizer
            )

            radius, spacing = 4.0, 0.5
            axis = torch.arange(-radius, radius + spacing / 2, spacing)
            offsets = torch.cartesian_prod(axis, axis, axis)
            offsets = offsets[torch.linalg.vector_norm(offsets, dim=1) <= radius]
            center = torch.cat((template, target_b)).mean(dim=0)
            points = center + offsets
            kernel = torch.exp(-1j * 2 * torch.pi * (hkl[valid] @ (points @ orth_to_frac.T).T))
            target_density = 2 * torch.real((map_coefficients[valid] - fixed_sf[valid]) @ kernel) / int(valid.sum())
            deposited_side_density = 2 * torch.real((deposited[valid] - fixed_sf[valid]) @ kernel) / int(valid.sum())
            density_normalizer = deposited_side_density.square().mean().clamp_min(1e-12)
            realspace = {}
            for label, value in (("A", sf_a), ("B", sf_b)):
                rho = 2 * torch.real((value[valid] - fixed_sf[valid]) @ kernel) / int(valid.sum())
                realspace[label] = float((rho - target_density).square().mean() / density_normalizer)
            learned_realspace_sf = fcalc(coords(learned_chi["realspace_local"][key]))
            learned_rho = 2 * torch.real(
                (learned_realspace_sf[valid] - fixed_sf[valid]) @ kernel
            ) / int(valid.sum())
            realspace["learned"] = float(
                (learned_rho - target_density).square().mean() / density_normalizer
            )

            results.append({
                "site": key,
                "kinematic_B_rmsd": float(torch.sqrt(torch.mean((candidate_b - target_b).square()))),
                "synthetic_fobs": synthetic,
                "localized_sf": localized_sf,
                "realspace_local": realspace,
            })

    out = ROOT / "probe4b_results/oracle_A_vs_B.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
