#!/usr/bin/env python3
"""Quick Probe 2: multi-start Cartesian recovery of 2O1K ARG A129 altloc B.

The target amplitudes are synthetic |F_obs| calculated from the deposited,
two-altloc model, exactly as specified for the README probe.  This keeps the
first run independent of phasing and scaling choices in the experimental MTZ.
"""
from pathlib import Path
import json
import os

import gemmi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).parent
PDB = ROOT / "data" / "2O1K.cif"
OUT = ROOT / "results"
CHAIN, RESIDUE, ALT = "A", 129, "B"
N_STARTS, N_STEPS, N_REFL = 50, 100, 1200
SEED = 7


def main():
    torch.manual_seed(SEED)
    OUT.mkdir(exist_ok=True)
    structure = gemmi.read_structure(str(PDB))
    structure.setup_entities()
    cell = structure.cell
    model = structure[0]

    all_atoms, target_b, target_a = [], [], []
    for chain in model:
        for residue in chain:
            for atom in residue:
                item = (atom, chain.name, residue.seqid.num)
                all_atoms.append(item)
                if chain.name == CHAIN and residue.seqid.num == RESIDUE and atom.name.strip() not in {"N", "CA", "C", "O"}:
                    if atom.altloc == ALT:
                        target_b.append(atom)
                    elif atom.altloc == "A":
                        target_a.append(atom)
    names = [a.name.strip() for a in target_b]
    a_by_name = {a.name.strip(): a for a in target_a}
    target_a = [a_by_name[n] for n in names]
    assert names and len(target_a) == len(target_b), (names, list(a_by_name))

    # Use measured Miller indices but calculate the target amplitudes from the
    # complete deposited ensemble (the probe's stipulated synthetic F_obs).
    mtz = gemmi.read_mtz_file(str(ROOT / "data" / "2O1K.mtz"))
    hkl = mtz.array[:, :3].astype(np.float32)
    hkl = hkl[np.unique(hkl, axis=0, return_index=True)[1]][:N_REFL]
    hkl_t = torch.tensor(hkl)
    ortho = np.array(cell.orth.mat.tolist(), dtype=np.float32)
    frac = np.array([cell.fractionalize(a.pos).tolist() for a, _, _ in all_atoms], dtype=np.float32)
    z = np.array([a.element.atomic_number * a.occ for a, _, _ in all_atoms], dtype=np.float32)

    # Atomic-number scattering is a deliberately lightweight stand-in for
    # SFcalculator in this plumbing probe; it is fully differentiable.
    phase = 2 * np.pi * (hkl @ frac.T)
    full_f = torch.exp(1j * torch.tensor(phase)) @ torch.tensor(z, dtype=torch.complex64)
    f_obs = full_f.abs().detach()

    b_indices = [i for i, (a, ch, seq) in enumerate(all_atoms)
                 if ch == CHAIN and seq == RESIDUE and a.altloc == ALT and a.name.strip() in names]
    base_frac = torch.tensor(frac)
    base_z = torch.tensor(z)
    old_b_frac = base_frac[b_indices]
    b_xyz = torch.tensor([a.pos.tolist() for a in target_b], dtype=torch.float32)
    a_xyz = torch.tensor([a.pos.tolist() for a in target_a], dtype=torch.float32)
    # Preserve bond lengths from the starting conformer A (not the answer B).
    bonds = [(i, i + 1) for i in range(len(names) - 1)]
    a_lengths = torch.stack([torch.linalg.vector_norm(a_xyz[i] - a_xyz[j]) for i, j in bonds])

    # Contribution of every atom except the variable altloc-B side chain.
    base_phase = 2 * torch.pi * (hkl_t @ base_frac.T)
    base_complex = torch.exp(1j * base_phase) @ base_z.to(torch.complex64)
    old_phase = 2 * torch.pi * (hkl_t @ old_b_frac.T)
    old_z = torch.tensor(z[b_indices])
    fixed_complex = base_complex - torch.exp(1j * old_phase) @ old_z.to(torch.complex64)
    inv_ortho = torch.tensor(np.linalg.inv(ortho), dtype=torch.float32)

    results = []
    for start in range(N_STARTS):
        xyz = (a_xyz + 0.75 * torch.randn_like(a_xyz)).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([xyz], lr=0.025)
        for _ in range(N_STEPS):
            optimizer.zero_grad()
            variable_frac = xyz @ inv_ortho.T
            phase = 2 * torch.pi * (hkl_t @ variable_frac.T)
            calc = fixed_complex + torch.exp(1j * phase) @ old_z.to(torch.complex64)
            sf_loss = torch.mean(((calc.abs() - f_obs) / f_obs.clamp_min(1.)).square())
            lengths = torch.stack([torch.linalg.vector_norm(xyz[i] - xyz[j]) for i, j in bonds])
            geom_loss = torch.mean((lengths - a_lengths).square())
            loss = sf_loss + 3.0 * geom_loss
            loss.backward()
            optimizer.step()
        rmsd_a = torch.sqrt(torch.mean((xyz.detach() - a_xyz).square())).item()
        rmsd_b = torch.sqrt(torch.mean((xyz.detach() - b_xyz).square())).item()
        results.append({"start": start, "rmsd_to_A": rmsd_a, "rmsd_to_B": rmsd_b,
                        "endpoint": "B" if rmsd_b < rmsd_a else "A"})

    with open(OUT / "probe2_2O1K_ARG129.json", "w") as handle:
        json.dump(results, handle, indent=2)
    a = [r["rmsd_to_A"] for r in results]
    b = [r["rmsd_to_B"] for r in results]
    colors = ["#e45756" if r["endpoint"] == "B" else "#4c78a8" for r in results]
    plt.figure(figsize=(5, 4))
    plt.scatter(a, b, c=colors, edgecolors="white", linewidths=.4)
    limit = max(a + b) * 1.05
    plt.plot([0, limit], [0, limit], "k--", linewidth=.8)
    plt.xlim(0, limit); plt.ylim(0, limit)
    plt.xlabel("endpoint RMSD to altloc A (Å)")
    plt.ylabel("endpoint RMSD to altloc B (Å)")
    plt.tight_layout(); plt.savefig(OUT / "probe2_2O1K_ARG129.png", dpi=180)
    hits = sum(r["endpoint"] == "B" for r in results)
    print(f"Probe 2 — 2O1K A:ARG129: {hits}/{N_STARTS} endpoints closer to altloc B")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()
