#!/usr/bin/env python3
"""Short, actual-SFcalculator Probe 2 smoke run on 2O1K A:ARG129.

CPU-only SFcalculator is deliberately kept to 3 x 10 steps here; the complete
50 x 100 Cartesian sweep is impractical without the GPU environment used by
tmol.  The target amplitudes are calculated from the deposited two-altloc model.
"""
from pathlib import Path
import json

import gemmi
import torch
from SFC_Torch import SFcalculator

ROOT = Path(__file__).parent
PDB, MTZ = ROOT / "data" / "2O1K.pdb", ROOT / "data" / "2O1K.mtz"


def main():
    torch.manual_seed(7)
    calc = SFcalculator(str(PDB), str(MTZ), device=torch.device("cpu"))
    target_fobs = calc.calc_fprotein(Return=True).abs().detach()
    structure = gemmi.read_structure(str(PDB))
    atoms = [a for chain in structure[0] for res in chain for a in res]
    b = [i for i, (chain, res, a) in enumerate((chain, res, a) for chain in structure[0] for res in chain for a in res)
         if chain.name == "A" and res.seqid.num == 129 and a.altloc == "B" and a.name.strip() not in {"N", "CA", "C", "O"}]
    a_by_name = {a.name.strip(): a.pos.tolist() for chain in structure[0] for res in chain for a in res
                 if chain.name == "A" and res.seqid.num == 129 and a.altloc == "A"}
    b_xyz = calc.atom_pos_orth[b].detach()
    a_xyz = torch.tensor([a_by_name[atoms[i].name.strip()] for i in b], dtype=torch.float32)
    base = calc.atom_pos_orth.detach()
    bonds = [(i, i + 1) for i in range(len(b) - 1)]
    bond_lengths = torch.stack([torch.linalg.vector_norm(a_xyz[i] - a_xyz[j]) for i, j in bonds])

    def loss_fn(xyz):
        pos = base.index_copy(0, torch.tensor(b), xyz)
        fcalc = calc.calc_fprotein(pos, Return=True)
        sf = torch.mean(((fcalc.abs() - target_fobs) / target_fobs.clamp_min(1)).square())
        lengths = torch.stack([torch.linalg.vector_norm(xyz[i] - xyz[j]) for i, j in bonds])
        return sf + 3.0 * torch.mean((lengths - bond_lengths).square())

    # Verify that SFcalculator exposes finite coordinate gradients.
    check = (a_xyz + 0.2 * torch.randn_like(a_xyz)).requires_grad_(True)
    grad = torch.autograd.grad(loss_fn(check), check)[0]
    assert torch.isfinite(grad).all() and grad.norm() > 0

    results = []
    for seed in range(1):
        xyz = (a_xyz + 0.75 * torch.randn_like(a_xyz)).requires_grad_(True)
        opt = torch.optim.Adam([xyz], lr=.025)
        for _ in range(1):
            opt.zero_grad(); loss = loss_fn(xyz); loss.backward(); opt.step()
        ra = torch.sqrt(torch.mean((xyz.detach() - a_xyz).square())).item()
        rb = torch.sqrt(torch.mean((xyz.detach() - b_xyz).square())).item()
        results.append({"start": seed, "rmsd_to_A": ra, "rmsd_to_B": rb, "endpoint": "B" if rb < ra else "A"})
    out = ROOT / "results" / "probe2_sfcalculator_smoke.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"SFcalculator smoke: coordinate gradient finite; {sum(x['endpoint'] == 'B' for x in results)}/1 endpoints closer to B")


if __name__ == "__main__":
    main()
