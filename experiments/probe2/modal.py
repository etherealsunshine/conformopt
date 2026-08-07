"""Modal GPU smoke job for the tmol + SFcalculator Probe 2 stack.

Submit with:
  UV_CACHE_DIR=/private/tmp/uv-modal uvx modal run probe2_modal.py
"""
from pathlib import Path
import json
import modal

APP = modal.App("probe2-tmol-sfcalculator")
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.40/"
    "tmol-0.1.40%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl"
)
IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-runtime-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.8.0", extra_options="--index-url https://download.pytorch.org/whl/cu128")
    .run_commands(f"pip install 'tmol @ {TMOL_WHEEL}'")
    .pip_install("SFcalculator-torch==0.3.3")
    .add_local_dir(Path(__file__).parent / "data", remote_path="/data", copy=True)
)


@APP.function(image=IMAGE, gpu="L4", timeout=600)
def smoke():
    import torch
    import tmol
    from SFC_Torch import SFcalculator

    device = torch.device("cuda")
    # 2O1K contains several incompletely modelled sidechains. This contiguous,
    # complete segment retains ARG129 and provides a valid tmol pose.
    pdb = "/data/2O1K_A108_130_complete.pdb"
    pose = tmol.pose_stack_from_pdb(pdb, device=device)
    scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(pose)
    tmol_xyz = pose.coords.detach().clone().requires_grad_(True)
    energy = scorer(tmol_xyz).sum()
    grad = torch.autograd.grad(energy, tmol_xyz)[0]
    calc = SFcalculator(pdb, "/data/2O1K.mtz", device=device)
    xyz = calc.atom_pos_orth.detach().clone().requires_grad_(True)
    fcalc = calc.calc_fprotein(xyz, Return=True)
    sf_grad = torch.autograd.grad(fcalc.abs().mean(), xyz)[0]
    result = {
        "tmol": tmol.__version__,
        "torch": torch.__version__,
        "energy": float(energy.detach().cpu()),
        "tmol_grad_norm": float(grad.norm().cpu()),
        "sfcalculator_grad_norm": float(sf_grad.norm().cpu()),
    }
    print(result)
    return result


@APP.function(image=IMAGE, gpu="L4", timeout=1800)
def sweep():
    """Probe 2: 50 Cartesian starts x 100 Adam steps on the shared energy."""
    import gemmi
    import torch
    import tmol
    from SFC_Torch import SFcalculator

    torch.manual_seed(7)
    device = torch.device("cuda")
    base_pdb = "/data/2O1K_A108_130_complete.pdb"
    alt_pdb = "/data/2O1K_A108_130_altlocs.pdb"
    mtz = "/data/2O1K.mtz"
    # SFcalculator sees the A conformer plus a variable B conformer. Its
    # deposited two-altloc model supplies the synthetic F_obs target.
    calc = SFcalculator(alt_pdb, mtz, device=device)
    f_obs = calc.calc_fprotein(Return=True).abs().detach()
    base_sf = calc.atom_pos_orth.detach()
    structure = gemmi.read_structure(alt_pdb)
    atoms = [a for ch in structure[0] for res in ch for a in res]
    b_idx = [i for i, (ch, res, a) in enumerate((ch, res, a) for ch in structure[0] for res in ch for a in res)
             if ch.name == "A" and res.seqid.num == 129 and a.altloc == "B" and a.name.strip() not in {"N", "CA", "C", "O"}]
    a_by_name = {a.name.strip(): a.pos.tolist() for ch in structure[0] for res in ch for a in res
                 if ch.name == "A" and res.seqid.num == 129 and a.altloc == "A"}
    b_by_name = {a.name.strip(): a.pos.tolist() for ch in structure[0] for res in ch for a in res
                 if ch.name == "A" and res.seqid.num == 129 and a.altloc == "B"}
    names = [atoms[i].name.strip() for i in b_idx]
    a_xyz = torch.tensor([a_by_name[n] for n in names], device=device)
    b_xyz = torch.tensor([b_by_name[n] for n in names], device=device)
    b_idx_t = torch.tensor(b_idx, device=device)

    # tmol sees the same variable B sidechain as a complete, physical ARG.
    pose = tmol.pose_stack_from_pdb(base_pdb, device=device)
    scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(pose)
    residue_index = 129 - 108
    block_type = int(pose.block_type_ind[0, residue_index])
    restype = pose.packed_block_types.active_block_types[block_type]
    atom_names = [str(restype.atom_name(i)).strip() for i in range(int(pose.packed_block_types.n_atoms[block_type]))]
    offset = int(pose.block_coord_offset[0, residue_index])
    tmol_idx = torch.tensor([offset + atom_names.index(n) for n in names], device=device)
    base_tmol = pose.coords.detach()
    # Calibrate at a perturbed A start.  Calibrating at the deposited model
    # would make the SF residual exactly zero and silently disable tmol.
    with torch.no_grad():
        calibration_xyz = a_xyz + .75 * torch.randn_like(a_xyz)
        calibration_sf = base_sf.index_copy(0, b_idx_t, calibration_xyz)
        sf0 = torch.mean(((calc.calc_fprotein(calibration_sf, Return=True).abs() - f_obs) / f_obs.clamp_min(1)).square())
        e0 = scorer(base_tmol).sum().abs().clamp_min(1.)
    tmol_weight = float((sf0 / e0).cpu())

    def composite(xyz):
        sf_xyz = base_sf.index_copy(0, b_idx_t, xyz)
        sf = torch.mean(((calc.calc_fprotein(sf_xyz, Return=True).abs() - f_obs) / f_obs.clamp_min(1)).square())
        tmol_xyz = base_tmol.index_copy(1, tmol_idx, xyz.unsqueeze(0))
        return sf + tmol_weight * scorer(tmol_xyz).sum(), sf

    results = []
    for start in range(50):
        xyz = (a_xyz + .75 * torch.randn_like(a_xyz)).requires_grad_(True)
        opt = torch.optim.Adam([xyz], lr=.025)
        for _ in range(100):
            opt.zero_grad(); loss, _ = composite(xyz); loss.backward(); opt.step()
        ra = torch.sqrt(torch.mean((xyz.detach() - a_xyz).square())).item()
        rb = torch.sqrt(torch.mean((xyz.detach() - b_xyz).square())).item()
        results.append({"start": start, "rmsd_to_A": ra, "rmsd_to_B": rb, "endpoint": "B" if rb < ra else "A"})
        if (start + 1) % 10 == 0:
            print(f"completed {start + 1}/50")
    print(json.dumps({"tmol_weight": tmol_weight, "hits_B": sum(r["endpoint"] == "B" for r in results), "results": results}))


@APP.function(image=IMAGE, gpu="L4", timeout=1800)
def torsion_sweep():
    """Probe 2 in ARG129 χ1–χ4 space; bonds/angles remain fixed."""
    import gemmi
    import torch
    import tmol
    from SFC_Torch import SFcalculator

    torch.manual_seed(11)
    device = torch.device("cuda")
    base_pdb, alt_pdb, mtz = "/data/2O1K_A108_130_complete.pdb", "/data/2O1K_A108_130_altlocs.pdb", "/data/2O1K.mtz"
    calc = SFcalculator(alt_pdb, mtz, device=device)
    f_obs, base_sf = calc.calc_fprotein(Return=True).abs().detach(), calc.atom_pos_orth.detach()
    structure = gemmi.read_structure(alt_pdb)
    atoms = [a for ch in structure[0] for res in ch for a in res]
    b_idx = [i for i, (ch,res,a) in enumerate((ch,res,a) for ch in structure[0] for res in ch for a in res)
             if ch.name == "A" and res.seqid.num == 129 and a.altloc == "B" and a.name.strip() not in {"N","CA","C","O"}]
    names = [atoms[i].name.strip() for i in b_idx]
    def coords(alt):
        return {a.name.strip(): torch.tensor(a.pos.tolist(), device=device) for ch in structure[0] for res in ch for a in res
                if ch.name == "A" and res.seqid.num == 129 and a.altloc == alt}
    a, b = coords("A"), coords("B")
    template = torch.stack([a[n] for n in names]); b_xyz = torch.stack([b[n] for n in names])
    name_to_i = {n:i for i,n in enumerate(names)}
    ca = next(torch.tensor(a0.pos.tolist(), device=device) for ch in structure[0] for res in ch for a0 in res
              if ch.name == "A" and res.seqid.num == 129 and a0.name.strip() == "CA")
    b_idx_t = torch.tensor(b_idx, device=device)

    pose = tmol.pose_stack_from_pdb(base_pdb, device=device)
    scorer = tmol.beta2016_score_function(device).render_whole_pose_scoring_module(pose)
    ri, bt = 21, int(pose.block_type_ind[0,21]); rt = pose.packed_block_types.active_block_types[bt]
    tnames = [str(rt.atom_name(i)).strip() for i in range(int(pose.packed_block_types.n_atoms[bt]))]
    off = int(pose.block_coord_offset[0,ri]); tmol_idx = torch.tensor([off+tnames.index(n) for n in names], device=device)
    base_tmol = pose.coords.detach()

    # Rodrigues rotations around the four ARG chi bonds, applied serially.
    chains = [(ca, "CB", ["CG","CD","NE","CZ","NH1","NH2"]), ("CB","CG", ["CD","NE","CZ","NH1","NH2"]),
              ("CG","CD", ["NE","CZ","NH1","NH2"]), ("CD","NE", ["CZ","NH1","NH2"])]
    def from_chi(theta):
        p = template.clone(); lookup = {n: p[i] for i,n in enumerate(names)}
        for k, (origin, endpoint, downstream) in enumerate(chains):
            o = origin if torch.is_tensor(origin) else lookup[origin]; axis = lookup[endpoint] - o; axis = axis / torch.linalg.vector_norm(axis)
            ids = torch.tensor([name_to_i[n] for n in downstream], device=device); v = p[ids] - o
            c, s = torch.cos(theta[k]), torch.sin(theta[k])
            p[ids] = o + v*c + torch.cross(axis.expand_as(v), v, dim=1)*s + axis.expand_as(v)*(v@axis).unsqueeze(1)*(1-c)
            lookup = {n: p[i] for i,n in enumerate(names)}
        return p
    with torch.no_grad():
        probe = from_chi(torch.randn(4,device=device))
        sf0 = torch.mean(((calc.calc_fprotein(base_sf.index_copy(0,b_idx_t,probe),Return=True).abs()-f_obs)/f_obs.clamp_min(1)).square())
        tw_composite = float((sf0/scorer(base_tmol).sum().abs().clamp_min(1)).cpu())
    # Local test of whether the diffraction term itself points toward B.
    x0 = template.detach().clone().requires_grad_(True)
    dloss0 = torch.mean(((calc.calc_fprotein(base_sf.index_copy(0,b_idx_t,x0),Return=True).abs()-f_obs)/f_obs.clamp_min(1)).square())
    g0 = torch.autograd.grad(dloss0, x0)[0]
    direction = b_xyz - template
    alignment = float(((-g0 * direction).sum() / (g0.norm() * direction.norm()).clamp_min(1e-12)).detach().cpu())
    with torch.no_grad():
        path_loss = [float(torch.mean(((calc.calc_fprotein(base_sf.index_copy(0,b_idx_t,template+t*direction),Return=True).abs()-f_obs)/f_obs.clamp_min(1)).square()).cpu()) for t in (0., .1, .25, .5, 1.)]
    tw = 0.0  # density-only control requested after the composite failure
    def loss(theta):
        xyz = from_chi(theta); sf = torch.mean(((calc.calc_fprotein(base_sf.index_copy(0,b_idx_t,xyz),Return=True).abs()-f_obs)/f_obs.clamp_min(1)).square())
        return sf if tw == 0.0 else sf + tw*scorer(base_tmol.index_copy(1,tmol_idx,xyz.unsqueeze(0))).sum(), xyz
    out=[]
    for start in range(50):
        theta = (torch.randn(4,device=device)*1.2).requires_grad_(True); opt=torch.optim.Adam([theta],lr=.06)
        _, initial_xyz = loss(theta); initial_rb = torch.sqrt(torch.mean((initial_xyz.detach()-b_xyz).square())).item()
        for _ in range(100): opt.zero_grad(); l,_=loss(theta); l.backward(); opt.step()
        _, xyz=loss(theta); ra=torch.sqrt(torch.mean((xyz.detach()-template).square())).item(); rb=torch.sqrt(torch.mean((xyz.detach()-b_xyz).square())).item()
        out.append({"start":start,"initial_rmsd_to_B":initial_rb,"rmsd_to_A":ra,"rmsd_to_B":rb,"endpoint":"B" if rb<ra else "A"})
        if (start+1)%10==0: print(f"completed {start+1}/50")
    print(json.dumps({"control":"density_only","composite_tmol_weight":tw_composite,"gradient_alignment_to_B":alignment,"path_loss_t_0_0.1_0.25_0.5_1":path_loss,"hits_B":sum(r["endpoint"]=="B" for r in out),"results":out}))


@APP.local_entrypoint()
def main():
    torsion_sweep.remote()
