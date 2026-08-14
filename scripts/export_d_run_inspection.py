#!/usr/bin/env python3
"""Export the 5OHJ D-run endpoint pair and local inspection maps."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qfit.structure import Structure
from qfit.xtal.transformer import get_transformer
from run_d1_tier_a_flips import source_path
from run_d6_tier2_realmap import make_map


SITE = ("5OHJ", "A", 540)
RUN_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/slot_coordination_replay_physics_5ohj_20260810")
SOURCE_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/slot_coordination_5ohj_20260810_v2")
# Measured from MapScaler(xmap).scale(full_structure, radius=0.5 + d_min/3,
# transformer="cctbx") on the D-run's 5OHJ input.
FULL_MAP_SCALE = 0.3505020632284488
FULL_MAP_OFFSET = 0.41251207410241775


def atom_line(serial, record, atom_name, altloc, resname, chain, resnum,
              xyz, occupancy, b_factor, element, icode=""):
    return (f"{record:<6}{serial:5d} {atom_name:>4s}{altloc:1s}{resname:>3s} "
            f"{chain:1s}{resnum:4d}{icode:1s}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}          {element:>2s}\n")


def element_from_name(name):
    letters = "".join(ch for ch in name if ch.isalpha())
    return (letters[:2] if letters[:2].upper() in {"CL", "BR"}
            else letters[:1]).upper()


def read_pdb_altloc_residue(path, chain, resnum, altloc):
    atoms = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[21].strip() != chain or int(line[22:26]) != resnum:
            continue
        line_altloc = line[16].strip()
        if line_altloc != altloc:
            continue
        atoms[line[12:16].strip()] = {
            "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
            "q": float(line[54:60]), "b": float(line[60:66]),
        }
    if not atoms:
        raise ValueError(f"No altloc {altloc} residue {chain}:{resnum} in {path}")
    return atoms


def residue_atom_names(residue):
    return [str(name) for name in residue.name.tolist()]


def write_window_pdb(path, window, coordinate_sets, occupancies, b_values,
                     title):
    lines = [f"REMARK  {title}\n"]
    serial = 1
    for slot_label, coordinates, occupancy in coordinate_sets:
        for residue_index, residue in enumerate(window.residues):
            names = residue_atom_names(residue)
            for atom_index, name in enumerate(names):
                record = "ATOM"
                b_factor = float(b_values[residue_index][atom_index])
                element = element_from_name(name)
                lines.append(atom_line(
                    serial, record, name, slot_label, residue.resn[0],
                    residue.chain[0], int(residue.id[0]),
                    coordinates[residue_index][atom_index], occupancy,
                    b_factor, element, residue.id[1] or "",
                ))
                serial += 1
        lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def window_coordinates(window, flat_coordinates):
    result = []
    cursor = 0
    for residue in window.residues:
        count = len(residue.name)
        result.append(np.asarray(flat_coordinates[cursor:cursor + count], dtype=float))
        cursor += count
    if cursor != len(flat_coordinates):
        raise ValueError("window coordinate length does not match residues")
    return result


def write_neighbours(path, structure, residue, padding):
    lines = ["REMARK  Frozen neighbour atoms subtracted by qFit\n"]
    serial = 1
    neighbours = structure.extract_neighbors(residue, padding)
    for atom_index in range(len(neighbours.name)):
        lines.append(atom_line(
            serial, "ATOM", str(neighbours.name[atom_index]), "",
            str(neighbours.resn[atom_index]), str(neighbours.chain[atom_index]),
            int(neighbours.resi[atom_index]), neighbours.coor[atom_index],
            float(neighbours.q[atom_index]), float(neighbours.b[atom_index]),
            element_from_name(str(neighbours.name[atom_index])),
            str(neighbours.icode[atom_index]) if str(neighbours.icode[atom_index]) != "None" else "",
        ))
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def write_map(xmap, array, path):
    output = xmap.zeros_like(xmap)
    output.array[:] = np.asarray(array, dtype=float)
    output.write_map_file(str(path))


def make_numpy_renderer(runner, target_map):
    from density_denoiser.differentiable_renderer import CCTBX_N_GAUSSIAN
    shape = target_map.array.shape
    grid_indices = np.argwhere(np.ones(shape, dtype=bool))
    n_real = np.asarray(target_map.unit_cell_shape, dtype=float)
    fractional = (grid_indices[:, [2, 1, 0]] + target_map.offset[None, :]) / n_real[None, :]
    orthogonalization = np.asarray(target_map.unit_cell.calc_orthogonalization_matrix(), dtype=float).reshape(3, 3)
    grid_cart_np = fractional @ orthogonalization.T
    central_xyz = np.asarray(runner.initial_window[runner.central_indices], dtype=float)
    render_selection = np.min(
        np.sum((grid_cart_np[:, None, :] - central_xyz[None, :, :]) ** 2, axis=-1), axis=1
    ) <= 25.0
    selected_flat_indices = np.flatnonzero(render_selection)
    grid_cart = grid_cart_np[render_selection]
    image_shifts = np.asarray([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float) @ orthogonalization.T
    elements = [str(element).strip().upper() for element in runner.a_residue.e]
    coefficients = np.asarray([CCTBX_N_GAUSSIAN[element] for element in elements], dtype=float)
    b_factors = np.asarray(runner.b_factors, dtype=float)
    u_base = 0.03242230106617484

    def render(coordinates):
        central = np.asarray(coordinates[runner.central_indices], dtype=float)
        fractional_atoms = np.linalg.solve(orthogonalization, central.T).T
        fractional_atoms -= np.floor(fractional_atoms)
        atom_xyz = (fractional_atoms @ orthogonalization.T)[:, None, :] + image_shifts[None, :, :]
        atom_xyz = atom_xyz.reshape(-1, 3)
        atom_coefficients = np.tile(coefficients, (27, 1, 1))
        atom_b = np.tile(b_factors, 27)
        beta = atom_coefficients[:, :, 1] + atom_b[:, None] + 8.0 * np.pi**2 * u_base
        scale = atom_coefficients[:, :, 0] * (4.0 * np.pi / beta) ** 1.5
        values = np.zeros(len(grid_cart), dtype=float)
        for start in range(0, len(grid_cart), 256):
            points = grid_cart[start:start + 256]
            distance2 = ((points[:, None, :] - atom_xyz[None, :, :]) ** 2).sum(axis=-1)
            exponent = -4.0 * np.pi**2 * distance2[:, :, None] / beta[None, :, :]
            values[start:start + len(points)] = np.sum(scale[None, :, :] * np.exp(exponent), axis=(1, 2))
        flat = np.zeros(int(np.prod(shape)), dtype=float)
        flat[selected_flat_indices] = values
        return flat.reshape(shape)

    return render


def cropped_model_map(render, coordinates, occupancy, intercept):
    return (
        float(occupancy[0]) * render(coordinates[0])
        + float(occupancy[1]) * render(coordinates[1])
        + float(intercept)
    )


def build_full_map_runner(output):
    """Build the D-run target while avoiding qFit's full-structure clash graph."""
    # The D-run used full-structure MapScaler calibration and full-structure
    # neighbour subtraction.  qFit's optional clash graph is not needed for
    # export and is prohibitively large for this deposited model, so construct
    # the same map state with the clash graph disabled by using the A-only
    # structure for the residue object and applying the frozen neighbour
    # transformer explicitly below.
    _, split = source_path(SITE[0])
    print("export: full structure", flush=True)
    full_structure = Structure.fromfile(f"/home/dev/qfit_unet_data/{split}/{SITE[0].lower()}.pdb")
    print("export: full loaded", flush=True)
    a_structure = full_structure.extract("altloc", ("", "A"))
    print("export: A extracted", flush=True)
    a_residue = a_structure[SITE[1]].conformers[0][(SITE[2], "")]
    print("export: A residue", flush=True)
    pdb_path = Path(f"/home/dev/qfit_unet_data/{split}/{SITE[0].lower()}.pdb")
    b_atoms = read_pdb_altloc_residue(pdb_path, SITE[1], SITE[2], "B")
    mtz = Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{SITE[0]}.mtz")
    print("export: map", flush=True)
    xmap, resolution, _, _ = make_map(mtz)
    qfit = SimpleNamespace()
    qfit.xmap = xmap
    qfit.options = SimpleNamespace(padding=3.0, waters_clash=False)
    qfit._rmask = 0.5 + resolution / 3.0
    model_map = xmap.zeros_like(xmap)
    print("export: transformer", flush=True)
    qfit._transformer = get_transformer(
        "cctbx", a_residue, model_map, smax=1.0 / (2.0 * resolution),
        smin=None, simple=False, em=False,
    )
    qfit._transformer.initialize()
    print("export: transformer ready", flush=True)
    segment = next(segment for segment in a_structure.segments if a_residue in segment)
    index = segment.find(a_residue.id)
    window = segment[index - 3:index + 4]
    print("export: neighbours", flush=True)
    neighbours = full_structure.extract_neighbors(a_residue, qfit.options.padding)
    if not qfit.options.waters_clash:
        neighbours = neighbours.extract("resn", "HOH", "!=")
    print("export: neighbour transformer", flush=True)
    sub = get_transformer(
        "cctbx", neighbours, model_map, smax=1.0 / (2.0 * resolution), smin=None,
        simple=False, em=False,
    )
    sub.initialize()
    print("export: full scale", flush=True)
    scale, offset = FULL_MAP_SCALE, FULL_MAP_OFFSET
    xmap.array[:] = scale * xmap.array + offset
    print("export: neighbour density", flush=True)
    sub.reset(full=True)
    sub.density()
    np.maximum(sub.xmap.array, 0.0, out=sub.xmap.array)
    qfit.xmap.array -= sub.xmap.array
    print("export: map runner ready", flush=True)

    central = window.residues[3]
    central_indices = [
        int(np.searchsorted(window.selection, int(central.select("name", name)[0])))
        for name in central.name.tolist()
    ]
    initial_window = window.coor.copy()
    b_by_name = {name: atom["xyz"] for name, atom in b_atoms.items()}
    deposited_b_window = initial_window.copy()
    for local, name in zip(central_indices, central.name.tolist()):
        deposited_b_window[local] = b_by_name[name]
    return SimpleNamespace(
        full_structure=full_structure, a_residue=a_residue, b_residue=None,
        qfit=qfit, resolution=resolution, map_scale=float(scale), map_offset=float(offset),
        window=window, initial_window=initial_window, central=central,
        central_indices=central_indices,
        deposited_b_window=deposited_b_window,
        deposited_occupancies=np.array([
            float(np.median(a_residue.q)), float(np.median([atom["q"] for atom in b_atoms.values()]))
        ]), mask=None, target=None,
        b_factors=a_residue.b.copy(),
    )


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    runner = build_full_map_runner(output)
    print("export: runner built", flush=True)
    replay_root = RUN_ROOT
    axis2 = np.load(replay_root / "D_null_axis2_30deg" / "final_slots.npz")
    axis3 = np.load(replay_root / "D_null_axis3_30deg" / "final_slots.npz")
    axis2_result = json.loads((replay_root / "D_null_axis2_30deg" / "result.json").read_text())
    axis3_result = json.loads((replay_root / "D_null_axis3_30deg" / "result.json").read_text())
    print("export: endpoints loaded", flush=True)

    target_map = runner.qfit.xmap.extract(runner.window.coor, padding=5.0)
    target_map.array[:] = np.maximum(target_map.array, 0.0)
    print("export: target cropped", flush=True)
    render = make_numpy_renderer(runner, target_map)

    d2_slot1 = np.asarray(axis2["slot1_window"], dtype=float)
    d2_slot2 = np.asarray(axis2["slot2_window"], dtype=float)
    d3_slot1 = np.asarray(axis3["slot1_window"], dtype=float)
    d3_slot2 = np.asarray(axis3["slot2_window"], dtype=float)
    deposited_a = np.asarray(runner.initial_window, dtype=float)
    deposited_b = np.asarray(runner.deposited_b_window, dtype=float)

    deposited_models = [render(deposited_a), render(deposited_b)]
    print("export: deposited models", flush=True)
    d2_occ = np.asarray(axis2_result["final_occupancies"], dtype=float)
    d2_intercept = float(axis2_result["final_intercept"])
    d3_occ = np.asarray(axis3_result["final_occupancies"], dtype=float)
    d3_intercept = float(axis3_result["final_intercept"])
    dep_occ = np.asarray(runner.deposited_occupancies, dtype=float)
    dep_intercept = float(np.mean(target_map.array - dep_occ[0] * deposited_models[0] - dep_occ[1] * deposited_models[1]))

    dep_crop = cropped_model_map(render, (deposited_a, deposited_b), dep_occ, dep_intercept)
    d2_crop = cropped_model_map(render, (d2_slot1, d2_slot2), d2_occ, d2_intercept)
    d3_crop = cropped_model_map(render, (d3_slot1, d3_slot2), d3_occ, d3_intercept)
    target_map_array = target_map.array.copy()
    write_map(target_map, target_map_array, output / "target.ccp4")
    write_map(target_map, target_map_array - dep_crop, output / "diff_deposited.ccp4")
    write_map(target_map, target_map_array - d2_crop, output / "diff_d_axis2.ccp4")

    b_values = []
    cursor = 0
    for residue in runner.window.residues:
        count = len(residue.name)
        b_values.append(np.asarray(runner.window.b[cursor:cursor + count], dtype=float))
        cursor += count
    d2_coords = window_coordinates(runner.window, d2_slot1), window_coordinates(runner.window, d2_slot2)
    write_window_pdb(
        output / "d_axis2_30_slots.pdb", runner.window,
        [("A", d2_coords[0], float(d2_occ[0])), ("B", d2_coords[1], float(d2_occ[1]))],
        d2_occ, b_values, "D null axis2 30 degree slots; occupancies are final joint QP weights",
    )
    d3_coords = window_coordinates(runner.window, d3_slot1), window_coordinates(runner.window, d3_slot2)
    write_window_pdb(
        output / "d_axis3_30_slots.pdb", runner.window,
        [("A", d3_coords[0], float(d3_occ[0])), ("B", d3_coords[1], float(d3_occ[1]))],
        d3_occ, b_values, "D null axis3 30 degree slots; occupancies are final joint QP weights",
    )
    dep_a_coords = window_coordinates(runner.window, deposited_a)
    dep_b_coords = window_coordinates(runner.window, deposited_b)
    write_window_pdb(
        output / "deposited_ab.pdb", runner.window,
        [("A", dep_a_coords, float(dep_occ[0])), ("B", dep_b_coords, float(dep_occ[1]))],
        dep_occ, b_values, "Deposited A/B model; occupancies are final fitted QP weights",
    )
    write_neighbours(output / "neighbours.pdb", runner.full_structure, runner.a_residue, runner.qfit.options.padding)

    spacing = np.asarray(target_map.voxelspacing, dtype=float)
    map_stats = {}
    for name, values in {
        "target.ccp4": target_map_array,
        "diff_deposited.ccp4": target_map_array - dep_crop,
        "diff_d_axis2.ccp4": target_map_array - d2_crop,
    }.items():
        values = np.asarray(values, dtype=float)
        map_stats[name] = {
            "mean": float(values.mean()),
            "sigma": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    target_level = map_stats["target.ccp4"]["mean"] + map_stats["target.ccp4"]["sigma"]
    dep_pos = map_stats["diff_deposited.ccp4"]["mean"] + 3.0 * map_stats["diff_deposited.ccp4"]["sigma"]
    dep_neg = map_stats["diff_deposited.ccp4"]["mean"] - 3.0 * map_stats["diff_deposited.ccp4"]["sigma"]
    d2_pos = map_stats["diff_d_axis2.ccp4"]["mean"] + 3.0 * map_stats["diff_d_axis2.ccp4"]["sigma"]
    d2_neg = map_stats["diff_d_axis2.ccp4"]["mean"] - 3.0 * map_stats["diff_d_axis2.ccp4"]["sigma"]
    (output / "inspect.pml").write_text(f"""reinitialize
load deposited_ab.pdb, deposited_ab
load d_axis2_30_slots.pdb, d_axis2_30
load d_axis3_30_slots.pdb, d_axis3_30
load neighbours.pdb, neighbours
load target.ccp4, target
load diff_deposited.ccp4, diff_deposited
load diff_d_axis2.ccp4, diff_d_axis2

color yellow, deposited_ab
color cyan, d_axis2_30
color orange, d_axis3_30
color grey70, neighbours
show sticks, (deposited_ab or d_axis2_30 or d_axis3_30) and name N+CA+C+O
show lines, (deposited_ab or d_axis2_30 or d_axis3_30) and not name N+CA+C+O
show lines, neighbours
set stick_radius, 0.15
set line_width, 1.0

select central, deposited_ab and chain A and resi 540
isomesh target_mesh, target, {target_level:.6f}, central, carve=5.0
color blue, target_mesh
isomesh diff_deposited_pos, diff_deposited, {dep_pos:.6f}, central, carve=5.0
isomesh diff_deposited_neg, diff_deposited, {dep_neg:.6f}, central, carve=5.0
color green, diff_deposited_pos
color red, diff_deposited_neg
isomesh diff_d_axis2_pos, diff_d_axis2, {d2_pos:.6f}, central, carve=5.0
isomesh diff_d_axis2_neg, diff_d_axis2, {d2_neg:.6f}, central, carve=5.0
color green, diff_d_axis2_pos
color red, diff_d_axis2_neg

orient central
zoom central, 10
""")
    manifest = {
        "site": "5OHJ A SER540",
        "window": "SER540 +/- 3 residues (7 residues total)",
        "source_run": str(replay_root),
        "map_source": "MapScaler-scaled full map, then qFit frozen-neighbour subtraction, followed by an explicit floor at 0.0",
        "grid_spacing_A_xyz": spacing.tolist(),
        "box_padding_A": 5.0,
        "box_shape_zyx": list(target_map.array.shape),
        "maps": map_stats,
        "models": {
            "deposited_ab": {"occupancies": dep_occ.tolist(), "intercept": dep_intercept, "result_source": str(SOURCE_ROOT / "B_joint_deposited_B" / "result.json")},
            "d_axis2_30": {"occupancies": d2_occ.tolist(), "intercept": d2_intercept, "result_source": str(replay_root / "D_null_axis2_30deg" / "result.json")},
            "d_axis3_30": {"occupancies": d3_occ.tolist(), "intercept": d3_intercept, "result_source": str(replay_root / "D_null_axis3_30deg" / "result.json")},
        },
        "files": ["d_axis2_30_slots.pdb", "d_axis3_30_slots.pdb", "deposited_ab.pdb", "neighbours.pdb", "target.ccp4", "diff_deposited.ccp4", "diff_d_axis2.ccp4", "inspect.pml", "report.json"],
    }
    (output / "report.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.output)
