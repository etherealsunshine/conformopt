#!/usr/bin/env python3
"""Export clean-D1 benchmark models, maps, and path-projection diagnostics.

This is an inspection-only exporter.  It reads completed neutral starts and
endpoint artifacts; it does not optimize or modify the benchmark outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from clean_d1_benchmark import site_key, source_path
from run_d1_8d_sequential_poc import atom_window_indices
from run_d1_aprime_sequential import APrimeSequential
from run_d1_tier_a_flips import BACKBONE_NAMES, atom_local_index
from run_d6_tier2_realmap import make_map


def atom_element(name: str) -> str:
    letters = "".join(ch for ch in name if ch.isalpha()).upper()
    return (letters[:2] if letters[:2] in {"CL", "BR"} else letters[:1]) or "C"


def atom_line(serial, name, altloc, resname, chain, resnum, icode,
              xyz, occupancy, b_factor):
    return (
        f"ATOM  {serial:5d} {name:>4s}{altloc:1s}{resname:>3s} "
        f"{chain:1s}{resnum:4d}{icode:1s}   "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"{occupancy:6.2f}{b_factor:6.2f}          {atom_element(name):>2s}\n"
    )


def write_window(path: Path, window, coordinates: np.ndarray,
                 b_factors: np.ndarray, occupancy: float, title: str,
                 altloc: str = "") -> None:
    lines = [f"REMARK  {title}\n"]
    serial = 1
    cursor = 0
    for residue in window.residues:
        resname = str(residue.resn[0])
        chain = str(residue.chain[0])
        resnum = int(residue.id[0])
        icode = str(residue.id[1] or "")
        for index, name in enumerate(residue.name.tolist()):
            lines.append(atom_line(
                serial, str(name), altloc, resname, chain, resnum, icode,
                coordinates[cursor + index], float(occupancy),
                float(b_factors[cursor + index]),
            ))
            serial += 1
        cursor += len(residue.name)
        lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def write_structure(path: Path, structure, title: str) -> None:
    lines = [f"REMARK  {title}\n"]
    serial = 1
    for atom_index in range(len(structure.name)):
        lines.append(atom_line(
            serial, str(structure.name[atom_index]), "",
            str(structure.resn[atom_index]), str(structure.chain[atom_index]),
            int(structure.resi[atom_index]),
            str(structure.icode[atom_index]) if str(structure.icode[atom_index]) != "None" else "",
            structure.coor[atom_index], float(structure.q[atom_index]),
            float(structure.b[atom_index]),
        ))
        serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def write_map(xmap, values: np.ndarray, path: Path) -> None:
    output = type(xmap).zeros_like(xmap)
    output.array[:] = np.asarray(values, dtype=float)
    output.write_map_file(str(path))


def model_on_full_grid(base, masked_values: np.ndarray) -> np.ndarray:
    full = np.zeros_like(base.qfit.xmap.array, dtype=float)
    full[base.mask] = np.asarray(masked_values, dtype=float)
    return full


def path_metrics(base, coordinates: np.ndarray, label: str) -> dict[str, float | str]:
    a = np.asarray(base.a_backbone, dtype=float)
    b = np.asarray(base.b_backbone, dtype=float)
    point = np.asarray(base.central_backbone(coordinates), dtype=float)
    vector = b - a
    flat_vector = vector.reshape(-1)
    flat_delta = (point - a).reshape(-1)
    denominator = float(np.dot(flat_vector, flat_vector))
    t = float(np.dot(flat_delta, flat_vector) / denominator)
    projection = a + t * vector
    conventional = lambda x, y: float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=1))))
    separation = conventional(a, b)
    perpendicular = conventional(point, projection)
    along = float(t * separation)
    return {
        "label": label,
        "projection_fraction_A_to_B": t,
        "projected_distance_A": along,
        "perpendicular_rmsd_A": perpendicular,
        "rmsd_to_A_A": conventional(point, a),
        "rmsd_to_B_A": conventional(point, b),
    }


def qfit_candidates(site: dict[str, object], start_pdb: Path):
    from qfit.qfit import QFitOptions, QFitRotamericResidue
    from qfit.structure import Structure

    _, split = source_path(str(site["pdb_id"]))
    mtz = Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{site['pdb_id']}.mtz")
    xmap, _, _, _ = make_map(mtz)
    structure = Structure.fromfile(str(start_pdb)).extract("altloc", ("", "A"))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    residue = structure[str(site["chain"])].conformers[0][residue_id]
    options = QFitOptions()
    options.qp_solver = options.miqp_solver = "CVXPYSolver"
    qfit = QFitRotamericResidue(residue, structure, xmap, options)
    qfit._sample_backbone()
    return residue, [np.asarray(item, dtype=float) for item in qfit._coor_set]


def replace_central(base, window_coordinates: np.ndarray,
                    candidate_coordinates: np.ndarray,
                    candidate_residue) -> np.ndarray:
    output = np.asarray(window_coordinates, dtype=float).copy()
    window_indices = {
        str(name): int(index)
        for name, index in zip(base.central.name.tolist(), base.central_indices)
    }
    candidate_by_name = {
        str(name): np.asarray(coordinate, dtype=float)
        for name, coordinate in zip(candidate_residue.name.tolist(), candidate_coordinates)
    }
    for name, index in window_indices.items():
        if name in candidate_by_name:
            output[index] = candidate_by_name[name]
    return output


def fit_pair_map(base, windows, slots, b_offset, occupancies, intercept):
    models = base.model_density_batch(
        np.asarray(windows), slots=np.asarray(slots, dtype=int), b_offset=float(b_offset)
    )
    masked = np.asarray(occupancies, dtype=float) @ models + float(intercept)
    return model_on_full_grid(base, masked)


def export_site(site: dict[str, object], start_pdb: Path, run_root: Path,
                output: Path, summary_path: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    site_id = site_key(site)
    summary = json.loads(summary_path.read_text())
    row = next(item for item in summary["rows"] if item["site"] == site_id)
    endpoint_root = run_root / "sites" / site_id
    endpoint = np.load(endpoint_root / "aprime_specs" / "D_null_axis2_30deg" / "final_slots.npz")
    endpoint_result = json.loads(
        (endpoint_root / "aprime_specs" / "D_null_axis2_30deg" / "result.json").read_text()
    )

    runner = APrimeSequential(
        output / "runner_tmp", 80, 6, str(site["pdb_id"]), str(site["chain"]),
        int(site["resnum"]), renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device="cpu",
        start_pdb=start_pdb, b_factor_mode="single_conformer",
    )
    base = runner.base
    start = np.asarray(base.initial_window, dtype=float)
    deposited_a = base.window_for_deposited_a()
    deposited_b = base.window_for_deposited_b()
    aprime_1 = np.asarray(endpoint["slot1_window"], dtype=float)
    aprime_2 = np.asarray(endpoint["slot2_window"], dtype=float)

    candidate_residue, candidates = qfit_candidates(site, start_pdb)
    qfit_central_indices = [atom_local_index(candidate_residue, name) for name in BACKBONE_NAMES]
    truth_a = np.asarray(base.a_backbone, dtype=float)
    truth_b = np.asarray(base.b_backbone, dtype=float)
    qfit_rows = []
    for index, candidate in enumerate(candidates):
        central = np.asarray(candidate[qfit_central_indices], dtype=float)
        qfit_rows.append({
            "candidate": index,
            "rmsd_to_A_A": float(np.sqrt(np.mean((central - truth_a) ** 2))),
            "rmsd_to_B_A": float(np.sqrt(np.mean((central - truth_b) ** 2))),
        })
    best_a_index = min(range(len(qfit_rows)), key=lambda i: qfit_rows[i]["rmsd_to_A_A"])
    best_b_index = min(range(len(qfit_rows)), key=lambda i: qfit_rows[i]["rmsd_to_B_A"])
    qfit_a = replace_central(base, start, candidates[best_a_index], candidate_residue)
    qfit_b = replace_central(base, start, candidates[best_b_index], candidate_residue)

    b_start = np.asarray(base.window.b, dtype=float)
    b_a = np.asarray(base.b_factors_a, dtype=float)
    b_b = np.asarray(base.b_factors_b, dtype=float)
    # The explicit oracle pair uses deposited A/B B arrays and a fresh affine fit.
    oracle = APrimeSequential(
        output / "oracle_runner_tmp", 80, 6, str(site["pdb_id"]), str(site["chain"]),
        int(site["resnum"]), renderer_backend="torch", residual_scale_mode="none",
        map_scaler_structure="full", mask_scope="window", device="cpu",
        start_pdb=start_pdb, b_factor_mode="oracle_deposited",
    )
    oracle_fit = oracle.base.profile_affine_b_offset(
        oracle.base.target, [deposited_a, deposited_b], [0, 1]
    )
    oracle_map = (
        model_on_full_grid(oracle.base, oracle_fit["models"][0] * oracle_fit["weights"][0]
                           + oracle_fit["models"][1] * oracle_fit["weights"][1]
                           + oracle_fit["intercept"])
    )
    target = np.asarray(base.qfit.xmap.array, dtype=float).copy()
    aprime_map = fit_pair_map(
        base, [aprime_1, aprime_2], [0, 1], endpoint_result["final_b_offset_A2"],
        endpoint_result["final_occupancies"], endpoint_result["final_intercept"],
    )
    write_map(base.qfit.xmap, target, output / "target_fitted.ccp4")
    write_map(base.qfit.xmap, target - oracle_map, output / "target_minus_deposited_pair.ccp4")
    write_map(base.qfit.xmap, target - aprime_map, output / "target_minus_aprime_pair.ccp4")

    write_window(output / "neutral_start.pdb", base.window, start, b_start, 1.0, "Neutral single-conformer A-prime start")
    write_window(output / "deposited_A.pdb", base.window, deposited_a, b_a, 1.0, "Deposited A reference")
    write_window(output / "deposited_B.pdb", base.window, deposited_b, b_b, 1.0, "Deposited B reference")
    write_window(output / "aprime_slot1.pdb", base.window, aprime_1, b_start, float(endpoint_result["final_occupancies"][0]), "A-prime mirror-descent slot 1")
    write_window(output / "aprime_slot2.pdb", base.window, aprime_2, b_start, float(endpoint_result["final_occupancies"][1]), "A-prime mirror-descent slot 2")
    write_window(output / "qfit_best_to_A.pdb", base.window, qfit_a, b_start, 1.0, f"qFit candidate {best_a_index}, best to deposited A")
    write_window(output / "qfit_best_to_B.pdb", base.window, qfit_b, b_start, 1.0, f"qFit candidate {best_b_index}, best to deposited B")
    neighbours = base.qfit_structure if hasattr(base, "qfit_structure") else base.full_structure
    from run_d1_8d_sequential_poc import extract_window_neighbors
    frozen = extract_window_neighbors(neighbours, base.window, base.qfit.options.padding)
    write_structure(output / "frozen_neighbours.pdb", frozen, "Frozen neighbours used by the benchmark target")

    map_stats = {}
    for name, values in {
        "target_fitted.ccp4": target,
        "target_minus_deposited_pair.ccp4": target - oracle_map,
        "target_minus_aprime_pair.ccp4": target - aprime_map,
    }.items():
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        sigma = float(values.std(ddof=0))
        map_stats[name] = {
            "mean": mean, "sigma": sigma, "min": float(values.min()), "max": float(values.max()),
            "positive_3sigma": mean + 3.0 * sigma,
            "negative_3sigma": mean - 3.0 * sigma,
        }
    target_level = map_stats["target_fitted.ccp4"]["mean"] + map_stats["target_fitted.ccp4"]["sigma"]
    central_sel = f"chain {site['chain']} and resi {site['resnum']}"
    (output / "inspect.pml").write_text(f"""reinitialize
load neutral_start.pdb, neutral_start
load deposited_A.pdb, deposited_A
load deposited_B.pdb, deposited_B
load aprime_slot1.pdb, aprime_slot1
load aprime_slot2.pdb, aprime_slot2
load qfit_best_to_A.pdb, qfit_best_to_A
load qfit_best_to_B.pdb, qfit_best_to_B
load frozen_neighbours.pdb, frozen_neighbours
load target_fitted.ccp4, target_fitted
load target_minus_deposited_pair.ccp4, target_minus_deposited_pair
load target_minus_aprime_pair.ccp4, target_minus_aprime_pair

color grey70, neutral_start
color yellow, deposited_A
color orange, deposited_B
color cyan, aprime_slot1
color magenta, aprime_slot2
color green, qfit_best_to_A
color blue, qfit_best_to_B
color grey50, frozen_neighbours
hide everything, all
show sticks, ({central_sel}) and (deposited_A or deposited_B or aprime_slot1 or aprime_slot2 or qfit_best_to_A or qfit_best_to_B)
show lines, frozen_neighbours
set stick_radius, 0.16
set line_width, 1.0

select central, deposited_A and {central_sel}
isomesh target_mesh, target_fitted, {target_level:.6f}, central, carve=5.0
color lightblue, target_mesh
isomesh deposited_pos, target_minus_deposited_pair, {map_stats['target_minus_deposited_pair.ccp4']['positive_3sigma']:.6f}, central, carve=5.0
isomesh deposited_neg, target_minus_deposited_pair, {map_stats['target_minus_deposited_pair.ccp4']['negative_3sigma']:.6f}, central, carve=5.0
color green, deposited_pos
color red, deposited_neg
isomesh aprime_pos, target_minus_aprime_pair, {map_stats['target_minus_aprime_pair.ccp4']['positive_3sigma']:.6f}, central, carve=5.0
isomesh aprime_neg, target_minus_aprime_pair, {map_stats['target_minus_aprime_pair.ccp4']['negative_3sigma']:.6f}, central, carve=5.0
color palegreen, aprime_pos
color salmon, aprime_neg

orient central
zoom central, 10
""")

    metrics = [
        path_metrics(base, deposited_a, "deposited_A"),
        path_metrics(base, deposited_b, "deposited_B"),
        path_metrics(base, start, "neutral_start"),
        path_metrics(base, aprime_1, "aprime_slot1"),
        path_metrics(base, aprime_2, "aprime_slot2"),
        path_metrics(base, qfit_a, "qfit_best_to_A"),
        path_metrics(base, qfit_b, "qfit_best_to_B"),
    ]
    report = {
        "status": "complete",
        "site": site_id,
        "source_run": str(run_root),
        "neutral_start": str(start_pdb),
        "a_b_separation_A": float(row["qfit"]["A_B_separation_A"]),
        "aprime_final_rss": float(row["aprime"]["final_rss"]),
        "deposited_oracle_full_fit": {
            "rss": float(oracle_fit["rss"]),
            "weights": np.asarray(oracle_fit["weights"]).tolist(),
            "intercept": float(oracle_fit["intercept"]),
            "b_offset_A2": float(oracle_fit["b_offset"]),
        },
        "map_definition": "MapScaler full-structure scaled map after frozen neighbour and single-conformer sidechain subtraction; model residuals use the benchmark mask.",
        "map_stats_and_contours": map_stats,
        "path_projection_definition": "central N/CA/C/O point projected onto the deposited A-to-B line in 12-D Cartesian coordinate space; perpendicular_rmsd is the orthogonal residual.",
        "path_metrics": metrics,
        "qfit_candidates": qfit_rows,
        "qfit_best_to_A_candidate": best_a_index,
        "qfit_best_to_B_candidate": best_b_index,
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    # Keep the temporary runner directories out of the delivered bundle.
    for temp in (output / "runner_tmp", output / "oracle_runner_tmp"):
        if temp.exists():
            import shutil
            shutil.rmtree(temp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    site = next(item for item in manifest if site_key(item) == args.site)
    export_site(site, args.start, args.run_root, args.output, args.summary)


if __name__ == "__main__":
    main()
