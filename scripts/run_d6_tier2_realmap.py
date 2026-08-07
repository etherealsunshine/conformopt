#!/usr/bin/env python3
"""Conditional Tier 2 real-map proxy audit.

The deposited MTZs contain FP and FC,PHIC rather than deposited FWT,PHWT.
This script therefore constructs a clearly labeled 2Fo-Fc proxy using model
phases and reports the resulting one-sided phase-bias limitation.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
from cctbx import miller
from iotbx.reflection_file_reader import any_reflection_file
from scitbx.array_family import flex

from qfit.structure import Structure
from qfit.xtal.transformer import fft_map_coefficients, get_transformer
from qfit.xtal.spacegroups import SpaceGroup
from qfit.xtal.unitcell import UnitCell
from qfit.xtal.volume import GridParameters, Resolution, XMap

from run_d6_tier1_synthetic import (
    ABSOLUTE_CULL,
    MIN_OCCUPANCY,
    THRESHOLDS,
    b_ic,
    find_residue,
    miqp_fit,
    one_state_b_optimization,
    qp_fit,
    render,
    rss,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def make_map(mtz_path: Path):
    arrays = {array.info().label_string().replace(" ", ""): array for array in any_reflection_file(str(mtz_path)).as_miller_arrays()}
    if "FWT,PHWT" in arrays:
        map_coeffs = arrays["FWT,PHWT"]
        map_source = "deposited_FWT_PHWT"
    else:
        try:
            fp = arrays["FP,SIGFP"]
            fc = arrays["FC,PHIC"]
        except KeyError as exc:
            raise KeyError(f"{mtz_path} has reflection labels {sorted(arrays)}") from exc
        fp, fc = fp.common_sets(fc)
        fobs = np.asarray(fp.data(), dtype=float)
        fcalc = np.asarray(fc.data(), dtype=np.complex128)
        phase = np.divide(fcalc, np.abs(fcalc), out=np.ones_like(fcalc), where=np.abs(fcalc) > 1e-12)
        coefficients = 2.0 * fobs * phase - fcalc
        map_coeffs = miller.array(
            miller_set=fp.set(), data=flex.complex_double(coefficients.tolist())
        )
        map_source = "derived_2Fo_minus_Fc_from_FP_FC_PHIC"
    grid = fft_map_coefficients(map_coeffs, nyquist=2.0, transformer="cctbx")
    unit_cell = UnitCell.from_cctbx(map_coeffs.unit_cell())
    unit_cell.space_group = SpaceGroup.from_cctbx(map_coeffs.space_group_info())
    spacing = unit_cell.abc / np.asarray(grid.shape[::-1], dtype=float)
    xmap = XMap(
        grid,
        GridParameters(spacing),
        unit_cell=unit_cell,
        resolution=Resolution(high=map_coeffs.d_min()),
        hkl=np.asarray(list(map_coeffs.indices()), dtype=np.int32),
    )
    return xmap, float(map_coeffs.d_min()), int(map_coeffs.data().size()), map_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-list", type=Path, required=True)
    parser.add_argument("--cc-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accepted = {
        (row["pdb_id"], row["chain"], row["resnum"])
        for row in csv.DictReader(args.cc_gate.open())
        if row.get("status") == "complete" and row.get("b_acceptable_prespecified") == "True"
    }
    with args.site_list.open() as handle:
        sites = [row for row in csv.DictReader(handle) if (row["pdb_id"], row["chain"], row["resnum"]) in accepted and min(float(row["occupancy_a"]), float(row["occupancy_b"])) > 0.0]
    rows = []
    checkpoint_dir = args.output_dir / "site_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for index, site in enumerate(sites, start=1):
        checkpoint = checkpoint_dir / f"site_{index:04d}.json"
        if checkpoint.exists():
            rows.append(json.loads(checkpoint.read_text()))
            continue
        mtz = Path(site["structure_factor_path"]).parent.parent / "mtz" / f"{site['pdb_id']}.mtz"
        if not mtz.exists():
            row = {
                "pdb_id": site["pdb_id"], "chain": site["chain"], "resnum": site["resnum"], "resname": site["resname"],
                "status": "unavailable_no_mtz", "reason": str(mtz),
            }
            rows.append(row)
            atomic_json(checkpoint, row)
            atomic_json(args.output_dir / "progress.json", {"sites_complete": index, "sites_total": len(sites)})
            continue
        structure = Structure.fromfile(site["source_path"])
        residue_a = find_residue(structure, site["chain"], int(site["resnum"]), site["insertion_code"], "A")
        residue_b = find_residue(structure, site["chain"], int(site["resnum"]), site["insertion_code"], "B")
        names_a = residue_a.name.tolist()
        b_by_name = {name: (coor, bf) for name, coor, bf in zip(residue_b.name, residue_b.coor, residue_b.b)}
        if set(names_a) != set(b_by_name):
            raise ValueError(f"A/B atom-name mismatch at {site['pdb_id']} {site['chain']} {site['resnum']}")
        b_coor = np.asarray([b_by_name[name][0] for name in names_a], dtype=float)
        b_b = np.asarray([b_by_name[name][1] for name in names_a], dtype=float)
        try:
            xmap, resolution, n_reflections, map_source = make_map(mtz)
        except KeyError as exc:
            row = {
                "pdb_id": site["pdb_id"], "chain": site["chain"], "resnum": site["resnum"], "resname": site["resname"],
                "status": "unavailable_no_FC_PHIC", "reason": str(exc),
            }
            rows.append(row)
            atomic_json(checkpoint, row)
            atomic_json(args.output_dir / "progress.json", {"sites_complete": index, "sites_total": len(sites)})
            continue
        local_map = xmap.extract(np.vstack([residue_a.coor, b_coor]), padding=8.0)
        transformer = get_transformer("cctbx", residue_a, local_map, smax=1.0 / (2.0 * resolution), simple=False, em=False)
        transformer.initialize()
        target_map = local_map.array.copy()
        mask = transformer.get_conformers_mask([residue_a.coor, b_coor], 0.5 + resolution / 3.0)
        model_a, model_b = render(transformer, [residue_a.coor, b_coor], [residue_a.b, b_b], mask)
        target = target_map[mask].astype(float, copy=False)
        models = np.vstack([model_a, model_b])
        direct_weights, direct_rss = qp_fit(target, models)
        voxel_volume = local_map.unit_cell.calc_volume() / local_map.array.size
        fit_by_threshold = {}
        for threshold in THRESHOLDS:
            weights, objective = miqp_fit(target, models, threshold)
            fit_rss = objective * voxel_volume
            nconfs = int(np.sum(weights >= MIN_OCCUPANCY))
            fit_bic, k = b_ic(fit_rss, len(target), len(residue_a.coor), nconfs)
            fit_by_threshold[threshold] = (weights, fit_rss, nconfs, k, fit_bic)
        lowest_bic = min(fit[4] for fit in fit_by_threshold.values())
        ties = [threshold for threshold in THRESHOLDS if np.isclose(fit_by_threshold[threshold][4], lowest_bic, rtol=1e-12, atol=1e-10)]
        selected = min(THRESHOLDS, key=lambda threshold: fit_by_threshold[threshold][4])
        pipeline_weights, pipeline_rss, pipeline_nconfs, _, _ = fit_by_threshold[selected]
        one_before, one_after, b_multiplier, scale = one_state_b_optimization(target, transformer, residue_a.coor, residue_a.b, mask)
        row = {
            "pdb_id": site["pdb_id"], "chain": site["chain"], "resnum": site["resnum"], "resname": site["resname"],
            "status": "proxy_complete",
            "map_source": map_source,
            "resolution_A": resolution, "n_reflections": n_reflections, "n_masked_voxels": len(target),
            "direct_weight_a": float(direct_weights[0]), "direct_weight_b": float(direct_weights[1]), "direct_rss": direct_rss,
            "pipeline_selected_threshold": selected, "pipeline_weight_a": float(pipeline_weights[0]), "pipeline_weight_b": float(pipeline_weights[1]), "pipeline_rss": pipeline_rss, "pipeline_nconfs_ge_0.002": pipeline_nconfs, "pipeline_success_two_conformers": pipeline_nconfs >= 2, "pipeline_both_survive_absolute_0.09_cull": int(np.sum(pipeline_weights >= ABSOLUTE_CULL)) >= 2, "pipeline_bic_tied": len(ties) > 1, "pipeline_bic_tie_thresholds": ";".join(str(value) for value in ties),
            "one_state_rss_before_B_scale": one_before, "one_state_rss_after_B_scale": one_after, "one_state_B_multiplier": b_multiplier, "one_state_scale_after": scale,
        }
        for threshold, fit in fit_by_threshold.items():
            weights, fit_rss, nconfs, k, fit_bic = fit
            row.update({f"miqp_{threshold}_n": len(target), f"miqp_{threshold}_k": k, f"miqp_{threshold}_rss": fit_rss, f"miqp_{threshold}_bic": fit_bic, f"miqp_{threshold}_nconfs_ge_0.002": nconfs, f"miqp_{threshold}_weight_a": float(weights[0]), f"miqp_{threshold}_weight_b": float(weights[1])})
        rows.append(row)
        atomic_json(checkpoint, row)
        atomic_json(args.output_dir / "progress.json", {"sites_complete": index, "sites_total": len(sites)})
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "per_site.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "complete",
        "sites": len(rows),
        "cc_gated_sites": len(sites),
        "map_proxy": "deposited FWT/PHWT where available; otherwise a 2Fo-Fc-style complex proxy from FP,SIGFP and FC,PHIC",
        "phase_bias": "one-sided: FWT/PHWT and FC,PHIC phases are refinement/model-derived, so this is not an unbiased phase-free test; PDBe was unavailable from the pod",
        "source_columns": ["FWT,PHWT", "FP,SIGFP", "FC,PHIC"],
        "map_source_counts": {source: sum(row.get("map_source") == source for row in rows) for source in ("deposited_FWT_PHWT", "derived_2Fo_minus_Fc_from_FP_FC_PHIC")},
        "proxy_sites": sum(row["status"] == "proxy_complete" for row in rows),
        "unavailable_sites": sum(row["status"] != "proxy_complete" for row in rows),
        "native_bic_success_two_conformers": sum(row["status"] == "proxy_complete" and bool(row["pipeline_success_two_conformers"]) for row in rows),
        "both_survive_absolute_0.09_cull": sum(row["status"] == "proxy_complete" and bool(row["pipeline_both_survive_absolute_0.09_cull"]) for row in rows),
        "bic_tied_cases": sum(row["status"] == "proxy_complete" and bool(row["pipeline_bic_tied"]) for row in rows),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"status": "complete", "sites_complete": len(rows), "sites_total": len(sites)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
