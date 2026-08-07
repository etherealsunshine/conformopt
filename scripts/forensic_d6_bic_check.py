#!/usr/bin/env python3
"""Forensic reconstruction of already-computed D6 Tier 1 cases.

This is a verification tool, not a new experiment: it replays the existing
case seeds and writes only arithmetic/mask/candidate diagnostics.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from qfit.structure import Structure
from run_d6_tier1_synthetic import (
    THRESHOLDS,
    b_ic,
    find_residue,
    local_transformer,
    miqp_fit,
    render,
)


RESOLUTIONS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SEED = 20260803


def case_specs(per_case: Path, site_list: Path) -> list[tuple[str, str, int]]:
    with site_list.open() as handle:
        sites = list(csv.DictReader(handle))
    with per_case.open() as handle:
        rows = list(csv.DictReader(handle))
    flip_rows = [row for row in rows if row["panel"] == "flip_filter"]
    # Prefer a case that survived the original +0.09 cull, then a selected-1
    # case, then a high-resolution case with a non-1 threshold.
    passed = next(row for row in flip_rows if row["resolution_A"] == "1.0" and row["pipeline_both_found_after_cull"] == "True")
    selected_one = next(row for row in flip_rows if row["resolution_A"] == "1.0" and row["pipeline_selected_threshold"] == "1.0" and float(row["true_occ_a"]) > 0.05 and float(row["true_occ_b"]) > 0.05)
    high_res = next(row for row in flip_rows if row["resolution_A"] == "1.8" and row["pipeline_selected_threshold"] != "1.0")
    return [(passed["pdb_id"], passed["chain"], int(passed["resnum"])),
            (selected_one["pdb_id"], selected_one["chain"], int(selected_one["resnum"])),
            (high_res["pdb_id"], high_res["chain"], int(high_res["resnum"])),]


def replay(site_rows: list[dict[str, str]], manifest: list[dict[str, str]], site_key: tuple[str, str, int], resolution: float):
    pdb, chain, resnum = site_key
    site = next(row for row in site_rows if row["pdb_id"] == pdb and row["chain"] == chain and int(row["resnum"]) == resnum)
    site_index = next(i for i, row in enumerate(manifest) if row["pdb_id"] == pdb and row["chain"] == chain and int(row["resnum"]) == resnum)
    resolution_index = RESOLUTIONS.index(resolution)
    rng = np.random.default_rng(SEED + site_index * len(RESOLUTIONS) + resolution_index + 1)
    structure = Structure.fromfile(site["source_path"])
    residue_a = find_residue(structure, chain, resnum, site["insertion_code"], "A")
    residue_b = find_residue(structure, chain, resnum, site["insertion_code"], "B")
    local_a, b_coor, b_b, transformer = local_transformer(residue_a, residue_b, resolution)
    a_coor = local_a.coor
    a_b = local_a.b
    occ = np.array([float(site["occupancy_a"]), float(site["occupancy_b"])], dtype=float)
    occ /= occ.sum()
    mask = transformer.get_conformers_mask([a_coor, b_coor], 0.5 + resolution / 3.0)
    density_a, density_b = render(transformer, [a_coor, b_coor], [a_b, b_b], mask)
    models = np.vstack([density_a, density_b])
    clean = occ @ models
    target = clean + rng.normal(0.0, 0.10 * float(np.std(clean)), size=clean.shape)
    fits = {}
    for threshold in THRESHOLDS:
        weights, fit_rss = miqp_fit(target, models, threshold)
        n = len(target)
        a_count = len(a_coor)
        k = 4.0 * a_count / threshold * 2.0
        fits[str(threshold)] = {
            "t_dmin": threshold,
            "n": n,
            "n_active_atoms": a_count,
            "k": k,
            "RSS": fit_rss,
            "n_ln_RSS_over_n": n * math.log(max(fit_rss / n, 1e-30)),
            "k_ln_n": k * math.log(n),
            "BIC": b_ic(fit_rss, n, a_count, threshold),
            "n_conformers_ge_0.002": int(np.sum(weights >= 0.002)),
            "n_conformers_ge_t_dmin": int(np.sum(weights >= threshold)),
            "occupancies": [float(x) for x in weights],
        }

    # Independent RSS calculation from the already-rendered density vectors.
    one_before_model = density_a
    one_before_rss = float(np.sum((target - one_before_model) ** 2))

    def evaluate_b(multiplier: float):
        model = render(transformer, [a_coor], [a_b * multiplier], mask)[0]
        scale = max(0.0, float(np.dot(target, model) / max(np.dot(model, model), 1e-30)))
        residual = target - scale * model
        return float(np.sum(residual * residual)), scale, model

    opt = minimize_scalar(lambda value: evaluate_b(float(value))[0], bounds=(0.25, 2.5), method="bounded")
    one_after_rss, one_scale, one_model = evaluate_b(float(opt.x))
    two_rss = float(np.sum((target - occ[0] * density_a - occ[1] * density_b) ** 2))

    # Nearest-grid-point atom inclusion diagnostic for the second conformer.
    xmap = transformer.xmap
    unit_shape = xmap.unit_cell_shape
    frac = b_coor @ xmap.unit_cell.orth_to_frac.T
    grid = np.rint(frac * unit_shape).astype(int)
    grid %= unit_shape
    inside = mask[grid[:, 2], grid[:, 1], grid[:, 0]]
    rmsd_aa = float(np.sqrt(np.mean(np.sum((a_coor - a_coor) ** 2, axis=1))))
    rmsd_ab = float(np.sqrt(np.mean(np.sum((a_coor - b_coor) ** 2, axis=1))))
    rmsd_ba = float(np.sqrt(np.mean(np.sum((b_coor - a_coor) ** 2, axis=1))))
    rmsd_bb = float(np.sqrt(np.mean(np.sum((b_coor - b_coor) ** 2, axis=1))))
    return {
        "site": f"{pdb}:{chain}:{resnum}",
        "resname": site["resname"],
        "resolution_A": resolution,
        "site_manifest_index_zero_based": site_index,
        "n_voxels": len(target),
        "n_active_atoms": len(a_coor),
        "mask_true_voxels": int(mask.sum()),
        "second_conformer_atoms_inside_mask_nearest_grid_fraction": float(np.mean(inside)),
        "candidate_count_passed_to_miqp": int(models.shape[0]),
        "candidate_rmsd_to_deposited": {"candidate_A_to_A": rmsd_aa, "candidate_A_to_B": rmsd_ab, "candidate_B_to_A": rmsd_ba, "candidate_B_to_B": rmsd_bb},
        "threshold_dump": fits,
        "independent_rss": {"one_state_B_scale_before": one_before_rss, "one_state_B_scale_after": one_after_rss, "optimized_B_multiplier": float(opt.x), "optimized_scale": one_scale, "two_state_deposited_occupancies": two_rss, "ratio_two_over_one_after": two_rss / one_after_rss},
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-case", type=Path, required=True)
    parser.add_argument("--site-list", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    site_rows = list(csv.DictReader(args.site_list.open()))
    manifest = json.loads(args.manifest.read_text())
    specs = case_specs(args.per_case, args.site_list)
    selected = [(specs[0], 1.0), (specs[1], 1.0), (specs[2], 1.8)]
    result = [replay(site_rows, manifest, site, resolution) for site, resolution in selected]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
