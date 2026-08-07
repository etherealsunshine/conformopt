#!/usr/bin/env python3
"""Narrow follow-up diagnostics for the D6 native-BIC audit.

This script intentionally does not launch or extend Tier 2.  It (1) joins the
completed Tier 1 high-minor control failures to their deposited A/B backbone
and O-atom deviations, and (2) repeats one real-map site with qFit's full-map
MapScaler plus an explicit local B/overall-scale control.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from qfit import MapScaler
from qfit.structure import Structure
from qfit.xtal.transformer import get_transformer

from run_d6_tier1_synthetic import (
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
from run_d6_tier2_realmap import make_map


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["pdb_id"], row["chain"], row["resnum"], row["insertion_code"])


def is_success(row: dict[str, str]) -> bool:
    return int(row["pipeline_nconfs_ge_0.002"]) >= 2


def control_detectability(
    candidate_rows: list[dict[str, str]], tier1_rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_by_key = {key(row): row for row in candidate_rows}
    controls = [
        row for row in tier1_rows
        if row["panel"] == "nonflip_control"
        and min(float(row["true_occ_a"]), float(row["true_occ_b"])) >= 0.2
    ]
    joined: list[dict[str, object]] = []
    for row in controls:
        metadata = candidate_by_key[key(row)]
        joined.append({
            **row,
            "deposited_resolution_A": float(metadata["resolution"]),
            "max_backbone_deviation_A": float(metadata["max_backbone_deviation"]),
            "o_atom_deviation_A": float(metadata["o_deviation"]),
            "native_bic_success": is_success(row),
        })
    failures = [row for row in joined if not row["native_bic_success"]]
    details: list[dict[str, object]] = []
    for row in failures:
        detail = {
            "site": f"{row['pdb_id']}:{row['chain']}:{row['resnum']}{row['insertion_code']} {row['resname']}",
            "resolution_A": float(row["resolution_A"]),
            "deposited_resolution_A": row["deposited_resolution_A"],
            "max_backbone_deviation_A": row["max_backbone_deviation_A"],
            "o_atom_deviation_A": row["o_atom_deviation_A"],
            "selected_threshold": row["pipeline_selected_threshold"],
            "bic_tie_thresholds": row["pipeline_bic_tie_thresholds"],
        }
        for threshold in THRESHOLDS:
            detail[f"rss_t{threshold}"] = float(row[f"miqp_{threshold}_rss"])
        details.append(detail)

    resolution_groups: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in joined:
        resolution_groups[float(row["resolution_A"])].append(row)
    by_resolution = [
        {
            "resolution_A": resolution,
            "failures": sum(not row["native_bic_success"] for row in rows),
            "total": len(rows),
            "failure_rate": sum(not row["native_bic_success"] for row in rows) / len(rows),
        }
        for resolution, rows in sorted(resolution_groups.items())
    ]

    def group_stats(rows: list[dict[str, object]]) -> dict[str, float]:
        return {
            "count": len(rows),
            "median_max_backbone_deviation_A": float(np.median([row["max_backbone_deviation_A"] for row in rows])),
            "median_o_atom_deviation_A": float(np.median([row["o_atom_deviation_A"] for row in rows])),
        }

    outcome = np.asarray([not row["native_bic_success"] for row in joined], dtype=float)
    association = {}
    for name in ("max_backbone_deviation_A", "o_atom_deviation_A", "resolution_A"):
        values = np.asarray([row[name] for row in joined], dtype=float)
        association[name] = {
            "point_biserial_failure_correlation": point_biserial(outcome, values),
        }
    summary = {
        "high_minor_controls": len(joined),
        "failures": len(failures),
        "successes": len(joined) - len(failures),
        "failure_group": group_stats(failures),
        "success_group": group_stats([row for row in joined if row["native_bic_success"]]),
        "failure_rate_by_resolution": by_resolution,
        "failure_association": association,
    }
    return details, summary


def target_stats(target: np.ndarray) -> dict[str, float | int]:
    centered = target - target.mean()
    return {
        "n_masked_voxels": int(target.size),
        "mean": float(target.mean()),
        "std": float(target.std(ddof=0)),
        "sum_squares_about_zero": float(np.dot(target, target)),
        "centered_density_variance_sum": float(np.dot(centered, centered)),
    }


def point_biserial(outcome: np.ndarray, values: np.ndarray) -> float:
    """Pearson correlation of a 0/1 outcome and a continuous variable."""
    if outcome.std(ddof=0) == 0 or values.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(outcome, values)[0, 1])


def fit_site(xmap, resolution: float, residue_a, b_coor: np.ndarray, b_b: np.ndarray) -> dict[str, object]:
    local_map = xmap.extract(np.vstack([residue_a.coor, b_coor]), padding=8.0)
    # get_conformers_mask repurposes the transformer's XMap as a calculated-
    # density workspace. Preserve observed density first, as the Tier 2 audit
    # does before constructing the mask.
    observed_map = local_map.array.copy()
    transformer = get_transformer(
        "cctbx", residue_a, local_map, smax=1.0 / (2.0 * resolution), simple=False, em=False
    )
    transformer.initialize()
    mask = transformer.get_conformers_mask(
        [residue_a.coor, b_coor], 0.5 + resolution / 3.0
    )
    target = observed_map[mask].astype(float, copy=False)
    model_a, model_b = render(transformer, [residue_a.coor, b_coor], [residue_a.b, b_b], mask)
    models = np.vstack([model_a, model_b])
    direct_weights, direct_rss = qp_fit(target, models)
    one_raw = rss(target, model_a[None, :], np.asarray([1.0]))
    one_b_scaled, one_b_scaled_rss, b_multiplier, local_scale = one_state_b_optimization(
        target, transformer, residue_a.coor, residue_a.b, mask
    )
    if not np.isclose(one_raw, one_b_scaled, rtol=1e-9, atol=1e-7):
        raise RuntimeError("one-state raw RSS consistency check failed")
    # This matches the audit/qFit code path, including its local-XMap voxel-volume convention.
    local_voxel_volume = local_map.unit_cell.calc_volume() / local_map.array.size
    by_threshold = {}
    for threshold in THRESHOLDS:
        weights, objective = miqp_fit(target, models, threshold)
        native_rss = objective * local_voxel_volume
        bic, k = b_ic(native_rss, len(target), len(residue_a.coor), int(np.sum(weights >= MIN_OCCUPANCY)))
        by_threshold[str(threshold)] = {
            "weights": [float(value) for value in weights],
            "solver_rss_sum": float(objective),
            "native_bic_rss": float(native_rss),
            "bic": float(bic),
            "k": float(k),
        }
    selected_threshold = min(THRESHOLDS, key=lambda threshold: by_threshold[str(threshold)]["bic"])
    return {
        "target": target_stats(target),
        "local_xmap_voxel_volume_used_by_audit": float(local_voxel_volume),
        "direct_two_state": {"weights": [float(value) for value in direct_weights], "rss_sum": float(direct_rss)},
        "one_state": {
            "rss_sum_unscaled_occupancy_1": float(one_raw),
            "rss_sum_after_joint_local_B_and_scale": float(one_b_scaled_rss),
            "after_over_before_rss_ratio": float(one_b_scaled_rss / one_raw),
            "B_multiplier": float(b_multiplier),
            "overall_scale": float(local_scale),
        },
        "native_bic_by_threshold": by_threshold,
        "selected_threshold": float(selected_threshold),
    }


def scaling_diagnostic(candidate_rows: list[dict[str, str]], site_spec: str) -> dict[str, object]:
    pdb_id, chain, resnum_text = site_spec.upper().split(":")
    candidates = [
        row for row in candidate_rows
        if row["pdb_id"].upper() == pdb_id and row["chain"] == chain and row["resnum"] == resnum_text
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one site for {site_spec}, found {len(candidates)}")
    site = candidates[0]
    structure = Structure.fromfile(site["source_path"])
    residue_a = find_residue(structure, site["chain"], int(site["resnum"]), site["insertion_code"], "A")
    residue_b = find_residue(structure, site["chain"], int(site["resnum"]), site["insertion_code"], "B")
    by_name = {name: (coor, bf) for name, coor, bf in zip(residue_b.name, residue_b.coor, residue_b.b)}
    b_coor = np.asarray([by_name[name][0] for name in residue_a.name], dtype=float)
    b_b = np.asarray([by_name[name][1] for name in residue_a.name], dtype=float)
    mtz = Path(site["structure_factor_path"]).parent.parent / "mtz" / f"{site['pdb_id']}.mtz"

    raw_xmap, resolution, reflections, map_source = make_map(mtz)
    raw = fit_site(raw_xmap, resolution, residue_a, b_coor, b_b)
    scaled_xmap, scaled_resolution, _, _ = make_map(mtz)
    if not np.isclose(resolution, scaled_resolution):
        raise RuntimeError("map reconstruction resolution changed between control arms")
    map_scale, map_offset = MapScaler(scaled_xmap).scale(
        structure,
        radius=0.5 + resolution / 3.0,
        transformer="cctbx",
    )
    scaled = fit_site(scaled_xmap, resolution, residue_a, b_coor, b_b)
    return {
        "site": f"{site['pdb_id']}:{site['chain']}:{site['resnum']}{site['insertion_code']} {site['resname']}",
        "source_path": site["source_path"],
        "mtz": str(mtz),
        "map_source": map_source,
        "resolution_A": float(resolution),
        "n_reflections": reflections,
        "qfit_MapScaler": {
            "full_structure_mask_radius_A": float(0.5 + resolution / 3.0),
            "map_multiplicative_scale": float(map_scale),
            "map_additive_offset": float(map_offset),
        },
        "unscaled_audit_map": raw,
        "qfit_MapScaler_map": scaled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sites", type=Path, required=True)
    parser.add_argument("--tier1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-site", default="7UTC:A:52")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_csv(args.candidate_sites)
    tier1 = read_csv(args.tier1_root / "per_case.csv")
    detailed, detection_summary = control_detectability(candidates, tier1)
    atomic_csv(args.output_dir / "control_failures_detail.csv", detailed)
    scaling = scaling_diagnostic(candidates, args.scale_site)
    atomic_json(args.output_dir / "control_detectability_summary.json", detection_summary)
    atomic_json(args.output_dir / "one_site_scaling_diagnostic.json", scaling)
    atomic_json(args.output_dir / "summary.json", {
        "status": "complete",
        "tier2_extended": False,
        "control_failures_reported": len(detailed),
        "scale_site": scaling["site"],
    })


if __name__ == "__main__":
    main()
