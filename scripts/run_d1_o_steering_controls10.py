#!/usr/bin/env python3
"""D1 control: run qFit's tier-(a) sampler with O as every steering atom."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np

from qfit.qfit import QFitOptions, QFitRotamericResidue
from qfit.structure import Structure

from run_d1_tier_a_flips import (
    BACKBONE_NAMES,
    analyze_site as actual_qfit_tier_a,
    atom_local_index,
    get_sampler_xmap,
    source_path,
)


BASELINE_ROOT = Path("/home/dev/qfit_unet_data/qfit_audit/d1_reachability_controls10_v2")
MANIFEST = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((np.asarray(first) - np.asarray(second)) ** 2, axis=1))))


def central_deposited_metrics(site: dict[str, object]) -> dict[str, float | str]:
    path, split = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
    a_residue = a_structure[str(site["chain"])].conformers[0][residue_id]
    b_residue = b_structure[str(site["chain"])].conformers[0][residue_id]
    a = np.asarray([a_residue.coor[atom_local_index(a_residue, name)] for name in BACKBONE_NAMES])
    b = np.asarray([b_residue.coor[atom_local_index(b_residue, name)] for name in BACKBONE_NAMES])
    return {
        "source_split": split,
        "deposited_A_to_B_central_backbone_rmsd_A": rmsd(a, b),
        "deposited_A_to_B_O_distance_A": float(np.linalg.norm(a[3] - b[3])),
        "deposited_A_to_B_max_backbone_atom_distance_A": float(np.max(np.linalg.norm(a - b, axis=1))),
    }


def anisou_status(site: dict[str, object]) -> dict[str, object]:
    """Check the actual deposited-A atoms qFit would use for directions."""
    path, _ = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    residue = structure[str(site["chain"])].conformers[0][residue_id]
    values: dict[str, object] = {}
    for atom_name in ("CB", "O"):
        try:
            atom = residue.extract("name", atom_name)
            anisou = atom.extract_anisous()[0]
            values[f"{atom_name}_has_anisou"] = True
            values[f"{atom_name}_anisou_trace"] = float(np.trace(np.asarray(anisou, dtype=float)))
        except AttributeError:
            values[f"{atom_name}_has_anisou"] = False
            values[f"{atom_name}_anisou_trace"] = None
    return values


def ols(rows: list[dict[str, object]], residual_key: str) -> dict[str, float | int]:
    x = np.asarray([float(row["deposited_A_to_B_max_backbone_atom_distance_A"]) for row in rows])
    y = np.asarray([float(row[residual_key]) for row in rows])
    design = np.column_stack((np.ones(len(rows)), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    error = y - fitted
    dof = len(rows) - 2
    covariance = error.dot(error) / dof * np.linalg.inv(design.T @ design)
    return {
        "n": len(rows), "intercept_A": float(beta[0]), "slope": float(beta[1]),
        "slope_standard_error": float(np.sqrt(covariance[1, 1])),
        "Pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "residual_standard_error_A": float(np.sqrt(error.dot(error) / dof)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = inspect.getsource(QFitRotamericResidue._sample_backbone)
    if 'atom_name = "O"' not in source:
        raise RuntimeError("qFit source is not in the required all-O steering configuration")
    manifest_rows = json.loads(MANIFEST.read_text())
    by_site = {site_key(site): site for site in manifest_rows}
    with (BASELINE_ROOT / "per_site.csv").open() as handle:
        baseline = [row for row in csv.DictReader(handle) if row["status"] == "complete"]
    if len(baseline) != 10:
        raise RuntimeError(f"Expected ten frozen baseline controls, found {len(baseline)}")
    sites = [by_site[row["site"]] for row in baseline]
    qfit_path = Path(inspect.getfile(QFitRotamericResidue))
    atomic_json(args.output / "run_config.json", {
        "panel": "exact ten complete controls from d1_reachability_controls10_v2",
        "baseline_root": str(BASELINE_ROOT), "candidate_generator": "QFitRotamericResidue._sample_backbone",
        "single_code_change": "atom_name = O for every residue", "qfit_source": str(qfit_path),
        "qfit_source_sha256": hashlib.sha256(qfit_path.read_bytes()).hexdigest(),
        "same_amplitudes_directions_IK_and_19_candidate_protocol": True,
    })
    rows: list[dict[str, object]] = []
    for site, baseline_row in zip(sites, baseline):
        try:
            row = {**actual_qfit_tier_a(site), **central_deposited_metrics(site), **anisou_status(site)}
            if row["status"] != "complete" or int(row["candidate_count_including_input"]) != 19:
                raise RuntimeError(f"expected 19 tier-(a) candidates, got {row.get('candidate_count_including_input')}")
            row["baseline_CB_central_backbone_rmsd_A"] = float(baseline_row["tier_a_central_backbone_rmsd_A"])
            row["baseline_CB_central_O_rmsd_A"] = float(baseline_row["tier_a_central_O_rmsd_A"])
            row["central_backbone_fraction_A_to_B_covered"] = 1.0 - float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["deposited_A_to_B_central_backbone_rmsd_A"])
            row["O_fraction_A_to_B_covered"] = 1.0 - float(row["tier_a_min_central_O_rmsd_A"]) / float(row["deposited_A_to_B_O_distance_A"])
            row["O_minus_CB_central_backbone_rmsd_A"] = float(row["tier_a_min_central_backbone_rmsd_A"]) - float(baseline_row["tier_a_central_backbone_rmsd_A"])
            row["O_minus_CB_central_O_rmsd_A"] = float(row["tier_a_min_central_O_rmsd_A"]) - float(baseline_row["tier_a_central_O_rmsd_A"])
            row["O_steering_makes_backbone_worse"] = bool(row["O_minus_CB_central_backbone_rmsd_A"] > 1e-12)
        except Exception as error:
            row = {"site": site_key(site), "status": "error", "error": repr(error), "traceback": traceback.format_exc()}
        rows.append(row)
        atomic_csv(args.output / "per_site.csv", rows)
        atomic_json(args.output / "progress.json", {"sites_recorded": len(rows), "sites_total": len(sites), "last_site": site_key(site)})
    complete = [row for row in rows if row["status"] == "complete"]
    if len(complete) != 10:
        raise RuntimeError(f"only {len(complete)}/10 controls completed")
    baseline_rows = [{**row, "deposited_A_to_B_max_backbone_atom_distance_A": row["deposited_A_to_B_max_backbone_atom_distance_A"], "baseline_residual": row["baseline_CB_central_backbone_rmsd_A"]} for row in complete]
    atomic_json(args.output / "summary.json", {
        "status": "complete", "sites_complete": len(complete),
        "O_steering_backbone_regression": ols(complete, "tier_a_min_central_backbone_rmsd_A"),
        "CB_baseline_regression_recomputed": ols(baseline_rows, "baseline_residual"),
        "median_O_steering_central_backbone_residual_A": float(np.median([float(row["tier_a_min_central_backbone_rmsd_A"]) for row in complete])),
        "sites_worse_under_O_steering": [row["site"] for row in complete if row["O_steering_makes_backbone_worse"]],
        "anisou_status_counts": {
            "both_CB_and_O": int(sum(row["CB_has_anisou"] and row["O_has_anisou"] for row in complete)),
            "CB_only": int(sum(row["CB_has_anisou"] and not row["O_has_anisou"] for row in complete)),
            "O_only": int(sum(not row["CB_has_anisou"] and row["O_has_anisou"] for row in complete)),
            "neither": int(sum(not row["CB_has_anisou"] and not row["O_has_anisou"] for row in complete)),
        },
    })
    atomic_json(args.output / "progress.json", {"status": "complete", "sites_total": len(sites), "sites_complete": len(complete)})


if __name__ == "__main__":
    main()
