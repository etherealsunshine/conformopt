#!/usr/bin/env python3
"""Run actual qFit tier-(a) on all frozen controls and stratify crankshaft visibility.

This is intentionally a thin wrapper around ``run_d1_tier_a_flips.analyze_site``:
the candidate generator remains qFit's actual ``_sample_backbone``.  The only
new measurements are deposited A-to-B atom displacements and an OLS
stratification of the already defined tier-(a) residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from qfit.structure import Structure

from run_d1_tier_a_flips import (
    BACKBONE_NAMES,
    analyze_site as actual_qfit_tier_a,
    atom_local_index,
    source_path,
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def atom_coordinate(residue, name: str) -> np.ndarray:
    return np.asarray(residue.coor[atom_local_index(residue, name)], dtype=float)


def deposited_displacements(site: dict[str, object]) -> dict[str, float | str]:
    """Measure A-to-B atom movement directly, independently of the sampler."""
    path, split = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_structure = Structure.fromfile(path).extract("altloc", ("", "A"))
    b_structure = Structure.fromfile(path).extract("altloc", ("", "B"))
    a_residue = a_structure[str(site["chain"])].conformers[0][residue_id]
    b_residue = b_structure[str(site["chain"])].conformers[0][residue_id]
    backbone = {
        name: float(np.linalg.norm(atom_coordinate(a_residue, name) - atom_coordinate(b_residue, name)))
        for name in BACKBONE_NAMES
    }
    visibility_atom = "O" if str(site["resname"]) == "GLY" else "CB"
    visibility_displacement = float(
        np.linalg.norm(atom_coordinate(a_residue, visibility_atom) - atom_coordinate(b_residue, visibility_atom))
    )
    oxygen_displacement = backbone["O"]
    if oxygen_displacement <= 1e-12:
        raise ValueError(f"O displacement is zero at {site['pdb_id']} {site['chain']}:{site['resnum']}")
    return {
        "source_split": split,
        "CB_visibility_atom": visibility_atom,
        "CB_or_O_deposited_displacement_A": visibility_displacement,
        "O_deposited_displacement_A": oxygen_displacement,
        "CB_visibility": visibility_displacement / oxygen_displacement,
        "N_deposited_displacement_A": backbone["N"],
        "CA_deposited_displacement_A": backbone["CA"],
        "C_deposited_displacement_A": backbone["C"],
        "backbone_max_atom_displacement_A": max(backbone.values()),
    }


def fit_visibility_model(rows: list[dict[str, object]]) -> dict[str, object]:
    usable = [row for row in rows if row.get("status") == "complete"]
    if len(usable) < 4:
        return {"n": len(usable), "status": "insufficient_rows"}
    deviation = np.asarray([float(row["backbone_max_atom_displacement_A"]) for row in usable])
    visibility = np.asarray([float(row["CB_visibility"]) for row in usable])
    residual_ratio = np.asarray([
        float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_max_atom_displacement_A"])
        for row in usable
    ])
    design = np.column_stack((np.ones(len(usable)), deviation, visibility))
    beta, _, _, _ = np.linalg.lstsq(design, residual_ratio, rcond=None)
    fitted = design @ beta
    errors = residual_ratio - fitted
    dof = len(usable) - design.shape[1]
    covariance = (errors @ errors / dof) * np.linalg.inv(design.T @ design)
    standard_error = np.sqrt(np.diag(covariance))
    t_value = beta[2] / standard_error[2]
    return {
        "status": "complete", "n": len(usable), "residual_degrees_of_freedom": int(dof),
        "model": "residual_over_max_backbone_deviation ~ intercept + max_backbone_deviation + CB_visibility",
        "intercept": float(beta[0]), "deviation_coefficient": float(beta[1]),
        "CB_visibility_coefficient": float(beta[2]),
        "CB_visibility_standard_error": float(standard_error[2]),
        "CB_visibility_t": float(t_value),
        "CB_visibility_two_sided_p": float(2.0 * student_t.sf(abs(t_value), dof)),
        "residual_standard_deviation": float(np.sqrt(errors @ errors / dof)),
    }


def make_scatter(rows: list[dict[str, object]], output: Path) -> None:
    usable = [row for row in rows if row.get("status") == "complete"]
    if not usable:
        return
    visibility = np.asarray([float(row["CB_visibility"]) for row in usable])
    residual_ratio = np.asarray([
        float(row["tier_a_min_central_backbone_rmsd_A"]) / float(row["backbone_max_atom_displacement_A"])
        for row in usable
    ])
    deviation = np.asarray([float(row["backbone_max_atom_displacement_A"]) for row in usable])
    figure, axis = plt.subplots(figsize=(8.4, 5.8))
    points = axis.scatter(visibility, residual_ratio, c=deviation, cmap="viridis", s=46, edgecolor="black", linewidth=.35)
    for x, y, row in zip(visibility, residual_ratio, usable):
        axis.annotate(str(row["site"]), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=5.7)
    figure.colorbar(points, ax=axis, label="deposited max {N, CA, C, O} displacement (Å)")
    axis.set(
        xlabel="Cβ-visibility = Cβ displacement / O displacement (O/O for Gly)",
        ylabel="tier-(a) residual / max backbone displacement",
        title="D1 controls: crankshaft-visibility stratification (raw points)",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=190)
    plt.close(figure)


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    checkpoints = args.output / "site_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    manifest_path = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")
    sites = [site for site in json.loads(manifest_path.read_text()) if site["panel"] == "nonflip_control"]
    if len(sites) != 33:
        raise RuntimeError(f"Expected 33 frozen non-flip controls, found {len(sites)}")
    atomic_json(args.output / "run_config.json", {
        "panel_manifest": str(manifest_path), "panel": "nonflip_control", "sites_requested": len(sites),
        "tier": "a_only_actual_qfit_sample_backbone", "candidate_generator": "QFitRotamericResidue._sample_backbone",
        "visibility": "CB displacement / O displacement; use O/O for GLY",
        "regression": "residual/max_backbone_displacement ~ max_backbone_displacement + CB_visibility",
    })
    rows: list[dict[str, object]] = []
    for site in sites:
        checkpoint = checkpoints / f"{site_key(site)}.json"
        if checkpoint.exists() and not (args.retry_errors and json.loads(checkpoint.read_text()).get("status") != "complete"):
            row = json.loads(checkpoint.read_text())
        else:
            try:
                # This call is deliberately shared with the flip runner.
                row = {**actual_qfit_tier_a(site), **deposited_displacements(site), "panel": "nonflip_control"}
            except Exception as error:
                row = {"site": site_key(site), "pdb_id": site["pdb_id"], "panel": "nonflip_control",
                       "status": "error", "error": repr(error), "error_traceback": traceback.format_exc()}
            atomic_json(checkpoint, row)
        rows.append(row)
        atomic_csv(args.output / "per_site.csv", rows)
        make_scatter(rows, args.output / "crankshaft_visibility_scatter.png")
        atomic_json(args.output / "progress.json", {
            "sites_recorded": len(rows), "sites_complete": sum(row.get("status") == "complete" for row in rows),
            "sites_error": sum(row.get("status") != "complete" for row in rows), "sites_total": len(sites),
            "last_site": site_key(site),
        })
    model = fit_visibility_model(rows)
    atomic_csv(args.output / "per_site.csv", rows)
    make_scatter(rows, args.output / "crankshaft_visibility_scatter.png")
    atomic_json(args.output / "summary.json", {
        "status": "complete", "sites_total": len(sites),
        "sites_complete": sum(row.get("status") == "complete" for row in rows),
        "sites_error": sum(row.get("status") != "complete" for row in rows),
        "visibility_regression": model,
    })
    atomic_json(args.output / "progress.json", {"status": "complete", "sites_total": len(sites)})


if __name__ == "__main__":
    main()
