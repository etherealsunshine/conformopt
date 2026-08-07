#!/usr/bin/env python3
"""Measure wider deposited A/B backbone-anchor agreement for D1 flip sites."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import numpy as np

from qfit.structure import Structure

from diagnose_d1_flip_closure import atom_coordinate, window_data
from run_d1_reachability import BACKBONE_NAMES


KS = (3, 4, 5, 6, 8, 10)


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


def atom_property(residue, name: str, property_name: str) -> float:
    selection = residue.select("name", name)
    if len(selection) != 1:
        raise ValueError(f"{residue.id} lacks unique {name}")
    global_index = int(selection[0])
    local = int(np.searchsorted(residue.selection, global_index))
    return float(getattr(residue, property_name)[local])


def backbone_properties(residue) -> tuple[float, float]:
    return (float(np.mean([atom_property(residue, name, "b") for name in BACKBONE_NAMES])),
            float(np.median([atom_property(residue, name, "q") for name in BACKBONE_NAMES])))


def coordinate_error_from_source(path: str) -> tuple[float | None, str]:
    """Read a PDB REMARK 3 coordinate error when present; mmCIF is often silent."""
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return None, "source_unreadable"
    for line in text.splitlines():
        if "ESTIMATED ERROR" in line.upper() and "COORDINATE" in line.upper():
            values = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            if values:
                return float(values[-1]), "PDB_REMARK3"
    return None, "not_reported_in_source"


def a_b_segments(site: dict[str, object]):
    a_structure, b_structure, a_residue, qfit, _, preflight, split = window_data(site)
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    a_index = qfit.segment.find(qfit.residue.id)
    # Segment membership compares residue IDs, not chain identity.  Searching
    # b_structure.segments directly can therefore select an identically
    # numbered segment from another chain (as it did for 8AJK B:231).  Scope
    # the lookup to the requested chain, exactly as strict_window does.
    b_chain = b_structure[str(site["chain"])].conformers[0]
    b_residue = b_chain[residue_id]
    b_segment = next(
        (segment for segment in b_chain.segments
         if any(residue.id == residue_id for residue in segment.residues)),
        None,
    )
    if b_segment is None:
        return a_structure, b_structure, qfit, None, a_index, None, preflight, split
    return a_structure, b_structure, qfit, b_segment, a_index, b_segment.find(b_residue.id), preflight, split


def row_for_k(site: dict[str, object], k: int, context=None) -> dict[str, object]:
    if context is None:
        context = a_b_segments(site)
    a_structure, _, qfit, b_segment, a_index, b_index, preflight, split = context
    path = f"/home/dev/qfit_unet_data/{split}/{str(site['pdb_id']).lower()}.pdb"
    error, error_source = coordinate_error_from_source(path)
    row = {"site": f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}", "k": k,
           "source_split": split, "coordinate_error_estimate_A": error, "coordinate_error_source": error_source,
           "qfit_preflight": preflight.get("reason", "complete")}
    if b_segment is None or a_index - k < 0 or a_index + k >= len(qfit.segment) or b_index - k < 0 or b_index + k >= len(b_segment):
        return {**row, "status": "anchor_unavailable"}
    a_minus, a_plus = qfit.segment.residues[a_index - k], qfit.segment.residues[a_index + k]
    b_minus, b_plus = b_segment.residues[b_index - k], b_segment.residues[b_index + k]
    if a_minus.id != b_minus.id or a_plus.id != b_plus.id:
        return {**row, "status": "A_B_anchor_residue_ids_differ", "a_minus_id": str(a_minus.id), "b_minus_id": str(b_minus.id), "a_plus_id": str(a_plus.id), "b_plus_id": str(b_plus.id)}
    def difference(a, b):
        return float(np.max(np.abs(np.asarray([atom_coordinate(a, name) for name in BACKBONE_NAMES]) - np.asarray([atom_coordinate(b, name) for name in BACKBONE_NAMES]))))
    a_minus_b, a_minus_q = backbone_properties(a_minus); b_minus_b, b_minus_q = backbone_properties(b_minus)
    a_plus_b, a_plus_q = backbone_properties(a_plus); b_plus_b, b_plus_q = backbone_properties(b_plus)
    minus, plus = difference(a_minus, b_minus), difference(a_plus, b_plus)
    return {**row, "status": "complete", "anchor_minus_difference_A": minus, "anchor_plus_difference_A": plus,
            "anchor_max_difference_A": max(minus, plus),
            "minus_residue": str(a_minus.id), "plus_residue": str(a_plus.id),
            "minus_A_backbone_mean_B": a_minus_b, "minus_B_backbone_mean_B": b_minus_b,
            "minus_A_backbone_median_occupancy": a_minus_q, "minus_B_backbone_median_occupancy": b_minus_q,
            "plus_A_backbone_mean_B": a_plus_b, "plus_B_backbone_mean_B": b_plus_b,
            "plus_A_backbone_median_occupancy": a_plus_q, "plus_B_backbone_median_occupancy": b_plus_q}


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson r, but never manufacture a number from a constant vector."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--site", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = Path("/home/dev/qfit_unet_data/qfit_audit/d6_tier1_native_bic_v1/panel_manifest.json")
    all_sites = [site for site in json.loads(manifest.read_text()) if site["panel"] == "flip_filter"]
    # Use the completed k=3 closure screen as the fixed 19-site denominator.
    closure_csv = Path("/home/dev/qfit_unet_data/qfit_audit/d1_flip_closure_anchors33_v2/per_site.csv")
    with closure_csv.open() as handle:
        testable = {row["site"] for row in csv.DictReader(handle) if row["status"] == "complete"}
    sites = [site for site in all_sites if f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}" in testable]
    if args.site:
        sites = [site for site in sites if f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}" == args.site]
    rows = []
    for site in sites:
        context = a_b_segments(site)
        site_rows = [row_for_k(site, k, context) for k in KS]
        rows.extend(site_rows)
        atomic_csv(args.output / "per_site_per_k.csv", rows)
        atomic_json(args.output / "progress.json", {"sites_complete": len(rows) // len(KS), "sites_total": len(sites), "last_site": site_rows[0]["site"]})
    summary = {"status": "complete", "fixed_k3_testable_sites": len(sites), "per_k": {}}
    for k in KS:
        complete = [row for row in rows if row["k"] == k and row["status"] == "complete"]
        values = np.asarray([row["anchor_max_difference_A"] for row in complete])
        mean_b = np.asarray([np.mean([
            row["minus_A_backbone_mean_B"], row["minus_B_backbone_mean_B"],
            row["plus_A_backbone_mean_B"], row["plus_B_backbone_mean_B"],
        ]) for row in complete])
        mean_occupancy = np.asarray([np.mean([
            row["minus_A_backbone_median_occupancy"], row["minus_B_backbone_median_occupancy"],
            row["plus_A_backbone_median_occupancy"], row["plus_B_backbone_median_occupancy"],
        ]) for row in complete])
        summary["per_k"][str(k)] = {
            "n_available": int(len(values)), "n_below_0p1_A": int(np.sum(values < .1)),
            "median_A": float(np.median(values)) if len(values) else None,
            "max_A": float(values.max()) if len(values) else None,
            "anchor_disagreement_vs_mean_anchor_B_Pearson_r": correlation(values, mean_b),
            "anchor_disagreement_vs_mean_anchor_occupancy_Pearson_r": correlation(values, mean_occupancy),
            "mean_anchor_B_median": float(np.median(mean_b)) if len(mean_b) else None,
            "mean_anchor_occupancy_median": float(np.median(mean_occupancy)) if len(mean_occupancy) else None,
        }
    per_site = {}
    for site in sorted({row["site"] for row in rows}):
        available = [row for row in rows if row["site"] == site and row["status"] == "complete" and row["anchor_max_difference_A"] < .1]
        per_site[site] = min((row["k"] for row in available), default=None)
    summary["smallest_requested_k_below_0p1_A_by_site"] = per_site
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", {"status": "complete", **summary})


if __name__ == "__main__":
    main()
