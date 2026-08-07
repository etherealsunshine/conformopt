#!/usr/bin/env python3
"""Per-conformer real-space CC gate for the deposited A/B Tier-2 panel."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np

from qfit.structure import Structure
from qfit.xtal.transformer import get_transformer

from run_d6_tier1_synthetic import find_residue, render
from run_d6_tier2_realmap import make_map


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def cc(observed: np.ndarray, calculated: np.ndarray) -> float:
    if observed.size < 3 or np.std(observed) == 0 or np.std(calculated) == 0:
        return float("nan")
    return float(np.corrcoef(observed, calculated)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sites = [row for row in csv.DictReader(args.site_list.open()) if min(float(row["occupancy_a"]), float(row["occupancy_b"])) > 0.0]
    rows = []
    checkpoints = args.output_dir / "site_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for index, site in enumerate(sites, start=1):
        checkpoint = checkpoints / f"site_{index:04d}.json"
        if checkpoint.exists():
            rows.append(json.loads(checkpoint.read_text()))
            continue
        mtz = Path(site["structure_factor_path"]).parent.parent / "mtz" / f"{site['pdb_id']}.mtz"
        base = {"pdb_id": site["pdb_id"], "chain": site["chain"], "resnum": site["resnum"], "resname": site["resname"]}
        if not mtz.exists():
            row = {**base, "status": "unavailable_no_mtz"}
        else:
            try:
                xmap, resolution, reflections, map_source = make_map(mtz)
                structure = Structure.fromfile(site["source_path"])
                residue_a = find_residue(structure, site["chain"], int(site["resnum"]), site["insertion_code"], "A")
                residue_b = find_residue(structure, site["chain"], int(site["resnum"]), site["insertion_code"], "B")
                by_name = {name: (coor, bf) for name, coor, bf in zip(residue_b.name, residue_b.coor, residue_b.b)}
                b_coor = np.asarray([by_name[name][0] for name in residue_a.name], dtype=float)
                b_b = np.asarray([by_name[name][1] for name in residue_a.name], dtype=float)
                local_map = xmap.extract(np.vstack([residue_a.coor, b_coor]), padding=8.0)
                transformer = get_transformer("cctbx", residue_a, local_map, smax=1.0 / (2.0 * resolution), simple=False, em=False)
                transformer.initialize()
                observed_map = local_map.array.copy()
                radius = 0.5 + resolution / 3.0
                mask_a = transformer.get_conformers_mask([residue_a.coor], radius)
                model_a = render(transformer, [residue_a.coor], [residue_a.b], mask_a)[0]
                mask_b = transformer.get_conformers_mask([b_coor], radius)
                model_b = render(transformer, [b_coor], [b_b], mask_b)[0]
                cc_a = cc(observed_map[mask_a].astype(float), model_a)
                cc_b = cc(observed_map[mask_b].astype(float), model_b)
                delta = cc_b - cc_a
                row = {**base, "status": "complete", "map_source": map_source, "resolution_A": resolution, "n_reflections": reflections, "cc_a": cc_a, "cc_b": cc_b, "cc_b_minus_a": delta, "a_mask_voxels": int(mask_a.sum()), "b_mask_voxels": int(mask_b.sum()), "b_cc_below_0.50": cc_b < 0.50, "b_much_worse_than_a_delta_le_minus_0.10": delta <= -0.10, "b_acceptable_prespecified": cc_b >= 0.50 and delta > -0.10}
            except Exception as exc:  # checkpoint unavailable sites instead of aborting the gate
                row = {**base, "status": "unavailable_error", "reason": str(exc)}
        rows.append(row)
        atomic_json(checkpoint, row)
        atomic_json(args.output_dir / "progress.json", {"sites_complete": index, "sites_total": len(sites)})
    fields = sorted({key for row in rows for key in row})
    with (args.output_dir / "per_site_cc.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    complete = [row for row in rows if row["status"] == "complete"]
    summary = {"sites_requested": len(sites), "sites_with_usable_map": len(complete), "sites_flagged_b_cc_below_0.50": sum(bool(row["b_cc_below_0.50"]) for row in complete), "sites_flagged_b_delta_le_minus_0.10": sum(bool(row["b_much_worse_than_a_delta_le_minus_0.10"]) for row in complete), "sites_acceptable_prespecified": sum(bool(row["b_acceptable_prespecified"]) for row in complete), "map_sources": {source: sum(row["map_source"] == source for row in complete) for source in sorted({row["map_source"] for row in complete})}}
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"status": "complete", "sites_complete": len(rows), "sites_total": len(sites)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
