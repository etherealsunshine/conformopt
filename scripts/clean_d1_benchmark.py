#!/usr/bin/env python3
"""Clean D1/qFit-vs-A-prime benchmark controller.

The module has three intentionally separate stages:

* ``screen`` measures site properties and reports site failures, scope
  exclusions, and harness errors separately;
* ``build-starts`` collapses altlocs, then delegates single-conformer
  refinement to Phenix and records neutral-start distances;
* ``run`` consumes only the frozen qualifying manifest and starts both methods
  from the same refined model.

No recovery result is read by the screen or start builder.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

BACKBONE_NAMES = ("N", "CA", "C", "O")


def rmsd(first, second):
    return float(np.sqrt(np.mean(np.sum((np.asarray(first) - np.asarray(second)) ** 2, axis=1))))


def source_path(pdb_id: str):
    wider_root = os.environ.get("CLEAN_D1_WIDER_ROOT")
    if wider_root:
        candidate = Path(wider_root) / "source" / f"{pdb_id.lower()}.pdb"
        if candidate.exists():
            return str(candidate), "wider"
    for split in ("train", "test"):
        candidate = Path(f"/home/dev/qfit_unet_data/{split}/{pdb_id.lower()}.pdb")
        if candidate.exists():
            return str(candidate), split
    raise FileNotFoundError(f"No source PDB for {pdb_id}")


SEPARATION_FLOOR_A = 0.30
BACKBONE_FRACTION_FLOOR = 0.80
DISCRIMINATING_FRACTION_FLOOR = 0.15
MAP_CORRELATION_FLOOR = 0.85
MINOR_OCCUPANCY_FLOOR = 0.25
DB_LIMIT_A2 = 15.0
RAMA_FLOOR = 0.02
RECOVERY_FRACTION = 0.30
MOTION_BINS = ((0.30, 0.80, "small"), (0.80, 1.50, "medium"), (1.50, None, "large"))


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
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def site_key(site: dict[str, object]) -> str:
    return f"{site['pdb_id']}_{site['chain']}_{site['resname']}{site['resnum']}"


def _atom_map(residue) -> dict[str, np.ndarray]:
    return {str(name): np.asarray(coordinate, dtype=float)
            for name, coordinate in zip(residue.name.tolist(), residue.coor)
            if str(name).strip()}


def _residue(site: dict[str, object], label: str):
    from qfit.structure import Structure

    path, _ = source_path(str(site["pdb_id"]))
    structure = Structure.fromfile(path).extract("altloc", ("", label))
    return structure, structure[str(site["chain"])].conformers[0][
        (int(site["resnum"]), str(site.get("insertion_code", "")))
    ]


def central_site_properties(site: dict[str, object]) -> dict[str, object]:
    """Measure central A/B geometry without using a recovery result."""
    a_structure, a = _residue(site, "A")
    b_structure, b = _residue(site, "B")
    amap, bmap = _atom_map(a), _atom_map(b)
    common = [name for name in amap if name in bmap and not name.startswith("H")]
    backbone = [name for name in BACKBONE_NAMES if name in amap and name in bmap]
    if not common or not backbone:
        raise ValueError("central A/B residue lacks matched heavy/backbone atoms")
    all_rmsd = rmsd(np.asarray([amap[name] for name in common]),
                    np.asarray([bmap[name] for name in common]))
    backbone_rmsd = rmsd(np.asarray([amap[name] for name in backbone]),
                         np.asarray([bmap[name] for name in backbone]))
    occ_a = float(np.median(a.q))
    occ_b = float(np.median(b.q))
    return {
        "backbone_fraction": float(backbone_rmsd / max(all_rmsd, 1e-12)),
        "central_allatom_A_B_rmsd_A": float(all_rmsd),
        "central_backbone_A_B_rmsd_A": float(backbone_rmsd),
        "minor_occupancy": float(min(occ_a, occ_b)),
        "occupancy_A": occ_a,
        "occupancy_B": occ_b,
        "matched_heavy_atoms": len(common),
    }


def strict_window_status(site: dict[str, object]) -> dict[str, object]:
    """Classify qFit's seven-residue scope exclusion without scoring density."""
    from qfit.structure import Structure

    path, _ = source_path(str(site["pdb_id"]))
    residue_id = (int(site["resnum"]), str(site.get("insertion_code", "")))
    structures = {
        label: Structure.fromfile(path).extract("altloc", ("", label))
        for label in ("A", "B")
    }
    segments = {}
    indices = {}
    for label, structure in structures.items():
        chain = structure[str(site["chain"])].conformers[0]
        segment = next((seg for seg in chain.segments
                        if any(residue.id == residue_id for residue in seg.residues)), None)
        if segment is None:
            return {"strict_window": False, "scope_exclusion": "chain_break", "scope_detail": f"{label} segment missing"}
        segments[label] = segment
        indices[label] = segment.find(residue_id)
    a_segment, b_segment = segments["A"], segments["B"]
    a_index, b_index = indices["A"], indices["B"]
    if a_index < 3 or a_index + 3 >= len(a_segment) or b_index < 3 or b_index + 3 >= len(b_segment):
        chain = structures["A"][str(site["chain"])].conformers[0]
        if len(chain.segments) > 1:
            cause = "chain_break"
        else:
            cause = "chain_terminus"
        return {"strict_window": False, "scope_exclusion": cause,
                "scope_detail": "qFit central +/-3 guard"}
    a_ids = [residue.id for residue in a_segment[a_index - 3:a_index + 4].residues]
    b_ids = [residue.id for residue in b_segment[b_index - 3:b_index + 4].residues]
    if a_ids != b_ids:
        if any(str(residue_id[1]) for residue_id in a_ids + b_ids):
            cause = "insertion_code"
        elif any(abs(int(left[0]) - int(right[0])) != 1
                 for left, right in zip(a_ids, a_ids[1:])):
            cause = "numbering_gap"
        else:
            cause = "chain_break"
        return {"strict_window": False, "scope_exclusion": cause,
                "scope_detail": "A/B seven-residue IDs do not align"}
    return {"strict_window": True, "scope_exclusion": None, "scope_detail": "complete and aligned"}


def full_structure_correlation(runner) -> float:
    from qfit.xtal.scaler import MapScaler
    from run_d6_tier2_realmap import make_map

    mtz = Path(f"/home/dev/qfit_unet_data/cache/{runner.split}/mtz/{runner.pdb_id}.mtz")
    xmap, resolution, _, _ = make_map(mtz)
    scaler = MapScaler(xmap)
    transformer = scaler._get_model_transformer(runner.full_structure, transformer="cctbx")
    transformer.mask(0.5 + resolution / 3.0)
    mask = scaler._model_map.array > 0
    transformer.reset(full=True)
    transformer.density()
    return float(np.corrcoef(xmap.array[mask], scaler._model_map.array[mask])[0, 1])


def reflection_map_capability(mtz: Path) -> tuple[bool, str]:
    """Check the labels needed by the fixed map constructor before fitting."""
    from iotbx.reflection_file_reader import any_reflection_file

    labels = {
        array.info().label_string().replace(" ", "")
        for array in any_reflection_file(str(mtz)).as_miller_arrays()
    }
    if "FWT,PHWT" in labels or ("FP,SIGFP" in labels and "FC,PHIC" in labels):
        return True, "amplitudes_and_phases"
    if ("FP,SIGFP" in labels and "FOM" in labels
            and any(label.startswith("HLA,HLB,HLC,HLD") for label in labels)):
        return True, "amplitudes_and_HL_phases"
    if "IMEAN,SIGIMEAN" in labels and "FC,PHIC" in labels:
        return True, "intensities_and_phases"
    return False, f"missing map-coefficient labels; available={sorted(labels)}"


def screen_one(site: dict[str, object], scratch: Path, device: str) -> dict[str, object]:
    properties = central_site_properties(site)
    scope = strict_window_status(site)
    path, split = source_path(str(site["pdb_id"]))
    row = {"status": "measured", "site": site_key(site), "source_split": split,
           "source_pdb": path, **site, **properties, **scope,
           "usable_reflection_data": bool(Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{site['pdb_id']}.mtz").is_file()),
           "motion_bin": next((name for low, high, name in MOTION_BINS
                                if properties["central_backbone_A_B_rmsd_A"] >= low and (high is None or properties["central_backbone_A_B_rmsd_A"] < high)), "below_floor")}
    mtz = Path(f"/home/dev/qfit_unet_data/cache/{split}/mtz/{site['pdb_id']}.mtz")
    map_capable, map_reason = reflection_map_capability(mtz) if row["usable_reflection_data"] else (False, "MTZ missing")
    row.update({"reflection_map_capability": map_capable, "reflection_map_detail": map_reason})
    if not map_capable:
        row.update({"reflection_failure": True, "category": "A_SITE_PROPERTY_FAILURE",
                    "criterion_failures": ["usable_reflection_data"], "all_criteria_pass": False})
        return row
    if not scope["strict_window"]:
        row.update({"scope_only": True, "category": "B_SCOPE_EXCLUSION",
                    "criterion_failures": [], "all_criteria_pass": False})
        return row
    from screen_deposited_two_state_support import (
        bic, classify, fixed_geometry_fits, flatten_fit, heldout_comparison,
    )
    from run_d1_8d_sequential_poc import SequentialBackbonePOC

    try:
        runner = SequentialBackbonePOC(
            str(site["pdb_id"]), str(site["chain"]), int(site["resnum"]),
            scratch / site_key(site), 0.25, 2.0, 0.0,
            renderer_backend="torch", map_scaler_structure="full",
            mask_scope="window", device=device, density_atom_scope="backbone",
            b_factor_mode="single_conformer",
        )
        fits = fixed_geometry_fits(runner)
        n_voxels = len(runner.target)
        one_label = min(("A_alone", "B_alone"), key=lambda label: float(fits[label]["rss"]))
        cv_rows = heldout_comparison(runner, one_label)
        density_a, density_b = np.asarray(fits["AB_free"]["models"], dtype=float)
        mean_density = np.mean((density_a + density_b) / 2.0)
        row.update({
            "discriminating_voxel_fraction": float((np.abs(density_a - density_b) > 0.05 * mean_density).mean()),
            "map_scaler_full_correlation": full_structure_correlation(runner),
            "two_state_classification": classify(fits[one_label], fits["AB_free"], cv_rows, n_voxels),
            "cv_two_minus_one_mean": float(np.mean([item["two_minus_one_heldout_rss"] for item in cv_rows])),
            "fitted_dB_A2": float(fits["AB_free"]["b_offset"]),
            "dB_converged": bool(fits["AB_free"].get("profile_converged", True)),
            "dB_bracketed": bool(fits["AB_free"].get("profile_bracketed", True)),
            "fit_b_factor_mode": runner.b_factor_mode,
            "qfit_b_factor_provenance": "QFitRotamericResidue.conformer.b from the supplied input structure; _sample_backbone appends the same conformer B vector to every candidate",
            "per_slot_b_factor_refinement": False,
        })
        for label, fit in fits.items():
            row.update(flatten_fit(label, fit, n_voxels, 3 if label.endswith("alone") else 4))
        row["criteria"] = {
            "usable_reflection_data": row["usable_reflection_data"],
            "backbone_fraction": row["backbone_fraction"] > BACKBONE_FRACTION_FLOOR,
            "discriminating_voxel_fraction": row["discriminating_voxel_fraction"] > DISCRIMINATING_FRACTION_FLOOR,
            "separation": row["central_backbone_A_B_rmsd_A"] > SEPARATION_FLOOR_A,
            "map_correlation": row["map_scaler_full_correlation"] > MAP_CORRELATION_FLOOR,
            "minor_occupancy": row["minor_occupancy"] >= MINOR_OCCUPANCY_FLOOR,
            "two_state_screen": row["two_state_classification"] == "TWO-STATE SUPPORTED",
            "dB": row["dB_converged"] and abs(row["fitted_dB_A2"]) <= DB_LIMIT_A2,
        }
        row["criterion_failures"] = [name for name, passed in row["criteria"].items() if not passed]
        row["category"] = "A_SITE_PROPERTY_FAILURE" if row["criterion_failures"] else "QUALIFIED"
        row["all_criteria_pass"] = not row["criterion_failures"]
        return row
    except Exception as error:
        return {**row, "category": "C_HARNESS_ERROR", "status": "error",
                "error_type": type(error).__name__, "error": repr(error)}


def sequential_counts(rows: list[dict[str, object]]) -> dict[str, object]:
    def as_bool(value: object) -> bool:
        return value is True or (isinstance(value, str) and value.strip().lower() == "true")

    for row in rows:
        if isinstance(row.get("criteria"), str):
            try:
                row["criteria"] = ast.literal_eval(str(row["criteria"]))
            except (SyntaxError, ValueError):
                row["criteria"] = {}
    measurable = [row for row in rows if row.get("category") in {"A_SITE_PROPERTY_FAILURE", "QUALIFIED"}]
    names = ("usable_reflection_data", "backbone_fraction", "discriminating_voxel_fraction",
             "separation", "map_correlation", "minor_occupancy", "two_state_screen", "dB")
    counts = []
    survivors = measurable
    for name in names:
        survivors = [row for row in survivors if bool(row.get("criteria", {}).get(name))]
        counts.append({"criterion": name, "surviving": len(survivors), "denominator": len(measurable)})
    bins = {}
    for _, _, name in MOTION_BINS:
        bins[name] = {
            "all_screened": sum(row.get("motion_bin") == name for row in rows),
            "measurable": sum(row in measurable and row.get("motion_bin") == name for row in rows),
            "qualified": sum(as_bool(row.get("all_criteria_pass")) and row.get("motion_bin") == name for row in rows),
        }
    return {"measurable_denominator": len(measurable), "counts": counts, "motion_bins": bins,
            "final_qualified_sites": [row["site"] for row in survivors]}


def run_screen(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    if args.limit is not None:
        manifest = manifest[:args.limit]
    rows = []
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "run_config.json", {
        "operation": "clean D1 screen", "manifest": str(args.manifest),
        "separation_floor_A": SEPARATION_FLOOR_A,
        "motion_bins": [{"low_A": low, "high_A": high, "name": name} for low, high, name in MOTION_BINS],
        "selection_does_not_read_recovery": True,
    })
    for site in manifest:
        row = screen_one(site, args.output / "scratch", args.device)
        rows.append(row)
        atomic_csv(args.output / "per_site.csv", rows)
        atomic_json(args.output / "progress.json", {"status": "running", "sites_recorded": len(rows), "total": len(manifest)})
    summary = {"status": "complete", "panel_sites": len(rows),
               "categories": {name: sum(row.get("category") == name for row in rows)
                              for name in ("A_SITE_PROPERTY_FAILURE", "B_SCOPE_EXCLUSION", "C_HARNESS_ERROR", "QUALIFIED")},
               "scope_exclusion_causes": {name: sum(row.get("scope_exclusion") == name for row in rows)
                                          for name in ("numbering_gap", "chain_break", "insertion_code", "chain_terminus")},
               "harness_error_types": {name: sum(row.get("error_type") == name for row in rows)
                                       for name in sorted({row.get("error_type") for row in rows if row.get("error_type")})},
               "criterion_counts": sequential_counts(rows),
               "rows": rows}
    atomic_json(args.output / "summary.json", summary)
    atomic_json(args.output / "progress.json", {"status": "complete", "sites_recorded": len(rows)})


def collapse_altlocs(source: Path, destination: Path) -> dict[str, object]:
    """Create a genuinely neutral single-conformer PDB by occupancy averaging."""
    # After altloc labels are blanked, source ANISOU records can no longer be
    # matched reliably to the collapsed atom labels.  The benchmark uses
    # isotropic B factors, so omit stale anisotropic records here.
    lines = [line for line in source.read_text(errors="replace").splitlines(True)
             if not line.startswith("ANISOU")]
    groups: dict[tuple[str, str, str, str, str, str], list[tuple[str, float]]] = {}
    order: list[tuple[str, str, str, str, str, str]] = []
    output_items: list[object] = []
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            output_items.append(line); continue
        alt = line[16].strip()
        if not alt:
            output_items.append(line); continue
        key = (line[0:6], line[12:16], line[17:21], line[21:22], line[22:27], line[76:78])
        if key not in groups:
            groups[key] = []; order.append(key); output_items.append(key)
        groups[key].append((line, float(line[54:60] or 1.0)))
    averaged = []
    for key in order:
        records = groups[key]
        weight = sum(max(occ, 0.0) for _, occ in records) or float(len(records))
        coords = np.asarray([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line, _ in records])
        occs = np.asarray([occ for _, occ in records])
        bvals = np.asarray([float(line[60:66] or 0.0) for line, _ in records])
        line = records[0][0]
        line = line[:16] + " " + line[17:]
        line = line[:30] + f"{np.average(coords[:,0], weights=occs):8.3f}{np.average(coords[:,1], weights=occs):8.3f}{np.average(coords[:,2], weights=occs):8.3f}" + line[54:]
        line = line[:54] + f"{1.0:6.2f}{np.average(bvals, weights=occs):6.2f}" + line[66:]
        averaged.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    replacements = dict(zip(order, averaged))
    destination.write_text("".join(replacements.get(item, item) if isinstance(item, tuple) else item
                                    for item in output_items))
    return {"source": str(source), "destination": str(destination), "collapsed_groups": len(groups),
            "definition": "occupancy-weighted coordinate/B average, occupancy reset to 1.00, altloc blank"}


def start_distances(start: Path, site: dict[str, object]) -> dict[str, object]:
    _, a_truth = _residue(site, "A"); _, b_truth = _residue(site, "B")
    from qfit.structure import Structure
    structure = Structure.fromfile(str(start)).extract("altloc", ("", "A"))
    residue = structure[str(site["chain"])].conformers[0][
        (int(site["resnum"]), str(site.get("insertion_code", "")))
    ]
    smap, amap, bmap = _atom_map(residue), _atom_map(a_truth), _atom_map(b_truth)
    names = [name for name in BACKBONE_NAMES if name in smap and name in amap and name in bmap]
    srmsd_a = rmsd(np.asarray([smap[name] for name in names]), np.asarray([amap[name] for name in names]))
    srmsd_b = rmsd(np.asarray([smap[name] for name in names]), np.asarray([bmap[name] for name in names]))
    separation = rmsd(np.asarray([amap[name] for name in names]), np.asarray([bmap[name] for name in names]))
    return {"start_rmsd_to_A_A": srmsd_a, "start_rmsd_to_B_A": srmsd_b,
            "start_fraction_of_AB": min(srmsd_a, srmsd_b) / max(separation, 1e-12),
            "start_landed_on_deposited_A": bool(srmsd_a <= 0.10 * separation),
            "start_meaningfully_nonzero": bool(srmsd_a > 0.05 and srmsd_b > 0.05),
            "A_B_separation_A": separation}


def run_start_builder(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    executable = shutil.which(args.phenix)
    rows = []
    for site in manifest:
        key = site_key(site); path, _ = source_path(str(site["pdb_id"]))
        root = args.output / "sites" / key; root.mkdir(parents=True, exist_ok=True)
        collapsed = root / "collapsed_single_conformer.pdb"
        collapse_record = collapse_altlocs(Path(path), collapsed)
        row = {"site": key, "collapse": collapse_record}
        if executable is None:
            row.update({"status": "error", "category": "C_HARNESS_ERROR", "error_type": "phenix_not_found",
                        "error": f"{args.phenix!r} not found; no neutral start was fabricated"})
        else:
            prefix = root / "phenix_refined"
            split = site.get("split")
            if split is None:
                _, split = source_path(str(site["pdb_id"]))
            mtz_path = site.get("mtz_path") or f"/home/dev/qfit_unet_data/cache/{split}/mtz/{site['pdb_id']}.mtz"
            command = [executable, str(collapsed), str(mtz_path),
                       f"output.prefix={prefix}", "strategy=individual_sites+individual_adp",
                       "main.number_of_macro_cycles=5"] + list(args.phenix_extra)
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
            outputs = sorted(root.glob("phenix_refined_*.pdb"))
            if completed.returncode or not outputs:
                row.update({"status": "error", "category": "C_HARNESS_ERROR", "error_type": "phenix_refine_failed",
                            "returncode": completed.returncode, "command": command,
                            "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]})
            else:
                refined = outputs[-1]
                row.update({"status": "complete", "refined_start": str(refined),
                            "command": command, **start_distances(refined, site)})
                atomic_json(root / "start.json", row)
        rows.append(row)
        atomic_json(args.output / "progress.json", {"status": "running", "completed": len(rows), "total": len(manifest)})
    atomic_csv(args.output / "per_site.csv", rows)
    atomic_json(args.output / "summary.json", {"status": "complete", "rows": rows})
    atomic_json(args.output / "progress.json", {"status": "complete", "completed": len(rows), "total": len(manifest)})


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("screen")
    screen.add_argument("--manifest", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--device", default="auto")
    screen.add_argument("--limit", type=int, default=None)
    start = sub.add_parser("build-starts")
    start.add_argument("--manifest", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)
    start.add_argument("--phenix", default="phenix.refine")
    start.add_argument("--phenix-extra", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "screen": run_screen(args)
    else: run_start_builder(args)


if __name__ == "__main__":
    main()
