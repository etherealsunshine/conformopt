#!/usr/bin/env python3
"""Run the full 11-site signature-panel metric and failure analysis.

This consumes the frozen A-prime endpoints and Phenix outputs.  It does not
re-optimize any geometry.  The four held-out-RSS arms, fixed-region local
metrics, B-factor refit, coordinate/half-map noise floors, ensemble
trade-off, slot separation, χ checkpoints, and filtered clash audits are
written site-by-site so a long analysis is restartable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def refined_pdb(root: Path, site: str) -> Path:
    candidates = (
        root / site / "refined_001.pdb",
        root / site / "native" / "refined_001.pdb",
        root / site / "generated_rfree" / "refined_001.pdb",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no refined_001.pdb for {site} under {root}")


def paired_statistics(left: dict, right: dict) -> dict[str, object]:
    """Return left-minus-right paired fold statistics."""
    import numpy as np

    values = np.asarray([
        float(a["heldout_rss"]) - float(b["heldout_rss"])
        for a, b in zip(left["folds"], right["folds"])
    ], dtype=float)
    n = int(values.size)
    mean = float(values.mean()) if n else 0.0
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    critical = 2.7764451051977996 if n == 5 else 1.96
    return {
        "definition": "left held-out RSS minus right held-out RSS on paired blocked folds; positive favors right",
        "n_folds": n,
        "mean_difference_left_minus_right": mean,
        "standard_error": se,
        "95pct_ci": [mean - critical * se, mean + critical * se],
        "folds_positive": int(np.count_nonzero(values > 0.0)),
        "fold_differences": values.tolist(),
        "excludes_zero_at_95pct": bool(
            n > 1 and ((mean - critical * se > 0.0) or (mean + critical * se < 0.0))
        ),
    }


def filtered_clash(pdb_path: Path, output: Path, clashscore_bin: Path,
                   monomer_root: Path) -> dict[str, object]:
    from analyze_phenix_clash_audit import Connectivity, load_json_log, serious_contacts

    output.mkdir(parents=True, exist_ok=True)
    log = output / "clashscore.log"
    command = [str(clashscore_bin), str(pdb_path), "json=True", "keep_hydrogens=False"]
    with log.open("w") as handle:
        process = subprocess.run(command, cwd=output, stdout=handle,
                                 stderr=subprocess.STDOUT, check=False)
    if process.returncode:
        return {
            "status": "failed", "returncode": int(process.returncode),
            "pdb": str(pdb_path), "command": command, "log": str(log),
        }
    obj = load_json_log(log)
    summary = obj["summary_results"][""]
    raw_count = int(summary["num_clashes"])
    raw_score = float(summary["clashscore"])
    contacts = serious_contacts(obj, Connectivity(pdb_path, monomer_root))
    return {
        "status": "complete", "pdb": str(pdb_path),
        "raw_clashes": raw_count, "raw_clashscore": raw_score,
        "filtered_clashes": len(contacts),
        "filtered_clashscore": raw_score * len(contacts) / raw_count if raw_count else 0.0,
        "filter": "CCP4 monomer connectivity; exclude graph distance 1-2; hydrogens omitted",
        "worst_filtered_contacts": [
            {"a": c["a"], "b": c["b"], "overlap_A": c["overlap"]}
            for c in contacts[:10]
        ],
    }


def load_sites(panel_root: Path) -> list[dict[str, str]]:
    with (panel_root / "selected_sites.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def run_site(row: dict[str, str], args, imports) -> dict[str, object]:
    import numpy as np

    (
        PairInitialAPrime, parse_pair, score_pair, local_map_audit,
        displacement_classes, cbeta_index, map_context, observed_model_metrics,
        bfactor_refit, hydrogen_audit, coordinate_noise_floor, half_map_noise,
        voxel_tradeoff, blocked_splits,
    ) = imports
    lower = row["pdb_id"].lower()
    label = f"{row['pdb_id']}_{row['chain']}_{row['resname']}{row['residue_number']}"
    resnum = int(row["residue_number"])
    root = args.execution_root / label
    context_root = args.output_root / "contexts" / label
    runner = PairInitialAPrime(
        context_root, 1, 1, row["pdb_id"], row["chain"], resnum,
        renderer_backend="torch", map_scaler_structure="full", mask_scope="window",
        density_atom_scope="all", b_factor_mode="deposited_A_B", device=args.device,
    )
    qfit_npz = np.load(root / "qfit_input.npz", allow_pickle=False)
    qfit_pair = np.stack((qfit_npz["slot1_window"], qfit_npz["slot2_window"]))
    qfit_objective = read_json(root / "qfit_input_objective.json")
    qfit_occ = np.asarray(qfit_objective["occupancies"], dtype=float)
    qfit_pdb = args.panel_root / "inputs" / "qfit" / f"{row['pdb_id']}_qFit.pdb"
    qfit_pair_parsed, qfit_b, qfit_occ_parsed, qfit_parse = parse_pair(
        qfit_pdb, runner, qfit_pair,
    )
    # Use the exact qFit input coordinates/occupancies for the optimizer arm;
    # retain the PDB-parsed B factors, which are part of qFit's published model.
    qfit_b = np.asarray(qfit_b, dtype=float)
    qfit_occ = np.asarray(qfit_occ, dtype=float)
    final = np.load(root / "aprime_sidechain_chi_2" / "final_slots.npz", allow_pickle=False)
    aprime_pair = np.stack((final["slot1_window"], final["slot2_window"]))
    aprime_raw_pdb = root / "phenix_input.pdb"
    qfit_phenix_pdb = refined_pdb(args.qfit_phenix_root, label)
    aprime_phenix_pdb = refined_pdb(args.aprime_phenix_root, label)

    qfit_phx_pair, qfit_phx_b, qfit_phx_occ, qfit_phx_parse = parse_pair(
        qfit_phenix_pdb, runner, qfit_pair,
    )
    aprime_phx_pair, aprime_phx_b, aprime_phx_occ, aprime_phx_parse = parse_pair(
        aprime_phenix_pdb, runner, aprime_pair,
    )
    stages = {
        "qfit_raw": (qfit_pair, qfit_occ, qfit_b),
        "qfit_phenix": (qfit_phx_pair, qfit_phx_occ, qfit_phx_b),
        "aprime": (aprime_pair, qfit_occ, qfit_b),
        "aprime_phenix": (aprime_phx_pair, aprime_phx_occ, aprime_phx_b),
    }
    heldout = {
        name: score_pair(runner, pair, b_values, occ)
        for name, (pair, occ, b_values) in stages.items()
    }
    paired = {
        "raw_qfit_minus_aprime": paired_statistics(heldout["qfit_raw"], heldout["aprime"]),
        "phenix_qfit_minus_aprime": paired_statistics(heldout["qfit_phenix"], heldout["aprime_phenix"]),
        "qfit_raw_minus_qfit_phenix": paired_statistics(heldout["qfit_raw"], heldout["qfit_phenix"]),
        "aprime_raw_minus_aprime_phenix": paired_statistics(heldout["aprime"], heldout["aprime_phenix"]),
    }

    reference = args.panel_root / "inputs" / "source" / f"{lower}.pdb"
    mtz = args.panel_root / "inputs" / "map_mtz" / f"{lower}.mtz"
    data, false_flags, symmetry, observed, obs_map, mask, half_maps = map_context(
        reference, mtz, row["chain"], resnum - 3, resnum + 3,
    )
    model_paths = {
        "deposited": reference,
        "qfit_raw": qfit_pdb,
        "qfit_phenix": qfit_phenix_pdb,
        "aprime": aprime_raw_pdb,
        "aprime_phenix": aprime_phenix_pdb,
    }
    local = {
        name: observed_model_metrics(
            path, data, false_flags, symmetry, observed, obs_map, mask,
            row["chain"], resnum - 3, resnum + 3,
        )
        for name, path in model_paths.items()
    }
    bfactor = bfactor_refit(
        aprime_raw_pdb, data, false_flags, symmetry, observed, obs_map, mask,
        row["chain"], resnum - 3, resnum + 3,
    )
    noise = {
        "coordinate_noise_floor": coordinate_noise_floor(
            aprime_raw_pdb, data, false_flags, symmetry, observed, obs_map, mask,
            row["chain"], resnum - 3, resnum + 3,
        ),
        "half_map_noise_floor": half_map_noise(
            aprime_raw_pdb, data, false_flags, symmetry, observed, obs_map, mask,
            half_maps, row["chain"], resnum - 3, resnum + 3,
        ),
    }
    tradeoff = voxel_tradeoff(runner, qfit_pair, aprime_pair)

    b1 = np.load(root / "aprime_backbone_1" / "final_slots.npz", allow_pickle=False)
    c1 = np.load(root / "aprime_sidechain_chi" / "final_slots.npz", allow_pickle=False)
    b2 = np.load(root / "aprime_backbone_2" / "final_slots.npz", allow_pickle=False)
    stage_pairs = {
        "qfit_raw": qfit_pair,
        "aprime_backbone_1": np.stack((b1["slot1_window"], b1["slot2_window"])),
        "aprime_chi_1": np.stack((c1["slot1_window"], c1["slot2_window"])),
        "aprime_backbone_2": np.stack((b2["slot1_window"], b2["slot2_window"])),
        "aprime": aprime_pair,
        "qfit_phenix": qfit_phx_pair,
        "aprime_phenix": aprime_phx_pair,
    }
    cbi = cbeta_index(runner, resnum)
    separations = {
        name: float(np.linalg.norm(pair[0, cbi] - pair[1, cbi]))
        for name, pair in stage_pairs.items()
    }
    dep_pair, _dep_b, _dep_occ, dep_parse = parse_pair(reference, runner, qfit_pair)
    separations["deposited_control"] = float(np.linalg.norm(dep_pair[0, cbi] - dep_pair[1, cbi]))
    decomposition = {
        "qfit_to_aprime": displacement_classes(runner, qfit_pair, aprime_pair),
        "aprime_to_aprime_phenix": displacement_classes(runner, aprime_pair, aprime_phx_pair),
        "qfit_to_qfit_phenix": displacement_classes(runner, qfit_pair, qfit_phx_pair),
    }
    chi_reports = {}
    for directory in ("aprime_sidechain_chi", "aprime_sidechain_chi_2"):
        progress = read_json(root / directory / "chi_progress.json")
        chi_reports[directory] = {
            key: progress.get(key)
            for key in ("status", "phase", "nfev", "njev", "evaluations",
                        "jacobian_evaluations", "n_parameters", "blocks")
        }
    clash_models = {
        "deposited_control": reference,
        "qfit_raw": qfit_pdb,
        "qfit_phenix": qfit_phenix_pdb,
        "aprime": aprime_raw_pdb,
        "aprime_phenix": aprime_phenix_pdb,
    }
    clash = {
        name: filtered_clash(
            path, args.output_root / "clash" / label / name,
            args.clashscore_bin, args.monomer_root,
        )
        for name, path in clash_models.items()
    }
    return {
        "site": label,
        "panel_metadata": {
            "panel_role": row.get("panel_role"),
            "ratio": float(row["ratio"]),
            "qfit_cb_displacement_A": float(row["qfit_cb_displacement_A"]),
            "deposited_cb_displacement_A": float(row["deposited_cb_displacement_A"]),
            "resolution_A": float(row["resolution_A"]),
            "structure_factors_available": row.get("structure_factors_available"),
            "map_labels": row.get("map_labels"),
        },
        "heldout_rss_five_splits": heldout,
        "paired_statistics": paired,
        "fixed_region_local_metrics": local,
        "bfactor_refit": bfactor,
        "noise_floors": noise,
        "ensemble_tradeoff": tradeoff,
        "cbeta_slot_separation_A": separations,
        "coordinate_residual_decomposition": decomposition,
        "chi_stage_reports": chi_reports,
        "hydrogen_treatment": hydrogen_audit(
            runner, {"deposited": reference, "qfit_raw": qfit_pdb,
                     "qfit_phenix": qfit_phenix_pdb, "aprime": aprime_raw_pdb,
                     "aprime_phenix": aprime_phenix_pdb},
            row["chain"], resnum - 3, resnum + 3,
        ),
        "filtered_clash_audit": clash,
        "parse_reports": {
            "qfit_raw": qfit_parse, "qfit_phenix": qfit_phx_parse,
            "aprime_phenix": aprime_phx_parse, "deposited": dep_parse,
        },
        "model_paths": {name: str(path) for name, path in model_paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--qfit-phenix-root", type=Path, required=True)
    parser.add_argument("--aprime-phenix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--clashscore-bin", type=Path,
        default=Path("/home/dev/qfit_unet_data/phenix-2.2-6143/bin/phenix.clashscore"),
    )
    parser.add_argument(
        "--monomer-root", type=Path,
        default=Path("/home/dev/qfit_unet_data/phenix-2.2-6143/lib/python3.11/site-packages/chem_data/chemical_components"),
    )
    parser.add_argument("--site", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    sites = load_sites(args.panel_root)
    if args.site:
        selected = set(args.site)
        sites = [
            row for row in sites
            if f"{row['pdb_id']}_{row['chain']}_{row['resname']}{row['residue_number']}" in selected
        ]
    # These modules read the panel/output roots at import time.
    os.environ["ZENODO_PANEL_ROOT"] = str(args.panel_root)
    os.environ["ZENODO_METRICS_OUTPUT"] = str(args.output_root)
    os.environ["ZENODO_EXECUTION_ROOT"] = str(args.execution_root)
    os.environ["ZENODO_METRICS_DEVICE"] = args.device
    os.environ["CLEAN_D1_WIDER_ROOT"] = str(args.panel_root / "inputs" / "source")
    os.environ["D1_MTZ_ROOT"] = str(args.panel_root / "inputs" / "map_mtz")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import qfit  # noqa: F401
    from run_zenodo_qfit_aprime_metrics import (
        PairInitialAPrime, parse_pair, score_pair, local_map_audit,
        displacement_classes, cbeta_index,
    )
    from diagnose_zenodo_ensemble_tradeoff import (
        map_context, observed_model_metrics, bfactor_refit,
        coordinate_noise_floor, half_map_noise, voxel_tradeoff, hydrogen_audit,
    )
    from run_d1_aprime_leakage_corrected_cv import blocked_splits
    imports = (
        PairInitialAPrime, parse_pair, score_pair, local_map_audit,
        displacement_classes, cbeta_index, map_context, observed_model_metrics,
        bfactor_refit, hydrogen_audit, coordinate_noise_floor, half_map_noise,
        voxel_tradeoff, blocked_splits,
    )
    progress_path = args.output_root / "progress.json"
    existing_rows = []
    if args.resume and progress_path.is_file():
        try:
            existing_rows = read_json(progress_path).get("rows", [])
        except json.JSONDecodeError:
            existing_rows = []
    by_site = {row["site"]: row for row in existing_rows}
    atomic_json(progress_path, {
        "status": "running", "completed_sites": len(by_site),
        "total_sites": len(sites), "rows": list(by_site.values()),
    })
    for row in sites:
        label = f"{row['pdb_id']}_{row['chain']}_{row['resname']}{row['residue_number']}"
        if label in by_site:
            continue
        try:
            result = run_site(row, args, imports)
        except Exception as exc:
            result = {"site": label, "status": "failed", "error": repr(exc)}
        by_site[label] = result
        ordered = [by_site[key] for key in sorted(by_site)]
        atomic_json(progress_path, {
            "status": "running", "completed_sites": len(ordered),
            "total_sites": len(sites), "rows": ordered,
        })
    ordered = [by_site[key] for key in sorted(by_site)]
    atomic_json(args.output_root / "zenodo_signature_full_analysis.json", {
        "status": "complete", "rows": ordered,
    })
    atomic_json(progress_path, {
        "status": "complete", "completed_sites": len(ordered),
        "total_sites": len(sites), "rows": ordered,
    })
    print(json.dumps({
        "status": "complete", "completed_sites": len(ordered),
        "failed_sites": [row["site"] for row in ordered if row.get("status") == "failed"],
    }, indent=2))


if __name__ == "__main__":
    main()
