#!/usr/bin/env python3
"""Run the frozen same-start qFit sampler and A-prime axis-2 benchmark."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from clean_d1_benchmark import atomic_json, site_key, source_path
from fit_provenance import assert_heldout_geometry_provenance
from run_d1_slot_coordination import build_specs, worker
from run_d1_tier_a_flips import BACKBONE_NAMES, atom_local_index
from run_d6_tier2_realmap import make_map


def qfit_sampler(site: dict[str, object], start_pdb: Path) -> dict[str, object]:
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
    if len(qfit._coor_set) != 19:
        raise RuntimeError(f"qFit returned {len(qfit._coor_set)} candidates, expected 19")
    truth_path, _ = source_path(str(site["pdb_id"]))
    truth_a = Structure.fromfile(truth_path).extract("altloc", ("", "A"))[str(site["chain"])].conformers[0][residue_id]
    truth_b = Structure.fromfile(truth_path).extract("altloc", ("", "B"))[str(site["chain"])].conformers[0][residue_id]
    indices = [atom_local_index(residue, name) for name in BACKBONE_NAMES]
    a = np.asarray([truth_a.coor[atom_local_index(truth_a, name)] for name in BACKBONE_NAMES])
    b = np.asarray([truth_b.coor[atom_local_index(truth_b, name)] for name in BACKBONE_NAMES])
    def metric(candidate, truth):
        return float(np.sqrt(np.mean(np.sum((candidate - truth) ** 2, axis=1))))
    rows = []
    for index, coordinates in enumerate(qfit._coor_set):
        central = np.asarray(coordinates[indices], dtype=float)
        rows.append({"candidate": index, "rmsd_to_A_A": metric(central, a), "rmsd_to_B_A": metric(central, b)})
    separation = metric(a, b)
    best_a = min(rows, key=lambda row: row["rmsd_to_A_A"])
    best_b = min(rows, key=lambda row: row["rmsd_to_B_A"])
    candidate0_b = np.asarray(qfit._bs[0], dtype=float)
    input_b = np.asarray(residue.b, dtype=float)
    return {"status": "complete", "candidate_count": len(rows), "candidates": rows,
            "best_to_A": best_a, "best_to_B": best_b, "A_B_separation_A": separation,
            "b_factor_provenance": "qFit inherits the single-conformer input residue B vector; no per-slot B refinement",
            "candidate_0_b_factors": candidate0_b.tolist(),
            "candidate_0_b_equals_input_residue": bool(np.array_equal(candidate0_b, input_b)),
            "input_residue_b_factors": input_b.tolist()}


def classify(slot_rmsds: list[dict[str, float]], separation: float) -> str:
    threshold = RECOVERY_FRACTION * separation
    assignments = ((slot_rmsds[0]["to_A"], slot_rmsds[1]["to_B"]),
                   (slot_rmsds[0]["to_B"], slot_rmsds[1]["to_A"]))
    if any(first <= threshold and second <= threshold for first, second in assignments):
        return "RECOVERED"
    if any(first <= threshold or second <= threshold for first, second in assignments):
        return "PARTIAL"
    return "FAILED"


def guarded_aprime_cv(site: dict[str, object], start_pdb: Path, root: Path,
                      flip_root: Path, device: str = "auto",
                      slot2_occupancy_floor: float = 0.0,
                      slot2_floor_outer_updates: int = 0,
                      occupancy_scheme: str = "qp",
                      mirror_eta: float = 0.0,
                      mirror_tau: float = 0.0) -> dict[str, object]:
    """Fit every A-prime parameter on five training masks, then score held out.

    The two B treatments are separate fits.  In particular, the zero-offset
    treatment does not reuse the fitted-offset geometry.  This is deliberately
    slower, but makes the provenance assertion meaningful.
    """
    from occupancy_selection import solve_affine_qp
    from run_d1_aprime_leakage_corrected_cv import blocked_splits
    from run_d1_aprime_sequential import APrimeSequential
    from run_d1_8d_sequential_poc import SequentialBackbonePOC

    site_tuple = (str(site["pdb_id"]), str(site["chain"]), int(site["resnum"]))
    base_runner = APrimeSequential(
        root / "cv_base", 80, 6, *site_tuple, renderer_backend="torch",
        residual_scale_mode="none", map_scaler_structure="full",
        mask_scope="window", device=device, start_pdb=start_pdb,
        b_factor_mode="single_conformer",
    )
    folds = blocked_splits(base_runner.base)
    all_rows = []
    oracle_runner = SequentialBackbonePOC(
        *site_tuple, root / "oracle_bound", 0.25, 2.0, 0.0,
        renderer_backend="torch", map_scaler_structure="full",
        mask_scope="window", device=device, density_atom_scope="backbone",
        b_factor_mode="oracle_deposited",
    )
    oracle_rows = []
    for fold, (train, test, direction) in enumerate(folds):
        windows = [oracle_runner.initial_window, oracle_runner.window_for_deposited_b()]
        profiled = oracle_runner.profile_affine_b_offset(
            oracle_runner.target[train], windows, [0, 1], voxel_indices=train
        )
        delta_b = float(profiled["b_offset"])
        models = oracle_runner.model_density_batch(
            windows, slots=np.array((0, 1)), b_offset=delta_b
        )
        weights, intercept, train_rss = solve_affine_qp(
            oracle_runner.target[train], models[:, train]
        )
        heldout = oracle_runner.target[test] - weights @ models[:, test] - intercept
        oracle_rows.append({
            "fold": fold, "split_direction": direction.tolist(),
            "delta_B_A2": delta_b, "training_rss": float(train_rss),
            "heldout_rss": float(np.square(heldout).sum()),
            "occupancies_refit_on_training": weights.tolist(),
            "intercept_refit_on_training": float(intercept),
            "bound": "deposited A/B geometry and deposited A/B B arrays; unavailable prospectively",
        })
    atomic_json(root / "oracle_bound_cv_summary.json", {
        "status": "complete", "bound": "oracle deposited geometry and B arrays",
        "unavailable_prospectively": True, "folds": oracle_rows,
        "mean_all_five_heldout_rss": float(np.mean([r["heldout_rss"] for r in oracle_rows])),
        "mean_excluding_fold_0_heldout_rss": float(np.mean([r["heldout_rss"] for r in oracle_rows[1:]])),
        "fold_0_known_unknown": True,
    })
    for treatment in ("dB_fitted", "dB_zero"):
        treatment_rows = []
        for fold, (train, test, direction) in enumerate(folds):
            fold_root = root / "cv" / treatment / f"fold_{fold}"
            fold_root.mkdir(parents=True, exist_ok=True)
            specs = build_specs(
                fold_root / "specs", flip_root, site=site_tuple,
                mask_scope="window", rama_floor=0.02,
                start_pdb=start_pdb, b_factor_mode="single_conformer",
                device=device,
                slot2_occupancy_floor=slot2_occupancy_floor,
                slot2_floor_outer_updates=slot2_floor_outer_updates,
                occupancy_scheme=occupancy_scheme,
                mirror_eta=mirror_eta,
                mirror_tau=mirror_tau,
            )
            spec = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
            spec["training_indices"] = train.tolist()
            if treatment == "dB_zero":
                spec["fixed_b_offset"] = 0.0
            result = worker(spec)
            saved = np.load(Path(spec["output"]) / "final_slots.npz")
            assert_heldout_geometry_provenance(
                result, saved, train, len(base_runner.base.target)
            )
            coordinates = np.stack((saved["slot1_window"], saved["slot2_window"]))
            delta_b = 0.0 if treatment == "dB_zero" else float(result["final_b_offset_A2"])
            models = base_runner.base.model_density_batch(
                coordinates, slots=np.array((0, 1)), b_offset=delta_b,
            )
            weights, intercept, train_rss = solve_affine_qp(
                base_runner.base.target[train], models[:, train]
            )
            heldout_residual = (
                base_runner.base.target[test]
                - weights @ models[:, test] - intercept
            )
            slot_rmsds = []
            for coordinate in coordinates:
                central = base_runner.base.central_backbone(coordinate)
                slot_rmsds.append({
                    "to_A": float(np.sqrt(np.mean(np.sum((central - base_runner.a_backbone) ** 2, axis=1)))),
                    "to_B": float(np.sqrt(np.mean(np.sum((central - base_runner.b_backbone) ** 2, axis=1)))),
                })
            assignments = (
                (slot_rmsds[0]["to_A"], slot_rmsds[1]["to_B"]),
                (slot_rmsds[0]["to_B"], slot_rmsds[1]["to_A"]),
            )
            assignment = min(assignments, key=lambda pair: sum(pair))
            rama = []
            for coordinate in coordinates:
                rama.append(base_runner.omega_and_rama(coordinate)[2])
            row = {
                "treatment": treatment, "fold": fold,
                "split_direction": direction.tolist(),
                "train_voxels": int(len(train)), "heldout_voxels": int(len(test)),
                "delta_B_A2": delta_b,
                "occupancies_refit_on_training": weights.tolist(),
                "intercept_refit_on_training": float(intercept),
                "training_rss": float(train_rss),
                "heldout_rss": float(np.square(heldout_residual).sum()),
                "slot_rmsds": slot_rmsds,
                "assigned_distances": list(map(float, assignment)),
                "rama_probabilities_all_seven": rama,
                "rama_floor": 0.02,
                "fit_provenance": result["fit_provenance"],
            }
            treatment_rows.append(row)
            all_rows.append(row)
        assigned = np.asarray([row["assigned_distances"] for row in treatment_rows], dtype=float)
        threshold = 0.30 * base_runner.ab_distance
        treatment_summary = {
            "treatment": treatment, "threshold_A": float(threshold),
            "folds": treatment_rows,
            "mean_all_five_assigned_distance_A": assigned.mean(axis=0).tolist(),
            "mean_excluding_fold_0_assigned_distance_A": assigned[1:].mean(axis=0).tolist(),
            "fold_sd_assigned_distance_A": assigned.std(axis=0, ddof=1).tolist(),
            "max_fold_sd_A": float(assigned.std(axis=0, ddof=1).max()),
            "fold_0_known_unknown": True,
            "recovery_threshold_passes_all_folds": bool(np.all(assigned <= threshold)),
            "scatter_below_threshold": bool(np.all(assigned.std(axis=0, ddof=1) < threshold)),
        }
        atomic_json(root / f"{treatment}_cv_summary.json", treatment_summary)
    prospective = json.loads((root / "dB_fitted_cv_summary.json").read_text())
    return {
        "status": "complete", "all_rows": all_rows,
        "prospective": prospective,
        "oracle_bound": json.loads((root / "oracle_bound_cv_summary.json").read_text()),
        "neighbour_subtraction_residual_mismatch": base_runner.base.start_sidechain_subtraction_mismatch(),
    }


RECOVERY_FRACTION = 0.30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--starts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flip-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--site", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--slot2-occupancy-floor", type=float, default=0.0,
                        help="Temporary joint-QP floor for slot 2 during early outer updates.")
    parser.add_argument("--slot2-floor-outer-updates", type=int, default=0,
                        help="Number of initial outer updates for the slot-2 floor; released thereafter.")
    parser.add_argument("--occupancy-scheme", choices=("qp", "mirror", "mirror_entropy"), default="qp")
    parser.add_argument("--mirror-eta", type=float, default=0.0)
    parser.add_argument("--mirror-tau", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.slot2_occupancy_floor < 1.0:
        parser.error("--slot2-occupancy-floor must be in [0, 1)")
    if not 0 <= args.slot2_floor_outer_updates <= 6:
        parser.error("--slot2-floor-outer-updates must be between 0 and 6")
    rows = []
    manifest = json.loads(args.manifest.read_text())
    if args.limit is not None:
        manifest = manifest[:args.limit]
    if args.site is not None:
        manifest = [site for site in manifest if site_key(site) == args.site]
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "run_config.json", {
        "operation": "same-start qFit versus A-prime", "recovery_fraction_of_A_B": RECOVERY_FRACTION,
        "rama_floor": 0.02, "nullspace": "axis2,30 degrees", "b_factor_mode": "single_conformer_for_both_A-prime_slots",
        "qfit_candidate_generator": "QFitRotamericResidue._sample_backbone, exactly 19 candidates",
        "per_slot_b_factor_refinement": False, "fold_0": "known unknown; report mean and mean excluding fold 0",
        "slot2_occupancy_floor": args.slot2_occupancy_floor,
        "slot2_floor_outer_updates": args.slot2_floor_outer_updates,
        "slot2_floor_release": "outer updates 4-6 and final selection" if args.slot2_floor_outer_updates == 3 else "configured schedule",
        "occupancy_scheme": args.occupancy_scheme,
        "mirror_eta": args.mirror_eta,
        "mirror_tau": args.mirror_tau,
    })
    starts_path = args.starts / "summary.json"
    if not starts_path.exists():
        starts_path = args.starts / "preflight.json"
    start_payload = json.loads(starts_path.read_text())
    if "rows" in start_payload:
        starts = start_payload["rows"]
    elif "site" in start_payload:
        starts = [{
            "status": "complete",
            "site": start_payload["site"],
            "refined_start": start_payload["procedure"]["refined_start"],
            "start_rmsd_to_A_A": start_payload["start_distances"]["start_rmsd_to_A_A"],
            "start_rmsd_to_B_A": start_payload["start_distances"]["start_rmsd_to_B_A"],
            "start_landed_on_deposited_A": start_payload["start_distances"]["start_landed_on_deposited_A"],
        }]
    else:
        starts = []
        for item in start_payload.get("sites", []):
            starts.append({
                "status": "complete",
                "site": item["site"],
                "refined_start": item["procedure"]["refined_start"],
                "start_rmsd_to_A_A": item["start_distances"]["start_rmsd_to_A_A"],
                "start_rmsd_to_B_A": item["start_distances"]["start_rmsd_to_B_A"],
                "start_landed_on_deposited_A": item["start_distances"]["start_landed_on_deposited_A"],
            })
    start_by_site = {row["site"]: row for row in starts if row.get("status") == "complete"}
    for site in manifest:
        key = site_key(site)
        try:
            start = start_by_site[key]
            start_pdb = Path(start["refined_start"])
            qfit = qfit_sampler(site, start_pdb)
            root = args.output / "sites" / key
            root.mkdir(parents=True, exist_ok=True)
            specs = build_specs(root / "aprime_specs", args.flip_root, site=(str(site["pdb_id"]), str(site["chain"]), int(site["resnum"])),
                                mask_scope="window", rama_floor=0.02, start_pdb=start_pdb,
                                b_factor_mode="single_conformer", device=args.device,
                                slot2_occupancy_floor=args.slot2_occupancy_floor,
                                slot2_floor_outer_updates=args.slot2_floor_outer_updates,
                                occupancy_scheme=args.occupancy_scheme,
                                mirror_eta=args.mirror_eta,
                                mirror_tau=args.mirror_tau)
            spec = next(item for item in specs if item["label"] == "D_null_axis2_30deg")
            aprime = worker(spec)
            separation = float(qfit["A_B_separation_A"])
            slot_rmsds = aprime["slot_rmsds"]
            cv = guarded_aprime_cv(
                site, start_pdb, root, args.flip_root, args.device,
                args.slot2_occupancy_floor, args.slot2_floor_outer_updates,
                args.occupancy_scheme, args.mirror_eta, args.mirror_tau,
            )
            prospective = cv["prospective"]
            full_verdict = classify(slot_rmsds, separation)
            if full_verdict == "RECOVERED":
                cv_ok = bool(
                    prospective["recovery_threshold_passes_all_folds"]
                    and prospective["scatter_below_threshold"]
                )
                verdict = "RECOVERED" if cv_ok else "PARTIAL"
            else:
                verdict = full_verdict
            rows.append({"status": "complete", "site": key, "motion_bin": site.get("motion_bin"),
                         "start": {name: start.get(name) for name in ("refined_start", "start_rmsd_to_A_A", "start_rmsd_to_B_A", "start_landed_on_deposited_A")},
                         "qfit": qfit, "aprime": aprime, "aprime_verdict": verdict,
                         "aprime_full_endpoint_verdict_before_cv": full_verdict,
                         "aprime_guarded_blocked_cv": cv,
                         "aprime_initial_slot_to_slot_rmsd_A": aprime["initial_slot_to_slot_rmsd_A"],
                         "fixed_recovery_threshold_A": RECOVERY_FRACTION * separation})
        except Exception as error:
            rows.append({"status": "error", "site": key, "error": repr(error), "traceback": traceback.format_exc()})
        atomic_json(args.output / "progress.json", {"status": "running", "completed": len(rows), "total": len(manifest)})
    atomic_json(args.output / "summary.json", {"status": "complete", "rows": rows})
    atomic_json(args.output / "progress.json", {"status": "complete", "completed": len(rows), "total": len(manifest)})


if __name__ == "__main__":
    main()
