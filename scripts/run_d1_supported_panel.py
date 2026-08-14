#!/usr/bin/env python3
"""Checkpointed A'' recovery and blocked-CV controller for supported sites.

This controller deliberately limits recovery to the deposited fixed-geometry
screen's TWO-STATE SUPPORTED sites.  Every candidate uses the corrected
seven-residue neighbour subtraction, full-window backbone mask, per-slot
deposited B arrays, a profiled occupancy/intercept, a fitted global B offset,
and the seven-residue Rama barrier.  It never reads held-out voxels while
choosing an endpoint: each blocked fold uses the one precommitted prospective
nullspace start, so it contains no hidden start-selection procedure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

# qFit/CCTBX and CUDA Torch use different Python environments on the pod.
QFIT_SITE = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/site-packages"
QFIT_DYNLIB = "/home/dev/qfit_unet_data/.venv-qfit-audit/lib/python3.12/lib-dynload"
WORKSPACE = "/home/dev/workspace"
QFIT_SRC = f"{WORKSPACE}/external/qfit-3.0/src"
if os.path.isdir(QFIT_SITE):
    sys.path.insert(0, QFIT_SITE)
    import numpy as np
    sys.path.remove(QFIT_SITE)
    import torch  # noqa: F401  # initialize CUDA before qFit extension imports
    sys.path[:0] = [QFIT_DYNLIB, QFIT_SITE, WORKSPACE, f"{WORKSPACE}/scripts", QFIT_SRC]
else:
    import numpy as np

from occupancy_selection import solve_affine_qp
from run_d1_aprime_leakage_corrected_cv import blocked_splits
from run_d1_aprime_sequential import APrimeSequential
from run_d1_slot_coordination import build_specs, worker
from run_d1_8d_sequential_poc import atomic_json
from fit_provenance import assert_heldout_geometry_provenance


SUPPORTED_SITES = (
    ("4HFS", "A", 200), ("4ZXG", "B", 13), ("5JBX", "C", 132),
    ("6ZWK", "B", 47), ("7SC4", "B", 2317), ("7UTC", "A", 52),
    ("7ZTL", "A", 257), ("8AJK", "A", 240), ("8R7O", "C", 1681),
)
FLIP_ROOT = Path(WORKSPACE) / "data/qfit_2015_s004"
# Fixed prospectively from the independent 5OHJ reachability experiment.  Do
# not select among nullspace axes by a site-specific training objective here:
# that would turn blocked CV into validation of a hidden multi-start selector.
NULL_LABELS = {"D_null_axis2_30deg"}


def site_name(site: tuple[str, str, int]) -> str:
    return f"{site[0]}_{site[1]}_{site[2]}"


def prospective_specs(root: Path, site: tuple[str, str, int], training_indices=None,
                      fixed_b_offset: float | None = None):
    """Return the precommitted axis-2, 30-degree nullspace proposal only."""
    specs = build_specs(root, FLIP_ROOT, site=site, mask_scope="window", rama_floor=0.02)
    answer = [spec for spec in specs if spec["label"] in NULL_LABELS]
    if len(answer) != 1:
        raise RuntimeError(f"expected one fixed nullspace start for {site}, got {len(answer)}")
    if training_indices is not None:
        for spec in answer:
            spec["training_indices"] = np.asarray(training_indices, dtype=int).tolist()
    if fixed_b_offset is not None:
        # A separate, fully train-only ΔB=0 fit is required for a fair
        # parameter-matched comparison.  Do not reuse a geometry optimized
        # with a free ΔB and merely rescore it at zero.
        for spec in answer:
            spec["fixed_b_offset"] = float(fixed_b_offset)
    return answer


def run_candidates(root: Path, site: tuple[str, str, int], training_indices=None,
                   fixed_b_offset: float | None = None) -> list[dict[str, object]]:
    """Run/reuse the prospective starts, selecting later by training objective."""
    specs = prospective_specs(
        root, site, training_indices=training_indices,
        fixed_b_offset=fixed_b_offset,
    )
    results = []
    for spec in specs:
        output = Path(spec["output"])
        result_path = output / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("status") != "complete":
                raise RuntimeError(f"incomplete existing candidate: {output}")
        else:
            result = worker(spec)
        results.append({"label": spec["label"], "output": str(output), "result": result})
        atomic_json(root / "candidate_progress.json", {
            "status": "running", "site": site_name(site),
            "completed_candidates": len(results), "total_candidates": len(specs),
        })
    return results


def choose_training_endpoint(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Return the sole precommitted start; there is no start-selection step."""
    if len(candidates) != 1:
        raise RuntimeError("fixed-initialization protocol requires exactly one candidate")
    return candidates[0]


def full_models(base, slots: np.ndarray, b_offset: float) -> np.ndarray:
    return base.model_density_batch(slots, slots=np.array((0, 1)), b_offset=b_offset)


def fixed_b_fold_fit(base, target: np.ndarray, windows: list[np.ndarray], slots: list[int],
                     b_offset: float, train: np.ndarray, test: np.ndarray) -> dict[str, object]:
    """Fit occupancies/intercept on train voxels at a specified shared ΔB."""
    models = base.model_density_batch(np.stack(windows), slots=np.asarray(slots), b_offset=b_offset)
    weights, intercept, training_rss = solve_affine_qp(target[train], models[:, train])
    heldout_rss = float(np.square(target[test] - weights @ models[:, test] - intercept).sum())
    return {"occupancies": np.asarray(weights).tolist(), "intercept": float(intercept),
            "dB_A2": float(b_offset), "training_rss": float(training_rss),
            "heldout_rss": heldout_rss}


def density_rank(runner: APrimeSequential, saved: object, weights: np.ndarray) -> dict[str, object]:
    """Rank the fixed-profile density Jacobian over 40 torsions plus ΔB."""
    import torch

    parameters = np.asarray(saved["parameters"], dtype=float)
    if parameters.shape != (41,):
        raise ValueError(f"expected 41 parameters, got {parameters.shape}")
    value = torch.as_tensor(parameters, dtype=torch.float64,
                            device=runner.base.torch_device).requires_grad_(True)
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=value.device)
    a_b = torch.as_tensor(runner.base.b_factors_a_model, dtype=torch.float64, device=value.device)
    b_b = torch.as_tensor(runner.base.b_factors_b_model, dtype=torch.float64, device=value.device)

    def density(parameters_t):
        coordinates = runner.torch_forward(parameters_t[:-1].reshape(2, -1))
        models = runner.base.model_density_torch(
            coordinates[:, runner.base.model_atom_indices],
            b_factors=torch.stack((a_b, b_b)) + parameters_t[-1],
        )
        return (weights_t[:, None] * models).sum(0)

    jacobian = torch.autograd.functional.jacobian(
        density, value, create_graph=False, vectorize=True, strategy="forward-mode",
    ).detach().cpu().numpy()
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = max(jacobian.shape) * np.finfo(float).eps * singular[0]
    return {
        "density_rank": int(np.count_nonzero(singular > threshold)),
        "density_parameter_count": int(jacobian.shape[1]),
        "rank_tolerance": float(threshold),
        "singular_values": singular.tolist(),
    }


def _evaluate_fold_treatment(full_runner: APrimeSequential, site: tuple[str, str, int],
                             output: Path, train: np.ndarray, test: np.ndarray,
                             fixed_b_offset: float | None) -> dict[str, object]:
    """Evaluate one parameter-matched ΔB treatment on a blocked fold."""
    label = "dB_zero" if fixed_b_offset is not None else "dB_fitted"
    candidates = run_candidates(
        output / f"candidates_{label}", site, training_indices=train,
        fixed_b_offset=fixed_b_offset,
    )
    selected = choose_training_endpoint(candidates)
    saved = np.load(Path(selected["output"]) / "final_slots.npz")
    assert_heldout_geometry_provenance(selected["result"], saved, train, len(full_runner.base.target))
    slots = np.stack((saved["slot1_window"], saved["slot2_window"]))
    b_offset = float(fixed_b_offset if fixed_b_offset is not None else saved["b_offset_A2"])
    models = full_models(full_runner.base, slots, b_offset)
    recovered_weights, recovered_intercept, recovered_train_rss = solve_affine_qp(
        full_runner.base.target[train], models[:, train]
    )
    recovered_heldout_rss = float(np.square(
        full_runner.base.target[test] - recovered_weights @ models[:, test] - recovered_intercept
    ).sum())
    deposited = [full_runner.base.initial_window, full_runner.base.window_for_deposited_b()]
    if fixed_b_offset is None:
        deposited_fit = full_runner.base.profile_affine_b_offset(
            full_runner.base.target[train], deposited, [0, 1], voxel_indices=train
        )
        deposited_b_offset = float(deposited_fit["b_offset"])
        deposited_weights = np.asarray(deposited_fit["weights"])
        deposited_intercept = float(deposited_fit["intercept"])
    else:
        deposited_fit = fixed_b_fold_fit(
            full_runner.base, full_runner.base.target, deposited, [0, 1],
            fixed_b_offset, train, test,
        )
        deposited_b_offset = float(fixed_b_offset)
        deposited_weights = np.asarray(deposited_fit["occupancies"])
        deposited_intercept = float(deposited_fit["intercept"])
    deposited_models = full_runner.base.model_density_batch(
        np.stack(deposited), slots=np.array((0, 1)), b_offset=deposited_b_offset
    )
    deposited_heldout_rss = float(np.square(
        full_runner.base.target[test] - deposited_weights @ deposited_models[:, test]
        - deposited_intercept
    ).sum())
    return {
        "fixed_dB_A2": fixed_b_offset,
        "selected_candidate": selected["label"],
        "selected_candidate_training_energy": float(selected["result"]["final_energy"]),
        "A_prime": {"training_rss": float(recovered_train_rss),
                     "heldout_rss": recovered_heldout_rss,
                     "occupancies": np.asarray(recovered_weights).tolist(),
                     "intercept": float(recovered_intercept), "dB_A2": b_offset},
        "deposited_two_state": {"heldout_rss": deposited_heldout_rss,
                                 "occupancies": deposited_weights.tolist(),
                                 "intercept": deposited_intercept,
                                 "dB_A2": deposited_b_offset},
        "deposited_minus_A_prime_heldout_rss": deposited_heldout_rss - recovered_heldout_rss,
    }


def evaluate_fold(site: tuple[str, str, int], output: Path, fold: int) -> dict[str, object]:
    """Fresh train-only A'' CV for both fitted-ΔB and ΔB=0 objectives."""
    full_runner = APrimeSequential(output / "full_evaluator", 8, 6, *site,
                                   renderer_backend="torch", residual_scale_mode="none",
                                   map_scaler_structure="full", mask_scope="window", device="cuda")
    full_runner.rama_floor = 0.02
    train, test, direction = blocked_splits(full_runner.base)[fold]
    return {
        "status": "complete", "fold": fold, "split_direction": direction.tolist(),
        "train_voxels": int(len(train)), "heldout_voxels": int(len(test)),
        "dB_fitted": _evaluate_fold_treatment(
            full_runner, site, output, train, test, fixed_b_offset=None
        ),
        "dB_zero": _evaluate_fold_treatment(
            full_runner, site, output, train, test, fixed_b_offset=0.0
        ),
    }


def run_site(root: Path, site: tuple[str, str, int], include_cv: bool) -> dict[str, object]:
    site_root = root / "sites" / site_name(site)
    site_root.mkdir(parents=True, exist_ok=True)
    candidates = run_candidates(site_root / "full_candidates", site)
    selected = choose_training_endpoint(candidates)
    saved = np.load(Path(selected["output"]) / "final_slots.npz")
    full_runner = APrimeSequential(site_root / "audit_runner", 8, 6, *site,
                                   renderer_backend="torch", residual_scale_mode="none",
                                   map_scaler_structure="full", mask_scope="window", device="cuda")
    full_runner.rama_floor = 0.02
    selection = selected["result"]["final_selection"]
    weights = np.asarray(selection["weights"], dtype=float)
    rank = density_rank(full_runner, saved, weights)
    record = {
        "status": "running", "site": site_name(site),
        "configuration": {
            "mask": "full-window backbone union of spheres",
            "density_atom_scope": "N/CA/C/O only; deposited-A window sidechains subtracted from target",
            "neighbour_subtraction": "exclude fitted seven-residue window plus its deposited-A sidechains",
            "slot_B_arrays": "slot 1 deposited A; slot 2 deposited B",
            "occupancy_intercept": "profiled affine QP", "global_dB": "explicit profiled density-width parameter",
            "rama": "all seven residues, floor 0.02", "omega_scale_deg": 5.0,
            "renderer": "Torch CUDA autodiff", "initialization": "fixed carbonyl-null axis 2, 30 degrees",
        },
        "selected_candidate": selected["label"],
        "candidate_results": [{"label": row["label"], "final_energy": row["result"]["final_energy"],
                               "final_rss": row["result"]["final_rss"]} for row in candidates],
        "endpoint": selected["result"], "density_rank": rank,
        # Slot RMSDs and this deposited A--B separation are computed on the
        # identical central-backbone atom set.  Fractions make a close A/B
        # deposition visible rather than overstating apparent recovery.
        "deposited_A_to_B_rmsd_A": float(full_runner.ab_distance),
        "slot_rmsd_fractions_of_A_B": [
            {key: (float(value) / float(full_runner.ab_distance)
                   if full_runner.ab_distance > 0.0 else None)
             for key, value in row.items()}
            for row in selected["result"]["slot_rmsds"]
        ],
    }
    endpoint_dB = float(selected["result"]["final_b_offset_A2"])
    record["summary_eligible"] = abs(endpoint_dB) <= 15.0
    record["summary_exclusion_reason"] = (
        None if record["summary_eligible"]
        else f"|fitted global dB|={abs(endpoint_dB):.3f} A^2 exceeds 15 A^2 model-fit threshold"
    )
    atomic_json(site_root / "site_result.json", record)
    if include_cv:
        folds = []
        for fold in range(5):
            fold_path = site_root / "blocked_cv" / f"fold_{fold}.json"
            if fold_path.is_file():
                current = json.loads(fold_path.read_text())
            else:
                current = evaluate_fold(site, site_root / "blocked_cv" / f"fold_{fold}", fold)
                atomic_json(fold_path, current)
            folds.append(current)
            record["blocked_cv"] = {"completed_folds": len(folds), "total_folds": 5}
            atomic_json(site_root / "site_result.json", record)
        deltas = [float(row["deposited_minus_A_prime_heldout_rss"]) for row in folds]
        record["blocked_cv"] = {
            "folds": folds, "paired_differences": deltas,
            "mean_difference": float(np.mean(deltas)),
            "sign_pattern": ["A_prime_better" if value > 0 else "deposited_better_or_tie" for value in deltas],
        }
        atomic_json(site_root / "site_result.json", record)
    record["status"] = "complete"
    atomic_json(site_root / "site_result.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {args.output}; pass --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "run_config.json", {
        "sites": [site_name(site) for site in SUPPORTED_SITES], "cv": not args.no_cv,
        "starts_per_fit": sorted(NULL_LABELS), "outer_updates": 6, "inner_nfev": 8,
        "protocol": "one fixed initialization for full fit and every blocked-CV fold; no start selection",
    })
    completed = []
    for site in SUPPORTED_SITES:
        try:
            path = args.output / "sites" / site_name(site) / "site_result.json"
            if args.resume and path.is_file() and json.loads(path.read_text()).get("status") == "complete":
                result = json.loads(path.read_text())
            else:
                result = run_site(args.output, site, include_cv=not args.no_cv)
            completed.append({"site": site_name(site), "status": result["status"]})
        except Exception as error:  # checkpoint errors and continue the independent panel.
            completed.append({"site": site_name(site), "status": "error", "error": repr(error),
                              "traceback": traceback.format_exc()})
        atomic_json(args.output / "progress.json", {
            "status": "running", "completed": completed, "total_sites": len(SUPPORTED_SITES),
        })
    atomic_json(args.output / "summary.json", {"status": "complete", "sites": completed})
    atomic_json(args.output / "progress.json", {"status": "complete", "completed": completed})


if __name__ == "__main__":
    main()
