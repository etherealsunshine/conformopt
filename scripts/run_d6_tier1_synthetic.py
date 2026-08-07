#!/usr/bin/env python3
"""Tier 1 D6 synthetic deposited-A/B diagnostic using qFit primitives."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from qfit.solvers import get_miqp_solver_class, get_qp_solver_class
from qfit.structure import Structure
from qfit.xtal.transformer import get_transformer
from qfit.xtal.unitcell import UnitCell
from qfit.xtal.volume import GridParameters, Resolution, XMap


THRESHOLDS = (1.0, 0.5, 0.33, 0.25, 0.2)
MIN_OCCUPANCY = 0.002
ABSOLUTE_CULL = 0.09


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp = Path(handle.name)
    tmp.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def bool_value(value: str) -> bool:
    return value.lower() in {"true", "1", "yes"}


def select_control_rows(candidate_path: Path, panel_rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rows = [row for row in read_rows(candidate_path) if not bool_value(row["passes_flip_filter"])]
    unique: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["pdb_id"], row["chain"], row["resnum"], row["insertion_code"])
        unique.setdefault(key, row)
    rng = random.Random(seed)
    n = min(len(panel_rows), len(unique))
    selected = rng.sample(list(unique.values()), n)
    selected_rows = []
    for row in selected:
        selected_row = row.copy()
        selected_row["panel"] = "nonflip_control"
        selected_rows.append(selected_row)
    return selected_rows


def find_residue(structure: Structure, chain_name: str, resnum: int, icode: str, altloc: str):
    # Direct atom selection also handles residues where only the backbone is
    # split A/B and qFit does not construct a complete residue conformer.
    chain_clause = f"chain {chain_name} and " if chain_name else ""
    icode_clause = f" and icode {icode}" if icode else ""
    selected = structure.extract(
        f"{chain_clause}resi {resnum}{icode_clause} and altloc {altloc}"
    )
    if selected.natoms == 0:
        raise KeyError(f"missing {chain_name}:{resnum}{icode} altloc {altloc}")
    return selected


def local_transformer(residue_a, residue_b, resolution: float):
    """Put the two deposited conformers in a small P1 box."""
    a = residue_a.copy()
    b = residue_b.copy()
    original = np.vstack([a.coor, b.coor])
    padding = 8.0
    lower = original.min(axis=0) - padding
    shift = -lower
    a.translate(shift)
    b.translate(shift)
    upper = original.max(axis=0) + padding + shift
    lengths = np.maximum(upper, 16.0)
    n_real = np.ceil(lengths / (resolution / 2.0)).astype(int)
    n_real = np.maximum(n_real, 16)
    n_real += n_real % 2
    spacing = lengths / n_real
    unit_cell = UnitCell(*lengths.tolist(), space_group="P1")
    xmap = XMap(
        np.zeros(tuple(n_real[::-1]), dtype=np.float64),
        GridParameters(spacing),
        unit_cell=unit_cell,
        resolution=Resolution(high=resolution),
    )
    a.crystal_symmetry = None
    b.crystal_symmetry = None
    a.unit_cell = unit_cell
    b.unit_cell = unit_cell
    transformer = get_transformer(
        "cctbx", a, xmap, smax=1.0 / (2.0 * resolution), simple=False, em=False
    )
    transformer.initialize()
    names_a = a.name.tolist()
    names_b = b.name.tolist()
    b_by_name = {name: (coor, bf) for name, coor, bf in zip(names_b, b.coor, b.b)}
    if set(names_a) != set(names_b):
        raise ValueError("deposited A/B atom-name sets differ")
    b_coor = np.asarray([b_by_name[name][0] for name in names_a], dtype=float)
    b_b = np.asarray([b_by_name[name][1] for name in names_a], dtype=float)
    return a, b_coor, b_b, transformer


def render(transformer, coor_set, b_set, mask):
    values = list(transformer.get_conformers_densities(coor_set, b_set))
    return [value[mask].astype(float, copy=False) for value in values]


def rss(target: np.ndarray, models: np.ndarray, weights: np.ndarray) -> float:
    return float(np.square(target - np.dot(weights, models)).sum())


def qp_fit(target: np.ndarray, models: np.ndarray) -> tuple[np.ndarray, float]:
    solver = get_qp_solver_class("CVXPYSolver")(target, models)
    solver.solve_qp()
    weights = np.asarray(solver.weights, dtype=float)
    return weights, rss(target, models, weights)


def miqp_fit(target: np.ndarray, models: np.ndarray, threshold: float) -> tuple[np.ndarray, float]:
    solver = get_miqp_solver_class("CVXPYSolver")(target, models)
    # Native qFit BIC selection uses cardinality=None; only the threshold
    # constraint limits active candidate conformers.
    solver.solve_miqp(cardinality=None, threshold=threshold)
    weights = np.asarray(solver.weights, dtype=float)
    return weights, float(solver.objective_value)


def b_ic(residual_sum_squares: float, n: int, n_atoms: int, n_conformers: int) -> tuple[float, float]:
    """Pinned qFit default BIC: 4 parameters/atom, 0.8 complexity factor."""
    k = 4.0 * n_atoms * n_conformers * 0.8
    return n * math.log(max(residual_sum_squares / n, 1e-30)) + k * math.log(n), k


def one_state_b_optimization(target, transformer, a_coor, a_b, mask):
    before = render(transformer, [a_coor], [a_b], mask)[0]
    before_rss = rss(target, before[None, :], np.array([1.0]))

    def evaluate(multiplier: float) -> tuple[float, float]:
        model = render(transformer, [a_coor], [a_b * multiplier], mask)[0]
        scale = max(0.0, float(np.dot(target, model) / max(np.dot(model, model), 1e-30)))
        return rss(target, model[None, :], np.array([scale])), scale

    result = minimize_scalar(lambda x: evaluate(float(x))[0], bounds=(0.25, 2.5), method="bounded")
    after_rss, scale = evaluate(float(result.x))
    return before_rss, after_rss, float(result.x), scale


def process_case(row: dict[str, str], structure: Structure, resolution: float, noise_fraction: float, rng: np.random.Generator) -> dict[str, object]:
    chain = row["chain"]
    resnum = int(row["resnum"])
    icode = row["insertion_code"]
    residue_a = find_residue(structure, chain, resnum, icode, "A")
    residue_b = find_residue(structure, chain, resnum, icode, "B")
    local_a, b_coor, b_b, transformer = local_transformer(residue_a, residue_b, resolution)
    a_coor = local_a.coor
    a_b = local_a.b
    occ_true = np.array([float(row["occupancy_a"]), float(row["occupancy_b"])], dtype=float)
    occ_true /= max(occ_true.sum(), 1e-30)
    mask = transformer.get_conformers_mask([a_coor, b_coor], 0.5 + resolution / 3.0)
    density_a, density_b = render(transformer, [a_coor, b_coor], [a_b, b_b], mask)
    models = np.vstack([density_a, density_b])
    clean_target = occ_true @ models
    noise_sigma = noise_fraction * float(np.std(clean_target))
    target = clean_target + rng.normal(0.0, noise_sigma, size=clean_target.shape)

    direct_weights, direct_rss = qp_fit(target, models)
    voxel_volume = transformer.xmap.unit_cell.calc_volume() / transformer.xmap.array.size
    threshold_fits = {}
    for threshold in THRESHOLDS:
        weights, objective_value = miqp_fit(target, models, threshold)
        fit_rss = objective_value * voxel_volume
        nconfs = int(np.sum(weights >= MIN_OCCUPANCY))
        fit_bic, k = b_ic(fit_rss, len(target), len(a_coor), nconfs)
        threshold_fits[str(threshold)] = {"weights": weights, "rss": fit_rss, "nconfs": nconfs, "k": k, "bic": fit_bic}
    lowest_bic = min(fit["bic"] for fit in threshold_fits.values())
    tied_thresholds = [threshold for threshold in THRESHOLDS if math.isclose(threshold_fits[str(threshold)]["bic"], lowest_bic, rel_tol=1e-12, abs_tol=1e-10)]
    selected_threshold = min(
        THRESHOLDS,
        key=lambda threshold: threshold_fits[str(threshold)]["bic"],
    )
    pipeline_weights = threshold_fits[str(selected_threshold)]["weights"]
    pipeline_rss = float(threshold_fits[str(selected_threshold)]["rss"])
    cull_threshold = ABSOLUTE_CULL
    culled_weights = pipeline_weights.copy()
    culled_weights[culled_weights < cull_threshold] = 0.0
    one_before, one_after, b_multiplier, scale = one_state_b_optimization(
        target, transformer, a_coor, a_b, mask
    )
    moved = [name for name, da, db in zip(local_a.name.tolist(), a_coor, b_coor) if np.linalg.norm(da - db) > 0.10]
    n_atoms = len(a_coor)
    row_out: dict[str, object] = {
        "pdb_id": row["pdb_id"],
        "chain": chain,
        "resnum": resnum,
        "insertion_code": icode,
        "resname": row["resname"],
        "panel": row.get("panel", "flip_filter"),
        "resolution_A": resolution,
        "max_backbone_deviation_A": float(row["max_backbone_deviation"]),
        "n_moved_atoms_gt_0.10": len(moved),
        "n_active_atoms": n_atoms,
        "true_occ_a": occ_true[0],
        "true_occ_b": occ_true[1],
        "noise_sigma": noise_sigma,
        "direct_weight_a": direct_weights[0],
        "direct_weight_b": direct_weights[1],
        "direct_rss": direct_rss,
        "direct_occ_l1_error": float(np.abs(direct_weights - occ_true).sum()),
        "pipeline_selected_threshold": selected_threshold,
        "pipeline_weight_a": pipeline_weights[0],
        "pipeline_weight_b": pipeline_weights[1],
        "pipeline_rss": pipeline_rss,
        "pipeline_nconfs_ge_0.002": threshold_fits[str(selected_threshold)]["nconfs"],
        "pipeline_bic_tie_thresholds": ";".join(str(value) for value in tied_thresholds),
        "pipeline_bic_tied": len(tied_thresholds) > 1,
        "pipeline_success_two_conformers": threshold_fits[str(selected_threshold)]["nconfs"] >= 2,
        "pipeline_occ_l1_error": float(np.abs(pipeline_weights - occ_true).sum()),
        "pipeline_cull_threshold_plus_0.09": cull_threshold,
        "pipeline_culled_weight_a": culled_weights[0],
        "pipeline_culled_weight_b": culled_weights[1],
        "one_state_rss_before_B_scale": one_before,
        "one_state_rss_after_B_scale": one_after,
        "one_state_B_multiplier": b_multiplier,
        "one_state_scale_after": scale,
        "direct_both_found_gt_0.09": bool(np.sum(direct_weights >= ABSOLUTE_CULL) >= 2),
        "pipeline_both_survive_absolute_0.09_cull": bool(np.sum(culled_weights >= ABSOLUTE_CULL) >= 2),
        "clean_target_std": float(np.std(clean_target)),
    }
    for threshold in THRESHOLDS:
        fit = threshold_fits[str(threshold)]
        row_out[f"miqp_{threshold}_weight_a"] = fit["weights"][0]
        row_out[f"miqp_{threshold}_weight_b"] = fit["weights"][1]
        row_out[f"miqp_{threshold}_rss"] = fit["rss"]
        row_out[f"miqp_{threshold}_n"] = len(target)
        row_out[f"miqp_{threshold}_k"] = fit["k"]
        row_out[f"miqp_{threshold}_bic"] = fit["bic"]
        row_out[f"miqp_{threshold}_nconfs_ge_0.002"] = fit["nconfs"]
    return row_out


def write_plots(output_dir: Path, rows: list[dict[str, object]]) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"matplotlib unavailable: {exc}"
    if not rows:
        return "no rows"
    x = np.array([float(row["max_backbone_deviation_A"]) for row in rows])
    direct = np.array([float(row["direct_bic_margin_tmin_0.20"]) for row in rows])
    pipe = np.array([float(row["pipeline_bic_margin_selected"]) for row in rows])
    resolution = np.array([float(row["resolution_A"]) for row in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, direct, c=resolution, cmap="viridis", alpha=0.65, label="DIRECT")
    ax.scatter(x, pipe, c=resolution, cmap="plasma", marker="x", alpha=0.65, label="PIPELINE")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="max deposited backbone A/B deviation (Å)", ylabel="one-state BIC − two-state BIC", title="D6 margin vs deposited deviation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "margin_vs_deviation.png", dpi=160)
    plt.close(fig)

    levels = sorted(set(resolution))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([[float(row["pipeline_bic_margin_selected"]) for row in rows if float(row["resolution_A"]) == level] for level in levels], labels=[f"{level:.1f}" for level in levels])
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="synthetic target resolution (Å)", ylabel="PIPELINE one-state BIC − two-state BIC", title="D6 margin distribution by resolution")
    fig.tight_layout()
    fig.savefig(output_dir / "margin_by_resolution.png", dpi=160)
    plt.close(fig)

    panels = sorted(set(str(row["panel"]) for row in rows))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([[float(row["pipeline_bic_margin_selected"]) for row in rows if str(row["panel"]) == panel] for panel in panels], labels=panels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(ylabel="PIPELINE one-state BIC − two-state BIC", title="D6 flip-filter versus nonflip control")
    fig.tight_layout()
    fig.savefig(output_dir / "margin_flip_vs_nonflip.png", dpi=160)
    plt.close(fig)
    return None


def write_occupancy_diagnostics(output_dir: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    """Write the requested deposited-occupancy and MIQP-path diagnostics."""
    flip_rows = [row for row in rows if row["panel"] == "flip_filter"]
    key_fields = ("pdb_id", "chain", "resnum", "insertion_code")
    by_site: dict[tuple[object, ...], dict[str, object]] = {}
    for row in flip_rows:
        by_site.setdefault(tuple(row[field] for field in key_fields), row)
    sites = list(by_site.values())

    site_rows = []
    for row in sites:
        occ_a, occ_b = float(row["true_occ_a"]), float(row["true_occ_b"])
        site_rows.append({
            **{field: row[field] for field in key_fields},
            "resname": row["resname"],
            "occupancy_a": occ_a,
            "occupancy_b": occ_b,
            "minor_state": min(occ_a, occ_b),
            "major_state": max(occ_a, occ_b),
            "minor_altloc": "A" if occ_a <= occ_b else "B",
            "minor_below_0.2": min(occ_a, occ_b) < 0.2,
        })
    site_rows.sort(key=lambda row: (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"])))
    with (output_dir / "deposited_occupancy_distribution_34_sites.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(site_rows[0]) if site_rows else [])
        writer.writeheader()
        writer.writerows(site_rows)

    verdict_rows = []
    for row in flip_rows:
        minor = min(float(row["true_occ_a"]), float(row["true_occ_b"]))
        verdict_rows.append({
            "minor_occupancy_bin": "<0.2" if minor < 0.2 else ">=0.2",
            "bic_verdict": "selected_t_dmin_1.0" if float(row["pipeline_selected_threshold"]) == 1.0 else "selected_t_dmin_below_1.0",
            "cull_verdict": "both_survived_cull" if bool(row["pipeline_both_found_after_cull"]) else "did_not_survive_cull",
        })
    crosstab = []
    for verdict_type, field in (("BIC threshold selection", "bic_verdict"), ("post-0.09 cull", "cull_verdict")):
        for minor_bin in ("<0.2", ">=0.2"):
            values = [row for row in verdict_rows if row["minor_occupancy_bin"] == minor_bin]
            for verdict in sorted({row[field] for row in verdict_rows}):
                crosstab.append({
                    "verdict_type": verdict_type,
                    "minor_occupancy_bin": minor_bin,
                    "verdict": verdict,
                    "cases": sum(row[field] == verdict for row in values),
                    "bin_total": len(values),
                })
    with (output_dir / "verdict_by_minor_occupancy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["verdict_type", "minor_occupancy_bin", "verdict", "cases", "bin_total"])
        writer.writeheader()
        writer.writerows(crosstab)

    failures = [row for row in flip_rows if float(row["pipeline_selected_threshold"]) == 1.0]
    failure_paths = []
    for row in failures:
        occ_a, occ_b = float(row["true_occ_a"]), float(row["true_occ_b"])
        minor_is_a = occ_a <= occ_b
        for threshold in THRESHOLDS:
            weight_a = float(row[f"miqp_{threshold}_weight_a"])
            weight_b = float(row[f"miqp_{threshold}_weight_b"])
            minor_weight = weight_a if minor_is_a else weight_b
            major_weight = weight_b if minor_is_a else weight_a
            failure_paths.append({
                **{field: row[field] for field in key_fields},
                "resname": row["resname"],
                "resolution_A": row["resolution_A"],
                "true_occupancy_a": occ_a,
                "true_occupancy_b": occ_b,
                "true_minor_occupancy": min(occ_a, occ_b),
                "minor_altloc": "A" if minor_is_a else "B",
                "minor_below_0.2": min(occ_a, occ_b) < 0.2,
                "t_dmin": threshold,
                "miqp_weight_a": weight_a,
                "miqp_weight_b": weight_b,
                "miqp_minor_weight": minor_weight,
                "miqp_major_weight": major_weight,
                "miqp_rss": float(row[f"miqp_{threshold}_rss"]),
                "minor_weight_nonzero_ge_0.002": minor_weight >= 0.002,
                "minor_weight_meets_t_dmin": minor_weight >= threshold - 1e-6,
            })
    failure_fields = list(failure_paths[0]) if failure_paths else [*key_fields, "t_dmin"]
    with (output_dir / "bic_t1_failures_miqp_occupancies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields)
        writer.writeheader()
        writer.writerows(failure_paths)

    path_summary = []
    for threshold in THRESHOLDS:
        values = [row for row in failure_paths if row["t_dmin"] == threshold]
        for minor_bin in ("<0.2", ">=0.2", "all"):
            subset = values if minor_bin == "all" else [row for row in values if ("<0.2" if row["minor_below_0.2"] else ">=0.2") == minor_bin]
            weights = [float(row["miqp_minor_weight"]) for row in subset]
            path_summary.append({
                "t_dmin": threshold,
                "true_minor_occupancy_bin": minor_bin,
                "cases": len(subset),
                "minor_nonzero_ge_0.002": sum(bool(row["minor_weight_nonzero_ge_0.002"]) for row in subset),
                "minor_meets_t_dmin": sum(bool(row["minor_weight_meets_t_dmin"]) for row in subset),
                "mean_miqp_minor_weight": float(np.mean(weights)) if weights else None,
                "median_miqp_minor_weight": float(np.median(weights)) if weights else None,
            })
    with (output_dir / "bic_t1_failures_miqp_occupancy_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(path_summary[0]))
        writer.writeheader()
        writer.writerows(path_summary)

    minors = np.asarray([float(row["minor_state"]) for row in site_rows], dtype=float)
    return {
        "deposited_occupancy_distribution": {
            "sites": len(site_rows),
            "minor_below_0.2": int(np.sum(minors < 0.2)),
            "minor_at_least_0.2": int(np.sum(minors >= 0.2)),
            "minor_min": float(np.min(minors)),
            "minor_median": float(np.median(minors)),
            "minor_max": float(np.max(minors)),
        },
        "flip_case_bic_selected_t1": len(failures),
        "flip_case_bic_selected_t1_minor_below_0.2": sum(float(row["true_occ_a"]) < 0.2 or float(row["true_occ_b"]) < 0.2 for row in failures),
    }


def write_native_diagnostics(output_dir: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    """Write native-BIC verdict, tie, and high-minor-failure evidence tables."""
    key_fields = ("pdb_id", "chain", "resnum", "insertion_code")
    flips = [row for row in rows if row["panel"] == "flip_filter"]
    by_site: dict[tuple[object, ...], dict[str, object]] = {}
    for row in flips:
        by_site.setdefault(tuple(row[field] for field in key_fields), row)
    site_rows = []
    for row in by_site.values():
        a, b = float(row["true_occ_a"]), float(row["true_occ_b"])
        site_rows.append({**{field: row[field] for field in key_fields}, "resname": row["resname"], "occupancy_a": a, "occupancy_b": b, "minor_occupancy": min(a, b)})
    site_rows.sort(key=lambda row: (str(row["pdb_id"]), str(row["chain"]), int(row["resnum"])))
    with (output_dir / "deposited_occupancy_distribution_33_sites.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(site_rows[0]))
        writer.writeheader(); writer.writerows(site_rows)

    crosstab = []
    for panel in sorted({str(row["panel"]) for row in rows}):
        panel_rows = [row for row in rows if row["panel"] == panel]
        for minor_bin in ("<0.2", ">=0.2"):
            subset = [row for row in panel_rows if ("<0.2" if min(float(row["true_occ_a"]), float(row["true_occ_b"])) < 0.2 else ">=0.2") == minor_bin]
            for metric, field in (("native_success", "pipeline_success_two_conformers"), ("survive_absolute_0.09_cull", "pipeline_both_survive_absolute_0.09_cull")):
                crosstab.append({"panel": panel, "metric": metric, "minor_occupancy_bin": minor_bin, "successes": sum(bool(row[field]) for row in subset), "failures": sum(not bool(row[field]) for row in subset), "total": len(subset)})
    with (output_dir / "success_by_minor_occupancy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["panel", "metric", "minor_occupancy_bin", "successes", "failures", "total"])
        writer.writeheader(); writer.writerows(crosstab)

    ties = [row for row in rows if bool(row["pipeline_bic_tied"])]
    tie_fields = [*key_fields, "resname", "panel", "resolution_A", "pipeline_selected_threshold", "pipeline_bic_tie_thresholds"]
    with (output_dir / "bic_ties.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tie_fields); writer.writeheader()
        writer.writerows([{field: row[field] for field in tie_fields} for row in ties])

    failures = [row for row in rows if not bool(row["pipeline_success_two_conformers"]) and min(float(row["true_occ_a"]), float(row["true_occ_b"])) >= 0.2]
    failure_fields = [*key_fields, "resname", "panel", "resolution_A", "true_occ_a", "true_occ_b", "pipeline_selected_threshold", "pipeline_bic_tie_thresholds"]
    for threshold in THRESHOLDS:
        failure_fields.extend([f"miqp_{threshold}_n", f"miqp_{threshold}_k", f"miqp_{threshold}_rss", f"miqp_{threshold}_bic", f"miqp_{threshold}_nconfs_ge_0.002", f"miqp_{threshold}_weight_a", f"miqp_{threshold}_weight_b"])
    with (output_dir / "native_bic_failures_minor_ge_0.2_full_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=failure_fields); writer.writeheader()
        writer.writerows([{field: row[field] for field in failure_fields} for row in failures])

    return {"flip_sites": len(site_rows), "native_bic_ties": len(ties), "minor_ge_0.2_failures_full_rows": len(failures)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--site-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--noise-fraction", type=float, default=0.10)
    parser.add_argument("--resolutions", type=float, nargs="+", default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flip_rows = [row for row in read_rows(args.site_list) if min(float(row["occupancy_a"]), float(row["occupancy_b"])) > 0.0]
    for row in flip_rows:
        row["panel"] = "flip_filter"
    rows = flip_rows
    if not args.no_control:
        controls = select_control_rows(args.site_list.parent / "candidate_sites.csv", flip_rows, args.seed + 1)
        rows = rows + controls
    if args.limit:
        rows = rows[:args.limit]
    panel_manifest = [{key: row[key] for key in ("pdb_id", "chain", "resnum", "insertion_code", "resname", "panel", "resolution", "max_backbone_deviation", "passes_flip_filter")} for row in rows]
    atomic_json(args.output_dir / "panel_manifest.json", panel_manifest)

    structures: dict[str, Structure] = {}
    output_rows: list[dict[str, object]] = []
    case_index = 0
    checkpoint_dir = args.output_dir / "site_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for site_index, row in enumerate(rows, start=1):
        checkpoint = checkpoint_dir / f"site_{site_index:04d}.json"
        if checkpoint.exists():
            site_rows = json.loads(checkpoint.read_text())
            output_rows.extend(site_rows)
            case_index += len(site_rows)
            continue
        source_path = row["source_path"]
        if source_path not in structures:
            structures[source_path] = Structure.fromfile(source_path)
        structure = structures[source_path]
        for resolution in args.resolutions:
            case_index += 1
            rng = np.random.default_rng(args.seed + case_index)
            result = process_case(row, structure, float(resolution), args.noise_fraction, rng)
            output_rows.append(result)
        atomic_json(checkpoint, output_rows[-len(args.resolutions):])
        atomic_json(args.output_dir / "progress.json", {"sites_complete": site_index, "sites_total": len(rows), "cases_complete": len(output_rows), "case_total": len(rows) * len(args.resolutions)})

    fields = sorted({key for row in output_rows for key in row})
    with (args.output_dir / "per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    plot_error = "omitted: native-BIC verdict replaces legacy one-state-margin plots"
    occupancy_diagnostics = write_native_diagnostics(args.output_dir, output_rows)
    summary = {
        "site_count": len(rows),
        "case_count": len(output_rows),
        "flip_filter_sites": sum(row["panel"] == "flip_filter" for row in rows),
        "nonflip_control_sites": sum(row["panel"] == "nonflip_control" for row in rows),
        "resolutions_A": args.resolutions,
        "noise_fraction_of_clean_target_std": args.noise_fraction,
        "seed": args.seed,
        "transformer": "qfit.xtal.transformer.get_transformer('cctbx')",
        "mask": "qFit transformer.get_conformers_mask for deposited A/B coordinates",
        "solver": "qFit CVXPYSolver wrappers; CVXPY SCIP backend for MIQP",
        "bic_definition": "Pinned qFit native default: BIC=n*log(RSS/n)+k*log(n), k=4*n_active_atoms*n_selected_conformers*0.8, n_selected_conformers counts weights >=0.002, and RSS is solver objective times voxel volume.",
        "pipeline_cull": "absolute: weights below 0.09 are set to zero; success remains a separate >=2-conformer criterion at 0.002",
        "moved_atom_sensitivity": {threshold: sum(float(row["max_backbone_deviation_A"]) > threshold for row in output_rows) for threshold in (0.10, 0.25, 0.50, 1.00)},
        "plot_error": plot_error,
        "occupancy_diagnostics": occupancy_diagnostics,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "progress.json", {"status": "complete", "sites_complete": len(rows), "sites_total": len(rows), "cases_complete": len(output_rows), "case_total": len(rows) * len(args.resolutions)})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
