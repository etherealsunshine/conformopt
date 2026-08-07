"""Measure coverage, occupancy, and endpoint margins on frozen-v3 endpoints.

This is a read-only diagnostic.  It reconstructs deposited and saved endpoint
densities on the production Stage-1 masked grid in native additive density
space, computes RSCC margins against the synthetic target, and normalizes the
coverage margin by the observed RSCC scatter of recovery-failed endpoints.
It never invokes an optimizer and never changes the frozen metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.diagnose_frozen_v3_occupancy_pooling import (
    atomic_csv,
    atomic_json,
    load_optimizer_rows,
    load_v3_payload,
    parse_vector,
    truth,
)
from scripts.diagnose_frozen_v3_residual_rotamers import (
    build_site_geometry,
    parse_matrix,
    pearson_correlation,
)


METRIC = "qfit-synth20-merge050-one-to-one-tmol044-v3"
EXPECTED_CASCADE = (742, 714, 710, 710, 710, 626)
OCCUPANCY_SPLITS = ((0.25, 0.75), (0.50, 0.50), (0.75, 0.25),
                    (0.90, 0.10))
MIN_RELIABLE_FAILURES = 5
REFERENCE_SIGMA = {"3A1C_B_ARG447": 9.4, "5Z8H_A_MET730": 0.29}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixed_label_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.square(delta).sum(axis=1).mean()))


def render_endpoint(
    optimizer: dict[str, str],
    geometry: dict[str, object],
) -> np.ndarray:
    chi = parse_matrix(optimizer["final_chi_radians"])
    occupancies = np.asarray(
        parse_vector(optimizer["occupancies"]), dtype=np.float64
    )
    if chi.shape[0] != len(occupancies):
        raise ValueError("endpoint chi/occupancy shape mismatch")
    render = geometry["render"]
    from_delta = geometry["from_delta"]
    return sum(
        occupancy * render(
            from_delta(torch.tensor(row, dtype=torch.float32)).detach()
        )
        for occupancy, row in zip(occupancies, chi)
    )


def candidate_rsccs(
    target: np.ndarray,
    density_a: np.ndarray,
    density_b: np.ndarray,
    target_a: float,
    target_b: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return all hand-built candidate RSCCs and three class margins."""
    correct = target_a * density_a + target_b * density_b
    a_alone = density_a.copy()
    b_alone = density_b.copy()
    a_duplicated = 0.5 * density_a + 0.5 * density_a
    candidates = [
        ("correct", "deposited", target_a, target_b, correct),
        ("A_alone", "coverage", 1.0, 0.0, a_alone),
        ("B_alone", "coverage", 0.0, 1.0, b_alone),
        ("A_duplicated", "coverage", 1.0, 0.0, a_duplicated),
    ]
    for occupancy_a, occupancy_b in OCCUPANCY_SPLITS:
        candidates.append((
            f"A{occupancy_a:.2f}_B{occupancy_b:.2f}",
            "occupancy",
            occupancy_a,
            occupancy_b,
            occupancy_a * density_a + occupancy_b * density_b,
        ))
    rows = [{
        "candidate": name,
        "candidate_class": candidate_class,
        "occupancy_A": occupancy_a,
        "occupancy_B": occupancy_b,
        "matches_deposited_occupancy": (
            math.isclose(occupancy_a, target_a, abs_tol=1e-6)
            and math.isclose(occupancy_b, target_b, abs_tol=1e-6)
        ),
        "rscc": pearson_correlation(target, density),
    } for name, candidate_class, occupancy_a, occupancy_b, density in candidates]
    correct_rscc = float(rows[0]["rscc"])
    class_summary: dict[str, object] = {
        "correct_rscc": correct_rscc,
        "duplicate_max_absolute_density_error": float(
            np.max(np.abs(a_duplicated - a_alone))
        ),
        "duplicate_rscc_absolute_error": abs(
            float(rows[3]["rscc"]) - float(rows[1]["rscc"])
        ),
    }
    for candidate_class in ("coverage", "occupancy"):
        eligible = [
            row for row in rows
            if row["candidate_class"] == candidate_class
            and not (
                candidate_class == "occupancy"
                and bool(row["matches_deposited_occupancy"])
            )
        ]
        best = max(eligible, key=lambda row: float(row["rscc"]))
        class_summary[f"{candidate_class}_best_wrong_candidate"] = (
            best["candidate"]
        )
        class_summary[f"{candidate_class}_best_wrong_rscc"] = float(
            best["rscc"]
        )
        class_summary[f"{candidate_class}_margin"] = (
            correct_rscc - float(best["rscc"])
        )
    return rows, class_summary


def cascade_guard(
    ensembles: dict[tuple[str, int], dict[str, str]],
    active_by_slot: dict[tuple[str, int, int], dict[str, str]],
) -> tuple[int, int, int, int, int, int]:
    active_by_id = {
        row["candidate_id"]: row for row in active_by_slot.values()
    }
    both = occupancy = rotamer = direct = symmetry = tmol = 0
    for row in ensembles.values():
        if not truth(row["both_found_conventional"]):
            continue
        both += 1
        if not truth(row["occupancy_accurate"]):
            continue
        occupancy += 1
        candidate_ids = (
            row["assigned_pair_candidate_A"],
            row["assigned_pair_candidate_B"],
        )
        if not all(candidate_id in active_by_id for candidate_id in candidate_ids):
            continue
        pair = [active_by_id[candidate_id] for candidate_id in candidate_ids]
        if not all(truth(candidate["rotamer_within_allowed_width"])
                   for candidate in pair):
            continue
        rotamer += 1
        if not all(truth(candidate["no_direct_clash"]) for candidate in pair):
            continue
        direct += 1
        if not all(truth(candidate["no_symmetry_clash"]) for candidate in pair):
            continue
        symmetry += 1
        if not all(truth(candidate["tmol_valid"]) for candidate in pair):
            continue
        tmol += 1
    return both, occupancy, rotamer, direct, symmetry, tmol


def load_separations(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    available = set(rows[0]) if rows else set()
    preferred = (
        "local_unsym_rmsd_A",
        "local_fixed_rmsd_A",
        "local_fixed_A",
    )
    field = next((name for name in preferred if name in available), None)
    if field is None:
        raise ValueError(
            f"no local fixed-label separation column in {path}; "
            f"available={sorted(available)}"
        )
    output = {row["site"]: float(row[field]) for row in rows}
    if len(output) != 20:
        raise ValueError(f"expected 20 separation rows, found {len(output)}")
    return output


def plot_discriminability(rows: list[dict[str, object]], output: Path) -> None:
    reliable = [
        row for row in rows
        if bool(row["discriminability_reliable"])
        and float(row["coverage_discriminability_sigma"]) > 0.0
    ]
    unreliable = [
        row for row in rows
        if not bool(row["discriminability_reliable"])
        and row["coverage_discriminability_sigma"] not in ("", None)
        and float(row["coverage_discriminability_sigma"]) > 0.0
    ]
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    if reliable:
        axis.scatter(
            [float(row["local_unsym_AB_separation_A"]) for row in reliable],
            [float(row["coverage_discriminability_sigma"]) for row in reliable],
            s=52, color="#1864ab", label=f"reliable (n≥{MIN_RELIABLE_FAILURES})",
            zorder=3,
        )
    if unreliable:
        axis.scatter(
            [float(row["local_unsym_AB_separation_A"]) for row in unreliable],
            [float(row["coverage_discriminability_sigma"]) for row in unreliable],
            s=52, facecolors="none", edgecolors="#868e96",
            label=f"unreliable (n<{MIN_RELIABLE_FAILURES})", zorder=3,
        )
    for row in rows:
        if row["site"] not in REFERENCE_SIGMA:
            continue
        sigma = row["coverage_discriminability_sigma"]
        if sigma in ("", None) or float(sigma) <= 0.0:
            continue
        is_right_anchor = row["site"] == "3A1C_B_ARG447"
        axis.annotate(
            str(row["site"]).split("_")[0],
            (
                float(row["local_unsym_AB_separation_A"]),
                float(sigma),
            ),
            xytext=((-8, 6) if is_right_anchor else (6, 6)),
            textcoords="offset points",
            horizontalalignment=("right" if is_right_anchor else "left"),
            fontsize=9,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Deposited A–B separation, local fixed labels (Å)")
    axis.set_ylabel("Coverage discriminability (σ; log scale)")
    axis.grid(True, which="both", linewidth=0.6, alpha=0.25)
    axis.legend(frameon=False, fontsize=9)
    axis.margins(x=0.04, y=0.10)
    figure.tight_layout()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def report_markdown(rows: list[dict[str, object]], guards: dict[str, object]) -> str:
    lines = [
        "# Frozen-v3 20-site coverage discriminability",
        "",
        f"Frozen control guard: **{' → '.join(map(str, guards['cascade']))}**.",
        "",
        "RSCCs use native additive density before z-scoring on the saved "
        "production Stage-1 mask. Scatter is the population standard deviation "
        "of RSCC across frozen-v3 endpoints that failed two-state recovery. "
        f"Values with fewer than {MIN_RELIABLE_FAILURES} failed endpoints are "
        "marked unreliable.",
        "",
        "| Site | Sep. Å | Correct | Coverage best wrong | Margin | Failed n | "
        "Scatter | σ | Reliability | Occ. margin | Endpoint margin |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in sorted(
        rows, key=lambda item: float(item["local_unsym_AB_separation_A"])
    ):
        sigma = row["coverage_discriminability_sigma"]
        sigma_text = "—" if sigma in ("", None) else f"{float(sigma):.3f}"
        scatter = row["failed_endpoint_rscc_scatter_population"]
        scatter_text = (
            "—" if scatter in ("", None) else f"{float(scatter):.6f}"
        )
        endpoint_margin = row["endpoint_margin"]
        endpoint_text = (
            "—" if endpoint_margin in ("", None)
            else f"{float(endpoint_margin):.6f}"
        )
        lines.append(
            f"| {row['site']} | {float(row['local_unsym_AB_separation_A']):.3f} "
            f"| {float(row['correct_candidate_rscc']):.6f} | "
            f"{row['coverage_best_wrong_candidate']} "
            f"({float(row['coverage_best_wrong_rscc']):.6f}) | "
            f"{float(row['coverage_margin']):.6f} | "
            f"{int(row['failed_endpoint_count'])} | "
            f"{scatter_text} "
            f"| {sigma_text} | {row['discriminability_status']} | "
            f"{float(row['occupancy_margin']):.6f} | {endpoint_text} |"
        )
    lines.extend(["", "## Anchor comparison", ""])
    for site, reference in REFERENCE_SIGMA.items():
        row = next(item for item in rows if item["site"] == site)
        measured = row["coverage_discriminability_sigma"]
        measured_text = "unavailable" if measured in ("", None) else (
            f"{float(measured):.4f}σ"
        )
        lines.append(
            f"- {site}: qfit {measured_text}; SampleWorks reference "
            f"{reference:.2f}σ; failed endpoint n={row['failed_endpoint_count']}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--replacement-root", type=Path, required=True)
    parser.add_argument("--separation-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    metric_root = (
        args.frozen_root / "analysis"
        / "metric_v3_protected_merge_sweep" / "0p5"
    )
    optimizer_rows, target_paths = load_optimizer_rows(
        args.baseline_root, args.replacement_root
    )
    active_by_slot, _, sites, ensembles, _ = load_v3_payload(metric_root)
    cascade = cascade_guard(ensembles, active_by_slot)
    if cascade != EXPECTED_CASCADE:
        raise RuntimeError(
            f"frozen-v3 cascade guard failed: {cascade} != {EXPECTED_CASCADE}"
        )
    separations = load_separations(args.separation_csv)

    per_candidate: list[dict[str, object]] = []
    per_endpoint: list[dict[str, object]] = []
    per_site: list[dict[str, object]] = []
    duplicate_failures = []
    target_validation = {}
    for site_name in sorted(sites):
        site = sites[site_name]
        geometry = build_site_geometry(site, target_paths[site_name])
        density_a = geometry["render"](geometry["deposited_A"])
        density_b = geometry["render"](geometry["deposited_B"])
        first_optimizer = optimizer_rows[(site_name, 0)]
        target_a = float(first_optimizer["target_A_occupancy"])
        target_b = float(first_optimizer["target_B_occupancy"])
        target = target_a * density_a + target_b * density_b
        saved_target = np.asarray(np.load(
            target_paths[site_name].parent
            / f"{site_name}_optimizer_synthetic_vector.npy"
        ), dtype=np.float64)
        normalized = (target - target.mean()) / max(
            target.std(ddof=1), 1e-6
        )
        target_validation[site_name] = float(
            np.linalg.norm(normalized - saved_target)
            / max(np.linalg.norm(saved_target), 1e-15)
        )

        candidate_rows, candidate_summary = candidate_rsccs(
            target, density_a, density_b, target_a, target_b
        )
        for row in candidate_rows:
            per_candidate.append({
                "site": site_name,
                "target_A_occupancy": target_a,
                "target_B_occupancy": target_b,
                **row,
            })
        if (
            candidate_summary["duplicate_max_absolute_density_error"] > 1e-12
            or candidate_summary["duplicate_rscc_absolute_error"] > 1e-12
        ):
            duplicate_failures.append(site_name)

        failed_rsccs = []
        for start in range(50):
            endpoint_density = render_endpoint(
                optimizer_rows[(site_name, start)], geometry
            )
            endpoint_rscc = pearson_correlation(target, endpoint_density)
            failed = not truth(
                ensembles[(site_name, start)]["both_found_conventional"]
            )
            per_endpoint.append({
                "site": site_name,
                "start": start,
                "both_found_frozen_v3": not failed,
                "recovery_failed_frozen_v3": failed,
                "endpoint_rscc": endpoint_rscc,
            })
            if failed:
                failed_rsccs.append(endpoint_rscc)

        failed_array = np.asarray(failed_rsccs, dtype=np.float64)
        failed_count = len(failed_array)
        population_scatter = (
            float(failed_array.std(ddof=0)) if failed_count else None
        )
        sample_scatter = (
            float(failed_array.std(ddof=1)) if failed_count >= 2 else None
        )
        coverage_margin = float(candidate_summary["coverage_margin"])
        sigma = (
            coverage_margin / population_scatter
            if population_scatter is not None and population_scatter > 0.0
            else None
        )
        best_endpoint_rscc = (
            float(failed_array.max()) if failed_count else None
        )
        endpoint_margin = (
            float(candidate_summary["correct_rscc"]) - best_endpoint_rscc
            if failed_count else None
        )
        reconstructed_separation = fixed_label_rmsd(
            geometry["deposited_A"], geometry["deposited_B"]
        )
        status = (
            "reliable" if failed_count >= MIN_RELIABLE_FAILURES
            else "unreliable_sparse"
        )
        if sigma is None or not math.isfinite(sigma):
            status = "unavailable_no_scatter"
        per_site.append({
            "site": site_name,
            "residue_type": geometry["residue_type"],
            "local_unsym_AB_separation_A": separations[site_name],
            "renderer_fixed_label_AB_rmsd_A": reconstructed_separation,
            "correct_candidate_rscc": candidate_summary["correct_rscc"],
            "coverage_best_wrong_candidate": (
                candidate_summary["coverage_best_wrong_candidate"]
            ),
            "coverage_best_wrong_rscc": (
                candidate_summary["coverage_best_wrong_rscc"]
            ),
            "coverage_margin": coverage_margin,
            "failed_endpoint_count": failed_count,
            "failed_endpoint_rscc_scatter_population": population_scatter,
            "failed_endpoint_rscc_scatter_sample": sample_scatter,
            "coverage_discriminability_sigma": sigma,
            "discriminability_reliable": (
                failed_count >= MIN_RELIABLE_FAILURES
                and sigma is not None and math.isfinite(sigma)
            ),
            "discriminability_status": status,
            "occupancy_best_wrong_candidate": (
                candidate_summary["occupancy_best_wrong_candidate"]
            ),
            "occupancy_best_wrong_rscc": (
                candidate_summary["occupancy_best_wrong_rscc"]
            ),
            "occupancy_margin": candidate_summary["occupancy_margin"],
            "endpoint_best_failed_start_rscc": best_endpoint_rscc,
            "endpoint_margin": endpoint_margin,
            "duplicate_max_absolute_density_error": (
                candidate_summary["duplicate_max_absolute_density_error"]
            ),
            "duplicate_rscc_absolute_error": (
                candidate_summary["duplicate_rscc_absolute_error"]
            ),
        })
        atomic_csv(args.output / "per_candidate.csv", per_candidate)
        atomic_csv(args.output / "per_endpoint.csv", per_endpoint)
        atomic_csv(args.output / "per_site.csv", per_site)
        atomic_json(args.output / "progress.json", {
            "status": "running",
            "completed_sites": [row["site"] for row in per_site],
        })

    if len(per_site) != 20 or len(per_endpoint) != 1000:
        raise RuntimeError("incomplete diagnostic output")
    guards = {
        "metric": METRIC,
        "cascade": list(cascade),
        "optimizer_runs": 0,
        "density_space": "native additive density before z-scoring",
        "mask": "saved production Stage-1 radial mask",
        "failed_endpoint_definition": (
            "not both_found_conventional under frozen-v3 protected "
            "one-to-one assignment"
        ),
        "scatter_definition": (
            "population standard deviation (ddof=0) of failed-endpoint RSCC"
        ),
        "duplicate_invariance_failures": duplicate_failures,
        "maximum_target_reconstruction_relative_l2": max(
            target_validation.values()
        ),
    }
    atomic_csv(args.output / "per_candidate.csv", per_candidate)
    atomic_csv(args.output / "per_endpoint.csv", per_endpoint)
    atomic_csv(args.output / "per_site.csv", per_site)
    atomic_json(args.output / "summary.json", {
        "guards": guards,
        "reference_sigma": REFERENCE_SIGMA,
        "sites": per_site,
    })
    plot_discriminability(
        per_site, args.output / "discriminability_vs_separation.png"
    )
    atomic_text(
        args.output / "report.md", report_markdown(per_site, guards)
    )
    atomic_json(args.output / "progress.json", {
        "status": "complete",
        "completed_sites": [row["site"] for row in per_site],
    })


if __name__ == "__main__":
    main()
