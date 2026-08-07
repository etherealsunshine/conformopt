"""Reframe frozen-v3 discriminability around deposited-candidate margins.

This reporting-only diagnostic consumes the already-computed per-site candidate
RSCC table from frozen_v3_coverage_discriminability_v4.  It does not render a
density, reconstruct an endpoint, recompute endpoint scatter, or invoke an
optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPECTED_CASCADE = [742, 714, 710, 710, 710, 626]
METRIC = "qfit-synth20-merge050-one-to-one-tmol044-v3"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, newline="", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including deterministic tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def normalize_site_name(site: str) -> str:
    return site.split("_", 1)[0]


def build_rows(source_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = sorted(
        source_rows,
        key=lambda row: float(row["local_unsym_AB_separation_A"]),
    )
    output = []
    for order, row in enumerate(rows, start=1):
        output.append({
            "separation_rank": order,
            "site": row["site"],
            "site_label": normalize_site_name(row["site"]),
            "local_unsym_AB_separation_A": float(
                row["local_unsym_AB_separation_A"]
            ),
            "correct_candidate_rscc": float(row["correct_candidate_rscc"]),
            "coverage_best_wrong_candidate": row[
                "coverage_best_wrong_candidate"
            ],
            "coverage_best_wrong_rscc": float(
                row["coverage_best_wrong_rscc"]
            ),
            "coverage_margin": float(row["coverage_margin"]),
            "occupancy_best_wrong_candidate": row[
                "occupancy_best_wrong_candidate"
            ],
            "occupancy_best_wrong_rscc": float(
                row["occupancy_best_wrong_rscc"]
            ),
            "occupancy_margin": float(row["occupancy_margin"]),
            "legacy_endpoint_scatter_sigma": (
                float(row["coverage_discriminability_sigma"])
                if row["coverage_discriminability_sigma"] else None
            ),
            "legacy_sigma_note": (
                "Confounded: denominator is failed-endpoint RSCC scatter, "
                "which measures failure consistency rather than detectability. "
                "Retained for provenance only; do not use as a headline."
            ),
            "duplicate_max_absolute_density_error": float(
                row["duplicate_max_absolute_density_error"]
            ),
            "duplicate_rscc_absolute_error": float(
                row["duplicate_rscc_absolute_error"]
            ),
        })
    return output


def plot_margins(rows: list[dict[str, object]], output: Path) -> None:
    labels = [str(row["site_label"]) for row in rows]
    separation = np.asarray([
        float(row["local_unsym_AB_separation_A"]) for row in rows
    ])
    coverage = np.asarray([float(row["coverage_margin"]) for row in rows])
    occupancy = np.asarray([float(row["occupancy_margin"]) for row in rows])
    positions = np.arange(len(rows))

    figure, (coverage_axis, occupancy_axis) = plt.subplots(
        2, 1, figsize=(15.0, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.14},
    )
    coverage_bars = coverage_axis.bar(
        positions, coverage, width=0.76, color="#2f6b9a"
    )
    occupancy_axis.bar(
        positions, occupancy, width=0.76, color="#c46f2c"
    )

    coverage_axis.set_ylabel("Coverage margin (ΔRSCC)")
    coverage_axis.set_title(
        "Coverage: correct A+B minus best single-state candidate",
        loc="left", fontsize=12,
    )
    rho = spearman_correlation(separation, coverage)
    coverage_axis.text(
        0.99, 0.95, f"Spearman ρ = {rho:.3f}",
        transform=coverage_axis.transAxes, ha="right", va="top", fontsize=10,
    )
    for bar, value in zip(coverage_bars, separation):
        coverage_axis.annotate(
            f"{value:.2f} Å",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", rotation=90, fontsize=7.5,
        )
    coverage_axis.set_ylim(0.0, max(coverage) * 1.28)

    occupancy_axis.set_ylabel("Occupancy margin (ΔRSCC)")
    occupancy_axis.set_title(
        "Occupancy: same positions, best wrong split "
        "(expanded y-axis; ≈6× smaller maximum)",
        loc="left", fontsize=12,
    )
    occupancy_axis.set_xlabel(
        "Site, ordered by local fixed-label A–B separation"
    )
    occupancy_axis.set_xticks(positions)
    occupancy_axis.set_xticklabels(labels, rotation=45, ha="right")
    occupancy_axis.set_ylim(0.0, max(occupancy) * 1.16)

    for axis in (coverage_axis, occupancy_axis):
        axis.grid(axis="y", linewidth=0.6, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.15)
    figure.savefig(output, dpi=240)
    plt.close(figure)


def report_markdown(
    rows: list[dict[str, object]],
    rho: float,
    maximum_reconstruction_error: float,
) -> str:
    lines = [
        "# Frozen-v3 deposited-candidate margin revision",
        "",
        f"Frozen control guard: **{' → '.join(map(str, EXPECTED_CASCADE))}**.",
        "",
        "Coverage margin is the primary quantity. It depends only on deposited "
        "candidate densities: `RSCC(correct A+B) - RSCC(best single-state "
        "candidate)`.",
        "",
        f"Across all 20 sites, Spearman correlation between local fixed-label "
        f"A–B separation and coverage margin is **ρ = {rho:.4f}**.",
        "",
        "The prior σ values remain in `per_site_margin_revision.csv` only for "
        "provenance. Their denominator is failed-endpoint scatter, so they are "
        "confounded by how consistently a site fails and are not interpreted.",
        "",
        "| Site | Separation Å | Correct RSCC | Best coverage wrong | "
        "Coverage margin | Best occupancy wrong | Occupancy margin |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['site_label']} | "
            f"{float(row['local_unsym_AB_separation_A']):.3f} | "
            f"{float(row['correct_candidate_rscc']):.6f} | "
            f"{row['coverage_best_wrong_candidate']} "
            f"({float(row['coverage_best_wrong_rscc']):.6f}) | "
            f"{float(row['coverage_margin']):.6f} | "
            f"{row['occupancy_best_wrong_candidate']} "
            f"({float(row['occupancy_best_wrong_rscc']):.6f}) | "
            f"{float(row['occupancy_margin']):.6f} |"
        )
    lines.extend([
        "",
        "## Verification",
        "",
        "- A duplicated at 0.5 + 0.5 was numerically identical to A alone "
        "at all 20 sites: zero density and RSCC discrepancy.",
        f"- Maximum raw-target reconstruction relative L2 error remained "
        f"`{maximum_reconstruction_error:.8g}`.",
        "- No density was re-rendered, no endpoint was read, and no σ or "
        "scatter value was recomputed.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    source_summary = json.loads((args.input / "summary.json").read_text())
    source_rows = read_csv(args.input / "per_site.csv")
    guards = source_summary["guards"]
    if guards["cascade"] != EXPECTED_CASCADE:
        raise RuntimeError(
            f"frozen-v3 guard failed: {guards['cascade']} != {EXPECTED_CASCADE}"
        )
    if guards["metric"] != METRIC:
        raise RuntimeError(f"unexpected metric: {guards['metric']}")
    if len(source_rows) != 20:
        raise RuntimeError(f"expected 20 source rows, found {len(source_rows)}")

    rows = build_rows(source_rows)
    if not all(
        math.isclose(float(row["correct_candidate_rscc"]), 1.0, abs_tol=1e-12)
        for row in rows
    ):
        raise RuntimeError("correct-candidate RSCC must be 1.0 by construction")
    if any(
        float(row["duplicate_max_absolute_density_error"]) != 0.0
        or float(row["duplicate_rscc_absolute_error"]) != 0.0
        for row in rows
    ):
        raise RuntimeError("A-duplicate numerical invariance failed")

    separation = np.asarray([
        float(row["local_unsym_AB_separation_A"]) for row in rows
    ])
    coverage = np.asarray([float(row["coverage_margin"]) for row in rows])
    rho = spearman_correlation(separation, coverage)
    best_wrong_counts = {
        candidate: sum(
            row["coverage_best_wrong_candidate"] == candidate for row in rows
        )
        for candidate in ("A_alone", "B_alone", "A_duplicated")
    }
    maximum_reconstruction_error = float(
        guards["maximum_target_reconstruction_relative_l2"]
    )

    atomic_csv(args.output / "per_site_margin_revision.csv", rows)
    atomic_json(args.output / "summary.json", {
        "provenance": {
            "source": str(args.input),
            "metric": METRIC,
            "cascade": EXPECTED_CASCADE,
            "density_renders": 0,
            "optimizer_runs": 0,
            "endpoint_rows_read": 0,
            "sigma_recomputed": False,
            "scatter_recomputed": False,
        },
        "primary_quantity": "coverage_margin",
        "spearman_separation_vs_coverage_margin": rho,
        "best_coverage_wrong_candidate_counts": best_wrong_counts,
        "verification": {
            "A_duplicate_density_invariance_failures": 0,
            "A_duplicate_rscc_invariance_failures": 0,
            "maximum_target_reconstruction_relative_l2": (
                maximum_reconstruction_error
            ),
        },
    })
    plot_margins(rows, args.output / "coverage_and_occupancy_margins.png")
    atomic_text(
        args.output / "report.md",
        report_markdown(rows, rho, maximum_reconstruction_error),
    )
    atomic_json(args.output / "progress.json", {
        "status": "complete",
        "sites": len(rows),
    })


if __name__ == "__main__":
    main()
