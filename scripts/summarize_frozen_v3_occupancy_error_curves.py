"""Reshape saved frozen-v3 occupancy candidates into error-margin curves.

This diagnostic deliberately reads only tables produced by
``diagnose_frozen_v3_coverage_discriminability.py``. It does not render density,
read optimizer endpoints, or recompute the frozen metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_CASCADE = [742, 714, 710, 710, 710, 626]
EXPECTED_METRIC = "qfit-synth20-merge050-one-to-one-tmol044-v3"
OCCUPANCY_CANDIDATES = {
    "A0.25_B0.75",
    "A0.50_B0.50",
    "A0.75_B0.25",
    "A0.90_B0.10",
}
ERROR_BIN_EDGES = np.arange(0.0, 0.71, 0.10)
SAMPLEWORKS_REFERENCE = {
    "3A1C": {
        0.06: 0.0217,
        0.19: 0.1183,
        0.31: 0.4365,
        0.46: 0.6563,
    },
    "5Z8H": {
        0.03: 0.000143,
        0.22: 0.0078,
        0.47: 0.0356,
        0.62: 0.0608,
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _site_label(site: str) -> str:
    return site.split("_", 1)[0]


def _average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = np.asarray(values)[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman requires equal-length vectors with >=2 values")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def error_bin_label(error: float) -> str:
    if error < 0 or error >= ERROR_BIN_EDGES[-1]:
        raise ValueError(f"Occupancy error outside configured bins: {error}")
    index = int(np.searchsorted(ERROR_BIN_EDGES, error, side="right") - 1)
    lower = ERROR_BIN_EDGES[index]
    upper = ERROR_BIN_EDGES[index + 1]
    return f"[{lower:.1f}, {upper:.1f})"


def _crosscheck(
    rows: list[dict],
    *,
    error_tolerance: float = 0.006,
) -> list[dict]:
    output = []
    for site, references in SAMPLEWORKS_REFERENCE.items():
        site_rows = [row for row in rows if row["site_label"] == site]
        for ref_error, ref_margin in references.items():
            nearest = min(
                site_rows,
                key=lambda row: abs(row["occupancy_error_A_fraction"] - ref_error),
            )
            error_delta = abs(nearest["occupancy_error_A_fraction"] - ref_error)
            if error_delta > error_tolerance:
                raise RuntimeError(
                    f"No qfit decoy near SampleWorks error {ref_error} for {site}"
                )
            qfit_margin = nearest["margin"]
            relative_delta = abs(qfit_margin - ref_margin) / ref_margin
            output.append(
                {
                    "site": site,
                    "occupancy_error": nearest["occupancy_error_A_fraction"],
                    "qfit_margin": qfit_margin,
                    "sampleworks_margin": ref_margin,
                    "absolute_delta": qfit_margin - ref_margin,
                    "relative_delta": relative_delta,
                    "material_discrepancy": relative_delta > 0.10,
                }
            )
    return output


def build_rows(
    candidates: list[dict[str, str]],
    per_site: list[dict[str, str]],
) -> tuple[list[dict], list[dict]]:
    coverage_by_site = {
        row["site"]: float(row["coverage_margin"]) for row in per_site
    }
    output: list[dict] = []
    for row in candidates:
        if row["candidate_class"] != "occupancy":
            continue
        if row["matches_deposited_occupancy"] == "True":
            # This is not a wrong-occupancy decoy. In the frozen table this
            # occurs once: 6H59's deposited and tested splits are both 0.50/0.50.
            continue
        if row["candidate"] not in OCCUPANCY_CANDIDATES:
            raise RuntimeError(f"Unexpected occupancy candidate: {row['candidate']}")
        target_a = float(row["target_A_occupancy"])
        target_b = float(row["target_B_occupancy"])
        decoy_a = float(row["occupancy_A"])
        decoy_b = float(row["occupancy_B"])
        rscc = float(row["rscc"])
        correct_rows = [
            candidate
            for candidate in candidates
            if candidate["site"] == row["site"]
            and candidate["candidate"] == "correct"
        ]
        if len(correct_rows) != 1:
            raise RuntimeError(f"Expected one correct row for {row['site']}")
        correct_rscc = float(correct_rows[0]["rscc"])
        output.append(
            {
                "site": row["site"],
                "site_label": _site_label(row["site"]),
                "deposited_A_occupancy": target_a,
                "deposited_B_occupancy": target_b,
                "decoy": row["candidate"],
                "decoy_A_occupancy": decoy_a,
                "decoy_B_occupancy": decoy_b,
                "occupancy_error_A_fraction": abs(decoy_a - target_a),
                "decoy_rscc": rscc,
                "correct_rscc": correct_rscc,
                "margin": correct_rscc - rscc,
                "coverage_margin": coverage_by_site[row["site"]],
            }
        )
    thresholds = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in output:
        grouped[row["site"]].append(row)
    for site, site_rows in sorted(grouped.items()):
        qualifying = [
            row for row in site_rows if row["margin"] > row["coverage_margin"]
        ]
        best = (
            min(qualifying, key=lambda row: row["occupancy_error_A_fraction"])
            if qualifying
            else None
        )
        thresholds.append(
            {
                "site": site,
                "site_label": _site_label(site),
                "coverage_margin": site_rows[0]["coverage_margin"],
                "smallest_tested_error_exceeding_coverage_margin": (
                    best["occupancy_error_A_fraction"] if best else None
                ),
                "decoy_at_threshold": best["decoy"] if best else "none_tested",
                "margin_at_threshold": best["margin"] if best else None,
                "threshold_status": (
                    "observed_on_tested_grid"
                    if best
                    else "not_reached_on_tested_grid"
                ),
                "maximum_tested_error": max(
                    row["occupancy_error_A_fraction"] for row in site_rows
                ),
                "maximum_tested_margin": max(row["margin"] for row in site_rows),
            }
        )
    return output, thresholds


def pooled_bins(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[error_bin_label(row["occupancy_error_A_fraction"])].append(
            row["margin"]
        )
    output = []
    for lower in ERROR_BIN_EDGES[:-1]:
        label = f"[{lower:.1f}, {lower + 0.1:.1f})"
        values = grouped.get(label, [])
        output.append(
            {
                "error_bin": label,
                "lower_bound_inclusive": lower,
                "upper_bound_exclusive": lower + 0.1,
                "pair_count": len(values),
                "median_margin": float(np.median(values)) if values else None,
            }
        )
    return output


def make_plot(
    rows: list[dict],
    bins: list[dict],
    output: Path,
    rho: float,
) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["site_label"]].append(row)

    fig, ax = plt.subplots(figsize=(15, 9))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(grouped)))
    for color, (site, site_rows) in zip(colors, sorted(grouped.items())):
        ordered = sorted(
            site_rows,
            key=lambda row: (
                row["occupancy_error_A_fraction"],
                row["decoy_A_occupancy"],
            ),
        )
        x_values = [row["occupancy_error_A_fraction"] for row in ordered]
        y_values = [row["margin"] for row in ordered]
        ax.scatter(
            x_values,
            y_values,
            marker="o",
            s=28,
            alpha=0.72,
            color=color,
            label=site,
        )
        # Preserve site trajectories without drawing vertical segments between
        # equal-error decoys. When both directions give the same absolute error
        # (6H59 at 0.25), the trajectory passes through their median margin while
        # both individual observations remain visible as markers.
        y_by_x: dict[float, list[float]] = defaultdict(list)
        for x_value, y_value in zip(x_values, y_values):
            y_by_x[round(x_value, 10)].append(y_value)
        unique_x = sorted(y_by_x)
        ax.plot(
            unique_x,
            [float(np.median(y_by_x[x_value])) for x_value in unique_x],
            linewidth=1.1,
            alpha=0.72,
            color=color,
        )

    populated = [row for row in bins if row["pair_count"]]
    bin_centers = [
        (row["lower_bound_inclusive"] + row["upper_bound_exclusive"]) / 2
        for row in populated
    ]
    ax.plot(
        bin_centers,
        [row["median_margin"] for row in populated],
        color="black",
        marker="D",
        markersize=6,
        linewidth=2.4,
        label="Pooled median (0.1 bins)",
        zorder=10,
    )
    ax.set_yscale("log")
    ax.set_xlim(-0.015, 0.68)
    ax.set_xlabel("Absolute occupancy error for conformer A")
    ax.set_ylabel("RSCC margin: correct split − decoy split")
    ax.set_title(
        "Occupancy-error curves from saved deposited-coordinate candidates"
    )
    ax.grid(axis="both", alpha=0.22)
    ax.text(
        0.02,
        0.96,
        f"Pooled Spearman ρ = {rho:.3f} (n = {len(rows)} site–decoy pairs)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=7,
        fontsize=8.5,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.25, left=0.09, right=0.98, top=0.92)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _fmt(value: float | None, digits: int = 6) -> str:
    return "not reached" if value is None else f"{value:.{digits}f}"


def make_report(
    *,
    rows: list[dict],
    thresholds: list[dict],
    bins: list[dict],
    crosscheck: list[dict],
    rho: float,
    summary: dict,
) -> str:
    lines = [
        "# Frozen-v3 occupancy-error curves",
        "",
        "Frozen control guard: **742 → 714 → 710 → 710 → 710 → 626**.",
        "",
        f"This report reshapes the {len(rows)} saved wrong-occupancy candidate "
        "RSCC values. "
        "It performs no density rendering, optimizer run, endpoint read, or "
        "metric recomputation.",
        "",
        "## SampleWorks cross-check — discrepancy precedes interpretation",
        "",
        "| Site | Occupancy error | qfit margin | SampleWorks margin | Relative difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in crosscheck:
        lines.append(
            f"| {row['site']} | {row['occupancy_error']:.2f} | "
            f"{row['qfit_margin']:.6f} | {row['sampleworks_margin']:.6f} | "
            f"{row['relative_delta'] * 100:.1f}% |"
        )
    if any(row["material_discrepancy"] for row in crosscheck):
        lines.extend(
            [
                "",
                "**The saved qfit margins materially disagree with the supplied "
                "SampleWorks values (>10% relative difference), so the panel-wide "
                "qfit trend should not be treated as a numerical reproduction of "
                "that two-site measurement.**",
            ]
        )
    lines.extend(
        [
            "",
            "## Pooled relationship",
            "",
            f"Across all 80 site–decoy pairs, Spearman correlation between "
            f"occupancy error and RSCC margin is **ρ = {rho:.4f}**.",
            "",
            "| Occupancy-error bin | Pairs | Median margin |",
            "|---|---:|---:|",
        ]
    )
    for row in bins:
        lines.append(
            f"| {row['error_bin']} | {row['pair_count']} | "
            f"{_fmt(row['median_margin'])} |"
        )
    lines.extend(
        [
            "",
            "## Smallest tested error exceeding the site's coverage margin",
            "",
            "These are sampled-grid thresholds, not interpolated physical limits.",
            "",
            "| Site | Coverage margin | Smallest tested error | Decoy | Margin at threshold | Status |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for row in thresholds:
        lines.append(
            f"| {row['site_label']} | {row['coverage_margin']:.6f} | "
            f"{_fmt(row['smallest_tested_error_exceeding_coverage_margin'], 3)} | "
            f"{row['decoy_at_threshold']} | "
            f"{_fmt(row['margin_at_threshold'])} | {row['threshold_status']} |"
        )
    lines.extend(
        [
            "",
            "## Sampling limitation",
            "",
            "The tested decoy grid is coarse and site-independent, so occupancy "
            "errors are unevenly sampled across sites. For example, 5Z8H has a "
            "decoy at 0.03 error, while 6H59's nearest tested error is 0.25. "
            "Plotting margin against the actual error mitigates the bar-chart "
            "comparability artifact but does not eliminate the sampling gap.",
            "",
            "A finer occupancy grid is worth computing if a quantitative "
            "detectability threshold is needed: the current four-point grid can "
            "only bracket thresholds coarsely and leaves several sites with no "
            "tested decoy beyond their coverage margin. It is not needed to "
            "establish the qualitative monotonic pooled relationship.",
            "",
            "## Verification",
            "",
            f"- Frozen metric: `{summary['provenance']['metric']}`.",
            f"- Candidate rows read: {len(rows)} wrong-occupancy decoys from "
            "`per_candidate.csv`.",
            "- Density renders: 0; optimizer runs: 0; endpoint rows read: 0.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_summary = json.loads((args.input / "summary.json").read_text())
    cascade = source_summary["guards"]["cascade"]
    metric = source_summary["guards"]["metric"]
    if cascade != EXPECTED_CASCADE:
        raise RuntimeError(f"Frozen cascade mismatch: {cascade}")
    if metric != EXPECTED_METRIC:
        raise RuntimeError(f"Frozen metric mismatch: {metric}")

    candidates = _read_csv(args.input / "per_candidate.csv")
    per_site = _read_csv(args.input / "per_site.csv")
    rows, thresholds = build_rows(candidates, per_site)
    if len(rows) != 79:
        raise RuntimeError(
            f"Expected 79 wrong-occupancy site-decoy rows, found {len(rows)}"
        )
    bins = pooled_bins(rows)
    rho = spearman(
        [row["occupancy_error_A_fraction"] for row in rows],
        [row["margin"] for row in rows],
    )
    crosscheck = _crosscheck(rows)

    args.output.mkdir(parents=True, exist_ok=False)
    _write_csv(
        args.output / "occupancy_error_curve.csv",
        rows,
        list(rows[0]),
    )
    _write_csv(
        args.output / "detectability_thresholds.csv",
        thresholds,
        list(thresholds[0]),
    )
    _write_csv(
        args.output / "pooled_error_bins.csv",
        bins,
        list(bins[0]),
    )
    _write_csv(
        args.output / "sampleworks_crosscheck.csv",
        crosscheck,
        list(crosscheck[0]),
    )
    make_plot(rows, bins, args.output / "occupancy_error_curves.png", rho)

    result_summary = {
        "control_guard": {
            "cascade": cascade,
            "metric": metric,
            "passed": True,
        },
        "pooled": {
            "site_count": len({row["site"] for row in rows}),
            "site_decoy_pair_count": len(rows),
            "spearman_occupancy_error_vs_margin": rho,
            "error_bins": bins,
        },
        "detectability": {
            "sites_reaching_coverage_margin_on_tested_grid": sum(
                row["threshold_status"] == "observed_on_tested_grid"
                for row in thresholds
            ),
            "sites_not_reaching_coverage_margin_on_tested_grid": sum(
                row["threshold_status"] == "not_reached_on_tested_grid"
                for row in thresholds
            ),
        },
        "sampleworks_crosscheck": {
            "material_discrepancy_definition": "relative difference > 10%",
            "material_discrepancy_count": sum(
                row["material_discrepancy"] for row in crosscheck
            ),
            "rows": crosscheck,
        },
        "provenance": {
            "source": str(args.input),
            "source_files": ["summary.json", "per_site.csv", "per_candidate.csv"],
            "density_renders": 0,
            "optimizer_runs": 0,
            "endpoint_rows_read": 0,
            "metric_recomputed": False,
        },
    }
    _write_json(args.output / "summary.json", result_summary)
    (args.output / "report.md").write_text(
        make_report(
            rows=rows,
            thresholds=thresholds,
            bins=bins,
            crosscheck=crosscheck,
            rho=rho,
            summary={"provenance": {"metric": metric}},
        )
    )
    _write_json(
        args.output / "progress.json",
        {"status": "complete", "site_decoy_pair_count": len(rows)},
    )


if __name__ == "__main__":
    main()
