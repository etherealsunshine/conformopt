#!/usr/bin/env python3
"""Render projector-ready 7UTC deposited-altloc anchor-displacement charts."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SITE = "7UTC_A_ARG52"
OFFSETS = (-3, -2, -1, 0, 1, 2, 3)
KS = (3, 4, 5, 6)
SIGMA = 0.02


def read_single_closure_site(path: Path):
    rows = list(csv.DictReader(path.open()))
    if len(rows) != 1 or rows[0]["site"] != SITE:
        raise ValueError(f"expected exactly one {SITE} row in {path}")
    record = rows[0]
    profile = ast.literal_eval(record["per_window_position_max_abs_coordinate_difference_A"])
    return {offset: float(profile[str(offset)]) for offset in OFFSETS}


def read_wider_sites(path: Path):
    rows = list(csv.DictReader(path.open()))
    selected = [row for row in rows if row["site"] == SITE]
    by_k = {int(row["k"]): row for row in selected}
    missing = set(KS) - set(by_k)
    if missing:
        raise ValueError(f"missing {SITE} k values: {sorted(missing)}")
    profile = {
        k: float(by_k[k]["anchor_max_difference_A"])
        for k in KS
        if by_k[k]["anchor_max_difference_A"]
    }
    below = {}
    for row in rows:
        value = row["anchor_max_difference_A"]
        if value and float(value) < 0.1:
            below.setdefault(row["site"], []).append(int(row["k"]))
    return profile, {site: min(ks) for site, ks in below.items()}


def style_axes(ax, ymax):
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=17, width=1.1, length=5, color="#374151")
    ax.axhline(SIGMA, color="#64748b", linestyle=(0, (3, 2)), linewidth=2, zorder=1)
    ax.text(
        0.02, 0.96, "bond restraint sigma = 0.02 Å\n(essentially at the axis on this scale)",
        transform=ax.transAxes, ha="left", va="top", fontsize=14,
        color="#475569",
    )


def label_bars(ax, bars, values, ymax):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * 0.025,
            f"{value:.3f}", ha="center", va="bottom", fontsize=16,
            color="#111827", fontweight="semibold",
        )


def displacement_chart(profile, output: Path):
    labels = ["-3", "-2", "-1", "centre", "+1", "+2", "+3"]
    values = [profile[offset] for offset in OFFSETS]
    colors = ["#d55e00", "#4e79a7", "#4e79a7", "#6a3d9a", "#4e79a7", "#4e79a7", "#d55e00"]
    ymax = max(values) * 1.18
    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=160)
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, width=0.68, edgecolor="#1f2937", linewidth=0.8, zorder=2)
    style_axes(ax, ymax)
    label_bars(ax, bars, values, ymax)
    for index in (0, 6):
        ax.text(index, ymax * 0.12, "anchor", ha="center", va="center", fontsize=15,
                color="#9a3412", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Window position relative to flip centre", fontsize=20, labelpad=13)
    ax.set_ylabel("Max backbone displacement A to B (Å)", fontsize=20, labelpad=14)
    ax.set_title("7UTC A:ARG52 -- backbone displacement between deposited altlocs",
                 fontsize=22, pad=20, weight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def widening_chart(profile, output: Path):
    values = [profile[k] for k in KS]
    ymax = max(values) * 1.20
    fig, ax = plt.subplots(figsize=(12.5, 7.5), dpi=160)
    x = np.arange(len(KS))
    bars = ax.bar(x, values, color="#4e79a7", width=0.60, edgecolor="#1f2937", linewidth=0.8, zorder=2)
    style_axes(ax, ymax)
    label_bars(ax, bars, values, ymax)
    ax.set_xticks(x, [str(k) for k in KS])
    ax.set_xlabel("Anchor offset k (residues from flip centre)", fontsize=20, labelpad=13)
    ax.set_ylabel("Max anchor displacement A to B (Å)", fontsize=20, labelpad=14)
    ax.set_title("Widening the window does not find a settled anchor", fontsize=22, pad=20, weight="bold")
    fig.text(0.5, 0.012, "7UTC A:ARG52. k = 8 and 10 are unavailable because the window runs off the chain end.",
             ha="center", fontsize=15, color="#475569")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closure-csv", type=Path, required=True)
    parser.add_argument("--wider-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    displacement = read_single_closure_site(args.closure_csv)
    widening, below = read_wider_sites(args.wider_csv)
    displacement_chart(displacement, args.output / "7utc_arg52_backbone_displacement_profile")
    widening_chart(widening, args.output / "7utc_arg52_widening_anchor_displacement")
    summary = {
        "status": "complete",
        "source_closure_csv": str(args.closure_csv),
        "source_wider_csv": str(args.wider_csv),
        "site": SITE,
        "backbone_displacement_by_offset_A": displacement,
        "max_anchor_displacement_by_k_A": widening,
        "unavailable_k": [8, 10],
        "sites_ever_below_0p1_A": below,
        "count_sites_ever_below_0p1_A": len(below),
        "outputs": [
            str(args.output / "7utc_arg52_backbone_displacement_profile.png"),
            str(args.output / "7utc_arg52_backbone_displacement_profile.pdf"),
            str(args.output / "7utc_arg52_widening_anchor_displacement.png"),
            str(args.output / "7utc_arg52_widening_anchor_displacement.pdf"),
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
