#!/usr/bin/env python3
"""Draw an editable two-panel schematic of qFit backbone sampling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch, Ellipse
import numpy as np


# Slide-adjustable visual constants.  The panels use an approximately
# pixel-coordinate canvas at the selected 2,000 px output width.
CANVAS_PX = 800
TARGET_STEP_PX = 15
TARGET_RADII_PX = (1, 2, 3)
RESIDUE_RADIUS_PX = 42
CBETA_RADIUS_PX = 31

CORAL = "#D55E00"       # welded ends and enumerated target points
AMBER = "#E69F00"       # Cbeta atom
PURPLE = "#6F4AA1"      # ADP ellipsoid and principal axes
NEUTRAL = "#9CA3AF"     # interior residues, bonds, and zoom connector
TEXT = "#111827"
BACKGROUND = "white"


def setup_panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, CANVAS_PX)
    ax.set_ylim(0, CANVAS_PX)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def text(ax, x, y, label, size=25, **kwargs):
    kwargs.setdefault("color", TEXT)
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    return ax.text(x, y, label, fontsize=size, **kwargs)


def draw_window(ax):
    residue_y = 470
    residue_x = np.linspace(110, 690, 7)
    for first, second in zip(residue_x[:-1], residue_x[1:]):
        ax.plot((first, second), (residue_y, residue_y), color=NEUTRAL, linewidth=7,
                solid_capstyle="round", zorder=1)
    for index, x in enumerate(residue_x):
        welded = index in (0, 6)
        ax.add_patch(Circle((x, residue_y), RESIDUE_RADIUS_PX,
                            facecolor=CORAL if welded else "#D1D5DB",
                            edgecolor=TEXT, linewidth=2.1, zorder=3))
        if welded:
            text(ax, x, residue_y + 84, "welded", size=22, color="#9A3412", fontweight="bold")

    centre_x = residue_x[3]
    cbeta_y = 272
    ax.plot((centre_x, centre_x), (residue_y - RESIDUE_RADIUS_PX, cbeta_y + CBETA_RADIUS_PX),
            color=NEUTRAL, linewidth=6, solid_capstyle="round", zorder=1)
    ax.add_patch(Circle((centre_x, cbeta_y), CBETA_RADIUS_PX, facecolor=AMBER,
                        edgecolor=TEXT, linewidth=2.1, zorder=4))
    text(ax, centre_x, cbeta_y - 63, "Cβ", size=25, fontweight="bold")
    text(ax, 400, 66, "7 residues, both ends fixed", size=16, fontweight="semibold")
    return centre_x, cbeta_y


def draw_targets(ax):
    cx, cy = 400, 410
    ellipse_width, ellipse_height = 270, 135
    ax.add_patch(Ellipse((cx, cy), width=ellipse_width, height=ellipse_height,
                         facecolor=PURPLE, edgecolor=PURPLE, linewidth=2.5, alpha=0.22, zorder=1))

    # Three projected principal axes.  Target positions deliberately use the
    # same normalized step on every axis: orientation comes from the ADP, not
    # its eigenvalues/ellipse extent.
    angles_deg = (8, 68, 128)
    axis_extent = 85
    for angle in angles_deg:
        direction = np.array((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))))
        start, end = np.array((cx, cy)) - axis_extent * direction, np.array((cx, cy)) + axis_extent * direction
        ax.plot((start[0], end[0]), (start[1], end[1]), color=PURPLE, alpha=0.42,
                linewidth=2.3, zorder=2)
        for sign in (-1, 1):
            for multiple in TARGET_RADII_PX:
                point = np.array((cx, cy)) + sign * multiple * TARGET_STEP_PX * direction
                ax.add_patch(Circle(point, 8, facecolor=CORAL, edgecolor=BACKGROUND,
                                    linewidth=0.8, zorder=4))

    ax.add_patch(Circle((cx, cy), 14, facecolor=AMBER, edgecolor=TEXT, linewidth=1.8, zorder=5))
    text(ax, cx, cy - 46, "Cβ", size=23, fontweight="bold")
    text(ax, 400, 77, "18 target points: +/-0.1 / +/-0.2 / +/-0.3 Å on 3 ADP axes", size=14)
    text(ax, 400, 40, "+ the unperturbed input = 19 candidates", size=14, fontweight="semibold")
    return cx - ellipse_width / 2, cy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    fig = plt.figure(figsize=(12.5, 6.5), dpi=160, facecolor=BACKGROUND)
    left = setup_panel(fig, (0.035, 0.11, 0.40, 0.80))
    right = setup_panel(fig, (0.565, 0.11, 0.40, 0.80))
    cbeta_x, cbeta_y = draw_window(left)
    ellipse_left, ellipse_y = draw_targets(right)

    for vertical_offset in (-16, 16):
        fig.add_artist(ConnectionPatch(
            xyA=(cbeta_x + CBETA_RADIUS_PX, cbeta_y + vertical_offset), coordsA=left.transData,
            xyB=(ellipse_left, ellipse_y + vertical_offset * 2.0), coordsB=right.transData,
            color=NEUTRAL, linewidth=1.7, linestyle=(0, (4, 4)), zorder=0,
        ))

    basename = args.output / "qfit_backbone_sampling_schematic"
    fig.savefig(basename.with_suffix(".png"), dpi=160, facecolor=BACKGROUND)
    fig.savefig(basename.with_suffix(".pdf"), facecolor=BACKGROUND)
    plt.close(fig)
    summary = {
        "status": "complete",
        "output_png": str(basename.with_suffix(".png")),
        "output_pdf": str(basename.with_suffix(".pdf")),
        "script": str(Path(__file__).resolve()),
        "target_spacing_pixels": TARGET_STEP_PX,
        "target_distances_pixels": [TARGET_STEP_PX * multiple for multiple in TARGET_RADII_PX],
        "target_count": 18,
        "total_candidates": 19,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
