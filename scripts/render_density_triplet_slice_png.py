"""Render matched central slices through an exported density triplet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from render_density_triplet_png import (
    ALT_COLORS,
    LABELS,
    NEGATIVE_COLOR,
    POSITIVE_COLOR,
    SIDECHAIN_BONDS,
    residue_atoms,
)


def add_slice_structure(ax, atoms, slice_z: float) -> None:
    for altloc, lookup in atoms.items():
        color = ALT_COLORS[altloc]
        for first, second in SIDECHAIN_BONDS:
            if first not in lookup or second not in lookup:
                continue
            xyz = np.stack([lookup[first], lookup[second]])
            distance = float(np.mean(np.abs(xyz[:, 2] - slice_z)))
            alpha = max(0.25, 1.0 - distance / 4.0)
            ax.plot(
                xyz[:, 0], xyz[:, 1], color=color, linewidth=4.0,
                solid_capstyle="round", alpha=alpha, zorder=8,
            )
        for name in {name for bond in SIDECHAIN_BONDS for name in bond}:
            if name not in lookup:
                continue
            xyz = lookup[name]
            distance = abs(float(xyz[2] - slice_z))
            alpha = max(0.25, 1.0 - distance / 4.0)
            ax.scatter(
                xyz[0], xyz[1], s=70 if name.startswith("O") else 55,
                color=color, edgecolor="white", linewidth=0.9,
                alpha=alpha, zorder=9,
            )


def render_slice(ax, volume, spacing, index, atoms, title, subtitle) -> None:
    coordinates = np.arange(volume.shape[0], dtype=np.float32) * spacing
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    values = volume[:, :, index].T
    maximum = float(values.max())
    minimum = float(values.min())
    if maximum > 1.0:
        upper = max(3.0, maximum)
        ax.contourf(
            x, y, values,
            levels=[1.0, 1.5, 2.0, 3.0, upper + 1e-5],
            colors=[POSITIVE_COLOR] * 4, alpha=0.13,
        )
        positive_levels = [level for level in (1.0, 2.0, 3.0) if level < maximum]
        ax.contour(
            x, y, values, levels=positive_levels, colors=POSITIVE_COLOR,
            linewidths=[1.5, 1.0, 0.8][:len(positive_levels)], alpha=0.9,
        )
    if minimum < -1.0:
        negative_levels = [level for level in (-3.0, -2.0, -1.0) if level > minimum]
        if negative_levels:
            ax.contour(
                x, y, values, levels=negative_levels,
                colors=NEGATIVE_COLOR, linewidths=0.9,
                linestyles="dashed", alpha=0.8,
            )
    slice_z = index * spacing
    add_slice_structure(ax, atoms, slice_z)
    ax.set_xlim(4.2, 10.7)
    ax.set_ylim(5.3, 11.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
        fontsize=20, fontweight="bold", color="#111827", zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.5},
    )
    ax.text(
        0.02, 0.905, subtitle, transform=ax.transAxes, ha="left", va="top",
        fontsize=12, color="#4b5563", zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.0},
    )
    x0, y0 = 8.2, 5.65
    ax.plot([x0, x0 + 2.0], [y0, y0], color="#111827", linewidth=2.0)
    ax.text(x0 + 1.0, y0 + 0.13, "2 Å", ha="center", va="bottom", fontsize=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.input / "manifest.json").read_text())
    with np.load(next(args.input.glob("*_density_triplet.npz")), allow_pickle=False) as archive:
        volumes = {key: np.asarray(archive[key]) for key in LABELS}
    pdb_path = next(args.input.glob("*_local_frame.pdb"))
    site_parts = manifest["site"].split("_")
    chain = site_parts[1]
    residue_label = site_parts[2]
    residue_name = "".join(c for c in residue_label if c.isalpha())
    residue_number = int("".join(c for c in residue_label if c.isdigit()))
    atoms = residue_atoms(pdb_path, chain, residue_number)
    spacing = float(manifest["spacing_angstrom"])
    index = int(round((volumes[next(iter(volumes))].shape[2] - 1) / 2.0))

    for key, volume in volumes.items():
        figure, axis = plt.subplots(figsize=(6.4, 5.6), dpi=300)
        figure.patch.set_alpha(0.0)
        axis.set_facecolor("none")
        render_slice(axis, volume, spacing, index, atoms, *LABELS[key])
        figure.tight_layout(pad=0.2)
        figure.savefig(
            args.output / f"{manifest['site']}_{key}_central_slice.png",
            dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04,
        )
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.3), dpi=300)
    figure.patch.set_facecolor("white")
    for axis, (key, volume) in zip(axes, volumes.items()):
        render_slice(axis, volume, spacing, index, atoms, *LABELS[key])
    figure.legend(
        handles=[
            Line2D([0], [0], color=ALT_COLORS["A"], lw=4, label="Deposited A (57%)"),
            Line2D([0], [0], color=ALT_COLORS["B"], lw=4, label="Deposited B (43%)"),
            Line2D([0], [0], color=POSITIVE_COLOR, lw=2, label="Positive density (1σ+)"),
            Line2D([0], [0], color=NEGATIVE_COLOR, lw=1, ls="--", label="Negative density"),
        ],
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005), fontsize=11,
    )
    figure.suptitle(
        f"{site_parts[0]}  {chain}:{residue_name}{residue_number}  ·  matched central slice",
        fontsize=22, fontweight="bold", y=0.995,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.12, wspace=0.04)
    figure.savefig(
        args.output / f"{manifest['site']}_density_comparison_central_slice.png",
        dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
