"""Render deck-ready 2D projections of an exported density triplet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LABELS = {
    "experimental_omit_mfo_dfc": (
        "Experimental density",
        "Normalized omit mFo–DFc",
    ),
    "denoised": (
        "Denoised density",
        "Frozen 3D U-Net output",
    ),
    "synthetic_ground_truth": (
        "Ground-truth density",
        "Deposited A/B synthetic target",
    ),
}

SIDECHAIN_BONDS = (
    ("CA", "CB"),
    ("CB", "CG"),
    ("CG", "OD1"),
    ("CG", "OD2"),
)

ALT_COLORS = {"A": "#f59e0b", "B": "#d946ef"}
POSITIVE_COLOR = "#168aad"
NEGATIVE_COLOR = "#dc2626"


def residue_atoms(pdb: Path, chain: str, residue_number: int) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
    for line in pdb.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[21].strip() != chain or int(line[22:26]) != residue_number:
            continue
        altloc = line[16].strip()
        if altloc not in result:
            continue
        result[altloc][line[12:16].strip()] = np.asarray([
            float(line[30:38]), float(line[38:46]), float(line[46:54])
        ])
    if not all(result.values()):
        raise RuntimeError(f"missing deposited A/B atoms for {chain}:{residue_number}")
    return result


def add_structure(ax, atoms: dict[str, dict[str, np.ndarray]]) -> None:
    for altloc, lookup in atoms.items():
        color = ALT_COLORS[altloc]
        for first, second in SIDECHAIN_BONDS:
            if first not in lookup or second not in lookup:
                continue
            xyz = np.stack([lookup[first], lookup[second]])
            ax.plot(
                xyz[:, 0], xyz[:, 1], color=color, linewidth=4.0,
                solid_capstyle="round", zorder=8,
            )
        for name in {name for bond in SIDECHAIN_BONDS for name in bond}:
            if name not in lookup:
                continue
            xyz = lookup[name]
            size = 70 if name.startswith("O") else 55
            ax.scatter(
                xyz[0], xyz[1], s=size, color=color, edgecolor="white",
                linewidth=0.9, zorder=9,
            )


def render_panel(
    ax,
    volume: np.ndarray,
    spacing: float,
    atoms: dict[str, dict[str, np.ndarray]],
    title: str,
    subtitle: str,
) -> None:
    coordinates = np.arange(volume.shape[0], dtype=np.float32) * spacing
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    positive = np.max(volume, axis=2).T
    negative = np.min(volume, axis=2).T

    maximum = float(positive.max())
    if maximum > 1.0:
        upper = max(3.0, maximum)
        ax.contourf(
            x, y, positive,
            levels=[1.0, 1.5, 2.0, 3.0, upper + 1e-5],
            colors=[POSITIVE_COLOR] * 4,
            alpha=0.13,
            antialiased=True,
        )
        line_levels = [level for level in (1.0, 2.0, 3.0) if level < maximum]
        if line_levels:
            ax.contour(
                x, y, positive, levels=line_levels,
                colors=POSITIVE_COLOR, linewidths=[1.5, 1.0, 0.8][:len(line_levels)],
                alpha=0.9,
            )
    minimum = float(negative.min())
    if minimum < -1.0:
        negative_levels = [level for level in (-3.0, -2.0, -1.0) if level > minimum]
        if negative_levels:
            ax.contour(
                x, y, negative, levels=negative_levels,
                colors=NEGATIVE_COLOR, linewidths=0.9,
                linestyles="dashed", alpha=0.8,
            )

    add_structure(ax, atoms)
    ax.set_xlim(4.2, 10.7)
    ax.set_ylim(5.3, 11.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top",
        fontsize=20, fontweight="bold", color="#111827",
    )
    ax.text(
        0.02, 0.905, subtitle, transform=ax.transAxes, ha="left", va="top",
        fontsize=12, color="#4b5563",
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
    npz_path = next(args.input.glob("*_density_triplet.npz"))
    pdb_path = next(args.input.glob("*_local_frame.pdb"))
    with np.load(npz_path, allow_pickle=False) as archive:
        volumes = {key: np.asarray(archive[key]) for key in LABELS}
    site_parts = manifest["site"].split("_")
    chain = site_parts[1]
    residue_name_number = site_parts[2]
    residue_name = "".join(c for c in residue_name_number if c.isalpha())
    residue_number = int("".join(c for c in residue_name_number if c.isdigit()))
    atoms = residue_atoms(pdb_path, chain, residue_number)
    spacing = float(manifest["spacing_angstrom"])

    for key, volume in volumes.items():
        title, subtitle = LABELS[key]
        figure, axis = plt.subplots(figsize=(6.4, 5.6), dpi=300)
        figure.patch.set_alpha(0.0)
        axis.set_facecolor("none")
        render_panel(axis, volume, spacing, atoms, title, subtitle)
        figure.tight_layout(pad=0.2)
        figure.savefig(
            args.output / f"{manifest['site']}_{key}.png",
            dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04,
        )
        plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.3), dpi=300)
    figure.patch.set_facecolor("white")
    for axis, (key, volume) in zip(axes, volumes.items()):
        render_panel(axis, volume, spacing, atoms, *LABELS[key])
    legend = [
        Line2D([0], [0], color=ALT_COLORS["A"], lw=4, label="Deposited A (57%)"),
        Line2D([0], [0], color=ALT_COLORS["B"], lw=4, label="Deposited B (43%)"),
        Line2D([0], [0], color=POSITIVE_COLOR, lw=2, label="Positive density (1σ+)"),
        Line2D([0], [0], color=NEGATIVE_COLOR, lw=1, ls="--", label="Negative density"),
    ]
    figure.legend(
        handles=legend, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005), fontsize=11,
    )
    figure.suptitle(
        f"{site_parts[0]}  {chain}:{residue_name}{residue_number}",
        fontsize=22, fontweight="bold", y=0.995,
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.12, wspace=0.04)
    figure.savefig(
        args.output / f"{manifest['site']}_density_comparison.png",
        dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
