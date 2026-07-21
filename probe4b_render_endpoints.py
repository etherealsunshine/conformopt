#!/usr/bin/env python3
"""Render density-backed 2D coordinate overlays for Probe 4b endpoints."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from SFC_Torch import SFcalculator


ROOT = Path(__file__).parent
AUDIT = ROOT / "probe4b_results/endpoint_audit"
COLORS = {
    "A": "#2ca02c",
    "B": "#17becf",
    "A_synthetic_fobs": "#e377c2",
    "B_localized_sf": "#ff7f0e",
    "C_realspace_local": "#bcbd22",
}
BONDS = {
    "ARG": (("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2")),
    "MET": (("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "ASP": (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
}


def main() -> None:
    data = json.loads((AUDIT / "tmol_inputs.json").read_text())
    summaries = {item["site"]: item["references"] for item in json.loads((AUDIT / "endpoint_summary.json").read_text())}
    calc = SFcalculator(str(ROOT / "data/2O1K.pdb"), str(ROOT / "data/2O1K.mtz"), device=torch.device("cpu"))
    fobs = calc.Fo.detach(); fcalc = calc.calc_fprotein(Return=True).detach()
    hkl = torch.tensor(calc.HKL_array, dtype=torch.float32); valid = torch.isfinite(fobs) & (fobs > 0)
    amp = fcalc.abs(); scale = (amp[valid] @ fobs[valid]) / amp[valid].square().sum()
    coefficients = (2 * fobs - scale * amp) * fcalc / amp.clamp_min(1e-8)
    orth_to_frac = calc.orth2frac_tensor.detach()

    def density(points: torch.Tensor) -> torch.Tensor:
        values = []
        for chunk in points.split(400):
            phase = 2 * torch.pi * (hkl[valid] @ (chunk @ orth_to_frac.T).T)
            values.append(2 * torch.real(coefficients[valid] @ torch.exp(-1j * phase)) / int(valid.sum()))
        return torch.cat(values)

    for site in data["sites"]:
        coordinates = {
            "A": np.asarray(site["A"]),
            "B": np.asarray(site["B"]),
            **{experiment: np.asarray(candidates[0]) for experiment, candidates in site["experiments"].items()},
        }
        stacked = np.concatenate(list(coordinates.values()))
        center = stacked.mean(axis=0)
        _, _, vh = np.linalg.svd(stacked - center, full_matrices=False)
        axes = vh[:2]
        projected = {label: (xyz - center) @ axes.T for label, xyz in coordinates.items()}
        all_projected = np.concatenate(list(projected.values()))
        low = all_projected.min(axis=0) - 2.0; high = all_projected.max(axis=0) + 2.0
        gx = np.linspace(low[0], high[0], 60); gy = np.linspace(low[1], high[1], 60)
        xx, yy = np.meshgrid(gx, gy)
        plane_points = center + xx.reshape(-1, 1) * axes[0] + yy.reshape(-1, 1) * axes[1]
        rho = density(torch.tensor(plane_points, dtype=torch.float32)).numpy().reshape(xx.shape)
        rho = (rho - rho.mean()) / max(rho.std(), 1e-6)

        fig, ax = plt.subplots(figsize=(8, 7))
        contour = ax.contourf(xx, yy, rho, levels=np.linspace(-1, 3, 17), cmap="Greys", alpha=.55, extend="both")
        ax.contour(xx, yy, rho, levels=[1.0, 1.5, 2.0], colors=["#888", "#555", "#222"], linewidths=.7)
        name_to_index = {name: index for index, name in enumerate(site["atom_names"])}
        for label, xy in projected.items():
            for first, second in BONDS[site["residue_type"]]:
                ax.plot([xy[name_to_index[first], 0], xy[name_to_index[second], 0]],
                        [xy[name_to_index[first], 1], xy[name_to_index[second], 1]],
                        color=COLORS[label], linewidth=2, alpha=.9)
            ax.scatter(xy[:, 0], xy[:, 1], s=28, color=COLORS[label], edgecolor="black", linewidth=.3, label=label)
        reference = summaries[site["site"]]
        clash_labels = [
            f"{label}: min {min(reference[label]['min_direct_distance'], reference[label]['min_symmetry_distance']):.2f} Å"
            for label in coordinates
        ]
        ax.text(.01, .01, "\n".join(clash_labels), transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "alpha": .75, "edgecolor": "none"})
        ax.set_title(f"{site['site']} — sidechains over observed 2Fo−Fc-like density")
        ax.set_xlabel("local PCA axis 1 (Å)"); ax.set_ylabel("local PCA axis 2 (Å)")
        ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
        fig.colorbar(contour, ax=ax, label="local density z-score")
        fig.tight_layout(); fig.savefig(AUDIT / "visualization" / f"{site['site']}_density_overlay.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
