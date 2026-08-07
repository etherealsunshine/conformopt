"""Mechanically inject downsampled landscape arrays into the inline visual."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
SOURCE = ROOT / "five_site_landscape_results"
OUTPUT = Path(
    "/Users/utkarsh/.codex/visualizations/2026/07/16/"
    "019f6938-ac85-71a2-bc32-9cd8f793e9d7/interactive-landscapes.html"
)
SITES = ("A_MET112", "A_ARG129", "B_MET112", "B_ASP114", "B_ARG129")


def normalized_log_shape(values: np.ndarray) -> np.ndarray:
    logged = np.log10(values + 1e-8)
    low, high = np.percentile(logged, [1.0, 99.0])
    return np.clip((logged - low) / max(high - low, 1e-8), 0.0, 1.0)


indices = np.linspace(0, 63, 32, dtype=int)
payload = {}
for site in SITES:
    values = np.load(SOURCE / f"{site}_landscape_values.npz")
    axis = np.round(values["chi_degrees"][indices], 1)
    payload[site] = {"axis": axis.tolist()}
    for source_key, output_key in (("complex_sf", "sf"), ("realspace", "real")):
        shape = normalized_log_shape(values[source_key])
        # Plotly expects rows to be y and columns to be x.
        sampled = shape[np.ix_(indices, indices)].T
        payload[site][output_key] = np.round(sampled, 4).tolist()

fragment = OUTPUT.read_text()
placeholder = "__LANDSCAPE_DATA__"
if placeholder not in fragment:
    raise RuntimeError("landscape data placeholder is missing")
OUTPUT.write_text(fragment.replace(placeholder, json.dumps(payload, separators=(",", ":"))))
