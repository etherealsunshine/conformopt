#!/usr/bin/env python3
"""Calibrate the prospective A-prime amplitude prior from the frozen panel scan.

The calibration deliberately reads only the 232,890-row deposited altloc scan.
It never reads benchmark targets, optimizer results, or site recovery outcomes.
The prior is represented in the optimizer's torsion-degree coordinates.  The
fixed 1.5 A lever arm converts the population C-alpha displacement scale into
an equivalent one-norm torsion scale, and lambda is chosen so that that norm
has quadratic prior energy one.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column", default="ca_deviation")
    parser.add_argument("--lever-arm-A", type=float, default=1.5)
    args = parser.parse_args()

    values = []
    rows = 0
    with args.csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            value = float(row[args.column])
            if math.isfinite(value):
                values.append(value)
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or args.lever_arm_A <= 0.0:
        raise ValueError("empty displacement population or invalid lever arm")

    sigma_A = float(values.std(ddof=1))
    sigma_q_deg = sigma_A / args.lever_arm_A * 180.0 / math.pi
    lambda_amp = 1.0 / sigma_q_deg ** 2
    iqr_sigma_A = float((np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.3489795003921634)
    report = {
        "status": "complete",
        "source_csv": str(args.csv),
        "source_rows": rows,
        "finite_displacement_rows": int(len(values)),
        "displacement_column": args.column,
        "displacement_units": "A",
        "distribution": {
            "mean_A": float(values.mean()),
            "std_A": sigma_A,
            "median_A": float(np.quantile(values, 0.50)),
            "q1_A": float(np.quantile(values, 0.25)),
            "q3_A": float(np.quantile(values, 0.75)),
            "p95_A": float(np.quantile(values, 0.95)),
            "min_A": float(values.min()),
            "max_A": float(values.max()),
            "robust_sigma_iqr_over_1p349_A": iqr_sigma_A,
        },
        "prior_calibration": {
            "parameterization": "two 20-component torsion vectors in degrees",
            "fixed_lever_arm_A": args.lever_arm_A,
            "population_one_sigma_A": sigma_A,
            "equivalent_one_norm_sigma_q_deg": sigma_q_deg,
            "lambda_amp_per_degree_squared": lambda_amp,
            "one_sigma_prior_energy": float(lambda_amp * sigma_q_deg ** 2),
            "prior_residual": "sqrt(lambda_amp) * (q - q_start)",
            "q_start": "zero neutral-start torsion vector for both slots",
            "derivation": "population-only C-alpha displacement standard deviation; no benchmark outcome or target used",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
