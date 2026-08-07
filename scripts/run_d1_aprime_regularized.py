#!/usr/bin/env python3
"""A′ 7UTC sequential PoC with a pre-registered BIC-normalized torsion prior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from run_d1_aprime_sequential import APrimeSequential
from run_d1_8d_sequential_poc import atomic_json


class RegularizedAPrime(APrimeSequential):
    """Add an L2 complexity prior without changing the sequential schedule."""

    def __init__(self, output: Path, inner_nfev: int, outer_updates: int):
        super().__init__(output, inner_nfev, outer_updates)
        spacing = np.asarray(self.base.qfit.xmap.voxelspacing, dtype=float)
        mask_volume = float(self.base.mask.sum() * np.prod(spacing))
        correlation_volume = float((self.base.resolution / 2.0) ** 3)
        self.n_eff = mask_volume / correlation_volume
        self.prior_scale_deg = 30.0
        # The density term is RSS/RSS_start.  Dividing BIC's ln(n_eff)
        # per-parameter cost by n_eff puts it on that normalized scale.
        self.prior_weight = math.log(self.n_eff) / self.n_eff

    def evaluate(self, parameters, target, capacity, normalizer, lambdas):
        state = super().evaluate(parameters, target, capacity, normalizer, lambdas)
        prior_residual = math.sqrt(self.prior_weight) * np.asarray(parameters) / self.prior_scale_deg
        state["residual"] = np.concatenate((state["residual"], prior_residual))
        state["torsion_prior_energy"] = float(np.dot(prior_residual, prior_residual))
        state["energy"] = float(np.dot(state["residual"], state["residual"]))
        return state

    def run(self):
        result = super().run()
        result["complexity_prior"] = {
            "type": "L2 torsion deviation from deposited A",
            "parameter_count": int(self.rotator.ndofs),
            "prior_scale_deg": self.prior_scale_deg,
            "weight": self.prior_weight,
            "n_eff": self.n_eff,
            "derivation": "ln(n_eff)/n_eff: BIC-scale cost on normalized RSS objective",
        }
        atomic_json(self.output / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inner-nfev", type=int, default=80)
    parser.add_argument("--outer-updates", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "run_config.json", {**vars(args), "output": str(args.output)})
    result = RegularizedAPrime(args.output, args.inner_nfev, args.outer_updates).run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
