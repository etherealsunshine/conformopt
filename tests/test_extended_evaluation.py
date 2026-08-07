import numpy as np
import torch

from density_denoiser.evaluate_extended import (
    _masked_mse,
    _masked_pearson,
    _native_metrics,
    _rank_correlation,
    _top_fraction_overlap,
)
from density_denoiser.summarize_altloc_evaluation import occupancy_bin


def test_masked_reconstruction_metrics_ignore_outside_voxels():
    target = torch.tensor([[[[1.0, 2.0, 99.0, 99.0]]]])
    prediction = torch.tensor([[[[1.0, 2.0, -99.0, -99.0]]]])
    mask = torch.tensor([[[[True, True, False, False]]]])
    assert _masked_mse(prediction, target, mask) == 0.0
    assert np.isclose(_masked_pearson(prediction, target, mask), 1.0)


def test_top_fraction_overlap_measures_peak_localization():
    target = torch.arange(10, dtype=torch.float32)
    identical = target.clone()
    reversed_prediction = target.flip(0)
    mask = torch.ones_like(target, dtype=torch.bool)
    assert _top_fraction_overlap(identical, target, mask, 0.2) == 1.0
    assert _top_fraction_overlap(reversed_prediction, target, mask, 0.2) == 0.0


def test_landscape_rank_and_native_metrics_have_expected_direction():
    oracle = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    reversed_values = oracle[::-1]
    assert np.isclose(_rank_correlation(oracle, oracle), 1.0)
    assert np.isclose(_rank_correlation(reversed_values, oracle), -1.0)
    metrics = _native_metrics(oracle)
    assert metrics["native_top1"]
    assert metrics["native_top3"]
    assert metrics["ab_beats_a_and_b_only"]
    assert metrics["native_margin"] == 1.0


def test_minor_occupancy_bins_have_stable_boundaries():
    assert occupancy_bin(0.05) == "<0.10"
    assert occupancy_bin(0.10) == "0.10-0.20"
    assert occupancy_bin(0.20) == "0.20-0.30"
    assert occupancy_bin(0.30) == "0.30-0.40"
    assert occupancy_bin(0.40) == ">=0.40"
