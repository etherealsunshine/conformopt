import math
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_d6_tier1_synthetic import b_ic  # noqa: E402


def test_bic_uses_pinned_qfit_default_parameter_count():
    rss, n, atoms, nconfs = 3.0, 500, 6, 2
    expected_k = 4.0 * atoms * nconfs * 0.8
    expected_bic = n * math.log(rss / n) + expected_k * math.log(n)
    bic, k = b_ic(rss, n, atoms, nconfs)
    assert math.isclose(k, expected_k)
    assert math.isclose(bic, expected_bic)
