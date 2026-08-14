import numpy as np
import torch
from scipy.optimize import least_squares as scipy_least_squares

from scripts.torch_trf import least_squares as torch_least_squares


def _residual_numpy(x: np.ndarray) -> np.ndarray:
    return np.asarray((x[0] ** 2 - 1.25, np.sin(x[1]) + 0.75), dtype=float)


def test_torch_trf_matches_scipy_on_same_residual_and_start():
    x0 = np.asarray([0.4, -0.2], dtype=float)
    lower = np.asarray([-2.0, -1.4], dtype=float)
    upper = np.asarray([2.0, 1.4], dtype=float)
    scipy_result = scipy_least_squares(
        _residual_numpy, x0, bounds=(lower, upper), method="trf",
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=40,
    )

    def residual_torch(x):
        return torch.stack((x[0] ** 2 - 1.25, torch.sin(x[1]) + 0.75))

    torch_result = torch_least_squares(
        residual_torch, torch.as_tensor(x0, dtype=torch.float64),
        lower=torch.as_tensor(lower, dtype=torch.float64),
        upper=torch.as_tensor(upper, dtype=torch.float64),
        max_nfev=40, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    np.testing.assert_allclose(torch_result.x.cpu().numpy(), scipy_result.x, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(torch_result.cost, 0.5 * np.dot(scipy_result.fun, scipy_result.fun), atol=1e-10)
    assert torch_result.projected_optimality < 1e-7
    assert scipy_result.optimality < 1e-7
