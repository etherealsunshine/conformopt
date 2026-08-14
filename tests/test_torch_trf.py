import torch

from scripts.torch_trf import least_squares


def test_torch_trf_converges_and_adapts_radius():
    dtype = torch.float64
    target = torch.tensor([1.25, -0.75], dtype=dtype)

    def residual(x):
        return torch.stack((x[0] ** 2 - target[0], torch.sin(x[1]) - target[1]))

    result = least_squares(
        residual, torch.tensor([0.4, -0.2], dtype=dtype),
        max_nfev=40, initial_radius=0.1,
    )
    assert result.message in {"gtol", "ftol", "xtol"}
    assert result.projected_optimality < 1e-7
    assert result.nfev <= 81
    assert len(result.trust_radius_trace) > 0
    assert all("radius_after" in row for row in result.trust_radius_trace)


def test_torch_trf_stays_on_requested_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def residual(x):
        return torch.stack((x[0] - 2.0, 2.0 * x[1] + 1.0))

    result = least_squares(residual, torch.zeros(2, device=device, dtype=torch.float64))
    assert result.x.device.type == device.type
