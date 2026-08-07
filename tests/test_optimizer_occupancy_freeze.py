import torch

from density_denoiser.five_site_optimizer import (
    _set_occupancy_learning_rate,
    _stage1_adam,
)


def _step(
    optimizer: torch.optim.Adam,
    chi: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = (
        (chi.sin() - 0.25).square().sum()
        + (torch.softmax(logits, dim=0) - torch.tensor([0.7, 0.3]))
        .square().sum()
    )
    loss.backward()
    optimizer.step()


def test_equal_group_lrs_reproduce_historical_single_group_adam() -> None:
    old_chi = torch.tensor([0.4, -0.8], requires_grad=True)
    old_logits = torch.tensor([0.1, -0.1], requires_grad=True)
    new_chi = old_chi.detach().clone().requires_grad_(True)
    new_logits = old_logits.detach().clone().requires_grad_(True)

    historical = torch.optim.Adam([old_chi, old_logits], lr=0.1)
    grouped = _stage1_adam(
        new_chi,
        new_logits,
        chi_learning_rate=0.1,
        occupancy_learning_rate=0.1,
    )
    for _ in range(20):
        _step(historical, old_chi, old_logits)
        _step(grouped, new_chi, new_logits)
        assert torch.equal(new_chi, old_chi)
        assert torch.equal(new_logits, old_logits)


def test_zero_occupancy_lr_warms_adam_state_without_moving_logits() -> None:
    chi = torch.tensor([0.4, -0.8], requires_grad=True)
    logits = torch.tensor([0.1, -0.1], requires_grad=True)
    initial_logits = logits.detach().clone()
    optimizer = _stage1_adam(
        chi,
        logits,
        chi_learning_rate=0.1,
        occupancy_learning_rate=0.0,
    )

    for _ in range(5):
        _step(optimizer, chi, logits)

    assert torch.equal(logits, initial_logits)
    state = optimizer.state[logits]
    assert int(state["step"]) == 5
    assert torch.count_nonzero(state["exp_avg"]) > 0
    assert torch.count_nonzero(state["exp_avg_sq"]) > 0

    _set_occupancy_learning_rate(optimizer, 0.1)
    _step(optimizer, chi, logits)
    assert not torch.equal(logits, initial_logits)
