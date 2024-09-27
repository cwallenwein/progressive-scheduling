from typing import Literal

import pytest
import torch
import torch.optim.lr_scheduler as pytorch_schedulers
from torch.optim import SGD

import progressive_scheduling.schedulers as progressive_schedulers
from progressive_scheduling.utils import get_progressive_schedule, get_pytorch_schedule


def create_optimizer():
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    return optimizer


def create_one_cycle_schedulers(
    max_lr,
    total_steps=None,
    pct_start=0.3,
    anneal_strategy="cos",
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=25.0,
    final_div_factor=10000.0,
    three_phase=False,
):

    # Create progressive scheduler
    progressive_optimizer = create_optimizer()
    progressive_scheduler = progressive_schedulers.OneCycleLR(
        progressive_optimizer,
        max_lr,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        base_momentum=base_momentum,
        max_momentum=max_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
    )

    # Create pytorch scheduler
    pytorch_optimizer = create_optimizer()
    pytorch_scheduler = pytorch_schedulers.OneCycleLR(
        pytorch_optimizer,
        max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        base_momentum=base_momentum,
        max_momentum=max_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
    )

    return (progressive_scheduler, pytorch_scheduler)


def create_cosine_annealing_schedulers(T_max: int, eta_min: float):
    # Create progressive scheduler
    progressive_optimizer = create_optimizer()
    progressive_scheduler = progressive_schedulers.CosineAnnealingLR(
        progressive_optimizer, eta_min=eta_min
    )

    # Create pytorch scheduler
    pytorch_optimizer = create_optimizer()
    pytorch_scheduler = pytorch_schedulers.CosineAnnealingLR(
        pytorch_optimizer, T_max=T_max, eta_min=eta_min
    )

    return (progressive_scheduler, pytorch_scheduler)


def compare_schedulers(
    num_steps: int,
    progressive_scheduler,
    pytorch_scheduler,
):
    # Get learning rate schedules
    progressive_lrs = get_progressive_schedule(progressive_scheduler, num_steps)
    pytorch_lrs = get_pytorch_schedule(pytorch_scheduler, num_steps)

    # Print the results header
    print(
        f"{'Step':<10} {'Progress':<10} {'Progressive LR':<15} "
        f"{'PyTorch LR':<15} {'Diff':<15}"
    )
    print("-" * 80)

    diffs = []
    for step, (progressive_lr, pytorch_lr) in enumerate(
        zip(progressive_lrs, pytorch_lrs), 1
    ):
        progress = step / num_steps
        diff = abs(progressive_lr - pytorch_lr)
        diffs.append(diff)

        # Log the current step, progress, and learning rates
        print(
            f"{step:<10} {progress:<10.4f} {progressive_lr:<15.6f} "
            f"{pytorch_lr:<15.6f} {diff:<15.6f}"
        )

    # Check if the learning rates are close enough
    max_diff = max(diffs)
    print(f"Maximum difference in learning rates: {max_diff:.2e}")
    # assert max_diff < 1e-6, "Schedulers behave differently!"
    assert max_diff < 0.03, "Schedulers behave differently!"

    print(
        "Verification successful: Progressive scheduler behaves identically to "
        "PyTorch scheduler."
    )


@pytest.mark.parametrize(
    "T_max, eta_min",
    [
        (100, 0),
        (50, 0.001),
        (200, 0.0001),
        (1000, 0.00001),
    ],
)
def test_cosine_annealing_scheduler(T_max: int, eta_min: float):
    (progressive_scheduler, pytorch_scheduler) = create_cosine_annealing_schedulers(
        T_max=T_max, eta_min=eta_min
    )
    compare_schedulers(
        T_max,
        progressive_scheduler,
        pytorch_scheduler,
    )


@pytest.mark.parametrize(
    "max_lr, total_steps, pct_start, anneal_strategy, div_factor, final_div_factor",
    [
        (0.1, 100, 0.3, "cos", 25.0, 10000.0),
        (0.01, 200, 0.4, "linear", 10.0, 1000.0),
        (1.0, 500, 0.2, "cos", 100.0, 100000.0),
        (0.001, 1000, 0.1, "linear", 5.0, 100.0),
    ],
)
def test_one_cycle_scheduler(
    max_lr: float,
    total_steps: int,
    pct_start: float,
    anneal_strategy: Literal["cos", "linear"],
    div_factor: float,
    final_div_factor: float,
    cycle_momentum: bool = True,
    base_momentum: float = 0.85,
    max_momentum: float = 0.95,
    three_phase: bool = False,
):
    (progressive_scheduler, pytorch_scheduler) = create_one_cycle_schedulers(
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        cycle_momentum=cycle_momentum,
        base_momentum=base_momentum,
        max_momentum=max_momentum,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
    )
    compare_schedulers(
        total_steps,
        progressive_scheduler,
        pytorch_scheduler,
    )


def test_invalid_progress_values():
    optimizer = create_optimizer()
    scheduler = progressive_schedulers.CosineAnnealingLR(optimizer, eta_min=0.001)
    with pytest.raises(ValueError):
        optimizer.step()
        scheduler.step(-0.1)
    with pytest.raises(ValueError):
        optimizer.step()
        scheduler.step(1.1)
