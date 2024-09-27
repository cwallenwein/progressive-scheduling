from typing import Literal

import pytest
import torch.optim.lr_scheduler as pytorch_schedulers

import progressive_scheduling.schedulers as progressive_schedulers

from .util import (
    compare_progressive_scheduler_with_reference_implementation,
    create_optimizer,
)


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
    compare_progressive_scheduler_with_reference_implementation(
        total_steps,
        progressive_scheduler,
        pytorch_scheduler,
    )


def test_invalid_progress_values():
    optimizer = create_optimizer()
    scheduler = progressive_schedulers.OneCycleLR(optimizer, max_lr=0.1)
    with pytest.raises(ValueError):
        optimizer.step()
        scheduler.step(-0.1)
    with pytest.raises(ValueError):
        optimizer.step()
        scheduler.step(1.1)
