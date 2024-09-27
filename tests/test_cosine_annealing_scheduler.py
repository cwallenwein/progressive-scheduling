import pytest
import torch.optim.lr_scheduler as pytorch_schedulers

import progressive_scheduling.schedulers as progressive_schedulers

from .util import (
    compare_progressive_scheduler_with_reference_implementation,
    create_optimizer,
)


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
    compare_progressive_scheduler_with_reference_implementation(
        T_max,
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
