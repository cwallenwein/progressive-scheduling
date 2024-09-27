from typing import List

import torch

from progressive_scheduling import ProgressiveScheduler


def get_pytorch_schedule(
    scheduler: torch.optim.lr_scheduler.LRScheduler, total_steps: int
) -> List[float]:
    optimizer = scheduler.optimizer

    learning_rates = []
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    return learning_rates


def get_progressive_schedule(
    scheduler: ProgressiveScheduler, total_steps: int
) -> List[float]:
    optimizer = scheduler.optimizer

    learning_rates = []
    for step in range(total_steps):
        progress = step / total_steps
        optimizer.step()
        scheduler.step(progress)
        learning_rates.append(optimizer.param_groups[0]["lr"])
    return learning_rates
