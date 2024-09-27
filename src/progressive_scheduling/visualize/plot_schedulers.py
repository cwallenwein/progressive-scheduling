from typing import Optional

import matplotlib.pyplot as plt
import torch

from progressive_scheduling.schedulers.base import ProgressiveScheduler
from progressive_scheduling.utils import get_progressive_schedule, get_pytorch_schedule


def plot_lr_scheduler(
    pytorch_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    progressive_scheduler: Optional[ProgressiveScheduler] = None,
    num_steps: int = 1000,
) -> None:
    if progressive_scheduler is None and pytorch_scheduler is None:
        raise ValueError("At least one scheduler must be provided")

    plt.figure(figsize=(10, 6))

    if progressive_scheduler:
        progressive_lrs = get_progressive_schedule(progressive_scheduler, num_steps)
        plt.plot(range(num_steps), progressive_lrs, label="Progressive Scheduler")

    if pytorch_scheduler:
        pytorch_lrs = get_pytorch_schedule(pytorch_scheduler, num_steps)
        plt.plot(
            range(num_steps), pytorch_lrs, label="PyTorch Scheduler", linestyle="--"
        )

    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.show()
