import math
from typing import List

from torch.optim import Optimizer

from progressive_scheduling.schedulers import ProgressiveScheduler


class CosineAnnealingLR(ProgressiveScheduler):
    """
    Implements the Cosine Annealing Learning Rate policy.

    This scheduler adjusts the learning rate using a cosine function,
    starting from the initial learning rate and gradually decreasing it
    to a minimum value (eta_min) over the course of training. This
    approach can help improve convergence and training stability.

    Args:
        optimizer (Optimizer): The optimizer to which this scheduler will be applied.
        eta_min (float): Minimum learning rate. Default is 0.0.
    """

    def __init__(self, optimizer: Optimizer, eta_min: float = 0.0):
        """
        Initializes the CosineAnnealingLRScheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            eta_min (float): Minimum learning rate. Default: 0.0.
        """
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self, training_progress: float = 0.0) -> List[float]:
        """
        Calculates the learning rate using the cosine annealing formula.

        Args:
            training_progress (float): Progress of the training
                between 0.0 (start) and 1.0 (end).

        Returns:
            List[float]: Learning rates for each parameter group.

        Raises:
            ValueError: If training_progress is not between 0.0 and 1.0.
        """
        if training_progress < 0.0 or training_progress > 1.0:
            raise ValueError("training_progress must be between 0.0 and 1.0.")

        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * training_progress)) / 2
            for base_lr in self.base_lrs
        ]
