import math
from typing import List, Literal

from torch.optim import Optimizer

from progressive_scheduling.schedulers import ProgressiveScheduler


class OneCycleLR(ProgressiveScheduler):
    """
    Implements the One Cycle Learning Rate policy.

    This scheduler adjusts the learning rate according to the 1cycle policy,
    which increases the learning rate from an initial value to a maximum value,
    and then decreases it to a minimum value. The policy is designed to improve
    training speed and model performance.

    Args:
        optimizer (Optimizer): The optimizer to which this scheduler will be applied.
        max_lr (float): The maximum learning rate during the cycle.
        pct_start (float, optional): The percentage of the cycle (in number of steps)
            spent increasing the learning rate. Default is 0.3.
        anneal_strategy (Literal["cos", "linear"], optional): The strategy used for
            annealing the learning rate. Can be either 'cos' for cosine annealing
            or 'linear' for linear annealing. Default is 'cos'.
        cycle_momentum (bool, optional): Whether to cycle momentum. Default is True.
        base_momentum (float, optional): The base momentum value. Default is 0.85.
        max_momentum (float, optional): The maximum momentum value. Default is 0.95.
        div_factor (float, optional): Determines the initial learning rate via
            initial_lr = max_lr / div_factor. Default is 25.0.
        final_div_factor (float, optional): Determines the minimum learning rate via
            min_lr = initial_lr / final_div_factor. Default is 10000.0.
        three_phase (bool, optional): If True, use a third phase of the schedule.
            Default is False.

        # TODO: add support for parameter groups
        # TODO: add support for cycle_momentum

    Raises:
        AssertionError: If three_phase is True, cycle_momentum is False, or if
            base_momentum and max_momentum are not equal to their default values.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        pct_start: float = 0.3,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        three_phase: bool = False,
    ):
        assert three_phase is False, (
            "Three phase is not supported."
            + "If you want this feature, let me know on Github: "
            + "https://github.com/cwallenwein/progressive-scheduling/issues"
        )
        assert cycle_momentum is True, (
            "Cycle momentum must not supported. "
            + "If you want this feature, let me know on Github: "
            + "https://github.com/cwallenwein/progressive-scheduling/issues"
        )

        self.max_lr = max_lr
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Calculate initial and minimum learning rates
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor

        super().__init__(optimizer)

    def _annealing_cos(self, start: float, end: float, pct: float) -> float:
        """
        Computes the cosine annealing from start to end based on the progress.

        Args:
            start (float): The starting learning rate.
            end (float): The ending learning rate.
            pct (float): The progress percentage (between 0 and 1).

        Returns:
            float: The annealed learning rate.
        """
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start: float, end: float, pct: float) -> float:
        """
        Computes the linear annealing from start to end based on the progress.

        Args:
            start (float): The starting learning rate.
            end (float): The ending learning rate.
            pct (float): The progress percentage (between 0 and 1).

        Returns:
            float: The annealed learning rate.
        """
        return (end - start) * pct + start

    def _get_annealed_lr(self, start_lr: float, end_lr: float, pct: float) -> float:
        """
        Retrieves the annealed learning rate based on the specified strategy.

        Args:
            start_lr (float): The starting learning rate.
            end_lr (float): The ending learning rate.
            pct (float): The progress percentage (between 0 and 1).

        Returns:
            float: The annealed learning rate.

        Raises:
            ValueError: If the annealing strategy is unknown.
        """
        if self.anneal_strategy == "cos":
            return self._annealing_cos(start_lr, end_lr, pct)
        elif self.anneal_strategy == "linear":
            return self._annealing_linear(start_lr, end_lr, pct)
        else:
            raise ValueError(f"Unknown annealing strategy: {self.anneal_strategy}")

    def get_lr(self, training_progress: float) -> List[float]:
        """
        Calculates the learning rate based on the training progress.

        Args:
            training_progress (float): The progress of the training, should be
                between 0.0 (start) and 1.0 (end).

        Returns:
            List[float]: The calculated learning rate.

        Raises:
            ValueError: If training_progress is not between 0.0 and 1.0.
        """
        if training_progress < 0.0 or training_progress > 1.0:
            raise ValueError("training_progress should be between 0.0 and 1.0")

        if training_progress < self.pct_start:
            warmup_pct = training_progress / self.pct_start
            lr = self._get_annealed_lr(self.initial_lr, self.max_lr, warmup_pct)
        else:
            cooldown_pct = (training_progress - self.pct_start) / (1 - self.pct_start)
            lr = self._get_annealed_lr(self.max_lr, self.min_lr, cooldown_pct)
        return [lr]
