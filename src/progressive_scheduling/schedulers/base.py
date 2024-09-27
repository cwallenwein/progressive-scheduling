import warnings
from typing import List, Optional

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class _enable_get_lr_call:
    def __init__(self, scheduler: LRScheduler):
        self.scheduler = scheduler

    def __enter__(self):
        self.scheduler._get_lr_called_within_step = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.scheduler._get_lr_called_within_step = False


class ProgressiveScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch, verbose="deprecated")

    def get_lr(self, training_progress: float) -> List[float]:
        """Compute learning rate based on the current progression of the training."""
        raise NotImplementedError("Subclasses must implement this method.")

    def step(self, training_progress: Optional[float] = 0.0):
        """Perform a step."""
        self._check_optimizer_step_order()

        self._step_count += 1

        with _enable_get_lr_call(self):
            self.last_epoch += 1
            values = self.get_lr(training_progress)

        self._update_learning_rates(values)

    def _check_optimizer_step_order(self):
        """Check if optimizer.step() is called before lr_scheduler.step()."""
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after "
                    "learning rate scheduler initialization. Please, make sure to "
                    "call `optimizer.step()` before `lr_scheduler.step()`. "
                    "See more details at https://pytorch.org/docs/stable/optim.html "
                    "#how-to-adjust-learning-rate ",
                    UserWarning,
                )
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite "
                    "order: `optimizer.step()` before `lr_scheduler.step()`. "
                    "Failure to do this will result in PyTorch skipping the first "
                    "value of the learning rate schedule. See more details at "
                    "https://pytorch.org/docs/stable/optim.html"
                    "#how-to-adjust-learning-rate",
                    UserWarning,
                )

    def _update_learning_rates(self, values: List[float]):
        """Update the learning rates for the optimizer's parameter groups."""
        for param_group, lr in zip(self.optimizer.param_groups, values):
            if isinstance(param_group["lr"], Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr

        self._last_lr: List[float] = [
            group["lr"] for group in self.optimizer.param_groups
        ]
