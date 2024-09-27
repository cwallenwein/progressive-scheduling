import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class AutoSchedulingCallback(pl.callbacks.Callback):
    def __init__(self, max_steps: int = -1, max_time_in_min: int = None):
        if max_steps == -1:
            assert max_time_in_min is not None
            self.total_training_duration = max_time_in_min * 60
            self.on_train_batch_end = self.on_train_batch_end_max_time
        else:
            assert max_time_in_min is None
            self.max_steps = max_steps
            self.on_train_batch_end = self.on_train_batch_end_max_steps
            self.once_outside_threshold = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.training_start = time.time()
        self.scheduler = pl_module.lr_schedulers()

    def on_train_batch_end_max_time(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        current_training_duration = time.time() - self.training_start
        training_progress = current_training_duration / self.total_training_duration
        if training_progress > 1:
            if self.once_outside_threshold:
                raise (
                    "training_progress must be between 0.0 and 1.0 but it's",
                    training_progress,
                )
            else:
                self.once_outside_threshold = True
                training_progress = 1.0

        self.scheduler.step(training_progress)

    def on_train_batch_end_max_steps(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        training_progress = batch_idx / self.max_steps
        self.scheduler.step(training_progress)
