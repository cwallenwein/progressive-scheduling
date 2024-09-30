import time
from datetime import timedelta
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class AutoSchedulingCallback(pl.callbacks.Callback):
    def __init__(self, training_duration: timedelta | dict):

        if isinstance(training_duration, dict):
            training_duration = timedelta(**training_duration)

        self.total_training_duration = training_duration.total_seconds()
        self.exceeded_training_duration = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.training_start = time.time()
        self.scheduler = pl_module.lr_schedulers()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ):
        current_training_duration = time.time() - self.training_start
        self.check_training_duration(current_training_duration)

        training_progress = current_training_duration / self.total_training_duration
        training_progress = min(training_progress, 1.0)

        self.scheduler.step(training_progress)

    def check_training_duration(self, current_training_duration: int):
        if current_training_duration > self.total_training_duration:
            # training duration can exceed once because training was not yet stopped
            # if it happens multiple times, somethings wrong
            if self.exceeded_training_duration:
                raise ("training_duration was exceeded but training wasn't stopped")
            else:
                self.exceeded_training_duration = True
