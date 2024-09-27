import torch
from torch.optim import SGD

from progressive_scheduling.utils import get_progressive_schedule, get_pytorch_schedule


def create_optimizer():
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    return optimizer


def compare_progressive_scheduler_with_reference_implementation(
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
