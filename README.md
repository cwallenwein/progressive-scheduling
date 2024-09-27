# Progressive Scheduling

**progressive_scheduling** is a PyTorch-compatible library that provides learning rate schedulers based on training progress rather than steps. This is useful when it's hard to estimate the total number of steps before starting the training (e.g. when training for exactly 24 hours, like https://arxiv.org/pdf/2212.14034).

## Features

- Progress-based learning rate schedulers
- Compatible with PyTorch optimizers
- Currently supports:
  - CosineAnnealingLR
  - OneCycleLR (only three_phase=False)

## Installation

You can install progressive_scheduling directly from GitHub:
```bash
pip install git+https://github.com/cwallenwein/progressive-scheduling.git
```

## Quick Start

Switching to **progressive_scheduling** requires only minimal changes to your existing code

1. Import the scheduler from progressive_scheduling instead of torch.optim.lr_scheduler.
2. Remove the total_steps parameter (or T_max, depending on the scheduler) when initializing the scheduler.
3. Pass the current progress (as float between 0 and 1) to scheduler.step()

```diff
import torch
from torch.optim import SGD
- from torch.optim.lr_scheduler import OneCycleLR
+ from progressive_scheduling import OneCycleLR

# Create your model
model = YourModel()

# Create an optimizer
optimizer = SGD(model.parameters(), lr=0.1)

# Create a scheduler
- scheduler = OneCycleLR(optimizer, max_lr=0.1, total_steps=100)
+ scheduler = OneCycleLR(optimizer, max_lr=0.1)


#In your training loop
for step in range(100):

    # Forward pass, loss computation, backward pass...
    optimizer.step()

    # Update the learning rate
-   scheduler.step()
+   progress = step / 100 # Calculate the current progress
+   scheduler.step(progress)
```

## Documentation

For more detailed information about the available schedulers and their parameters, please refer to the docstrings in the source code.

## Contributing

We welcome contributions from the community! If you'd like to contribute, please follow these steps to submit a Pull Request:

1. Clone the repository:
   ```bash
   git clone https://github.com/cwallenwein/progressive-scheduling.git
   ```
2. Navigate into the project directory:
   ```bash
   cd progressive-scheduling
   ```
3. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

Your contributions help improve the library for everyone. Thank you for your support!

## TODO

- Implement three_phase option for OneCycleLR
- Setup GitHub actions to run tests, linting, and type checking automatically
- Add support for more schedulers


<!---  
All schedulers

lr_scheduler.LambdaLR
lr_scheduler.MultiplicativeLR
lr_scheduler.StepLR
lr_scheduler.MultiStepLR
lr_scheduler.ConstantLR
lr_scheduler.LinearLR
lr_scheduler.ExponentialLR
lr_scheduler.PolynomialLR
lr_scheduler.CosineAnnealingLR
lr_scheduler.ChainedScheduler
lr_scheduler.SequentialLR
lr_scheduler.ReduceLROnPlateau
lr_scheduler.CyclicLR
lr_scheduler.OneCycleLR
lr_scheduler.CosineAnnealingWarmRestarts
--->
