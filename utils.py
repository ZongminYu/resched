"""
Utility functions for ReSched.
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_lower_bound_makespan(finish_times_lb: torch.Tensor) -> torch.Tensor:
    """
    Compute the estimated lower-bound makespan.
    finish_times_lb: (batch, num_ops) - lower bound finish time for each op.
    Returns: (batch,) - lower bound makespan for each instance.
    """
    return finish_times_lb.max(dim=-1)[0]


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
