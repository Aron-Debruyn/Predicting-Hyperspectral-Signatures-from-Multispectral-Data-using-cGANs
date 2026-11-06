"""Simple differentiable error measures shared by training validation and evaluation.

Contents
--------
rmse_loss  Root mean squared error.
mrae_loss  Mean relative absolute error.
"""

import torch
import torch.nn.functional as F


def rmse_loss(pred, target):
    return torch.sqrt(F.mse_loss(pred, target))


def mrae_loss(pred, target, epsilon=1e-7):
    relative_errors = torch.abs(target - pred) / (torch.abs(target) + epsilon)
    return torch.mean(relative_errors)
