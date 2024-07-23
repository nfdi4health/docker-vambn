import logging

import torch
from torch import Tensor

from vambn import HIVAE_EPS

logger = logging.getLogger(__name__)


def accuracy(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Calculate the accuracy of predictions for a categorical variable.

    Args:
        pred (Tensor): Predictions of shape (batch_size, n_categories).
        target (Tensor): Ground truth of shape (batch_size, n_categories).
        mask (Tensor): Mask of shape (batch_size, n_categories).

    Returns:
        Tensor: The accuracy of predictions.
    """
    n_correct = torch.sum((pred != target) * mask)
    n_total = mask.sum()
    return n_correct / (n_total + HIVAE_EPS)
