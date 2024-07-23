import logging

import torch

from vambn import HIVAE_EPS

logger = logging.getLogger(__name__)


def nrmse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Calculate the normalized root mean squared error (NRMSE).

    Args:
        pred (torch.Tensor): The predicted values.
        target (torch.Tensor): The target values.
        mask (torch.LongTensor): The mask to be applied, must be the same size as pred and target.

    Returns:
        torch.Tensor: The normalized root mean squared error.
    """
    norm_term = torch.max(target) - torch.min(target)

    # Calculate the error for only the masked values
    new_pred = torch.masked_select(pred, mask.to(torch.bool))
    new_target = torch.masked_select(target, mask.to(torch.bool))

    # Calculate the normalized root mean squared error
    return torch.sqrt(torch.nn.functional.mse_loss(new_pred, new_target)) / (
        norm_term + HIVAE_EPS
    )
