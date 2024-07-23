import logging
from typing import Tuple

import torch


logger = logging.getLogger(__name__)


def sample_gumbel(
    shape: Tuple[int, int],
    eps: float = 1e-9,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate a sample from the Gumbel distribution.

    Args:
        shape (Tuple[int, int]): Shape of the sample.
        eps (float, optional): Value to be added to avoid numerical issues. Defaults to 1e-9.
        device (torch.device, optional): The device to generate the sample on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Sample from the Gumbel distribution.
    """
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(
    logits: torch.Tensor,
    shape: Tuple[int, int],
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    Gumbel-Softmax implementation.

    Args:
        logits (torch.Tensor): Logits to be used for the Gumbel-Softmax.
        shape (Tuple[int, int]): Shape of the logits. Required for torchscript.
        tau (float, optional): Temperature factor. Defaults to 1.0.
        hard (bool, optional): Hard sampling or soft. Defaults to False.

    Returns:
        torch.Tensor: Sampled categorical distribution.

    Raises:
        ValueError: If logits contain NaN values.
    """
    if torch.isnan(logits).any():
        raise ValueError("Logits contain NaN values")

    gumbel_noise = sample_gumbel(shape, device=logits.device)
    y = logits + gumbel_noise
    tau = max(tau, 1e-9)
    y_soft = torch.softmax(y / (tau), dim=-1)

    if hard:
        _, ind = y_soft.max(dim=-1)
        y_hard = torch.zeros_like(y_soft).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(shape[0], shape[1])
        y_soft = (y_hard - y_soft).detach() + y_soft

    return y_soft
