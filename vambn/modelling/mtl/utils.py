"""
This script includes code adapted from the 'impartial-vaes' repository 
with minor modifications. The original code can be found at:
https://github.com/adrianjav/impartial-vaes

Credit to the original authors: Adrian Javaloy, Maryam Meghdadi, and Isabel Valera 
for their valuable work.
"""


import torch


def batch_product(batch: torch.Tensor, weight: torch.Tensor):
    r"""
    Multiplies each slice of the first dimension of batch by the corresponding scalar in the weight vector.

    Args:
        batch (torch.Tensor): Tensor of size [B, ...].
        weight (torch.Tensor): Tensor of size [B].

    Returns:
        torch.Tensor: A tensor such that `result[i] = batch[i] * weight[i]`.
    """
    assert batch.size(0) == weight.size(0)
    return (batch.T * weight.T).T
