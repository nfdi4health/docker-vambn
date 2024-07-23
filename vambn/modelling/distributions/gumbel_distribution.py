from __future__ import annotations

import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical


class GumbelDistribution(ExpRelaxedCategorical):
    """
    Gumbel distribution based on the ExpRelaxedCategorical distribution.
    """

    @property
    def probs(self):
        """
        Returns the probabilities associated with the Gumbel distribution.

        Returns:
            torch.Tensor: The probabilities.
        """
        return torch.exp(self.logits).clip(1e-6, 1 - 1e-6)

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        """
        Draws a sample from the Gumbel distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample to draw. Defaults to torch.Size().

        Returns:
            torch.Tensor: The drawn sample.
        """
        probs = self.probs.clip(1e-6, 1 - 1e-6)
        return OneHotCategorical(probs=probs).sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling for the Gumbel distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample to draw. Defaults to torch.Size().

        Returns:
            torch.Tensor: The reparameterized sample.
        """
        return torch.exp(super().rsample(sample_shape))

    @property
    def mean(self):
        """
        Returns the mean of the Gumbel distribution.

        Returns:
            torch.Tensor: The mean of the distribution.
        """
        return self.probs.clip(1e-6, 1 - 1e-6)

    @property
    def mode(self):
        """
        Returns the mode of the Gumbel distribution.

        Returns:
            torch.Tensor: The mode of the distribution.
        """
        probs = self.probs.clip(1e-6, 1 - 1e-6)
        return OneHotCategorical(probs=probs).mode

    def expand(self, batch_shape, _instance=None):
        """
        Expands the Gumbel distribution to the given batch shape.

        Args:
            batch_shape (torch.Size): The desired batch shape.
            _instance: The instance to expand.

        Returns:
            GumbelDistribution: The expanded Gumbel distribution.
        """
        return super().expand(batch_shape[:-1], _instance)

    def log_prob(self, value):
        """
        Calculates the log probability of a value under the Gumbel distribution.

        Args:
            value (torch.Tensor): The value for which to calculate the log probability.

        Returns:
            torch.Tensor: The log probability of the value.
        """
        probs = self.probs.clip(1e-6, 1 - 1e-6)
        return OneHotCategorical(probs=probs).log_prob(value)
