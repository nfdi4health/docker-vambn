from typing import Dict, Optional

import torch
from torch.distributions import Distribution, constraints

from vambn.modelling.distributions.gumbel_softmax import gumbel_softmax


class ReparameterizedCategorical(Distribution):
    """
    A categorical distribution with reparameterized sampling using the Gumbel-Softmax trick.

    This class extends the torch.distributions.Categorical distribution to allow for
    reparameterized sampling, which enables gradient-based optimization techniques.

    Attributes:
        _categorical (torch.distributions.Categorical): The underlying categorical distribution.
        temperature (float): The temperature parameter for the Gumbel-Softmax distribution.
    """

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ):
        """
        Initialize the Reparameterized Categorical Distribution.
        Args:
            logits (Optional[torch.Tensor]): A tensor of logits (unnormalized log probabilities).
            probs (Optional[torch.Tensor]): A tensor of probabilities.
            temperature (float): A temperature parameter for the Gumbel-Softmax distribution.
        """
        self._categorical = torch.distributions.Categorical(
            logits=logits, probs=probs
        )
        self.temperature = temperature

    @property
    def param_shape(self) -> torch.Size:
        """
        Returns the shape of the parameter tensor.
        Returns:
            torch.Size: The shape of the parameter tensor.
        """
        return self._categorical.param_shape

    @property
    def batch_shape(self) -> torch.Size:
        """
        Returns the shape of the batch of distributions.
        Returns:
            torch.Size: The shape of the batch of distributions.
        """
        return self._categorical.batch_shape

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the shape of the event of the distribution.
        Returns:
            torch.Size: The shape of the event of the distribution.
        """
        return self._categorical.event_shape

    @property
    def support(self) -> torch.Tensor:
        """
        Returns the support of the distribution.
        Returns:
            torch.Tensor: The support of the distribution.
        """
        return self._categorical.support

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Draws a sample from the distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample to draw. Defaults to torch.Size().

        Returns:
            torch.Tensor: The drawn sample.
        """
        return self._categorical.sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Reparameterized sampling using Gumbel-Softmax trick.
        """
        if torch.any(torch.isnan(self._categorical.logits)):
            raise Exception("NaN values in logits")
        samples = gumbel_softmax(
            logits=self._categorical.logits,
            shape=tuple(self._categorical.logits.shape),
            tau=self.temperature,
            hard=False,
        )
        return samples

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log_prob of a value.

        Args:
            value (torch.Tensor): Input value.

        Returns:
            torch.Tensor: Log probability of the input value.
        """

        return self._categorical.log_prob(value)

    @constraints.dependent_property
    def arg_constraints(self) -> Dict[str, constraints.Constraint]:
        """
        Returns the argument constraints of the distribution.

        Returns:
            Dict[str, Constraint]: Constraint dictionary.
        """
        return self._categorical.arg_constraints

    @property
    def mode(self) -> torch.Tensor:
        """
        Returns the mode of the distribution.
        """
        return self._categorical.mode.unsqueeze(-1)

    def mean(self) -> torch.Tensor:
        """
        Returns the mean of the distribution.
        """
        return self._categorical.mean
