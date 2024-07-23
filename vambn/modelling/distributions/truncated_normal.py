from typing import Optional
import torch
from torch.distributions import Normal, constraints


class TruncatedNormal(Normal):
    """
    TruncatedNormal distribution with support [low, high].

    This class extends the Normal distribution to support truncation, i.e., the
    distribution is limited to the range [low, high].
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "low": constraints.real,
        "high": constraints.real,
    }

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: Optional[torch.Tensor] = -torch.tensor(float("inf")),
        high: Optional[torch.Tensor] = torch.tensor(float("inf")),
        validate_args: Optional[bool] = None,
    ):
        """
        Initialize the TruncatedNormal distribution.

        Args:
            loc (torch.Tensor): The mean of the normal distribution.
            scale (torch.Tensor): The standard deviation of the normal distribution.
            low (Optional[torch.Tensor]): The lower bound of the truncation. Defaults to -inf.
            high (Optional[torch.Tensor]): The upper bound of the truncation. Defaults to inf.
            validate_args (Optional[bool]): Whether to validate arguments. Defaults to None.
        """
        self.low = low
        self.high = high
        super().__init__(loc, scale, validate_args=validate_args)

    def _clamp(self, x):
        """
        Clamp the values to the range [low, high].

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The clamped tensor.
        """
        clamped_x = torch.clamp(x, self.low, self.high)
        x_mask = (x < self.low) | (x > self.high)
        x_fill = torch.where(x < self.low, self.low, self.high)
        return torch.where(x_mask, x_fill, clamped_x)

    def sample(self, sample_shape=torch.Size()):
        """
        Draws a sample from the distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample to draw. Defaults to torch.Size().

        Returns:
            torch.Tensor: The drawn sample.
        """
        with torch.no_grad():
            return self._clamp(super().sample(sample_shape))

    def rsample(self, sample_shape=torch.Size()):
        """
        Draws a reparameterized sample from the distribution.

        Args:
            sample_shape (torch.Size, optional): The shape of the sample to draw. Defaults to torch.Size().

        Returns:
            torch.Tensor: The reparameterized sample.
        """
        with torch.no_grad():
            return self._clamp(super().rsample(sample_shape))

    def log_prob(self, value):
        """
        Calculate the log probability of a value.

        Args:
            value (torch.Tensor): Input value.

        Returns:
            torch.Tensor: Log probability of the input value.
        """
        if self._validate_args:
            self._validate_sample(value)
        log_prob = super().log_prob(value)
        log_prob = torch.where(
            (value < self.low) | (value > self.high),
            torch.log(torch.tensor(1e-12)),
            log_prob,
        )
        normalizer = torch.log(self.cdf(self.high) - self.cdf(self.low))
        return log_prob - normalizer

    def cdf(self, value):
        """
        Calculate the cumulative distribution function (CDF) of a value.

        Args:
            value (torch.Tensor): Input value.

        Returns:
            torch.Tensor: CDF of the input value.
        """
        if self._validate_args:
            self._validate_sample(value)
        cdf = super().cdf(value)
        low_cdf = super().cdf(self.low)
        high_cdf = super().cdf(self.high)
        return (cdf - low_cdf) / (high_cdf - low_cdf)

    def icdf(self, value):
        """
        Calculate the inverse cumulative distribution function (ICDF) of a value.

        Args:
            value (torch.Tensor): Input value.

        Returns:
            torch.Tensor: ICDF of the input value.
        """
        if self._validate_args:
            self._validate_sample(value)
        low_cdf = super().cdf(self.low)
        high_cdf = super().cdf(self.high)
        rescaled_value = low_cdf + (high_cdf - low_cdf) * value
        return super().icdf(rescaled_value)
