import logging
from typing import Optional

import torch
import torch.distributions as dists
import torch.nn as nn
import typeguard
from torch import Tensor
from torch.nn import functional as F

from vambn.modelling.distributions.gumbel_distribution import GumbelDistribution
from vambn.modelling.models.hivae.outputs import EncoderOutput
from vambn.modelling.models.layers import ModifiedLinear

logger = logging.getLogger()


class Encoder(nn.Module):
    """HIVAE Encoder.

    Args:
        input_dim (int): Dimension of input data (e.g., columns in dataframe).
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.

    Attributes:
        input_dim (int): Dimension of input data.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        encoder_s (ModifiedLinear): Linear layer for s encoding.
        encoder_z (nn.Module): Identity layer for z encoding.
        param_z (ModifiedLinear): Linear layer for z parameterization.
        _tau (float): Temperature parameter for Gumbel softmax.
        _decoding (bool): Flag indicating whether the model is in decoding mode.
    """


    def __init__(self, input_dim: int, dim_s: int, dim_z: int):
        """HIVAE Encoder

        Args:
            input_dim (int): Dimension of input data (e.g. columns in dataframe)
            dim_s (int): Dimension of s space
            dim_z (int): Dimension of z space
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(
                f"Input dimension must be positive, got {input_dim}"
            )

        if dim_s <= 0:
            raise ValueError(f"S dimension must be positive, got {dim_s}")

        if dim_z <= 0:
            raise ValueError(f"Z dimension must be positive, got {dim_z}")

        self.input_dim = input_dim
        self.dim_s = dim_s
        self.dim_z = dim_z

        self.encoder_s = ModifiedLinear(self.input_dim, dim_s, bias=True)
        self.encoder_z = nn.Identity()
        self.param_z = ModifiedLinear(
            self.input_dim + dim_s, dim_z * 2, bias=True
        )

        self._tau = 1.0
        self._decoding = True

    @property
    def decoding(self) -> bool:
        """bool: Flag indicating whether the model is in decoding mode."""
        return self._decoding

    @decoding.setter
    def decoding(self, value: bool) -> None:
        """Sets the decoding flag.

        Args:
            value (bool): Decoding flag.
        """
        self._decoding = value

    @property
    def tau(self) -> float:
        """float: Temperature parameter for Gumbel softmax."""
        return self._tau

    @tau.setter
    def tau(self, value: float) -> None:
        """Sets the temperature parameter for Gumbel softmax.

        Args:
            value (float): Temperature value.

        Raises:
            ValueError: If the temperature value is not positive.
        """
        if value <= 0:
            raise ValueError(f"Tau must be positive, got {value}")
        self._tau = value

    @staticmethod
    def q_z(loc, scale: Tensor) -> dists.Normal:
        """Creates a normal distribution for z.

        Args:
            loc (Tensor): Mean of the distribution.
            scale (Tensor): Standard deviation of the distribution.

        Returns:
            dists.Normal: Normal distribution.
        """
        return dists.Normal(loc, scale)

    def q_s(self, probs: Tensor) -> GumbelDistribution:
        """Creates a Gumbel distribution for s.

        Args:
            probs (Tensor): Probabilities for the Gumbel distribution.

        Returns:
            GumbelDistribution: Gumbel distribution.
        """
        return GumbelDistribution(probs=probs, temperature=self.tau)

    def forward(self, x: Tensor) -> EncoderOutput:
        """Forward pass of the encoder.

        Args:
            x (Tensor): Normalized input data.

        Raises:
            Exception: If samples contain NaN values.

        Returns:
            EncoderOutput: Contains samples, logits, and parameters.
        """
        logits_s = self.encoder_s(x)
        probs_s = F.softmax(logits_s, dim=-1).clamp(1e-6, 1 - 1e-6)

        if self.training:
            samples_s = self.q_s(probs_s).rsample()
        elif self.decoding:
            # NOTE:
            # The idea behind this was that we can use e.g. the mode during decoding
            # Experiments indicate that this is not helpful; the correlation between
            # the real and decoded data is improved, but the JSD is significantly worse.
            samples_s = self.q_s(probs_s).sample()
        else:
            samples_s = self.q_s(probs_s).sample()

        x_and_s = torch.cat((x, samples_s), dim=-1)

        h = self.encoder_z(x_and_s)

        loc_z, scale_z_unnorm = torch.chunk(self.param_z(h), 2, dim=-1)
        scale_z = F.softplus(scale_z_unnorm) + 1e-6

        samples_z = (
            self.q_z(loc_z, scale_z).rsample()
            if self.training
            else self.q_z(loc_z, scale_z).sample()
            if self.decoding
            else self.q_z(loc_z, scale_z).sample()
        )

        return EncoderOutput(
            samples_s=samples_s,
            samples_z=samples_z,
            logits_s=logits_s,
            mean_z=loc_z,
            scale_z=scale_z,
        )


class LstmEncoder(nn.Module):
    """Encoder for longitudinal input.

    Args:
        input_dimension (int): Dimension of input data.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        n_layers (int): Number of LSTM layers.
        hidden_size (Optional[int], optional): Size of the hidden layer. Defaults to None.

    Attributes:
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        hidden_size (int): Size of the hidden layer.
        lstm (nn.LSTM): LSTM layer.
        encoder (Encoder): Encoder module.
    """

    @typeguard.typechecked
    def __init__(
        self,
        input_dimension: int,
        dim_s: int,
        dim_z: int,
        n_layers: int,
        hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim_s = dim_s
        self.dim_z = dim_z

        if hidden_size is None:
            hidden_size = input_dimension

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_dimension,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.encoder = Encoder(input_dim=hidden_size, dim_s=dim_s, dim_z=dim_z)

    def forward(self, input_data: Tensor) -> EncoderOutput:
        """Forward pass of the LSTM encoder.

        Args:
            input_data (Tensor): Time points/visits x batch size x variables/input size.

        Returns:
            EncoderOutput: Output for each time point.
        """
        out, (_, _) = self.lstm(input_data)
        # out: batch_size x time points x hidden size
        last_out = out[:, -1, :]
        encoder_output = self.encoder.forward(last_out)

        return encoder_output

    def q_z(self, loc, scale: Tensor) -> dists.Normal:
        """Creates a normal distribution for z.

        Args:
            loc (Tensor): Mean of the distribution.
            scale (Tensor): Standard deviation of the distribution.

        Returns:
            dists.Normal: Normal distribution.
        """
        return self.encoder.q_z(loc, scale)

    def q_s(self, probs: Tensor) -> GumbelDistribution:
        """Creates a Gumbel distribution for s.

        Args:
            probs (Tensor): Probabilities for the Gumbel distribution.

        Returns:
            GumbelDistribution: Gumbel distribution.
        """
        return self.encoder.q_s(probs)
