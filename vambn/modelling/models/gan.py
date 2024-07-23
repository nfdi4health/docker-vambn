import logging
from typing import Tuple

import torch
import torch.functional as F
import torch.nn as nn
import typeguard


logger = logging.getLogger()


class Generator(nn.Module):
    @typeguard.typechecked
    def __init__(
        self,
        input_dim: int,
        layer_sizes: Tuple[int, ...],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        """Initialize the Generator network.

        Args:
            input_dim (int): The dimension of the input tensor.
            layer_sizes (Tuple[int, ...]): A tuple specifying the sizes of the hidden layers.
            output_dim (int): The dimension of the output tensor.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ReLU().
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = activation

        layers = [
            nn.Linear(input_dim, layer_sizes[0]),
            nn.BatchNorm1d(layer_sizes[0], eps=1e-5, momentum=0.1),
            activation,
        ]

        for i in range(len(layer_sizes) - 1):
            layers += [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-5, momentum=0.1),
                activation,
            ]

        layers += [
            nn.Linear(layer_sizes[-1], output_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Generator network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.net(x)


class Classifier(nn.Module):
    @typeguard.typechecked
    def __init__(
        self,
        input_dim: int,
        layer_sizes: Tuple[int, ...],
        output_dim: int = 1,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        """Initialize the Classifier network.

        Args:
            input_dim (int): The dimension of the input tensor.
            layer_sizes (Tuple[int, ...]): A tuple specifying the sizes of the hidden layers.
            output_dim (int, optional): The dimension of the output tensor. Defaults to 1.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.ReLU().
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = activation

        layers = [
            nn.Linear(input_dim, layer_sizes[0]),
            nn.BatchNorm1d(layer_sizes[0], eps=1e-5, momentum=0.1),
            activation,
        ]

        for i in range(len(layer_sizes) - 1):
            layers += [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-5, momentum=0.1),
                activation,
            ]

        layers += [
            nn.Linear(layer_sizes[-1], output_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Classifier network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor with sigmoid activation.
        """
        return F.sigmoid(self.net(x))


class Discriminator(nn.Module):
    @typeguard.typechecked
    def __init__(
        self,
        input_dim: int,
        layer_sizes: Tuple[int, ...],
        output_dim: int = 1,
        activation: nn.Module = nn.LeakyReLU(),
    ) -> None:
        """Initialize the Discriminator network.

        Args:
            input_dim (int): The dimension of the input tensor.
            layer_sizes (Tuple[int, ...]): A tuple specifying the sizes of the hidden layers.
            output_dim (int, optional): The dimension of the output tensor. Defaults to 1.
            activation (nn.Module, optional): The activation function to use. Defaults to nn.LeakyReLU().
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = activation

        layers = [
            nn.Linear(input_dim, layer_sizes[0]),
            nn.BatchNorm1d(layer_sizes[0], eps=1e-5, momentum=0.1),
            activation,
        ]

        for i in range(len(layer_sizes) - 1):
            layers += [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-5, momentum=0.1),
                activation,
            ]

        layers += [
            nn.Linear(layer_sizes[-1], output_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Discriminator network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return torch.mean(self.net(x))
