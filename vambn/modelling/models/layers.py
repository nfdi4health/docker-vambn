import logging

import torch
import torch.nn as nn

logger = logging.getLogger()


def init_weights(module: nn.Module):
    """Initialize the weights of a module.

    Args:
        module (nn.Module): The module to initialize.
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.05)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


class ModifiedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """Initialize the ModifiedLinear layer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
            device: The device on which to create the tensor. Defaults to None.
            dtype: The desired data type of the tensor. Defaults to None.
        """
        super().__init__(in_features, out_features, bias, device, dtype)
        nn.init.orthogonal_(self.weight)

        if bias:
            nn.init.constant_(self.bias, 0)


class ImputationLayer(nn.Module):
    """Imputation layer capable of handling both 2D and 3D data."""

    def __init__(self, feature_size: int) -> None:
        """Initialize the imputation layer.

        Args:
            feature_size (int): Size of the features dimension.

        Raises:
            ValueError: If `feature_size` is not positive.
        """
        super().__init__()

        if feature_size <= 0:
            raise ValueError(
                f"Feature size should be positive, got {feature_size}"
            )

        self.imputation_matrix = nn.Parameter(
            torch.zeros(feature_size, requires_grad=True)
        )
        nn.init.normal_(self.imputation_matrix)
        self.register_parameter("imputation_matrix", self.imputation_matrix)

    def forward(
        self, input_data: torch.Tensor, missing_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform the forward pass for data imputation.

        Args:
            input_data (torch.Tensor): Input data matrix, can be 2D (batch x features) or 3D (batch x time x features).
            missing_mask (torch.Tensor): Binary mask indicating missing values in `input_data`.

        Returns:
            torch.Tensor: Imputed data matrix.

        Raises:
            ValueError: If `input_data` is not 2D or 3D.
        """
        if input_data.dim() == 2:
            imputation = self.imputation_matrix
        elif input_data.dim() == 3:
            imputation = self.imputation_matrix.unsqueeze(0).expand(
                input_data.shape[1], -1
            )
        else:
            raise ValueError("Input data must be either 2D or 3D.")

        return input_data * missing_mask + imputation * (1.0 - missing_mask)
