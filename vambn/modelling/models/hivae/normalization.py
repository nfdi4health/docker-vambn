import logging
from dataclasses import dataclass
from typing import Tuple

import torch

from vambn.data.dataclasses import VarTypes
from vambn.modelling.distributions.parameters import (
    LogNormalParameters,
    NormalParameters,
    Parameters,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class NormalizationParameters:
    """
    Data class for normalization parameters, including mean and standard deviation.
    
    This class is only used for the parameters of real and pos typed variables.

    Args:
        mean (torch.Tensor): The mean values for normalization.
        std (torch.Tensor): The standard deviation values for normalization.
    """

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_tensors(
        cls, mean: torch.Tensor, std: torch.Tensor
    ) -> "NormalizationParameters":
        """
        Create NormalizationParameters from mean and std tensors.

        Args:
            mean (torch.Tensor): The mean tensor.
            std (torch.Tensor): The standard deviation tensor.

        Returns:
            NormalizationParameters: An instance of NormalizationParameters.
        """
        return NormalizationParameters(mean, std)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor] | "NormalizationParameters":  # type: ignore
        """
        Get normalization parameters by index.

        Args:
            idx (int): The index to retrieve parameters for.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] | NormalizationParameters: The mean and std tensors or a new NormalizationParameters instance.
        """
        if self.mean.ndim == 1:
            return self.mean[idx], self.std[idx]
        elif self.mean.ndim == 2:
            mean = self.mean[idx, :]
            std = self.std[idx, :]
            return NormalizationParameters(mean, std)

    def __setitem__(
        self, idx: int, value: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """
        Set normalization parameters by index.

        Args:
            idx (int): The index to set parameters for.
            value (Tuple[torch.Tensor, torch.Tensor]): The mean and std tensors to set.
        """
        self.mean[idx], self.std[idx] = value

    def __add__(
        self, other: "NormalizationParameters"
    ) -> "NormalizationParameters":
        """
        Add two NormalizationParameters instances.

        Args:
            other (NormalizationParameters): Another instance to add.

        Returns:
            NormalizationParameters: A new instance with combined parameters.
        """
        if self.mean.ndim == 1:
            # create a 2d tensor by stacking the tensors
            mean = torch.stack([self.mean, other.mean])
            std = torch.stack([self.std, other.std])
        elif self.mean.ndim == 2:
            mean = torch.cat([self.mean, other.mean.unsqueeze(0)], dim=0)
            std = torch.cat([self.std, other.std.unsqueeze(0)], dim=0)
        else:
            raise ValueError("Invalid dimension for mean tensor")

        return NormalizationParameters(mean, std)


class Normalization:
    """
    Class for normalization utilities, including broadcasting masks and normalizing/denormalizing data.
    """
    @staticmethod
    def _broadcast_mask(
        mask: torch.Tensor, variable_types: VarTypes
    ) -> torch.Tensor:
        """
        Broadcast the mask tensor to match the shape required by variable types.

        Args:
            mask (torch.Tensor): The input mask tensor.
            variable_types (VarTypes): The variable types information.

        Returns:
            torch.Tensor: The broadcasted mask tensor.
        """
        # if all(d.n_parameters == 1 for d in variable_types):
        #     return mask

        new_mask = []
        for i, vtype in enumerate(variable_types):
            if vtype.data_type == "cat":
                new_mask.append(
                    mask[:, i].unsqueeze(1).expand(-1, vtype.n_parameters)
                )
            else:
                new_mask.append(mask[:, i].unsqueeze(1))
        return torch.cat(new_mask, dim=1)

    @staticmethod
    def normalize_data(
        x: torch.Tensor,
        mask: torch.Tensor,
        variable_types: VarTypes,
        prior_parameters: NormalizationParameters,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, NormalizationParameters]:
        """
        Normalize the input data based on variable types and prior parameters.

        Args:
            x (torch.Tensor): The input data tensor.
            mask (torch.Tensor): The mask tensor indicating missing values.
            variable_types (VarTypes): The variable types information.
            prior_parameters (NormalizationParameters): The prior normalization parameters.
            eps (float, optional): A small value to prevent division by zero. Defaults to 1e-6.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, NormalizationParameters]: The normalized data, updated mask, and new normalization parameters.
        """
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
            mask = mask.squeeze(1)

        assert len(variable_types) == x.shape[-1]
        mean_data = prior_parameters.mean
        std_data = prior_parameters.std
        new_x = []
        for i, vtype in enumerate(variable_types):
            x_i = torch.masked_select(x[..., i], mask[..., i].bool())
            new_x_i = torch.unsqueeze(x[..., i], -1)

            if vtype.data_type == "real" or vtype.data_type == "truncate_norm":
                if x_i.shape[0] >= 4:
                    mean_data[i] = x_i.mean()
                    std_data[i] = x_i.std().clamp(min=eps, max=1e20)

                new_x_i = (new_x_i - mean_data[i]) / std_data[i]
            elif vtype.data_type == "pos":
                x_i = torch.log1p(x_i)
                if x_i.shape[0] >= 4:
                    mean_data[i] = x_i.mean()
                    std_data[i] = x_i.std().clamp(min=eps, max=1e20)

                new_x_i = (torch.log1p(new_x_i) - mean_data[i]) / std_data[i]
            elif vtype.data_type == "gamma":
                x_i = torch.log1p(x_i)
                if x_i.shape[0] >= 4:
                    mean_data[i] = x_i.mean()
                    std_data[i] = x_i.std().clamp(min=eps, max=1e20)

                new_x_i = (torch.log1p(new_x_i) - mean_data[i]) / std_data[i]
            elif vtype.data_type == "count":
                new_x_i = torch.log1p(new_x_i)
            elif vtype.data_type == "cat":
                # convert to one hot
                new_x_i = torch.nn.functional.one_hot(
                    new_x_i.long().squeeze(1), vtype.n_parameters
                )

            if torch.isnan(new_x_i).any():
                raise ValueError(
                    f"NaN values found in normalized data for {vtype}"
                )
            if torch.isnan(mean_data[i]) or torch.isnan(std_data[i]):
                raise ValueError(
                    f"NaN values found in normalization parameters for {vtype}"
                )
            new_x.append(new_x_i)

        new_x = torch.cat(new_x, dim=-1)
        mask = Normalization._broadcast_mask(mask, variable_types)
        new_x = new_x * mask
        return (
            new_x,
            mask,
            NormalizationParameters.from_tensors(mean_data, std_data),
        )

    @staticmethod
    def denormalize_params(
        params: Tuple[Parameters, ...],
        variable_types: VarTypes,
        normalization_params: NormalizationParameters,
    ) -> Tuple[Parameters, ...]:
        """
        Denormalize parameters based on variable types and normalization parameters.

        Args:
            etas (Tuple[Etas, ...]): The parameters to denormalize.
            variable_types (VarTypes): The variable types information.
            normalization_params (NormalizationParameters): The normalization parameters.

        Returns:
            Tuple[Parameters, ...]: The denormalized parameters.
        """
        out_params = []
        for i, vtype in enumerate(variable_types):
            param_i = params[i]
            if vtype.data_type in ["truncate_norm", "real"] and isinstance(
                param_i, NormalParameters
            ):
                mean_data, std_data = normalization_params[i]
                std_data = std_data

                mean, std = param_i.loc, param_i.scale
                mean = mean * std_data + mean_data
                std = std * std_data

                out_params.append(NormalParameters(mean, std))
            elif vtype.data_type == "pos" and isinstance(
                param_i, LogNormalParameters
            ):
                mean_data, std_data = normalization_params[i]

                mean, std = param_i.loc, param_i.scale
                mean = mean * std_data + mean_data
                std = std * std_data

                out_params.append(LogNormalParameters(mean, std))
            elif vtype.data_type == "count":
                out_params.append(param_i)
            elif vtype.data_type == "cat":
                out_params.append(param_i)
            else:
                raise ValueError(f"Unknown data type {vtype.data_type}")
        return tuple(params)
