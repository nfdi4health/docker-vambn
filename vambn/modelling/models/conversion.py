from typing import List, Tuple

import torch
from torch import Tensor

from vambn.data.dataclasses import VariableType


class Conversion:
    @staticmethod
    def _encode_categorical(
        x: Tensor,
        mask: Tensor,
        n: int,
    ) -> Tuple[Tensor, Tensor]:
        """Encode categorical data into one-hot vectors.

        Args:
            x (Tensor): The input tensor containing categorical data.
            mask (Tensor): The mask tensor indicating valid data points.
            n (int): The number of categories.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the one-hot encoded data tensor and the updated mask tensor.
        """
        ohe_data = torch.zeros((mask.shape[0], n), device=x.device)
        ohe_data[mask == 1, :] = torch.nn.functional.one_hot(
            torch.masked_select(x, mask.bool()).long(), num_classes=n
        ).float()
        ohe_mask = mask.view(-1, 1).repeat(1, n)
        return ohe_data, ohe_mask

    @staticmethod
    def cat_to_one_hot(
        variable_types: List[VariableType],
        x: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Normalize the input data batch-wise for various data types.

        Args:
            variable_types (List[VariableType]): List of variable types indicating the data type of each variable.
            x (Tensor): The input tensor containing data to be normalized.
            mask (Tensor): The mask tensor indicating valid data points.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the normalized data tensor and the updated mask tensor.
        """
        normalized_data, normalized_mask = [], []
        for i, (var_type, d, m) in enumerate(zip(variable_types, x.T, mask.T)):
            if var_type.data_type == "cat":
                normalized, mask_i = Conversion._encode_categorical(
                    x=d, mask=m, n=var_type.input_dim
                )
            else:
                normalized, mask_i = d, m

            normalized_data.append(normalized)
            normalized_mask.append(mask_i)

        if any([x.ndim == 2 for x in normalized_data]):
            normalized_data = [
                x if x.ndim == 2 else x.unsqueeze(1) for x in normalized_data
            ]
            normalized_mask = [
                x if x.ndim == 2 else x.unsqueeze(1) for x in normalized_mask
            ]
            new_x = torch.cat(normalized_data, dim=1).float()
            new_mask = torch.cat(normalized_mask, dim=1)
        else:
            new_x = torch.stack(normalized_data, dim=1).float()
            new_mask = torch.stack(normalized_mask, dim=1)
        new_x *= new_mask

        return new_x, new_mask
