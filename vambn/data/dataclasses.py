from dataclasses import dataclass
from typing import List, Optional
from sklearn.discriminant_analysis import StandardScaler

import torch
import typeguard


@dataclass
class VariableType:
    """Dataclass for storing type information of the input variables"""

    name: str = "default"
    data_type: str = "default"
    n_parameters: int = -1
    input_dim: int = 1
    scaler: Optional[StandardScaler] = None

    def __eq__(self, __value: "VariableType") -> bool:
        """
        Test if two VariableTypes are equal

        Args:
            __value (VariableType): Second VariableType object

        Returns:
            bool: True or false
        """
        if not isinstance(__value, VariableType):
            return False
        return (
            self.name == __value.name
            and self.data_type == __value.data_type
            and self.n_parameters == __value.n_parameters
            and self.input_dim == __value.input_dim
        )

    def __post_init__(self):
        if isinstance(self.input_dim, float):
            raise TypeError("input_dim must be int, not float")
        if isinstance(self.n_parameters, float):
            raise TypeError("n_parameters must be int, not float")

    @typeguard.typechecked
    def reverse_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use the variable's scaler to invert the transformation so that the original input format is restored.

        Args:
            x (torch.tensor): The transformed input tensor.

        Returns:
            torch.Tensor: Inverse transformed output
        """
        if self.scaler is not None:
            # x can be of shape (time, batch) or (batch)
            orig_shape = x.shape
            dev = x.device
            if x.ndim == 2:
                # time x batch
                transformed = torch.from_numpy(
                    self.scaler.inverse_transform(x.cpu().numpy())
                ).to(dev)
                if orig_shape != transformed.shape:
                    raise RuntimeError(
                        f"Shape of transformed data is {transformed.shape}, expected {orig_shape}"
                    )
                return transformed
            elif x.ndim == 1:
                # batch
                x = x.view(-1, 1)
                transformed = (
                    torch.from_numpy(
                        self.scaler.inverse_transform(x.cpu().numpy())
                    )
                    .to(dev)
                    .view(-1)
                )
                if orig_shape != transformed.shape:
                    raise RuntimeError(
                        f"Shape of transformed data is {transformed.shape}, expected {orig_shape}"
                    )
                return transformed
            else:
                raise RuntimeError(
                    f"Input data must have 1 or 2 dimensions, got {x.ndim}"
                )
        else:
            # raise RuntimeError("Scaler is not defined")
            return x


VarTypes = List[VariableType]


def get_input_dim(types: List[VariableType]) -> int:
    """
    Get the input dimension of a list of variable types

    Args:
        types (List[VariableType]): List of variable types

    Returns:
        int: Sum of input dimensions
    """
    return int(sum([x.input_dim for x in types]))


def check_equal_types(a: List[VariableType], b: List[VariableType]) -> bool:
    """
    Check if two lists of variable types are equal

    Args:
        a (List[VariableType]): First list of variable types
        b (List[VariableType]): Second list of variable types

    Returns:
        bool: True if equal, False otherwise
    """
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if x != y:
            return False

    return True
