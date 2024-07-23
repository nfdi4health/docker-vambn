from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from vambn.data.dataclasses import VarTypes, get_input_dim


@dataclass
class ModelConfig:
    """Configuration for the model."""

    pass


@dataclass
class DataModuleConfig:
    """Configuration for the data module.

    Attributes:
        name (str): The name of the data module.
        variable_types (VarTypes): Types of variables used in the data module.
        num_timepoints (int): The number of timepoints. Must be at least 1.
        n_layers (Optional[int]): The number of layers. Defaults to None.
        noise_size (Optional[int]): The size of the noise. Defaults to None.
    """

    name: str
    variable_types: VarTypes
    num_timepoints: int
    n_layers: Optional[int] = None
    noise_size: Optional[int] = None

    def __post_init__(self):
        """Validate that the number of timepoints is at least 1.

        Raises:
            Exception: If the number of timepoints is less than 1.
        """
        if self.num_timepoints < 1:
            raise Exception("Number of timepoints must be at least 1")

    @cached_property
    def input_dim(self):
        """Get the input dimension based on variable types.

        Returns:
            int: The input dimension.
        """
        return get_input_dim(self.variable_types)

    @cached_property
    def is_longitudinal(self) -> bool:
        """Check if the data is longitudinal.

        Returns:
            bool: True if the number of timepoints is greater than 1, False otherwise.
        """
        return self.num_timepoints > 1
