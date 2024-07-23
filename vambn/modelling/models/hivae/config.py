from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Optional, Tuple, TypeVar

from vambn.data.dataclasses import VarTypes, get_input_dim
from vambn.modelling.models.config import DataModuleConfig, ModelConfig


@dataclass
class ModularHivaeConfig(ModelConfig):
    """Configuration class for Modular HIVAE models.

    Attributes:
        module_config (Tuple[DataModuleConfig]): Configuration for the data modules. See DataModuleConfig.
        dim_s (int | Tuple[int, ...] | Dict[str, int]): Dimension of the latent space S.
        dim_z (int): Dimension of the latent space Z.
        dim_ys (int): Dimension of the YS space.
        dim_y (int | Tuple[int, ...] | Dict[str, int]): Dimension of the Y space.         
        mtl_method (Tuple[str, ...]): Methods for multi-task learning. Tested
            possibilities are combinations of "identity", "gradnorm", "graddrop".
            Further implementations and details can be found in the mtl.py file.
        use_imputation_layer (bool): Flag to use imputation layer.
        dropout (float): Dropout rate.
        n_layers (int): Number of layers.
        shared_element (str): Shared element type. Possible values are "none", 
            "sharedLinear", "concatMtl", "concatIndiv", "avgMtl", "maxMtl", "encoder", "encoderMtl".
            Default is "none".
    """    
    module_config: Tuple[DataModuleConfig]
    dim_s: int | Tuple[int, ...] | Dict[str, int]
    dim_z: int
    dim_ys: int
    dim_y: int | Tuple[int, ...] | Dict[str, int]
    mtl_method: Tuple[str, ...]
    use_imputation_layer: bool
    dropout: float
    n_layers: int
    shared_element: str = "none"

    def __post_init__(self):
        """Validates and sets up the configuration after initialization.

        Raises:
            Exception: If the number of layers is less than 1.
        """
        if self.n_layers < 1:
            raise Exception("Number of layers must be at least 1")

        for module in self.module_config:
            module.n_layers = self.n_layers


@dataclass
class HivaeConfig(ModelConfig):
    """Configuration class for HIVAE models.

    Attributes:
        name (str): Name of the configuration. Typically the module name.
        variable_types (VarTypes): Definition of the variable types. See VarTypes.
        dim_s (int): Dimension of the latent space S.
        dim_y (int): Dimension of the Y space.
        dim_z (int): Dimension of the latent space Z.
        mtl_methods (Tuple[str, ...]): Methods for multi-task learning. Tested 
            possibilities are combinations of "identity", "gradnorm", "graddrop". 
            Further implementations and details can be found in the mtl.py file.
        use_imputation_layer (bool): Flag to use imputation layer.
        dropout (float): Dropout rate.
        n_layers (Optional[int]): Number of layers. Needed for longitudinal data.
        num_timepoints (int): Number of timepoints for longitudinal data. Default is 1.
    """
    name: str
    variable_types: VarTypes
    dim_s: int
    dim_y: int
    dim_z: int
    mtl_methods: Tuple[str, ...]
    use_imputation_layer: bool
    dropout: float

    # Only needed for longitudinal data
    n_layers: Optional[int] = None
    num_timepoints: int = 1

    def __post_init__(self):
        """Converts mtl_methods to a tuple if it is a list."""
        if isinstance(self.mtl_methods, List):
            self.mtl_methods = tuple(self.mtl_methods)

    @cached_property
    def input_dim(self):
        """Gets the input dimension based on variable types.

        Returns:
            int: The input dimension.
        """
        return get_input_dim(self.variable_types)

    @cached_property
    def is_longitudinal(self) -> bool:
        """Checks if the data is longitudinal.

        Returns:
            bool: True if the data has more than one timepoint, False otherwise.
        """
        return self.num_timepoints > 1


ModelConfiguration = TypeVar("ModelConfiguration", bound=ModelConfig)
