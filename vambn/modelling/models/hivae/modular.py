import logging
from typing import Dict, List, Tuple

import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from vambn.modelling.models.hivae.config import DataModuleConfig
from vambn.modelling.models.hivae.hivae import Hivae, LstmHivae
from vambn.modelling.models.hivae.outputs import (
    EncoderOutput,
    ModularHivaeEncoding,
    ModularHivaeOutput,
)
from vambn.modelling.models.hivae.shared import SHARED_MODULES
from vambn.modelling.models.templates import (
    AbstractModularModel,
)

logger = logging.getLogger(__name__)


class ModularHivae(
    AbstractModularModel[
        Tuple[Tensor, ...],
        Tuple[Tensor, ...],
        ModularHivaeOutput,
        ModularHivaeEncoding,
    ]
):
    """
    Modular HIVAE model containing multiple data modules.

    Args:
        module_config (Tuple[DataModuleConfig]): Configuration for each data module. See DataModuleConfig for details.
        dim_s (int | Dict[str, int]): Number of mixture components for each module individually (dict) or a single value for all modules (int).
        dim_z (int): Dimension of the latent space. Equal for all modules.
        dim_ys (int): Dimension of the latent space ys. Equal for all modules.
        dim_y (int | Dict[str, int]): Dimension of the latent variable y for each module or a single value for all modules.
        shared_element_type (str, optional): Type of shared element. Possible values are "none", "sharedLinear", "concatMtl", "concatIndiv", "avgMtl", "maxMtl", "encoder", "encoderMtl". Defaults to "none".
        mtl_method (Tuple[str, ...], optional): Methods for multi-task learning. Tested possibilities are combinations of "identity", "gradnorm", "graddrop". Further implementations and details can be found in the mtl.py file. Defaults to ("identity",).
        use_imputation_layer (bool, optional): Flag to indicate if imputation layer should be used. Defaults to False.
    """

    def __init__(
        self,
        module_config: Tuple[DataModuleConfig],
        dim_s: int | Dict[str, int],
        dim_z: int,
        dim_ys: int,
        dim_y: int | Dict[str, int],
        shared_element_type: str = "none",
        mtl_method: Tuple[str, ...] = ("identity",),
        use_imputation_layer: bool = False,
    ):
        """
        Initialize the modular HIVAE model.

        Args:
            module_config (Tuple[DataModuleConfig]): Configuration for each data module.
            dim_s (int | Dict[str, int]): Number of the mixture components for each module or a single value for all modules.
            dim_z (int): Dimension of the latent space. Equal for all modules.
            dim_ys (int): Dimension of the latent space ys. Equal for all modules.
            dim_y (int | Dict[str, int]): Dimension of the latent variable y for each module or a single value for all modules.
            shared_element_type (str, optional): Type of shared element. Defaults to "none".
            mtl_method (Tuple[str, ...], optional): Methods for MTL. Defaults to ("identity",).
            use_imputation_layer (bool, optional): Flag to indicate if imputation layer should be used. Defaults to False.
        """

        super().__init__()
        self.module_configs = module_config
        self.mtl_method = mtl_method
        self.use_imputation_layer = use_imputation_layer
        self.dim_s = dim_s
        self.dim_z = dim_z
        self.dim_ys = dim_ys
        self.dim_y = dim_y

        if not isinstance(dim_s, int) and len(dim_s) != len(module_config):
            raise ValueError(
                "If dim_s is a tuple, it must have the same length as module_config"
            )

        # Initialize the shared element
        # The shared element is a module that takes the z samples from the encoder outputs
        # and generates a shared representation ys, finally a representation y for each module
        shared_element_class = SHARED_MODULES[shared_element_type]
        if shared_element_class is None:
            raise ValueError(
                f"Shared element {shared_element_type} is not available"
            )
        self.shared_element = shared_element_class(
            z_dim=self.dim_z,
            n_modules=len(module_config),
            ys_dim=self.dim_ys,
            y_dim=self.dim_y
            if isinstance(dim_y, int)
            else tuple([dim_y[module.name] for module in self.module_configs]),
            mtl_method=mtl_method if mtl_method else ("graddrop",),
            module_names=[module.name for module in module_config],
        )

        module_models = {}
        for module in module_config:
            module_name = module.name
            if module.is_longitudinal:
                module_models[module_name] = LstmHivae(
                    dim_s=dim_s
                    if isinstance(dim_s, int)
                    else dim_s[module_name],
                    dim_y=dim_y
                    if isinstance(dim_y, int)
                    else dim_y[module_name],
                    dim_z=dim_z,
                    individual_model=False,
                    input_dim=module.input_dim,
                    module_name=module_name,
                    mtl_method=mtl_method,
                    n_layers=module.n_layers,
                    num_timepoints=module.num_timepoints,
                    use_imputation_layer=use_imputation_layer,
                    variable_types=module.variable_types,
                )
            else:
                module_models[module_name] = Hivae(
                    dim_s=dim_s
                    if isinstance(dim_s, int)
                    else dim_s[module_name],
                    dim_y=dim_y
                    if isinstance(dim_y, int)
                    else dim_y[module_name],
                    dim_z=dim_z,
                    individual_model=False,
                    input_dim=module.input_dim,
                    module_name=module_name,
                    mtl_method=mtl_method,
                    use_imputation_layer=use_imputation_layer,
                    variable_types=module.variable_types,
                )

        self.module_models = nn.ModuleDict(module_models)
        self._tau = 1.0

    @property
    def decoding(self) -> bool:
        """
        Decoding flag indicating if the encoder and decoder are in decoding mode.

        Returns:
            bool: Decoding flag.
        """
        assert all([module.decoding for module in self.module_models.values()])
        return self.module_models[self.module_configs[0].name].decoding


    @decoding.setter
    def decoding(self, value: bool) -> None:
        """
        Sets the decoding flag for all modules.

        Args:
            value (bool): The decoding flag to set.
        """
        for module in self.module_models.values():
            module.decoding = value


    def colnames(self, module_name: str) -> Tuple[str, ...]:
        """
        Get column names for a specific module.

        Args:
            module_name (str): Name of the module.

        Returns:
            Tuple[str, ...]: Column names for the specified module.
        """
        return self.module_models[module_name].colnames


    def is_longitudinal(self, module_name: str) -> bool:
        """
        Check if a specific module is longitudinal.

        Args:
            module_name (str): Name of the module.

        Returns:
            bool: True if the module is longitudinal, False otherwise.
        """
        return self.module_models[module_name].is_longitudinal


    def forward(
        self, data: Tuple[Tensor, ...], mask: Tuple[Tensor, ...]
    ) -> ModularHivaeOutput:
        """
        Forward pass through the modular HIVAE model.

        Args:
            data (Tuple[Tensor, ...]): Input data tensors for each module.
            mask (Tuple[Tensor, ...]): Mask tensors indicating missing values for each module.

        Returns:
            ModularHivaeOutput: The output of the modular HIVAE model.
        """

        # Generate the encoder outputs and data for each module
        # This step does not differ from the normal HIVAE model
        encoder_outputs: List[EncoderOutput] = []
        modified_data = []
        modified_mask = []
        for i, module in enumerate(self.module_configs):
            mdata, mmask, output = self.module_models[module.name].encoder_part(
                data[i], mask[i]
            )
            encoder_outputs.append(output)
            modified_data.append(mdata)
            modified_mask.append(mmask)

        # Then retrieve the z samples from the encoder outputs
        # These z samples are then passed through the shared element to
        # achieve the modularity
        # The shared representation is a tuple of tensors, one for each module
        z_samples = tuple([x.samples_z for x in encoder_outputs])
        shared_representations = self.shared_element(z=z_samples)

        # This shared representation is then passed back to the encoder outputs
        for i, enc in enumerate(encoder_outputs):
            enc.decoder_representation = shared_representations[i]

        # Then we continue with the decoder part as usual
        decoder_outputs = []
        for i, module in enumerate(self.module_configs):
            output = self.module_models[module.name].decoder_part(
                data=modified_data[i],
                mask=modified_mask[i],
                encoder_output=encoder_outputs[i],
            )
            decoder_outputs.append(output)

        gathered_output = ModularHivaeOutput(outputs=tuple(decoder_outputs))
        return gathered_output

    @property
    def tau(self):
        """
        Get the temperature parameter for the model.

        Returns:
            float: The temperature parameter.
        """

        return self._tau

    @tau.setter
    def tau(self, value: float):
        """
        Set the temperature parameter for the model.

        Args:
            value (float): The temperature parameter to set.
        """

        self._tau = value
        for module in self.module_models.values():
            module.tau = value

    def decode(self, encoding: ModularHivaeEncoding) -> Tuple[Tensor, ...]:
        """
        Decode the given encoding to reconstruct the input data.

        Args:
            encoding (ModularHivaeEncoding): The encoding to decode.

        Returns:
            Tuple[Tensor, ...]: The reconstructed data tensors for each module.
        """

        self.eval()
        self.tau = 1e-3
        z_samples = tuple([x.z for x in encoding])

        modified_output = self.shared_element(z=z_samples)
        output_mapping = {
            x: i for i, x in enumerate(self.shared_element.module_names)
        }

        for enc in encoding.encodings:
            out_pos = output_mapping[enc.module]
            enc.decoder_representation = modified_output[out_pos]

        outputs = []
        for i, module in enumerate(self.module_configs):
            module_enc = encoding.get(module.name)
            output = self.module_models[module.name].decode(module_enc)
            outputs.append(output)
        assert len(outputs) == len(self.module_configs)
        return tuple(outputs)

    def _training_step(
        self,
        data: Tuple[Tensor],
        mask: Tuple[Tensor],
        optimizer: Tuple[optim.Optimizer],
    ) -> float:
        """
        Perform a training step to update the model parameters.

        Args:
            data (Tuple[Tensor]): Input data tensors for each module.
            mask (Tuple[Tensor]): Mask tensors indicating missing values for each module.
            optimizer (Tuple[optim.Optimizer]): Optimizers for updating model parameters.

        Returns:
            float: The training loss.
        """

        # set all gradients to zero
        for opt in optimizer:
            if opt is None:
                continue
            opt.zero_grad()
        output = self.forward(data, mask)
        self.fabric.backward(output.loss)
        for opt in optimizer:
            if opt is None:
                continue
            opt.step()
        return output.loss.item()

    def _validation_step(
        self, data: Tuple[Tensor], mask: Tuple[Tensor]
    ) -> float:
        """
        Perform a validation step to evaluate the model.

        Args:
            data (Tuple[Tensor]): Input data tensors for each module.
            mask (Tuple[Tensor]): Mask tensors indicating missing values for each module.

        Returns:
            float: The validation loss.
        """

        output = self.forward(data, mask)
        return output.loss.item()

    def _test_step(self, data: Tuple[Tensor], mask: Tuple[Tensor]) -> float:
        """
        Perform a test step to evaluate the model on test data.

        Args:
            data (Tuple[Tensor]): Input data tensors for each module.
            mask (Tuple[Tensor]): Mask tensors indicating missing values for each module.

        Returns:
            float: The test loss.
        """

        output = self.forward(data, mask)
        return output.loss.item()

    def _predict_step(
        self, data: Tuple[Tensor], mask: Tuple[Tensor]
    ) -> ModularHivaeOutput:
        """
        Perform a prediction step without gradient calculation.

        Args:
            data (Tuple[Tensor]): Input data tensors for each module.
            mask (Tuple[Tensor]): Mask tensors indicating missing values for each module.

        Returns:
            ModularHivaeOutput: The output of the modular HIVAE model.
        """

        return self.forward(data, mask)
