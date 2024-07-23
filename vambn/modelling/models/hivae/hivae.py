import logging
from functools import cached_property, reduce
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from vambn.data.dataclasses import VarTypes
from vambn.modelling.models.hivae.decoder import Decoder, LstmDecoder
from vambn.modelling.models.hivae.encoder import Encoder, LstmEncoder
from vambn.modelling.models.hivae.normalization import (
    Normalization,
    NormalizationParameters,
)
from vambn.modelling.models.hivae.outputs import (
    EncoderOutput,
    HivaeEncoding,
    HivaeOutput,
)
from vambn.modelling.models.layers import ImputationLayer
from vambn.modelling.models.templates import AbstractNormalModel

logger = logging.getLogger()


class Hivae(
    AbstractNormalModel[torch.Tensor, torch.Tensor, HivaeOutput, HivaeEncoding]
):
    """
    Entire HIVAE model containing Encoder and Decoder structure.

    Args:
        variable_types (VarTypes): List of VariableType objects defining the types
            of the variables in the data.
        input_dim (int): Dimension of input data (number of columns in the dataframe).
            If the data contains categorical variables, the input dimension is
            larger than the number of features.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_y (int): Dimension of y space.
        module_name (str, optional): Name of the module this HIVAE is associated with. Defaults to 'HIVAE'.
        mtl_method (Tuple[str], optional): List of methods to use for multi-task learning.
            Assessed possibilities are combinations of "identity", "gradnorm", "graddrop".
            Further implementations and details can be found in the mtl.py file. Defaults to ("identity",).
        use_imputation_layer (bool, optional): Flag to indicate if imputation layer should be used. Defaults to False.
        individual_model (bool, optional): Flag to indicate if the current model
            is applied individually or as part of e.g. a modular HIVAE. Defaults to True.
    """


    def __init__(
        self,
        variable_types: VarTypes,
        input_dim: int,
        dim_s: int,
        dim_z: int,
        dim_y: int,
        module_name: Optional[str] = "HIVAE",
        mtl_method: Tuple[str, ...] = ("identity",),
        use_imputation_layer: bool = False,
        individual_model: bool = True,
    ) -> None:


        super().__init__()

        self.variable_types = variable_types
        self.input_dim = input_dim
        self.dim_s = dim_s
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.individual_model = individual_model

        self.use_imp_layer = use_imputation_layer
        self.module_name = module_name

        # Imputation layer
        if use_imputation_layer:
            self.imputation_layer = ImputationLayer(self.input_dim)
        else:
            self.imputation_layer = None

        # Normalization parameters
        self.register_buffer(
            "_mean_data",
            torch.zeros(len(self.variable_types), requires_grad=False),
        )
        self.register_buffer(
            "_std_data",
            torch.ones(len(self.variable_types), requires_grad=False),
        )
        self._batch_mean_data = self._batch_std_data = None

        self.encoder = Encoder(input_dim=input_dim, dim_s=dim_s, dim_z=dim_z)
        self.decoder = Decoder(
            variable_types=variable_types,
            s_dim=dim_s,
            z_dim=dim_z,
            y_dim=dim_y,
            mtl_method=mtl_method,
            decoder_shared=nn.Identity(),
        )
        self._temperature = 1.0
        self.tau = self._temperature

        # set to cpu by default
        self.device = torch.device("cpu")
        self.module_name = module_name

    @cached_property
    def colnames(self) -> Tuple[str, ...]:
        """
        Tuple of column names derived from variable types.

        Returns:
            Tuple[str, ...]: A tuple containing column names.
        """

        return tuple([v.name for v in self.variable_types])

    @property
    def decoding(self) -> bool:
        """
        Decoding flag indicating if the encoder and decoder are in decoding mode.

        Returns:
            bool: Decoding flag.
        """
        assert self.encoder.decoding == self.decoder.decoding
        return self.encoder.decoding

    @decoding.setter
    def decoding(self, value: bool) -> None:
        """
        Sets the decoding flag for both encoder and decoder.

        Args:
            value (bool): The decoding flag to set.
        """
        self.encoder.decoding = value
        self.decoder.decoding = value

    @property
    def tau(self) -> float:
        """
        Gets the temperature parameter for the model.

        Returns:
            float: The temperature parameter.
        """
        return self._temperature

    @tau.setter
    def tau(self, value: float) -> None:
        """
        Sets the temperature parameter for the model.

        Args:
            value (float): The temperature parameter to set.

        Raises:
            ValueError: If the value is not positive.
        """

        if value <= 0:
            raise ValueError(f"Tau must be positive, got {value}")
        self._temperature = value
        self.encoder.tau = value

    @property
    def normalization_parameters(self) -> NormalizationParameters:
        """
        Gets the normalization parameters (mean and standard deviation).

        Returns:
            NormalizationParameters: The normalization parameters.
        """

        if self._batch_std_data is not None:
            return NormalizationParameters(
                mean=self._batch_mean_data, std=self._batch_std_data
            )
        else:
            return NormalizationParameters(
                mean=self._mean_data, std=self._std_data
            )

    @normalization_parameters.setter
    def normalization_parameters(self, value: NormalizationParameters) -> None:
        """
        Sets the normalization parameters (mean and standard deviation).

        Args:
            value (NormalizationParameters): The normalization parameters to set.
        """

        if self.training:
            # calculate the mean and std of the data with momentum
            momentum = 0.01
            # Choosing a momentum of 0.01 to update the mean and std of the data
            # this leads to a smoother update of the mean and std of the data
            new_mean = (1 - momentum) * self._mean_data + momentum * value.mean
            new_std = (1 - momentum) * self._std_data + momentum * value.std
            self._mean_data = new_mean
            self._std_data = new_std
            self._batch_mean_data = value.mean
            self._batch_std_data = value.std
        else:
            logger.warning(
                "Running parameters are not updated during evaluation"
            )
            self._batch_mean_data = value.mean
            self._batch_std_data = value.std

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> HivaeOutput:
        """
        Forward pass through the HIVAE model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            HivaeOutput: The output of the HIVAE model.
        """

        data, mask, encoder_output = self.encoder_part(data, mask)
        decoder_output = self.decoder_part(data, mask, encoder_output)
        return decoder_output

    def decoder_part(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        encoder_output: EncoderOutput,
    ) -> HivaeOutput:
        """
        Pass through the decoder part of the model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.
            encoder_output (EncoderOutput): Output from the encoder.

        Returns:
            HivaeOutput: The output of the decoder.
        """

        decoder_output = self.decoder(
            data=data,
            mask=mask,
            encoder_output=encoder_output,
            normalization_parameters=self.normalization_parameters,
        )

        return decoder_output

    def encoder_part(
        self, data: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, EncoderOutput]:
        """
        Pass through the encoder part of the model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, EncoderOutput]: Processed data, mask, and encoder output.
        
        Raises:
            ValueError: If data tensor has invalid shape.
        """

        if data.ndim == 3 and data.shape[1] == 1:
            data = data.squeeze(1)
            mask = mask.squeeze(1)
        elif data.ndim == 3:
            raise ValueError(
                f"Data should have shape (batch_size, n_features), got {data.shape}"
            )

        self._batch_mean_data = None
        self._batch_std_data = None
        (
            new_x,
            new_mask,
            self.normalization_parameters,
        ) = Normalization.normalize_data(
            data, mask, variable_types=self.variable_types, prior_parameters=self.normalization_parameters
        )

        if self.use_imp_layer and self.imputation_layer is not None:
            new_x = self.imputation_layer(new_x, new_mask)

        encoder_output = self.encoder(new_x)
        return data, mask, encoder_output

    def decode(self, encoding: HivaeEncoding) -> torch.Tensor:
        """
        Decode the given encoding to reconstruct the input data.

        Args:
            encoding (HivaeEncoding): The encoding to decode.

        Returns:
            torch.Tensor: The reconstructed data tensor.
        """

        return self.decoder.decode(
            encoding_s=encoding.s,
            encoding_z=encoding.decoder_representation,
            normalization_params=self.normalization_parameters,
        )

    def _predict_step(
        self, data: torch.Tensor, mask: torch.Tensor
    ) -> HivaeOutput:
        """
        Prediction step without gradient calculation.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            HivaeOutput: The output of the HIVAE model.
        """

        with torch.no_grad():
            return self.forward(data, mask)

    def _test_step(self, data: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Test step to evaluate the model on test data.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            float: The test loss.
        """

        with torch.no_grad():
            return self.forward(data, mask).loss.detach()

    def _training_step(
        self, data: torch.Tensor, mask: torch.Tensor, optimizer: Optimizer
    ) -> float:
        """
        Training step to update the model parameters.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.
            optimizer (Optimizer): Optimizer for updating model parameters.

        Returns:
            float: The training loss.
        """

        optimizer.zero_grad()
        output = self.forward(data, mask)
        self.fabric.backward(output.loss)
        optimizer.step()
        return output.loss.detach()

    def _validation_step(self, data: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Validation step to evaluate the model on validation data.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            float: The validation loss.
        """

        with torch.no_grad():
            return self.forward(data, mask).loss.detach()


class LstmHivae(Hivae):
    """
    LSTM-based HIVAE model with Encoder and Decoder structure for temporal data.

    Args:
        variable_types (VarTypes): List of VariableType objects defining the types
            of the variables in the data.
        input_dim (int): Dimension of input data (number of columns in the dataframe).
            If the data contains categorical variables, the input dimension is
            larger than the number of features.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_y (int): Dimension of y space.
        n_layers (int): Number of layers in the LSTM.
        num_timepoints (int): Number of time points in the temporal data.
        module_name (str, optional): Name of the module this HIVAE is associated with. Defaults to 'HIVAE'.
        mtl_method (Tuple[str], optional): List of methods to use for multi-task learning.
            Assessed possibilities are combinations of "identity", "gradnorm", "graddrop".
            Further implementations and details can be found in the mtl.py file. Defaults to ("identity",).
        use_imputation_layer (bool, optional): Flag to indicate if imputation layer should be used. Defaults to False.
        individual_model (bool, optional): Flag to indicate if the current model
            is applied individually or as part of e.g. a modular HIVAE. Defaults to True.
    """

    def __init__(
        self,
        variable_types: VarTypes,
        input_dim: int,
        dim_s: int,
        dim_z: int,
        dim_y: int,
        n_layers: int,
        num_timepoints: int,
        module_name: str | None = "HIVAE",
        mtl_method: Tuple[str] = ("identity",),
        use_imputation_layer: bool = False,
        individual_model: bool = True,
    ) -> None:
        super().__init__(
            variable_types,
            input_dim,
            dim_s,
            dim_z,
            dim_y,
            module_name,
            mtl_method,
            use_imputation_layer,
            individual_model,
        )
        self.n_layers = n_layers
        self.num_timepoints = num_timepoints

        self.encoder = LstmEncoder(
            input_dimension=input_dim,
            dim_s=dim_s,
            dim_z=dim_z,
            n_layers=n_layers,
            hidden_size=input_dim,
        )
        self.decoder = LstmDecoder(
            mtl_method=mtl_method,
            n_layers=n_layers,
            num_timepoints=num_timepoints,
            s_dim=dim_s,
            variable_types=variable_types,
            y_dim=dim_y,
            z_dim=dim_z,
            decoder_shared=nn.Identity(),
        )
        self.imputation_layer = nn.ModuleList(
            [ImputationLayer(input_dim) for _ in range(num_timepoints)]
        )

        # set normalization params for each timepoint
        self.register_buffer(
            "_mean_data",
            torch.zeros(
                num_timepoints, len(self.variable_types), requires_grad=False
            ),
        )
        self.register_buffer(
            "_std_data",
            torch.ones(
                num_timepoints, len(self.variable_types), requires_grad=False
            ),
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> HivaeOutput:
        """
        Forward pass through the LSTM HIVAE model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            HivaeOutput: The output of the LSTM HIVAE model.
        """

        data, mask, encoder_output = self.encoder_part(data, mask)
        decoder_output = self.decoder_part(data, mask, encoder_output)
        return decoder_output

    def decoder_part(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        encoder_output: EncoderOutput,
    ) -> HivaeOutput:
        """
        Pass through the decoder part of the model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.
            encoder_output (EncoderOutput): Output from the encoder.

        Returns:
            HivaeOutput: The output of the decoder.
        """

        decoder_output = self.decoder(
            data=data,
            mask=mask,
            encoder_output=encoder_output,
            normalization_parameters=self.normalization_parameters,
        )

        return decoder_output

    def encoder_part(
        self, data: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, EncoderOutput]:
        """
        Pass through the encoder part of the model.

        Args:
            data (torch.Tensor): Input data tensor.
            mask (torch.Tensor): Mask tensor indicating missing values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, EncoderOutput]: Processed data, mask, and encoder output.
        """

        time_point_x = []
        time_point_mask = []
        normalization_parameters = [None] * self.num_timepoints
        for i in range(self.num_timepoints):
            (
                new_x,
                new_mask,
                normalization_parameters[i],
            ) = Normalization.normalize_data(
                data[:, i], mask[:, i], variable_types=self.variable_types, prior_parameters=self.normalization_parameters[i]
            )
            time_point_x.append(new_x)
            time_point_mask.append(new_mask)

        self.normalization_parameters = reduce(
            lambda x, y: x + y, normalization_parameters
        )
        new_x = torch.stack(time_point_x, dim=1)
        new_mask = torch.stack(time_point_mask, dim=1)

        if self.use_imp_layer and self.imputation_layer is not None:
            for i in range(self.num_timepoints):
                new_x[:, i] = self.imputation_layer[i](
                    new_x[:, i], new_mask[:, i]
                )

        encoder_output = self.encoder(new_x)
        return data, mask, encoder_output

    def decode(self, encoding: HivaeEncoding) -> torch.Tensor:
        """
        Decode the given encoding to reconstruct the input data.

        Args:
            encoding (HivaeEncoding): The encoding to decode.

        Returns:
            torch.Tensor: The reconstructed data tensor.
        """

        return self.decoder.decode(
            encoding_s=encoding.s,
            encoding_z=encoding.decoder_representation,
            normalization_params=self.normalization_parameters,
        )
