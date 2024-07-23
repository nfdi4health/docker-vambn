import logging
from functools import cached_property
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from vambn.data.dataclasses import VarTypes
from vambn.modelling.models.config import DataModuleConfig
from vambn.modelling.models.gan import Discriminator, Generator
from vambn.modelling.models.hivae.hivae import Hivae, LstmHivae
from vambn.modelling.models.hivae.modular import ModularHivae
from vambn.modelling.models.hivae.normalization import NormalizationParameters
from vambn.modelling.models.hivae.outputs import (
    EncoderOutput,
    HivaeEncoding,
    HivaeOutput,
    ModularHivaeEncoding,
    ModularHivaeOutput,
)
from vambn.modelling.models.templates import (
    AbstractGanModel,
    AbstractGanModularModel,
)

logger = logging.getLogger()


class GanHivae(
    AbstractGanModel[torch.Tensor, torch.Tensor, HivaeOutput, HivaeEncoding]
):
    """GAN-enhanced HIVAE model.

    Args:
        variable_types (VarTypes): Types of variables.
        input_dim (int): Dimension of input data.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_y (int): Dimension of y space.
        n_layers (int): Number of layers.
        noise_size (int, optional): Size of the noise vector. Defaults to 10.
        num_timepoints (Optional[int], optional): Number of time points. Defaults to None.
        module_name (Optional[str], optional): Name of the module. Defaults to "GanHivae".
        mtl_method (Tuple[str, ...], optional): Methods for multi-task learning. Defaults to ("identity",).
        use_imputation_layer (bool, optional): Whether to use an imputation layer. Defaults to False.
        individual_model (bool, optional): Whether to use individual models. Defaults to True.

    Attributes:
        noise_size (int): Size of the noise vector.
        is_longitudinal (bool): Flag for longitudinal data.
        model (Hivae or LstmHivae): HIVAE model.
        generator (Generator): GAN generator.
        discriminator (Discriminator): GAN discriminator.
        device (torch.device): Device to run the model on.
        one (nn.Parameter): Parameter for GAN training.
        mone (nn.Parameter): Parameter for GAN training.
    """

    def __init__(
        self,
        variable_types: VarTypes,
        input_dim: int,
        dim_s: int,
        dim_z: int,
        dim_y: int,
        n_layers: int,
        noise_size: int = 10,
        num_timepoints: Optional[int] = None,
        module_name: Optional[str] = "GanHivae",
        mtl_method: Tuple[str, ...] = ("identity",),
        use_imputation_layer: bool = False,
        individual_model: bool = True,
    ):
        super().__init__()

        self.noise_size = noise_size
        if num_timepoints is not None and num_timepoints > 1:
            self.is_longitudinal = True
            self.model = LstmHivae(
                variable_types=variable_types,
                input_dim=input_dim,
                dim_s=dim_s,
                dim_z=dim_z,
                dim_y=dim_y,
                n_layers=n_layers,
                num_timepoints=num_timepoints,
                module_name=module_name,
                mtl_method=mtl_method,
                use_imputation_layer=use_imputation_layer,
                individual_model=individual_model,
            )
        else:
            self.is_longitudinal = False
            self.model = Hivae(
                variable_types=variable_types,
                input_dim=input_dim,
                dim_s=dim_s,
                dim_z=dim_z,
                dim_y=dim_y,
                module_name=module_name,
                mtl_method=mtl_method,
                use_imputation_layer=use_imputation_layer,
                individual_model=individual_model,
            )

        self.generator = Generator(noise_size, (8, 4), dim_z)
        self.discriminator = Discriminator(dim_z, (8, 4), 1)
        #
        self.device = torch.device("cpu")
        self.one = nn.Parameter(
            torch.FloatTensor([1.0])[0], requires_grad=False
        )
        self.mone = nn.Parameter(
            torch.FloatTensor([-1.0])[0], requires_grad=False
        )

    def _train_gan_generator_step(
        self, data: torch.Tensor, mask: torch.Tensor, optimizer: Optimizer
    ) -> torch.Tensor:
        """Performs a training step for the GAN generator.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.
            optimizer (Optimizer): Optimizer for the generator.

        Returns:
            torch.Tensor: Generator loss.
        """
        optimizer.zero_grad()
        noise = torch.randn(data.shape[0], self.noise_size)
        fake_hidden = self.generator(noise)
        errG = self.discriminator(fake_hidden)
        self.fabric.backward(errG, self.one)
        optimizer.step()
        return errG

    def _train_gan_discriminator_step(
        self, data: torch.Tensor, mask: torch.Tensor, optimizer: Optimizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a training step for the GAN discriminator.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.
            optimizer (Optimizer): Optimizer for the discriminator.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Discriminator loss, real loss, and fake loss.
        """
        optimizer.zero_grad()
        _, _, encoder_output = self.model.encoder_part(data, mask)
        real_hidden = encoder_output.samples_z
        errD_real = self.discriminator(real_hidden.detach())
        self.fabric.backward(errD_real, self.one)

        noise = torch.randn(data.shape[0], self.noise_size)
        fake_hidden = self.generator(noise)
        errD_fake = self.discriminator(fake_hidden.detach())
        self.fabric.backward(errD_fake, self.mone)

        gradient_penaltiy = self._calc_gradient_penalty(
            real_hidden, fake_hidden
        )
        self.fabric.backward(gradient_penaltiy)
        optimizer.step()
        errD = -(errD_real - errD_fake)
        return errD, errD_real, errD_fake

    def _train_model_step(
        self, data: torch.Tensor, mask: torch.Tensor, optimizer: Optimizer
    ) -> HivaeOutput:
        """Performs a training step for the HIVAE model.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.
            optimizer (Optimizer): Optimizer for the model.

        Returns:
            HivaeOutput: Model output.
        """
        optimizer.zero_grad()
        output = self.model.forward(data, mask)
        loss = output.loss
        # if loss > 1e10:
        #     raise optuna.TrialPruned("Unsuited hyperparameters. High loss")
        self.fabric.backward(loss)
        optimizer.step()
        return output

    def _train_model_from_discriminator_step(
        self, data: torch.Tensor, mask: torch.Tensor, optimizer: Optimizer
    ) -> torch.Tensor:
        """Performs a training step for the model from the discriminator's perspective.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.
            optimizer (Optimizer): Optimizer for the model.

        Returns:
            torch.Tensor: Discriminator real loss.
        """
        optimizer.zero_grad()
        _, _, encoder_output = self.model.encoder_part(data, mask)
        real_hidden = encoder_output.samples_z
        errD_real = self.discriminator(real_hidden)
        self.fabric.backward(errD_real, self.mone)
        optimizer.step()
        return errD_real

    def _get_loss_from_output(self, output: HivaeOutput) -> torch.Tensor:
        """Extracts the loss from the model output.

        Args:
            output (HivaeOutput): Model output.

        Returns:
            torch.Tensor: Loss value.
        """
        return output.loss

    def _get_number_of_items(self, mask: torch.Tensor) -> int:
        """Gets the number of items in the mask.

        Args:
            mask (torch.Tensor): Data mask.

        Returns:
            int: Number of items.
        """
        return mask.sum()

    @cached_property
    def colnames(self) -> Tuple[str]:
        """Gets the column names of the data.

        Returns:
            Tuple[str]: Column names.
        """
        return self.model.colnames

    @property
    def decoding(self) -> bool:
        """bool: Flag indicating whether the model is in decoding mode."""
        return self.model.decoding

    @decoding.setter
    def decoding(self, value: bool) -> None:
        """Sets the decoding flag.

        Args:
            value (bool): Decoding flag.
        """
        self.model.decoding = value

    @property
    def tau(self) -> float:
        """float: Temperature parameter for Gumbel softmax."""
        return self.model.tau

    @tau.setter
    def tau(self, value: float) -> None:
        """Sets the temperature parameter for Gumbel softmax.

        Args:
            value (float): Temperature value.
        """
        self.model.tau = value

    @property
    def normalization_parameters(self) -> NormalizationParameters:
        """NormalizationParameters: Parameters for normalization."""
        return self.model.normalization_parameters

    @normalization_parameters.setter
    def normalization_parameters(self, value: NormalizationParameters) -> None:
        """Sets the normalization parameters.

        Args:
            value (NormalizationParameters): Normalization parameters.
        """
        self.model.normalization_parameters = value

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> HivaeOutput:
        """Forward pass of the model.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.

        Returns:
            HivaeOutput: Model output.
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
        """Performs the decoder part of the forward pass.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.
            encoder_output (EncoderOutput): Output from the encoder.

        Returns:
            HivaeOutput: Model output.
        """
        return self.model.decoder_part(data, mask, encoder_output)

    def encoder_part(
        self, data: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, EncoderOutput]:
        """Performs the encoder part of the forward pass.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, EncoderOutput]: Encoder output and modified inputs.
        """
        return self.model.encoder_part(data, mask)

    def decode(self, encoding: HivaeEncoding) -> torch.Tensor:
        """Decodes the given encoding.

        Args:
            encoding (HivaeEncoding): Encoding to decode.

        Returns:
            torch.Tensor: Decoded data.
        """
        return self.model.decode(encoding)

    def _training_step(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        optimizer: Tuple[Optimizer],
    ) -> float:
        """Training step is not needed for this class."""
        raise Exception(f"Method not needed for {self.__class__.__name__}")

    def _validation_step(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Performs a validation step.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.

        Returns:
            float: Validation loss.
        """
        return self.model._validation_step(data, mask)

    def _test_step(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Performs a test step.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.

        Returns:
            float: Test loss.
        """
        return self.model._test_step(data, mask)

    def _predict_step(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
    ) -> HivaeOutput:
        """Performs a prediction step.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Data mask.

        Returns:
            HivaeOutput: Prediction output.
        """
        return self.model._predict_step(data, mask)


class GanModularHivae(
    AbstractGanModularModel[
        Tuple[Tensor, ...],
        Tuple[Tensor, ...],
        ModularHivaeOutput,
        ModularHivaeEncoding,
    ]
):
    """GAN-enhanced Modular HIVAE model.

    Args:
        module_config (Tuple[DataModuleConfig]): Configuration for the data modules.
        dim_s (int | Dict[str, int]): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_ys (int): Dimension of YS space.
        dim_y (int | Dict[str, int]): Dimension of y space.
        noise_size (int, optional): Size of the noise vector. Defaults to 10.
        shared_element_type (str, optional): Type of shared element. Defaults to "none".
        mtl_method (Tuple[str, ...], optional): Methods for multi-task learning. Defaults to ("identity",).
        use_imputation_layer (bool, optional): Whether to use an imputation layer. Defaults to False.

    Attributes:
        module_configs (Tuple[DataModuleConfig]): Configuration for the data modules.
        mtl_method (Tuple[str, ...]): Methods for multi-task learning.
        use_imputation_layer (bool): Whether to use an imputation layer.
        dim_s (int | Dict[str, int]): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_ys (int): Dimension of YS space.
        dim_y (int | Dict[str, int]): Dimension of y space.
        model (ModularHivae): Modular HIVAE model.
        generators (nn.ModuleList): List of GAN generators.
        discriminators (nn.ModuleList): List of GAN discriminators.
        device (torch.device): Device to run the model on.
        one (nn.Parameter): Parameter for GAN training.
        mone (nn.Parameter): Parameter for GAN training.
        noise_size (int): Size of the noise vector.
    """

    def __init__(
        self,
        module_config: Tuple[DataModuleConfig],
        dim_s: int | Dict[str, int],
        dim_z: int,
        dim_ys: int,
        dim_y: int | Dict[str, int],
        noise_size: int = 10,
        shared_element_type: str = "none",
        mtl_method: Tuple[str, ...] = ("identity",),
        use_imputation_layer: bool = False,
    ):
        super().__init__()
        self.module_configs = module_config
        self.mtl_method = mtl_method
        self.use_imputation_layer = use_imputation_layer
        self.dim_s = dim_s
        self.dim_z = dim_z
        self.dim_ys = dim_ys
        self.dim_y = dim_y

        self.model = ModularHivae(
            module_config=module_config,
            dim_s=dim_s,
            dim_z=dim_z,
            dim_ys=dim_ys,
            dim_y=dim_y,
            shared_element_type=shared_element_type,
            mtl_method=mtl_method,
            use_imputation_layer=use_imputation_layer,
        )

        self.generators = nn.ModuleList(
            [
                Generator(noise_size, (8, 4), dim_z)
                for _ in range(len(module_config))
            ]
        )
        self.discriminators = nn.ModuleList(
            [Discriminator(dim_z, (8, 4), 1) for _ in range(len(module_config))]
        )

        self.device = torch.device("cpu")
        self.one = nn.Parameter(
            torch.FloatTensor([1.0])[0], requires_grad=False
        )
        self.mone = nn.Parameter(
            torch.FloatTensor([-1.0])[0], requires_grad=False
        )
        self.noise_size = noise_size

    def _train_gan_generator_step(
        self,
        data: Tuple[Tensor],
        mask: Tuple[Tensor],
        optimizer: Tuple[Optimizer],
    ) -> Tuple[Tensor]:
        """Performs a training step for the GAN generators.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.
            optimizer (Tuple[Optimizer]): Optimizers for the generators.

        Returns:
            Tuple[Tensor]: Generator losses.
        """
        errG_list = []
        for data, mask, generator, discriminator, opt in zip(
            data, mask, self.generators, self.discriminators, optimizer
        ):
            opt.zero_grad()
            noise = torch.randn(data.shape[0], self.noise_size)
            fake_hidden = generator(noise)
            errG = discriminator(fake_hidden)
            self.fabric.backward(errG, self.one)
            opt.step()
            errG_list.append(errG.detach())
        return tuple(errG_list)

    def _train_gan_discriminator_step(
        self,
        data: Tuple[Tensor],
        mask: Tuple[Tensor],
        optimizer: Tuple[Optimizer],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Performs a training step for the GAN discriminators.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.
            optimizer (Tuple[Optimizer]): Optimizers for the discriminators.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Discriminator loss, real loss, and fake loss.
        """
        errD_list = []
        errD_real_list = []
        errD_fake_list = []
        for data, mask, generator, discriminator, opt, module in zip(
            data,
            mask,
            self.generators,
            self.discriminators,
            optimizer,
            self.model.module_models.values(),
        ):
            opt.zero_grad()
            _, _, encoder_output = module.encoder_part(data, mask)
            real_hidden = encoder_output.samples_z
            errD_real = discriminator(real_hidden.detach())
            self.fabric.backward(errD_real, self.one)

            noise = torch.randn((data.shape[0], self.noise_size))
            fake_hidden = generator(noise)
            errD_fake = discriminator(fake_hidden.detach())
            self.fabric.backward(errD_fake, self.mone)

            gradient_penaltiy = self._calc_gradient_penalty(
                real_hidden, fake_hidden, discriminator
            )
            self.fabric.backward(gradient_penaltiy)
            opt.step()
            errD = -(errD_real - errD_fake)
            errD_list.append(errD.detach())
            errD_real_list.append(errD_real.detach())
            errD_fake_list.append(errD_fake.detach())
        return tuple(errD_list), tuple(errD_real_list), tuple(errD_fake_list)

    def _train_model_step(
        self,
        data: Tuple[Tensor],
        mask: Tuple[Tensor],
        optimizer: Tuple[Optimizer],
    ) -> ModularHivaeOutput:
        """Performs a training step for the Modular HIVAE model.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.
            optimizer (Tuple[Optimizer]): Optimizers for the model.

        Returns:
            ModularHivaeOutput: Model output.
        """
        for opt in optimizer:
            if opt is None:
                continue
            opt.zero_grad()
        output = self.forward(data, mask)
        self.fabric.backward(output.loss)
        # if output.loss > 1e10:
        #     raise optuna.TrialPruned("Unsuited hyperparameters. High loss")
        for opt in optimizer:
            if opt is None:
                continue
            opt.step()
        return output

    def _train_model_from_discriminator_step(
        self,
        data: Tuple[Tensor],
        mask: Tuple[Tensor],
        optimizer: Tuple[Optimizer],
    ) -> Tuple[Tensor]:
        """Performs a training step for the model from the discriminator's perspective.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.
            optimizer (Tuple[Optimizer]): Optimizers for the model.

        Returns:
            Tuple[Tensor]: Discriminator real losses.
        """
        errD_real_list = []
        for data, mask, discriminator, opt, module in zip(
            data,
            mask,
            self.discriminators,
            optimizer,
            self.model.module_models.values(),
        ):
            opt.zero_grad()
            _, _, encoder_output = module.encoder_part(data, mask)
            real_hidden = encoder_output.samples_z
            errD_real = discriminator(real_hidden)
            self.fabric.backward(errD_real, self.mone)
            opt.step()
            errD_real_list.append(errD_real.detach())
        return tuple(errD_real_list)

    def _get_loss_from_output(self, output: ModularHivaeOutput) -> float:
        """Extracts the loss from the model output.

        Args:
            output (ModularHivaeOutput): Model output.

        Returns:
            float: Loss value.
        """
        return output.avg_loss

    def _get_number_of_items(self, mask: Tuple[Tensor]) -> int:
        """Gets the number of items in the mask.

        Args:
            mask (Tuple[Tensor]): Data mask.

        Returns:
            int: Number of items.
        """
        return sum([m.sum() for m in mask])

    @cached_property
    def colnames(self, module_name: str) -> Tuple[str, ...]:
        """Gets the column names for a specific module.

        Args:
            module_name (str): Name of the module.

        Returns:
            Tuple[str, ...]: Column names.
        """
        return self.model.module_models[module_name].colnames

    @property
    def decoding(self) -> bool:
        """bool: Flag indicating whether the model is in decoding mode."""
        return self.model.decoding

    @decoding.setter
    def decoding(self, value: bool) -> None:
        """Sets the decoding flag.

        Args:
            value (bool): Decoding flag.
        """
        self.model.decoding = value

    @property
    def tau(self) -> float:
        """float: Temperature parameter for Gumbel softmax."""
        return self.model.tau

    @tau.setter
    def tau(self, value: float) -> None:
        """Sets the temperature parameter for Gumbel softmax.

        Args:
            value (float): Temperature value.
        """
        self.model.tau = value

    def forward(
        self, data: Tuple[Tensor], mask: Tuple[Tensor]
    ) -> ModularHivaeOutput:
        """Forward pass of the model.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.

        Returns:
            ModularHivaeOutput: Model output.
        """
        return self.model.forward(data, mask)

    def decode(self, encoding: ModularHivaeEncoding) -> Tuple[Tensor, ...]:
        """Decodes the given encoding.

        Args:
            encoding (ModularHivaeEncoding): Encoding to decode.

        Returns:
            Tuple[Tensor, ...]: Decoded data.
        """
        return self.model.decode(encoding)

    def _training_step(
        self,
        data: Tuple[Tensor],
        mask: Tuple[Tensor],
        optimizer: Tuple[Optimizer],
    ) -> float:
        """Training step is not needed for this class."""
        raise Exception(f"Method not needed for {self.__class__.__name__}")

    def _validation_step(
        self, data: Tuple[Tensor], mask: Tuple[Tensor]
    ) -> float:
        """Performs a validation step.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.

        Returns:
            float: Validation loss.
        """
        return self.model._validation_step(data, mask)

    def _test_step(self, data: Tuple[Tensor], mask: Tuple[Tensor]) -> float:
        """Performs a test step.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.

        Returns:
            float: Test loss.
        """
        return self.model._test_step(data, mask)

    def _predict_step(
        self, data: Tuple[Tensor], mask: Tuple[Tensor]
    ) -> ModularHivaeOutput:
        """Performs a prediction step.

        Args:
            data (Tuple[Tensor]): Input data.
            mask (Tuple[Tensor]): Data mask.

        Returns:
            ModularHivaeOutput: Prediction output.
        """
        return self.model._predict_step(data, mask)
