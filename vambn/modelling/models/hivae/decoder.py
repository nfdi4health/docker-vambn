import logging
from functools import cached_property
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from vambn.data.dataclasses import VarTypes
from vambn.modelling.models.hivae.heads import (
    HEAD_DICT,
    CatHead,
    CountHead,
    PosHead,
    RealHead,
)
from vambn.modelling.models.hivae.normalization import (
    Normalization,
    NormalizationParameters,
)
from vambn.modelling.models.hivae.outputs import EncoderOutput, HivaeOutput
from vambn.modelling.models.layers import ModifiedLinear
from vambn.modelling.mtl import moo
from vambn.modelling.mtl.parameters import MtlMethodParams

logger = logging.getLogger()


class Decoder(nn.Module):
    """HIVAE Decoder class.

    Args:
        variable_types (VarTypes): List of VariableType objects. See VarTypes in
            data/dataclasses.py.
        s_dim (int): Dimension of s space.
        z_dim (int): Dimension of z space.
        y_dim (int): Dimension of y space.
        mtl_method (Tuple[str, ...], optional): List of methods to use for multi-task learning.
            Assessed possibilities are combinations of "identity", "gradnorm", "graddrop".
            Further implementations and details can be found in the mtl.py file. Defaults to ("identity",).
        decoder_shared (nn.Module, optional): Shared decoder module. Defaults to nn.Identity().

    Attributes:
        prior_s_val (nn.Parameter): Prior distribution for s values.
        prior_loc_z (ModifiedLinear): Linear layer for z prior distribution.
        s_dim (int): Dimension of s space.
        z_dim (int): Dimension of z space.
        y_dim (int): Dimension of y space.
        variable_types (VarTypes): List of variable types.
        decoder_shared (nn.Module): Shared decoder module.
        internal_layer_norm (nn.Module): Layer normalization module.
        heads (nn.ModuleList): List of head modules for each variable type.
        mtl_methods (Tuple[str, ...]): Methods for multi-task learning.
        _mtl_module_y (nn.Module): Multi-task learning module for y.
        _mtl_module_s (nn.Module): Multi-task learning module for s.
        moo_block (moo.MultiMOOForLoop): Multi-task learning block.
        _decoding (bool): Decoding flag.
    """

    def __init__(
        self,
        variable_types: VarTypes,
        s_dim: int,
        z_dim: int,
        y_dim: int,
        mtl_method: Tuple[str, ...] = ("identity",),
        decoder_shared: nn.Module = nn.Identity(),
    ):
        """
        Initialize the HIVAE Decoder.

        Args:
            variable_types (List[VariableType]): List of VariableType objects.
            s_dim (int): Dimension of s space.
            z_dim (int): Dimension of z space.
            y_dim (int): Dimension of y space.
            mtl_method (Tuple[str]): List of methods to use for multi-task learning.
        """
        super().__init__()

        # prior distributions
        self.prior_s_val = nn.Parameter(
            torch.ones(s_dim) / s_dim, requires_grad=False
        )
        self.prior_loc_z = ModifiedLinear(s_dim, z_dim, bias=True)

        self.s_dim = s_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.variable_types = variable_types

        self.decoder_shared = decoder_shared
        self.internal_layer_norm = nn.Identity()  # nn.LayerNorm(self.z_dim)
        self.heads: List[
            RealHead | PosHead | CountHead | CatHead
        ] | nn.ModuleList = nn.ModuleList(
            [
                HEAD_DICT[variable_types[i].data_type](
                    variable_types[i], s_dim, z_dim, y_dim
                )
                for i in range(len(variable_types))
            ]
        )

        self.mtl_methods = mtl_method
        self._mtl_module_y: nn.Module = moo.setup_moo(
            [MtlMethodParams(x) for x in mtl_method],
            num_tasks=len(self.heads),
        )
        self._mtl_module_s: nn.Module = moo.setup_moo(
            [MtlMethodParams(x) for x in mtl_method],
            num_tasks=len(self.heads),
        )
        self.moo_block = moo.MultiMOOForLoop(
            len(self.heads),
            moo_methods=(self._mtl_module_y, self._mtl_module_s),
        )

        self._decoding = True

    @property
    def decoding(self) -> bool:
        """bool: Flag indicating whether the model is in decoding mode."""
        return self._decoding

    @decoding.setter
    def decoding(self, value: bool) -> None:
        """Sets the decoding flag.

        Args:
            value (bool): Decoding flag.
        """
        self._decoding = value

    @cached_property
    def colnames(self) -> List[str]:
        """Gets the column names of the data.

        Returns:
            List[str]: List of column names.
        """
        return [var.name for var in self.variable_types]

    @property
    def prior_s(self) -> torch.distributions.OneHotCategorical:
        """Gets the prior distribution for s.

        Returns:
            torch.distributions.OneHotCategorical: Prior distribution for s.
        """
        return torch.distributions.OneHotCategorical(
            probs=self.prior_s_val, validate_args=False
        )

    def prior_z(self, loc: torch.Tensor) -> torch.distributions.Normal:
        """Gets the prior distribution for z.

        Args:
            loc (torch.Tensor): Location parameter for z.

        Returns:
            torch.distributions.Normal: Prior distribution for z.
        """
        return torch.distributions.Normal(loc, torch.ones_like(loc))

    def kl_s(self, encoder_output: EncoderOutput) -> torch.Tensor:
        """Computes the KL divergence for s.

        Args:
            encoder_output (EncoderOutput): Encoder output.

        Returns:
            torch.Tensor: KL divergence for s.
        """
        return torch.distributions.kl.kl_divergence(
            torch.distributions.OneHotCategorical(
                logits=encoder_output.logits_s, validate_args=False
            ),
            self.prior_s,
        )

    def kl_z(
        self,
        mean_qz: torch.Tensor,
        std_qz: torch.Tensor,
        mean_pz: torch.Tensor,
        std_pz: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the KL divergence for z.

        Args:
            mean_qz (torch.Tensor): Mean of the posterior distribution.
            std_qz (torch.Tensor): Standard deviation of the posterior distribution.
            mean_pz (torch.Tensor): Mean of the prior distribution.
            std_pz (torch.Tensor): Standard deviation of the prior distribution.

        Returns:
            torch.Tensor: KL divergence for z.
        """
        return torch.distributions.kl.kl_divergence(
            torch.distributions.Normal(mean_qz, std_qz),
            torch.distributions.Normal(mean_pz, std_pz),
        ).sum(dim=-1)

    def _cat_samples(self, samples: List[torch.Tensor]) -> torch.Tensor:
        """Concatenates samples.

        Args:
            samples (List[torch.Tensor]): List of samples.

        Returns:
            torch.Tensor: Concatenated samples.

        Raises:
            ValueError: If no samples were drawn or if samples have an incorrect shape.
        """
        if len(samples) == 0 or all(x is None for x in samples):
            raise ValueError("No samples were drawn")
        else:
            sample_stack = torch.stack(samples, dim=1)
            if sample_stack.ndim == 2:
                return sample_stack
            elif sample_stack.ndim == 3 and sample_stack.shape[2] == 1:
                return sample_stack.squeeze(2)
            else:
                raise ValueError(
                    "Samples should be of shape (batch, features) or (batch, features, 1)"
                )

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        encoder_output: EncoderOutput,
        normalization_parameters: NormalizationParameters,
    ) -> HivaeOutput:
        """Forward pass of the decoder.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Mask for the data.
            encoder_output (EncoderOutput): Output from the encoder.
            normalization_parameters (NormalizationParameters): Parameters for normalization.

        Returns:
            HivaeOutput: Output from the decoder.
        """
        samples_z = encoder_output.samples_z
        decoder_representation = encoder_output.decoder_representation
        samples_s = encoder_output.samples_s

        # Obtaining the parameters for the decoder
        # decoder representation of shape batch x dim_z
        interim_decoder_representation = self.decoder_shared(
            decoder_representation
        )  # identity
        interim_decoder_representation = self.internal_layer_norm(
            interim_decoder_representation
        )  # identity
        if self.training:
            moo_out_s, moo_out_z = self.moo_block(
                samples_s, interim_decoder_representation
            )
            x_params = []
            for head, s_i, k_i in zip(self.heads, moo_out_s, moo_out_z):
                x_params.append(head(k_i, s_i))
            x_params = tuple(x_params)
        else:
            x_params = tuple(
                [
                    head(interim_decoder_representation, samples_s)
                    for head in self.heads
                ]
            )

        x_params = Normalization.denormalize_params(
            x_params, self.variable_types, normalization_parameters
        )

        # Compute the likelihood and kl divergences
        log_probs: List[torch.Tensor] = [torch.Tensor([-1])] * len(
            self.variable_types
        )
        samples: List[torch.Tensor] = [torch.Tensor([-1])] * len(
            self.variable_types
        )
        for i, (x_i, m_i, head_i, params_i) in enumerate(
            zip(data.T, mask.T, self.heads, x_params)
        ):
            head_i.dist(params_i)
            log_probs[i] = head_i.log_prob(x_i) * m_i

            if not self.training and self.decoding:
                # draw samples for evaluation and decoding
                samples[i] = head_i.sample()
            elif not self.training and not self.decoding:
                # draw samples for evaluation
                samples[i] = head_i.sample()

        # Stack the log likelihoods
        log_prob = torch.stack(log_probs, dim=1)  # batch, features
        log_prob = log_prob.sum(dim=1)  # / (mask.sum(dim=1) + 1e-6)  # batch
        cat_samples = self._cat_samples(samples)  # batch, features

        # Compute the KL divergences
        # KL divergence for s
        # samples_s of shape (batch, dim_s)
        kl_s = self.kl_s(encoder_output)  # shape (batch,)

        # KL divergence for z
        pz_loc = self.prior_loc_z(encoder_output.samples_s)  # batch, dim_z
        mean_pz, std_pz = pz_loc, torch.ones_like(pz_loc)
        mean_qz, std_qz = encoder_output.mean_z, encoder_output.scale_z
        kl_z = self.kl_z(mean_qz, std_qz, mean_pz, std_pz)  # shape (batch,)

        loss = -torch.sum(log_prob - kl_s - kl_z) / (
            (torch.sum(mask) / mask.shape[-1]) + 1e-6
        )
        print(f"Loss: {loss}, num_samples: {mask.sum()}")
        return HivaeOutput(
            enc_s=samples_s,
            enc_z=samples_z,
            samples=cat_samples,
            loss=loss,
        )

    @torch.no_grad()
    def decode(
        self,
        encoding_z: torch.Tensor,
        encoding_s: torch.Tensor,
        normalization_params: NormalizationParameters,
    ) -> torch.Tensor:
        """Decoding logic for the decoder.

        Args:
            encoding_z (torch.Tensor): Encoding for z.
            encoding_s (torch.Tensor): Encoding for s.
            normalization_params (NormalizationParameters): Parameters for normalization.

        Returns:
            torch.Tensor: Decoded samples.

        Raises:
            ValueError: If no samples were drawn.
        """
        # Implement the decoding logic here
        assert not self.training, "Model should be in eval mode"
        # shared_y of shape (batch, dim_y)
        decoder_interim_representation = self.decoder_shared(
            encoding_z
        )  # identity
        decoder_interim_representation = self.internal_layer_norm(
            decoder_interim_representation
        )  # identity

        if encoding_s.shape[1] != self.s_dim:
            encoding_s = F.one_hot(
                encoding_s.squeeze(1).long(), num_classes=self.s_dim
            ).float()  # batch, dim_s

        x_params = tuple(
            [
                head(decoder_interim_representation, encoding_s)
                for head in self.heads
            ]
        )
        x_params = Normalization.denormalize_params(
            x_params, self.variable_types, normalization_params
        )
        _ = [
            head.dist(
                params=params,
            )
            for i, (head, params) in enumerate(zip(self.heads, x_params))
        ]
        if self.decoding:
            samples = [head.sample() for head in self.heads]
        else:
            samples = [head.sample() for head in self.heads]
        cat_samples = self._cat_samples(samples)
        if cat_samples is not None:
            return cat_samples
        else:
            raise ValueError("No samples were drawn")


class LstmDecoder(Decoder):
    """LSTM-based HIVAE Decoder class.

    Args:
        variable_types (VarTypes): List of VariableType objects. See VarTypes in
            data/dataclasses.py.
        s_dim (int): Dimension of s space.
        z_dim (int): Dimension of z space.
        y_dim (int): Dimension of y space.
        num_timepoints (int): Number of timepoints.
        n_layers (int, optional): Number of LSTM layers. Defaults to 1.
        mtl_method (Tuple[str, ...], optional): List of methods to use for multi-task learning. 
            Assessed possibilities are combinations of "identity", "gradnorm", "graddrop".  
            Further implementations and details can be found in the mtl.py file. Defaults to ("identity",).
        decoder_shared (nn.Module, optional): Shared decoder module. Defaults to nn.Identity().

    Attributes:
        num_timepoints (int): Number of timepoints.
        n_layers (int): Number of LSTM layers.
        lstm_decoder (nn.LSTM): LSTM decoder module.
        fc (nn.Linear): Fully connected layer.
    """

    def __init__(
        self,
        variable_types: VarTypes,
        s_dim: int,
        z_dim: int,
        y_dim: int,
        num_timepoints: int,
        n_layers: int = 1,
        mtl_method: Tuple[str, ...] = ("identity",),
        decoder_shared: nn.Module = nn.Identity(),
    ) -> None:
        """HIVAE Decoder

        Args:
            type_array (np.ndarray): Array containing the data type information (type, class, ndim)
            s_dim (int): Dimension of s space
            z_dim (int): Dimension of z space
            y_dim (int): Dimension of y space
            num_timepoints (int): Number of num_timepoints.
            mtl_method (Tuple[str], optional): List of methods to use for multi-task learning. Defaults to
                ["gradnorm", "pcgrad"].
        """
        super().__init__(
            variable_types=variable_types,
            s_dim=s_dim,
            z_dim=z_dim,
            y_dim=y_dim,
            mtl_method=mtl_method,
            decoder_shared=decoder_shared,
        )
        self.num_timepoints = num_timepoints
        self.n_layers = n_layers
        self.lstm_decoder = nn.LSTM(
            input_size=z_dim,
            hidden_size=z_dim,
            num_layers=self.n_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(z_dim, z_dim)

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        encoder_output: EncoderOutput,
        normalization_parameters: NormalizationParameters,
    ) -> HivaeOutput:
        """Forward pass of the LSTM decoder.

        Args:
            data (torch.Tensor): Input data.
            mask (torch.Tensor): Mask for the data.
            encoder_output (EncoderOutput): Output from the encoder.
            normalization_parameters (NormalizationParameters): Parameters for normalization.

        Returns:
            HivaeOutput: Output from the decoder.
        """
        samples_z = encoder_output.samples_z
        decoder_representation = encoder_output.decoder_representation
        samples_s = encoder_output.samples_s

        # Obtaining the parameters for the decoder
        # decoder representation of shape batch x dim_z
        decoder_interim_representation = self.decoder_shared(
            decoder_representation
        )  # identity
        decoder_interim_representation = self.internal_layer_norm(
            decoder_interim_representation
        )  # identity

        h0 = self.fc(decoder_interim_representation).repeat(self.n_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        y_repeated = decoder_interim_representation.unsqueeze(1).repeat(
            1, self.num_timepoints, 1
        )
        decoder_interim_representation, _ = self.lstm_decoder(
            y_repeated, (h0, c0)
        )

        log_probs = [None] * self.num_timepoints
        samples = [None] * self.num_timepoints
        for t in range(self.num_timepoints):
            sub_data = data[:, t, :]
            sub_mask = mask[:, t, :]
            if self.training:
                x_params = tuple(
                    [
                        head(y_i, s_i)
                        for head, s_i, y_i in zip(  # type: ignore
                            self.heads,
                            *self.moo_block.forward(
                                samples_s,
                                decoder_interim_representation[:, t, :],
                            ),
                        )
                    ]
                )
            else:
                x_params = tuple(
                    [
                        head(decoder_interim_representation[:, t, :], samples_s)
                        for head in self.heads
                    ]
                )

            x_params = Normalization.denormalize_params(
                x_params, self.variable_types, normalization_parameters[t]
            )

            log_probs[t] = torch.stack(
                [
                    head_i.log_prob(
                        params=params_i,
                        data=d_i,
                    )
                    * m_i
                    for i, (head_i, params_i, d_i, m_i) in enumerate(
                        zip(self.heads, x_params, sub_data.T, sub_mask.T)
                    )
                ],
                dim=1,
            ).sum(-1)  # batch
            assert isinstance(
                log_probs[t], torch.Tensor
            ), f"Log probs: {log_probs[t]}"
            assert log_probs[t].shape == (
                encoder_output.samples_s.shape[0],
            ), f"Log probs shape: {log_probs[t].shape}"

            if not self.training and self.decoding:
                samples[t] = self._cat_samples(
                    [
                        head.sample()
                        for head, params in zip(self.heads, x_params)
                    ]
                )
            elif not self.training and not self.decoding:
                samples[t] = self._cat_samples(
                    [
                        head.sample()
                        for head, params in zip(self.heads, x_params)
                    ]
                )

        log_prob = torch.stack(log_probs, dim=1)  # batch, timepoints
        log_prob = log_prob.sum(
            dim=1
        )  # / (mask.sum(dim=(1, 2)) + 1e-9)  # batch
        if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
            raise ValueError("Log likelihood is nan or inf")
        samples = (
            torch.stack(samples, dim=1) if samples[0] is not None else None
        )

        # Compute the KL divergences
        # # KL divergence for s
        kl_s = self.kl_s(encoder_output)
        if torch.isnan(kl_s).any() or torch.isinf(kl_s).any():
            raise ValueError("KL divergence for s is nan or inf")

        # KL divergence for z
        pz_loc = self.prior_loc_z(encoder_output.samples_s)
        mean_pz, std_pz = pz_loc, torch.ones_like(pz_loc)
        mean_qz, std_qz = encoder_output.mean_z, encoder_output.scale_z
        kl_z = self.kl_z(mean_qz, std_qz, mean_pz, std_pz)

        assert kl_z.shape == (encoder_output.samples_s.shape[0],)
        if torch.isnan(kl_z).any() or torch.isinf(kl_z).any():
            raise ValueError("KL divergence for z is nan or inf")

        # Compute the loss
        loss = -torch.sum(log_prob - kl_s - kl_z) / (
            torch.sum(mask) / mask.shape[-1]
        )

        return HivaeOutput(
            enc_s=encoder_output.samples_s,
            enc_z=samples_z,
            samples=samples,
            loss=loss,
        )

    @torch.no_grad()
    def decode(
        self,
        encoding_z: torch.Tensor,
        encoding_s: torch.Tensor,
        normalization_params: NormalizationParameters,
    ):
        """Decoding logic for 3D data.

        Args:
            encoding_z (torch.Tensor): Encoding for z.
            encoding_s (torch.Tensor): Encoding for s.
            normalization_params (NormalizationParameters): Parameters for normalization.

        Returns:
            torch.Tensor: Decoded samples.
        """
        assert not self.training, "Model should be in eval mode"

        # shared_y of shape (batch, dim_y)
        decoder_interim_representation = self.decoder_shared(
            encoding_z
        )  # identity
        decoder_interim_representation = self.internal_layer_norm(
            decoder_interim_representation
        )  # identity

        decoder_interim_representation = (
            decoder_interim_representation.unsqueeze(1).repeat(
                1, self.num_timepoints, 1
            )
        )
        decoder_interim_representation, _ = self.lstm_decoder(
            decoder_interim_representation
        )

        if encoding_s.shape[1] != self.s_dim:
            encoding_s = F.one_hot(
                encoding_s.squeeze(1).long(), num_classes=self.s_dim
            ).float()

        samples = [None] * self.num_timepoints
        for t in range(self.num_timepoints):
            x_params = tuple(
                [
                    head(decoder_interim_representation[:, t, :], encoding_s)
                    for head in self.heads
                ]
            )
            x_params = Normalization.denormalize_params(
                x_params, self.variable_types, normalization_params[t]
            )
            [head.dist(params) for head, params in zip(self.heads, x_params)]
            if self.decoding:
                samples[t] = self._cat_samples(
                    [head.sample() for head in self.heads]
                )
            else:
                samples[t] = self._cat_samples(
                    [
                        head.sample()
                        for head, params in zip(self.heads, x_params)
                    ]
                )
        time_dim_samples = torch.stack(samples, dim=1)
        return time_dim_samples
