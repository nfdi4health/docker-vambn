import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar, Union

import torch
import torch.distributions as dists
import torch.nn as nn
from torch import Tensor

from vambn.data.dataclasses import VariableType
from vambn.modelling.distributions.categorical import ReparameterizedCategorical
from vambn.modelling.distributions.parameters import (
    CategoricalParameters,
    LogNormalParameters,
    NormalParameters,
    PoissonParameters,
)
from vambn.modelling.distributions.truncated_normal import TruncatedNormal
from vambn.modelling.models import ModifiedLinear

logger = logging.getLogger()

ParameterType = TypeVar("ParameterType")
DistributionType = TypeVar("DistributionType")


# Model Classes
class BaseModuleHead(Generic[ParameterType, DistributionType], nn.Module, ABC):
    """Base class for different data types.

    Args:
        variable_type (VariableType): Dataclass containing the type information.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_y (int): Dimension of y space.

    Attributes:
        types (VariableType): Dataclass containing the type information.
        dim_s (int): Dimension of s space.
        dim_z (int): Dimension of z space.
        dim_y (int): Dimension of y space.
        internal_pass (ModifiedLinear): Internal pass module.

    Properties:
        num_parameters (int): Number of parameters.

    Methods:
        forward(samples_z: Tensor, samples_s: Tensor) -> ParameterType: Forward pass of the module.
        dist(params: ParameterType) -> DistributionType: Compute the distribution given the parameters.
        log_prob(data: Tensor, params: Optional[ParameterType] = None) -> Tensor: Compute the log probability of the data given the parameters.
        sample() -> Tensor: Sample from the distribution.
        rsample() -> Tensor: Sample using the reparameterization trick.
        mode() -> Tensor: Compute the mode of the distribution.

    Raises:
        RuntimeError: If the distribution is not initialized.

    """

    def __init__(
        self, variable_type: VariableType, dim_s: int, dim_z: int, dim_y: int
    ) -> None:
        """Base class for the different datatypes

        Args:
            types (VariableType): Array containing the data type information (type, class, ndim)
            dim_s (int): Dimension of s space
            dim_z (int): Dimension of z space
            dim_y (int): Dimension of y space
        """
        super().__init__()
        # General parameters
        self.types = variable_type
        self.dim_s = dim_s
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.internal_pass = ModifiedLinear(self.dim_z, self.dim_y, bias=True)

        # Specific parameters
        self._dist = None
        self._n_pars = None

    @abstractmethod
    def forward(self, samples_z: Tensor, samples_s: Tensor) -> ParameterType:
        """Forward pass of the module.

        Args:
            samples_z (Tensor): Samples from the z space.
            samples_s (Tensor): Samples from the s space.

        Returns:
            ParameterType: The output of the forward pass.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def dist(self, params: ParameterType) -> DistributionType:
        """Compute the distribution given the parameters.

        Args:
            params (ParameterType): The parameters of the distribution.

        Returns:
            DistributionType: The computed distribution.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    def log_prob(
        self, data: Tensor, params: Optional[ParameterType] = None
    ) -> Tensor:
        """Compute the log probability of the data given the parameters.

        Args:
            data (Tensor): The input data.
            params (Optional[ParameterType], optional): The parameters of the distribution. Defaults to None.

        Returns:
            Tensor: The log probability of the data.

        Raises:
            RuntimeError: If the distribution is not initialized.
        """
        if self._dist is None and params is None:
            raise RuntimeError("Distribution is not initialized.")
        elif params is not None:
            self.dist(params)

        return self._dist.log_prob(data)

    def sample(self) -> Tensor:
        """Sample from the distribution.

        Returns:
            Tensor: The sampled data.

        Raises:
            RuntimeError: If the distribution is not initialized.
        """
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")

        gen_sample = self._dist.sample()
        if gen_sample.ndim == 1:
            return gen_sample.unsqueeze(1)
        elif gen_sample.ndim == 0:
            # ensure 2 dim output
            return gen_sample.unsqueeze(0).unsqueeze(1)
        return gen_sample

    def rsample(self) -> Tensor:
        """Sample using the reparameterization trick.

        Returns:
            Tensor: The sampled data.

        Raises:
            RuntimeError: If the distribution is not initialized.
        """
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")

        gen_sample = self._dist.rsample()
        if gen_sample.ndim == 1:
            return gen_sample.unsqueeze(1)
        return gen_sample

    @property
    def mode(self) -> Tensor:
        """Compute the mode of the distribution.

        Returns:
            Tensor: The mode of the distribution.

        Raises:
            RuntimeError: If the distribution is not initialized.
        """
        if self._dist is None:
            raise RuntimeError("Distribution is not initialized.")
        res = self._dist.mode
        if res.ndim == 1:
            return res.unsqueeze(1)
        return res

    @property
    def num_parameters(self) -> int:
        """Number of parameters.

        Returns:
            int: The number of parameters.
        """
        return self._n_pars


class RealHead(BaseModuleHead[NormalParameters, dists.Normal]):
    """Class representing the RealHead module.

    This module is used for data of type real or pos.

    Args:
        variable_type (VariableType): Array containing the data type information (type, class, ndim)
        dim_s (int): Dimension of s space
        dim_z (int): Dimension of z space
        dim_y (int): Dimension of y space

    Attributes:
        loc_layer (ModifiedLinear): Linear layer for computing the location parameter
        scale_layer (ModifiedLinear): Linear layer for computing the scale parameter
        _dist_class (type): Class representing the distribution
        _n_pars (int): Number of parameters in the distribution

    """

    def __init__(
        self, variable_type: VariableType, dim_s: int, dim_z: int, dim_y: int
    ) -> None:
        """Initialize the RealHead module.

        Args:
            variable_type (VariableType): Array containing the data type information (type, class, ndim)
            dim_s (int): Dimension of s space
            dim_z (int): Dimension of z space
            dim_y (int): Dimension of y space
        """
        super().__init__(variable_type, dim_s, dim_z, dim_y)
        self.loc_layer = ModifiedLinear(self.dim_y + self.dim_s, 1, bias=False)
        self.scale_layer = ModifiedLinear(self.dim_s, 1, bias=False)

        self._parameter_class = NormalParameters
        self._dist_class = dists.Normal
        self._n_pars = 2

    def forward(self, samples_z: Tensor, samples_s: Tensor) -> NormalParameters:
        """Forward pass of the RealHead module.

        Args:
            samples_z (Tensor): Samples from the z space
            samples_s (Tensor): Samples from the s space

        Returns:
            NormalParameters: Parameters of the Normal distribution
        """
        y = self.internal_pass(samples_z)
        s_and_y = torch.cat([y, samples_s], dim=-1)
        loc = self.loc_layer(s_and_y)
        scale = nn.functional.softplus(self.scale_layer(samples_s))
        return NormalParameters(loc, scale)

    def dist(self, params: NormalParameters) -> dists.Normal:
        """Create a Normal distribution based on the given parameters.

        Args:
            params (NormalParameters): Parameters of the Normal distribution

        Returns:
            dists.Normal: Normal distribution
        """
        self._dist = self._dist_class(
            params.loc.squeeze(), params.scale.squeeze()
        )
        return self._dist


class TruncatedNormalHead(BaseModuleHead[NormalParameters, TruncatedNormal]):
    """
    Class representing the TruncatedNormalHead module.

    This module is used for data of type real or pos.

    Args:
        variable_type (VariableType): Array containing the data type information (type, class, ndim)
        dim_s (int): Dimension of s space
        dim_z (int): Dimension of z space
        dim_y (int): Dimension of y space

    Attributes:
        loc_layer (ModifiedLinear): Linear layer for computing the location parameter
        scale_layer (ModifiedLinear): Linear layer for computing the scale parameter
        _dist_class (type): Class representing the distribution
        _n_pars (int): Number of parameters in the distribution

    """

    def __init__(
        self, variable_type: VariableType, dim_s: int, dim_z: int, dim_y: int
    ) -> None:
        """
        Initializes a TruncatedNormalHead object.

        Args:
            variable_type (VariableType): The type of variable.
            dim_s (int): The dimensionality of s.
            dim_z (int): The dimensionality of z.
            dim_y (int): The dimensionality of y.
        """
        super().__init__(variable_type, dim_s, dim_z, dim_y)
        self.loc_layer = ModifiedLinear(self.dim_y + self.dim_s, 1, bias=False)
        self.scale_layer = ModifiedLinear(self.dim_s, 1, bias=False)

        self._dist_class = TruncatedNormal
        self._n_pars = 2

    def forward(self, samples_z: Tensor, samples_s: Tensor) -> NormalParameters:
        """
        Performs a forward pass through the TruncatedNormalHead.

        Args:
            samples_z (Tensor): The z samples.
            samples_s (Tensor): The s samples.

        Returns:
            NormalParameters: The output of the forward pass.
        """
        y = self.internal_pass(samples_z)
        s_and_y = torch.cat([y, samples_s], dim=-1)
        loc = self.loc_layer(s_and_y)
        scale = nn.functional.softplus(self.scale_layer(samples_s))
        return NormalParameters(loc, scale)

    def dist(self, params: NormalParameters) -> TruncatedNormal:
        """
        Creates a TruncatedNormal distribution based on the given parameters.

        Args:
            params (NormalParameters): The parameters of the distribution.

        Returns:
            TruncatedNormal: The created TruncatedNormal distribution.
        """
        self._dist = self._dist_class(
            params.loc.squeeze(), params.scale.squeeze(), low=torch.tensor(0.0)
        )
        return self._dist

    def log_prob(
        self, data: Tensor, params: NormalParameters | None = None
    ) -> Tensor:
        """
        Computes the log probability of the data given the parameters.

        Args:
            data (Tensor): The input data.
            params (NormalParameters | None): The parameters of the distribution.

        Returns:
            Tensor: The log probability of the data.
        """
        return super().log_prob(data.clamp(min=1e-3), params)


class PosHead(BaseModuleHead[LogNormalParameters, dists.LogNormal]):
    """
    Head module for the LogNormal (pos) distribution

    Attributes:
        variable_type (VariableType): The type of variable.
        dim_s (int): The dimension of s.
        dim_z (int): The dimension of z.
        dim_y (int): The dimension of y.
        loc_layer (ModifiedLinear): The linear layer for computing the location parameter.
        scale_layer (ModifiedLinear): The linear layer for computing the scale parameter.
        _dist_class (dists.LogNormal): The class representing the LogNormal distribution.
        _n_pars (int): The number of parameters in the distribution.

    """

    def __init__(
        self, variable_type: VariableType, dim_s: int, dim_z: int, dim_y: int
    ) -> None:
        """
        Initializes the PosHead class.

        Args:
            variable_type (VariableType): The type of variable.
            dim_s (int): The dimension of s.
            dim_z (int): The dimension of z.
            dim_y (int): The dimension of y.
        """
        super().__init__(variable_type, dim_s, dim_z, dim_y)
        self.loc_layer = ModifiedLinear(self.dim_y + self.dim_s, 1, bias=False)
        self.scale_layer = ModifiedLinear(self.dim_s, 1, bias=False)

        self._dist_class = dists.LogNormal
        self._n_pars = 2

    def forward(
        self, samples_z: Tensor, samples_s: Tensor
    ) -> LogNormalParameters:
        """
        Performs the forward pass of the PosHead.

        Args:
            samples_z (Tensor): The z samples.
            samples_s (Tensor): The s samples.

        Returns:
            LogNormalParameters: The output of the forward pass.
        """
        y = self.internal_pass(samples_z)
        s_and_y = torch.cat([y, samples_s], dim=-1)

        loc = self.loc_layer(s_and_y)
        scale = nn.functional.softplus(self.scale_layer(samples_s))
        return LogNormalParameters(loc, scale)

    def dist(self, params: LogNormalParameters) -> dists.LogNormal:
        """
        Creates a LogNormal distribution based on the given parameters.

        Args:
            params (LogNormalParameters): The parameters of the distribution.

        Returns:
            dists.LogNormal: The LogNormal distribution.
        """
        self._dist = self._dist_class(
            params.loc.squeeze(), params.scale.squeeze()
        )
        return self._dist

    def log_prob(
        self, data: Tensor, params: LogNormalParameters | None = None
    ) -> Tensor:
        """
        Computes the log probability of the data given the parameters.

        Args:
            data (Tensor): The input data.
            params (LogNormalParameters | None): The parameters of the distribution.

        Returns:
            Tensor: The log probability of the data.
        """
        return super().log_prob(data.clamp(min=1e-3), params)


class CountHead(BaseModuleHead[PoissonParameters, dists.Poisson]):
    """Head module for the Poisson distribution (Count data).

    Args:
        variable_type (VariableType): Array containing the data type information (type, class, ndim)
        dim_s (int): Dimension of s space
        dim_z (int): Dimension of z space
        dim_y (int): Dimension of y space

    Attributes:
        lambda_layer (ModifiedLinear): Linear layer for computing the rate parameter
        _dist_class (dists.Poisson): Class for representing the Poisson distribution

    """

    def __init__(
        self, variable_type: VariableType, dim_s: int, dim_z: int, dim_y: int
    ) -> None:
        """Initializes the CountHead class.

        Args:
            variable_type (VariableType): Array containing the data type information (type, class, ndim)
            dim_s (int): Dimension of s space
            dim_z (int): Dimension of z space
            dim_y (int): Dimension of y space
        """
        super().__init__(variable_type, dim_s, dim_z, dim_y)
        self.lambda_layer = ModifiedLinear(
            self.dim_y + self.dim_s, 1, bias=False
        )
        self._dist_class = dists.Poisson

    def forward(
        self, samples_z: Tensor, samples_s: Tensor
    ) -> PoissonParameters:
        """Performs the forward pass of the CountHead.

        Args:
            samples_z (Tensor): Samples from the z space
            samples_s (Tensor): Samples from the s space

        Returns:
            PoissonParameters: The Poisson parameter
        """
        y = self.internal_pass(samples_z)
        s_and_y = torch.cat([y, samples_s], dim=-1)
        rate = self.lambda_layer(s_and_y)
        return PoissonParameters(rate)

    def dist(self, params: PoissonParameters) -> dists.Poisson:
        """Creates a Poisson distribution from the given parameters.

        Args:
            params (PoissonParameters): The Poisson parameters

        Returns:
            dists.Poisson: The Poisson distribution
        """
        self._dist = self._dist_class(params.rate.squeeze())
        return self._dist


class CatHead(
    BaseModuleHead[CategoricalParameters, ReparameterizedCategorical]
):
    """Class representing the categorical head of a model.

    Attributes:
        variable_type (VariableType): Array containing the data type information (type, class, ndim)
        dim_s (int): Dimension of s space
        dim_z (int): Dimension of z space
        dim_y (int): Dimension of y space
        logit_layer (ModifiedLinear): Linear layer for computing logits
        _dist_class (ReparameterizedCategorical): Class for representing reparameterized categorical distribution
    """

    # @typechecked
    def __init__(
        self, variable_type: VariableType, dim_s: int, dim_z: int, dim_y: int
    ) -> None:
        """Initialize the CatHead class.

        Args:
            variable_type (VariableType): Array containing the data type information (type, class, ndim)
            dim_s (int): Dimension of s space
            dim_z (int): Dimension of z space
            dim_y (int): Dimension of y space
        """
        super().__init__(variable_type, dim_s, dim_z, dim_y)
        self.logit_layer = ModifiedLinear(
            self.dim_y + self.dim_s, variable_type.n_parameters, bias=False
        )
        self._dist_class = ReparameterizedCategorical

    def forward(
        self, samples_z: Tensor, samples_s: Tensor
    ) -> CategoricalParameters:
        """Forward pass of the CatHead.

        Args:
            samples_z (Tensor): Samples from the z space
            samples_s (Tensor): Samples from the s space

        Returns:
            CategoricalParameters: Categorical parameters
        """
        y = self.internal_pass(samples_z)
        s_and_y = torch.cat([y, samples_s], dim=-1)
        logits = self.logit_layer(s_and_y)
        return CategoricalParameters(logits)

    def dist(self, params: CategoricalParameters) -> ReparameterizedCategorical:
        """Compute the reparameterized categorical distribution.

        Args:
            params (CategoricalParameters): Categorical parameters

        Returns:
            ReparameterizedCategorical: Reparameterized categorical distribution
        """
        self._dist = self._dist_class(logits=params.logits)
        return self._dist


# Dictionary with the class mapping
HEAD_DICT = {
    "real": RealHead,
    "truncate_norm": TruncatedNormalHead,
    "pos": PosHead,
    "cat": CatHead,
    "count": CountHead,
    "categorical": CatHead,
}
HEADS = Union[RealHead, PosHead, CatHead, CountHead, TruncatedNormalHead]
