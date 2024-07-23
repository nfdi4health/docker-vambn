import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    List,
    Optional,
    TypeVar,
)

import torch
from torch import Tensor
from torch.distributions.utils import logits_to_probs

logger = logging.getLogger()


################################################################################
# Generic definitions
################################################################################

@dataclass
class Parameters(ABC):
    """Dataclass for parameter output"""

    name: str = field(kw_only=True, default="")

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass


################################################################################
# Parameter classes
################################################################################


@dataclass
class NormalParameters(Parameters):
    """Dataclass for real output"""

    loc: Tensor
    scale: Tensor
    name: str = field(kw_only=True, default="real")

    @property
    def device(self) -> torch.device:
        return self.loc.device

    def __str__(self) -> str:
        return f"RealOutput: Avg. Mean {self.loc.mean()}; Avg. Std {self.scale.mean()}); shape {self.loc.shape}"



@dataclass
class LogNormalParameters(Parameters):
    loc: Tensor
    scale: Tensor
    name: str = field(kw_only=True, default="pos")

    @property
    def device(self) -> torch.device:
        return self.loc.device

    def __str__(self) -> str:
        return f"PosOutput: Avg. Mean {self.loc.mean()}; Avg. Std {self.scale.mean()}; shape {self.loc.shape}, Std max {self.scale.max()}, loc max {self.loc.max()}"


@dataclass
class PoissonParameters(Parameters):
    """Dataclass for count output"""

    rate: Tensor
    name: str = field(kw_only=True, default="count")

    @property
    def device(self) -> torch.device:
        return self.rate.device

    def __str__(self) -> str:
        return f"CountOutput: Avg. Lambda {self.rate.mean()}; shape {self.rate.shape}"



@dataclass
class CategoricalParameters(Parameters):
    """Dataclass for categorical output"""

    logits: Tensor
    name: str = field(kw_only=True, default="cat")

    @property
    def device(self) -> torch.device:
        return self.logits.device

    @property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits)

    def __str__(self) -> str:
        return f"CatOutput: Avg. logits {self.logits.mean()}; shape {self.logits.shape}"




ParameterType = TypeVar("ParameterType", bound=Parameters)
HivaeParameters = List[Optional[ParameterType]]
