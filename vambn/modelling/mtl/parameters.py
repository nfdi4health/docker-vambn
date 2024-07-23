from dataclasses import dataclass
from typing import Optional


@dataclass
class MtlMethodParams:
    """
    Params and method description for multi-task learning.

    Attributes:
        name (str): Name of the MTL method.
        update_at (Optional[int]): Update interval, specific to certain methods.
        alpha (Optional[float]): Alpha parameter, specific to certain methods.
    """

    name: str
    update_at: Optional[int] = None
    alpha: Optional[float] = None

    def __post_init__(self):
        """
        Post-initialization to set default values for specific methods.
        """
        if self.name == "nsgd":
            if self.update_at is None:
                self.update_at = 1
        elif self.name == "gradnorm":
            if self.update_at is None:
                self.update_at = 1
            if self.alpha is None:
                self.alpha = 1.0
        elif self.name == "pcgrad":
            if self.update_at is None:
                self.update_at = 1
        elif self.name == "cagrad":
            if self.alpha is None:
                self.alpha = 10
