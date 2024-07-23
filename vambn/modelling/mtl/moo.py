"""
This script includes code adapted from the 'impartial-vaes' repository 
with minor modifications. The original code can be found at:
https://github.com/adrianjav/impartial-vaes

Credit to the original authors: Adrian Javaloy, Maryam Meghdadi, and Isabel Valera 
for their valuable work.
"""


import logging
from typing import Any, Generator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from vambn.modelling.mtl import mtl
from vambn.modelling.mtl.parameters import MtlMethodParams

logger = logging.getLogger(__name__)


class MooMulti(nn.Module):
    """A PyTorch Module for Multiple Objective Optimization (MOO) within a loop."""

    inputs: Optional[torch.Tensor]

    def __init__(
        self, num_modules: int, moo_method: Optional[nn.Module] = None
    ):
        """
        Initialize the MooMulti module.

        Args:
            num_modules (int): Number of heads for extending the input.
            moo_method (nn.Module, optional): The MOO method to be used. Default is None.
        """
        super().__init__()

        self._moo_method = [moo_method]
        self.num_heads = num_modules
        self.inputs = None
        self.outputs = None

        if self.moo_method is not None:
            self.register_full_backward_hook(MooMulti._hook)

    @property
    def moo_method(self):
        """Get the MOO method."""
        return self._moo_method[0]

    def _hook(
        self, grads_input: Tuple[torch.Tensor], grads_output: Any
    ) -> Tuple[torch.Tensor]:
        """
        Hook function to replace gradients with MOO directions.

        Args:
            grads_input (Tuple[torch.Tensor]): Gradients of the module's inputs.
            grads_output (Any): Gradients of the module's outputs.

        Returns:
            Tuple[torch.Tensor]: Modified gradients.
        """
        moo_directions = self.moo_method(
            grads_output[0], self.inputs, self.outputs
        )
        self.outputs = None

        if grads_output[0].shape != moo_directions.shape:
            raise ValueError(
                f"MOO directions shape {moo_directions.shape} does not match grads_output shape {grads_output[0].shape}"
            )

        original_norm = grads_output[0].norm(p=2)
        moo_norm = moo_directions.norm(p=2).clamp_min(1e-10)
        scaling_factor = original_norm / moo_norm
        scaled_moo_directions = moo_directions * scaling_factor

        if grads_input[0].shape != scaled_moo_directions.shape:
            raise ValueError(
                f"Scaled MOO directions shape {scaled_moo_directions.shape} does not match grads_input shape {grads_input[0].shape}"
            )
        return (scaled_moo_directions,)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Extend the input to the number of heads and store it.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extended input tensor.
        """
        return z

    def __str__(self) -> str:
        return f"MooMulti({self.moo_method})"


class MOOForLoop(nn.Module):
    """A PyTorch Module for Multiple Objective Optimization (MOO) within a loop."""

    inputs: Optional[torch.Tensor]

    def __init__(self, num_heads: int, moo_method: Optional[nn.Module] = None):
        """
        Initialize the MOOForLoop module.

        Args:
            num_heads (int): Number of heads for extending the input.
            moo_method (nn.Module, optional): The MOO method to be used. Default is None.
        """
        super().__init__()

        self._moo_method = [moo_method]
        self.num_heads = num_heads
        self.inputs = None
        self.outputs = None

        if self.moo_method is not None:
            self.register_full_backward_hook(MOOForLoop._hook)

    @property
    def moo_method(self):
        """Get the MOO method."""
        return self._moo_method[0]

    def _hook(
        self, grads_input: Tuple[torch.Tensor], grads_output: Any
    ) -> Tuple[torch.Tensor]:
        """
        Hook function to replace gradients with MOO directions.

        Args:
            grads_input (Tuple[torch.Tensor]): Gradients of the module's inputs.
            grads_output (Any): Gradients of the module's outputs.

        Returns:
            Tuple[torch.Tensor]: Modified gradients.
        """
        moo_directions = self.moo_method(
            grads_output[0], self.inputs, self.outputs
        )
        self.outputs = None

        original_norm = grads_output[0].sum(dim=0).norm(p=2)
        moo_norm = moo_directions.sum(dim=0).norm(p=2).clamp_min(1e-10)
        moo_directions.mul_(original_norm / moo_norm)

        return (moo_directions.sum(dim=0),)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Extend the input to the number of heads and store it.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Extended input tensor.
        """
        extended_shape = [self.num_heads] + [-1 for _ in range(z.ndim)]
        if self.moo_method.requires_input and z.requires_grad:
            self.inputs = z.detach()
        extended_z = z.unsqueeze(0).expand(extended_shape)
        return extended_z

    def __str__(self) -> str:
        return f"MOOForLoop({self.moo_method})"


class MultiMOOForLoop(nn.Module):
    """A PyTorch Module for applying multiple MOOForLoop modules in parallel."""

    def __init__(self, num_heads: int, moo_methods: Sequence[nn.Module]):
        """
        Initialize the MultiMOOForLoop module.

        Args:
            num_heads (int): Number of heads for each MOOForLoop.
            moo_methods (Sequence[nn.Module]): List of MOO methods to be used.
        """
        super().__init__()

        self.num_inputs = len(moo_methods)
        self.loops = [MOOForLoop(num_heads, method) for method in moo_methods]

    def forward(self, *args) -> Generator[torch.Tensor, None, None]:
        """
        Forward pass. Applies each MOOForLoop to its corresponding input.

        Args:
            *args (torch.Tensor): Variable number of input tensors.

        Returns:
            Generator: A generator of extended input tensors after applying MOOForLoop.
        """
        if len(args) != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} inputs, got {len(args)} instead."
            )
        return (loop(z) for z, loop in zip(args, self.loops))


def setup_moo(hparams: List[MtlMethodParams], num_tasks: int) -> nn.Module:
    """
    Setup the multi-task learning module.

    Args:
        hparams (List[MtlMethodParams]): MTL method parameters.
        num_tasks (int): Number of tasks to perform.

    Raises:
        ValueError: If invalid method name is provided.

    Returns:
        nn.Module: Module for MTL objective.
    """
    if len(hparams) == 0:
        return mtl.Identity()

    modules = []
    for obj in hparams:
        try:
            method = mtl.MtlMethods[obj.name].value
        except KeyError:
            raise ValueError(f"Invalid method name: {obj.name}")

        if obj.name in ["nsgd"]:
            modules.append(method(num_tasks=num_tasks, update_at=obj.update_at))
        elif obj.name in ["gradnorm"]:
            modules.append(
                method(
                    num_tasks=num_tasks,
                    alpha=obj.alpha,
                    update_at=obj.update_at,
                )
            )
        elif obj.name in ["cagrad"]:
            modules.append(method(alpha=obj.alpha))
        elif obj.name in ["graddrop"]:
            modules.append(method(leakage=[0.2] * num_tasks))
        else:
            modules.append(method())

    return mtl.Compose(*modules) if len(modules) != 0 else None
