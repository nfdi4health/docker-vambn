from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from vambn.modelling.models.layers import ModifiedLinear
from vambn.modelling.mtl import moo
from vambn.modelling.mtl.parameters import MtlMethodParams


class BaseElement(nn.Module):
    """
    A base neural network element that consists of a single modified linear layer.

    Args:
        input_dim (int): Input dimension of the layer.
        output_dim (int): Output dimension of the layer.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = ModifiedLinear(self.input_dim, self.output_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass through the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the layer.
        """
        return self.layer(x)


class BaseModule(nn.Module, ABC):
    """
    Base class for all modules in the HIVAE model.

    Args:
        z_dim (int): Input dimension of the latent samples z.
        ys_dim (int): Intermediate dimension of the latent samples ys.
        y_dim (int or Tuple[int, ...]): Output dimension of the latent samples y, which can be different for each module.
        n_modules (int): Number of modules.
        module_names (Tuple[str, ...]): Names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): MTL methods used to avoid conflicting gradients. Defaults to None.

    Raises:
        ValueError: If the length of y_dim is not equal to the number of modules.
    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """
        Base class for all modules in the HIVAE model.

        Args:
            z_dim (int): Input dimension of the latent samples z.
            ys_dim (int): Intermediate dimension of the latent samples ys.
            y_dim (int | Tuple[int, ...]): Output dimension of the latent samples y, which can be different for each module.
            n_modules (int): Number of modules.
            module_names (Tuple[str, ...]): Names of the modules.
            mtl_method (Optional[Tuple[str, ...]], optional): MTL methods used to avoid conflicting gradients. Defaults to None.

        Raises:
            ValueError: If the length of y_dim is not equal to the number of modules.
        """
        super().__init__()
        self.z_dim = z_dim
        self.ys_dim = ys_dim
        self.y_dim = y_dim
        self.n_modules = n_modules
        self.mtl_method = mtl_method
        self.module_names = module_names
        self.has_params = True

        if (
            isinstance(self.y_dim, Iterable)
            and len(self.y_dim) != self.n_modules
        ):
            raise ValueError(
                f"Length of y_dim must be equal to the number of modules: {self.n_modules}"
            )


    @abstractmethod
    def order_layers(self, module_names: Tuple[str, ...]) -> None:
        """
        Order the layers of the module according to the module names.

        Args:
            module_names (Tuple[str, ...]): Names of the modules
        """
        pass

    @abstractmethod
    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        pass

    def _y_dim(self, i: int) -> int:
        """
        Retruns the output dimension of the ith module.

        Args:
            i (int): Index of the module.

        Returns:
            int: Output dimension of the ith module.
        """
        return self.y_dim if isinstance(self.y_dim, int) else self.y_dim[i]


class ImposterModule(BaseModule):
    """
    This module is used as a placeholder when no modularity is desired. It simply
    returns the input z as the output.

    Args:
        z_dim (int): Input dimension of the latent samples z.
        ys_dim (int): Intermediate dimension of the latent samples ys.
        y_dim (int or Tuple[int, ...]): Output dimension of the latent samples y, which can be different for each module.
        n_modules (int): Number of modules.
        module_names (Tuple[str, ...]): Names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): MTL methods used to avoid conflicting gradients. Defaults to None.

    Attributes:
        has_params (bool): Whether the module has parameters.
    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        self.has_params = False

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Order the layers based on the module names. Since this module has no parameters,
        this method does nothing in the case of the ImposterModule.

        Args:
            module_names (Tuple[str]): Names of the modules.
        """
        pass

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the module. Since this module has no parameters, it simply
        returns the input z.

        Args:
            z (Tuple[torch.Tensor, ...]): Input tensor.

        Returns:
            Tuple[torch.Tensor, ...]: Output tensor.
        """
        return z


class SharedLinearModule(BaseModule):
    """
    This module passes the z's through a single shared dense layer that generates the
    individual outputs for each module using the same weights. Outputs are generated
    one by one. The assumption is that z shares the same dimensional space across
    modules. This is the simplest form of modularity.

    Args:
        z_dim (int): Input dimension of the latent samples z.
        ys_dim (int): Intermediate dimension of the latent samples ys.
        y_dim (int or Tuple[int, ...]): Output dimension of the latent samples y, which can be different for each module.
        n_modules (int): Number of modules.
        module_names (Tuple[str, ...]): Names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): MTL methods used to avoid conflicting gradients. Defaults to None.

    Attributes:
        shared_layer (BaseElement): Shared dense layer.
        scaling_layers (nn.ModuleList): List of individual dense layers for each module.
    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        self.shared_layer = BaseElement(self.z_dim, self.ys_dim)
        self.scaling_layers = nn.ModuleList(
            [
                BaseElement(self.ys_dim, self.z_dim)
                for i in range(self.n_modules)
            ]
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Order the layers based on the module names.

        Args:
            module_names (Tuple[str]): Names of the modules.
        """
        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the module. The z's are passed through the shared layer
        to generate one output which is identical for all modules. The output is then
        passed through individual layers to generate the final outputs.

        Args:
            z (Tuple[torch.Tensor, ...]): Input tensor.

        Returns:
            Tuple[torch.Tensor, ...]: Output tensor with individual outputs for each module.
        """
        ys = tuple([self.shared_layer(zi) for zi in z])
        return tuple([self.scaling_layers[i](ysi) for i, ysi in enumerate(ys)])


class ConcatModuleMtl(BaseModule):
    """
    This module concatenates the z's and passes them through a shared layer before
    passing through the MOO block. The output is then passed through individual layers
    to generate the final outputs. This is the same as the ConcatModule, but with the
    addition of the MOO block.
    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        self.mtl_method = mtl_method
        self.shared_layer = BaseElement(z_dim * n_modules, ys_dim)
        self._mtl_module = moo.setup_moo(
            [MtlMethodParams(x) for x in mtl_method],
            num_tasks=n_modules,
        )
        self.moo_block = moo.MultiMOOForLoop(
            n_modules, moo_methods=(self._mtl_module,)
        )
        self.scaling_layers = nn.ModuleList(
            [
                BaseElement(self.ys_dim, self.z_dim)
                for i in range(self.n_modules)
            ]
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Order the layers based on the module names.

        Args:
            module_names (Tuple[str]): Names of the modules.
        """
        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the module. The z's are concatenated and passed through
        the shared layer to generate one output which is identical for all modules. The
        output is then passed through the MOO block and individual layers to generate
        the final outputs.

        Args:
            z (Tuple[torch.Tensor, ...]): Input tensor.

        Returns:
            Tuple[torch.Tensor, ...]: Output tensor with individual outputs for each module.
        """
        # Concatenate the z's and pass through the
        # shared layer to generate one output which is identical for all modules
        h = self.shared_layer(torch.cat(z, dim=1))
        # Pass through the MOO block
        (hs,) = self.moo_block(h)
        # Pass through the individual layers
        return tuple(
            [
                self.scaling_layers[i](hi)
                for i, hi in zip(range(self.n_modules), hs)
            ]
        )


class ConcatModule(BaseModule):
    """
    A module that concatenates multiple input tensors and applies scaling layers.

    Args:
        z_dim (int): The dimension of the input z tensors.
        ys_dim (int): The dimension of the output ys tensors.
        y_dim (int or Tuple[int, ...]): The dimension of the output y tensors.
        n_modules (int): The number of modules.
        module_names (Tuple[str, ...]): The names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): The method used for multi-task learning (default: None).

    Attributes:
        shared_layer (BaseElement): The shared layer that concatenates the input z tensors.
        scaling_layers (nn.ModuleList): The list of scaling layers for each module.

    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        self.shared_layer = BaseElement(
            self.z_dim * n_modules, self.ys_dim * n_modules
        )
        self.scaling_layers = nn.ModuleList(
            [
                BaseElement(self.ys_dim, self.z_dim)
                for i in range(self.n_modules)
            ]
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Reorders the scaling layers based on the given module names.

        Args:
            module_names (Tuple[str]): The new order of the module names.

        """
        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Performs forward pass through the ConcatModule. The input tensors z are concatenated
        and passed through the shared layer to generate a single output tensor. The output
        tensor is then passed through the scaling layers to generate the final output tensors.

        Args:
            z (Tuple[torch.Tensor, ...]): The input tensors z.

        Returns:
            Tuple[torch.Tensor, ...]: The output tensors after applying scaling layers.

        """
        # z-type: Tuple[torch.Tensor, ...], with each z having shape (batch_size, z_dim)
        h = self.shared_layer(torch.cat(z, dim=1))
        # size of h: (batch_size, ys_dim * n_modules)
        return tuple(
            [
                self.scaling_layers[i](
                    h[:, i * self.ys_dim : (i + 1) * self.ys_dim]
                )
                for i in range(self.n_modules)
            ]
        )  # size of h[:, i * self.ys_dim : (i + 1) * self.ys_dim]: (batch_size, ys_dim)


class AvgModuleMtl(BaseModule):
    """
    This module averages the z's and passes them through the MOO block. The output is then
    passed through individual layers to generate the final outputs.

    Args:
        z_dim (int): The dimension of the input z.
        ys_dim (int): The dimension of the shared output ys.
        y_dim (int | Tuple[int, ...]): The dimension of the individual outputs y.
        n_modules (int): The number of modules.
        module_names (Tuple[str, ...]): The names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): The method used for multi-task learning. Defaults to None.

    Attributes:
        _mtl_module (MultiObjectiveOptimization): The multi-objective optimization module.
        moo_block (MultiMOOForLoop): The multi-objective optimization block.
        scaling_layers (nn.ModuleList): The list of scaling layers.

    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        self._mtl_module = moo.setup_moo(
            [MtlMethodParams(x) for x in mtl_method],
            num_tasks=n_modules,
        )
        self.moo_block = moo.MultiMOOForLoop(
            n_modules, moo_methods=(self._mtl_module,)
        )
        self.scaling_layers = nn.ModuleList(
            [BaseElement(self.z_dim, self.z_dim) for i in range(self.n_modules)]
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Orders the scaling layers based on the given module names.

        Args:
            module_names (Tuple[str]): The names of the modules.

        """
        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Performs the forward pass of the module. The input z's are averaged and passed through
        the MOO block. The output is then passed through the individual scaling layers to
        generate the final output tensors.

        Args:
            z (Tuple[torch.Tensor, ...]): The input z.

        Returns:
            Tuple[torch.Tensor, ...]: The output tensors.

        """
        # Average the z's and pass through the shared layer
        (h,) = self.moo_block(torch.mean(torch.stack(z), dim=0))
        # Pass through the individual layers
        return tuple(
            [
                self.scaling_layers[i](hi)
                for i, hi in zip(range(self.n_modules), h)
            ]
        )


class MaxModuleMtl(BaseModule):
    """
    This module takes the maximum of the z's and passes them through the MOO block. The
    output is then passed through individual layers to generate the final outputs.

    Args:
        z_dim (int): The dimension of the input z.
        ys_dim (int): The dimension of the ys.
        y_dim (int | Tuple[int, ...]): The dimension of the output y.
        n_modules (int): The number of modules.
        module_names (Tuple[str, ...]): The names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): The method for multi-task learning. Defaults to None.

    Attributes:
        _mtl_module (MultiObjectiveOptimization): The multi-objective optimization module.
        moo_block (MultiMOOForLoop): The multi-objective optimization block.
        scaling_layers (nn.ModuleList): The list of scaling layers.
    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        self._mtl_module = moo.setup_moo(
            [MtlMethodParams(x) for x in mtl_method],
            num_tasks=n_modules,
        )
        self.moo_block = moo.MultiMOOForLoop(
            n_modules, moo_methods=(self._mtl_module,)
        )
        self.scaling_layers = nn.ModuleList(
            [BaseElement(self.z_dim, self.z_dim) for i in range(self.n_modules)]
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Orders the scaling layers based on the given module names.

        Args:
            module_names (Tuple[str]): The names of the modules.
        """
        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Performs forward pass through the module. The maximum of the z's is passed through
        the MOO block. The output is then passed through individual scaling layers to
        generate the final output tensors.

        Args:
            z (Tuple[torch.Tensor, ...]): The input z.

        Returns:
            Tuple[torch.Tensor, ...]: The output tensors.
        """
        # Average the z's and pass through the shared layer
        (h,) = self.moo_block(torch.max(torch.stack(z), dim=0).values)
        # Pass through the individual layers
        return tuple(
            [
                self.scaling_layers[i](hi)
                for i, hi in zip(range(self.n_modules), h)
            ]
        )


class SelfAttention(nn.Module):
    """
    Self-Attention module.

    Args:
        hidden_dim (int): The dimension of the input and output tensors.

    Attributes:
        query (nn.Linear): Linear layer for computing the query tensor.
        key (nn.Linear): Linear layer for computing the key tensor.
        value (nn.Linear): Linear layer for computing the value tensor.
        softmax (nn.Softmax): Softmax function for computing attention weights.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        """
        Forward pass of the SelfAttention module. Computes the query, key, and value tensors
        and uses them to compute the attention weights. The attention weights are then used
        to compute the attended values.

        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        q = self.query(z)
        k = self.key(z)
        v = self.value(z)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)

        attended_values = torch.matmul(attention_weights, v)
        return attended_values


class EncoderModule(BaseModule):
    """
    EncoderModule class represents a module for encoding input data.

    Args:
        z_dim (int): The dimension of the latent space.
        ys_dim (int): The dimension of the output space.
        y_dim (int or Tuple[int, ...]): The dimension(s) of the input data.
        n_modules (int): The number of modules.
        module_names (Tuple[str, ...]): The names of the modules.
        mtl_method (Optional[Tuple[str, ...]]): The method(s) for multi-task learning. Defaults to None.

    Attributes:
        attention (SelfAttention): The self-attention layer.
        feed_forward (nn.Sequential): The feed-forward neural network.
        dropout (nn.Dropout): The dropout layer.
        layer_norm_1 (nn.LayerNorm): The layer normalization layer.
        layer_norm_2 (nn.LayerNorm): The layer normalization layer.
        ys_layer (nn.Linear): The linear layer for output.
        scaling_layers (nn.ModuleList): The list of scaling layers.

    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        cat_dim = z_dim * n_modules
        self.attention = SelfAttention(cat_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(cat_dim, ys_dim),
            nn.GELU(),
            nn.Linear(ys_dim, cat_dim),
            nn.Dropout(0.1),
        )
        self.dropout = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(cat_dim)
        self.layer_norm_2 = nn.LayerNorm(cat_dim)
        self.ys_layer = nn.Linear(cat_dim, ys_dim)
        self.scaling_layers = nn.ModuleList(
            [
                BaseElement(self.ys_dim, self.z_dim)
                for i in range(self.n_modules)
            ]
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """
        Orders the scaling layers based on the given module names.

        Args:
            module_names (Tuple[str]): The names of the modules.

        Returns:
            None

        """

        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """
        Performs forward pass through the encoder module. The input tensors z are concatenated
        and passed through the self-attention layer. The output is then passed through the feed-forward
        neural network and the output layer. The output is then passed through the scaling layers to
        generate the final output tensors.

        Args:
            z (Tuple[torch.Tensor, ...]): The input tensors.

        Returns:
            Tuple[torch.Tensor, ...]: The output tensors.

        """

        x = torch.cat(z, dim=1)
        attended = self.attention(torch.cat(z, dim=1))
        x = self.layer_norm_1(x + self.dropout(attended))

        h = self.feed_forward(x)
        h = self.layer_norm_2(x + self.dropout(h))
        yb = self.ys_layer(h)
        y = tuple([self.scaling_layers[i](yb) for i in range(self.n_modules)])
        return y


class EncoderModuleMtl(BaseModule):
    """Encoder module for multi-task learning.

    Args:
        z_dim (int): The dimension of the latent space.
        ys_dim (int): The dimension of the shared representation.
        y_dim (int | Tuple[int, ...]): The dimension(s) of the task-specific representations.
        n_modules (int): The number of task-specific modules.
        module_names (Tuple[str, ...]): The names of the task-specific modules.
        mtl_method (Optional[Tuple[str, ...]], optional): The multi-task learning method(s) to use. Defaults to None.

    Attributes:
        attention (SelfAttention): The self-attention layer.
        feed_forward (nn.Sequential): The feed-forward neural network.
        dropout (nn.Dropout): The dropout layer.
        layer_norm_1 (nn.LayerNorm): The first layer normalization.
        layer_norm_2 (nn.LayerNorm): The second layer normalization.
        ys_layer (nn.Linear): The linear layer for shared representation.
        scaling_layers (nn.ModuleList): The list of scaling layers for task-specific representations.
        _mtl_module (moo.MultiObjectiveOptimization): The multi-objective optimization module.
        moo_block (moo.MultiMOOForLoop): The multi-objective optimization block.

    Raises:
        Exception: This class should no longer be used.

    """

    def __init__(
        self,
        z_dim: int,
        ys_dim: int,
        y_dim: int | Tuple[int, ...],
        n_modules: int,
        module_names: Tuple[str, ...],
        mtl_method: Optional[Tuple[str, ...]] = None,
    ) -> None:
        super().__init__(
            z_dim=z_dim,
            ys_dim=ys_dim,
            y_dim=y_dim,
            n_modules=n_modules,
            mtl_method=mtl_method,
            module_names=module_names,
        )
        cat_dim = z_dim * n_modules
        self.attention = SelfAttention(cat_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(cat_dim, ys_dim),
            nn.GELU(),
            nn.Linear(ys_dim, cat_dim),
            nn.Dropout(0.1),
        )
        self.dropout = nn.Dropout(0.1)
        self.layer_norm_1 = nn.LayerNorm(cat_dim)
        self.layer_norm_2 = nn.LayerNorm(cat_dim)
        self.ys_layer = nn.Linear(cat_dim, ys_dim)
        self.scaling_layers = nn.ModuleList(
            [
                BaseElement(self.ys_dim, self.z_dim)
                for i in range(self.n_modules)
            ]
        )
        self._mtl_module = moo.setup_moo(
            [MtlMethodParams(x) for x in mtl_method],
            num_tasks=n_modules,
        )
        self.moo_block = moo.MultiMOOForLoop(
            n_modules, moo_methods=(self._mtl_module,)
        )

    def order_layers(self, module_names: Tuple[str]) -> None:
        """Order the scaling layers based on the given module names.

        Args:
            module_names (Tuple[str]): The names of the task-specific modules.

        """
        prior_map = {name: i for i, name in enumerate(self.module_names)}
        self.scaling_layers = nn.ModuleList(
            [self.scaling_layers[prior_map[name]] for name in module_names]
        )

    def forward(self, z: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Forward pass of the module. The input tensors z are concatenated and passed through
        the self-attention layer. The output is then passed through the feed-forward neural network
        combined with the residual connection and layer normalization. The output is then passed through
        the MTL block and the scaling layers to generate the final output tensors.

        Args:
            z (Tuple[torch.Tensor, ...]): The input tensors.

        Returns:
            Tuple[torch.Tensor, ...]: The output tensors.

        """
        x = torch.cat(z, dim=1)
        attended = self.attention(torch.cat(z, dim=1))
        x = self.layer_norm_1(x + self.dropout(attended))

        h = self.feed_forward(x)
        h = self.layer_norm_2(x + self.dropout(h))
        yb = self.ys_layer(h)
        (yb_moo,) = self.moo_block(yb)
        y = tuple(
            [
                self.scaling_layers[i](yi)
                for i, yi in zip(range(self.n_modules), yb_moo)
            ]
        )
        return y


SHARED_MODULES = {
    "sharedLinear": SharedLinearModule,
    "concatMtl": ConcatModule,
    "concatIndiv": ConcatModuleMtl,
    "none": ImposterModule,
    "avgMtl": AvgModuleMtl,
    "maxMtl": MaxModuleMtl,
    "encoder": EncoderModule,
    "encoderMtl": EncoderModuleMtl,
}
