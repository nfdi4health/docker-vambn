"""
This script includes code adapted from the 'impartial-vaes' repository 
with minor modifications. The original code can be found at:
https://github.com/adrianjav/impartial-vaes

Credit to the original authors: Adrian Javaloy, Maryam Meghdadi, and Isabel Valera 
for their valuable work.
"""


import math
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

from vambn.modelling.mtl.utils import batch_product


def norm(tensor):
    """
    Compute the L2 norm of a tensor along the last dimension.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: L2 norm of the input tensor.
    """
    return tensor.norm(p=2, dim=-1, keepdim=True)


def divide(numer, denom):
    """
    Numerically stable division.

    Args:
        numer (torch.Tensor): Numerator tensor.
        denom (torch.Tensor): Denominator tensor.

    Returns:
        torch.Tensor: Result of numerically stable division.
    """
    epsilon = 1e-15
    return (
        torch.sign(numer)
        * torch.sign(denom)
        * torch.exp(
            torch.log(numer.abs() + epsilon) - torch.log(denom.abs() + epsilon)
        )
    )


def unitary(tensor):
    """
    Normalize the tensor to unit norm.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Unitary (normalized) tensor.
    """
    return divide(tensor, norm(tensor) + 1e-15)


def projection(u, v):
    """
    Project vector u onto vector v.

    Args:
        u (torch.Tensor): Vector to be projected.
        v (torch.Tensor): Vector onto which u is projected.

    Returns:
        torch.Tensor: Projection of u onto v.
    """
    numer = torch.dot(u, v)
    denom = torch.dot(v, v)

    return numer / denom.clamp_min(1e-15) * v


class MOOMethod(nn.Module, metaclass=ABCMeta):
    """Base class for multiple objective optimization (MOO) methods."""

    requires_input: bool = False

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        grads: torch.Tensor,
        inputs: Optional[torch.Tensor],
        outputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the new task gradients based on the original ones.

        Given K gradients of size D, returns a new set of K gradients of size D based on some criterion.

        Args:
            grads (torch.Tensor): Tensor of size K x D with the different gradients.
            inputs (torch.Tensor, optional): Tensor with the input of the forward pass (if requires_input is set to True).
            outputs (torch.Tensor, optional): Tensor with the K outputs of the module (not used currently).

        Returns:
            torch.Tensor: A tensor of the same size as `grads` with the new gradients to use during backpropagation.
        """
        raise NotImplementedError("You need to implement the forward pass.")


class Compose(MOOMethod):
    """
    Compose multiple MOO methods.

    Args:
        modules (MOOMethod): List of MOO methods to compose.

    Attributes:
        methods (nn.ModuleList): List of MOO methods.
        requires_input (bool): Flag indicating if input is required.

    """

    def __init__(self, *modules: MOOMethod):
        super().__init__()
        self.methods = nn.ModuleList(modules)
        self.requires_input = any([m.requires_input for m in modules])

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply composed MOO methods sequentially.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: Modified gradients.
        """
        for module in self.methods:
            grads = module(grads, inputs, outputs)
        return grads

class Identity(MOOMethod):
    """Identity MOO method that returns the input gradients unchanged."""

    def forward(
        self,
        grads: torch.Tensor,
        inputs: Optional[torch.Tensor],
        outputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Return the input gradients unchanged.

        Args:
            grads (torch.Tensor): Input gradients.
            inputs (torch.Tensor, optional): Input tensor.
            outputs (torch.Tensor, optional): Output tensor.

        Returns:
            torch.Tensor: Unchanged input gradients.
        """
        return grads


class IMTLG(MOOMethod):
    """IMTLG method for multiple objective optimization."""

    requires_input: bool = False

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using IMTLG method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        flatten_grads = grads.flatten(start_dim=1)
        num_tasks = len(grads)
        if num_tasks == 1:
            return grads

        grad_diffs, unit_diffs = [], []
        for i in range(1, num_tasks):
            grad_diffs.append(flatten_grads[0] - flatten_grads[i])
            unit_diffs.append(
                unitary(flatten_grads[0]) - unitary(flatten_grads[i])
            )
        grad_diffs = torch.stack(grad_diffs, dim=0)
        unit_diffs = torch.stack(unit_diffs, dim=0)

        DU_T = torch.einsum("ik,jk->ij", grad_diffs, unit_diffs)
        DU_T_inv = torch.pinverse(DU_T)

        alphas = torch.einsum(
            "i,ki,kj->j", grads[0].flatten(), unit_diffs, DU_T_inv
        )
        alphas = torch.cat(
            (1 - alphas.sum(dim=0).unsqueeze(dim=0), alphas), dim=0
        )

        return batch_product(grads, alphas)


class NSGD(MOOMethod):
    """Normalized Stochastic Gradient Descent (NSGD) method for MOO."""

    initial_grads: torch.Tensor
    requires_input: bool = False

    def __init__(self, num_tasks: int, update_at: int = 20):
        """
        Initialize NSGD method.

        Args:
            num_tasks (int): Number of tasks.
            update_at (int): Update interval.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.update_at = update_at
        self.register_buffer("initial_grads", torch.ones(num_tasks))
        self.counter = 0

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using NSGD method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        grad_norms = grads.flatten(start_dim=1).norm(dim=1)

        if self.initial_grads is None or self.counter == self.update_at:
            self.initial_grads = grad_norms

        self.counter += 1

        conv_ratios = grad_norms / self.initial_grads.clamp_min(1e-15)
        alphas = conv_ratios / conv_ratios.sum().clamp_min(1e-15)
        alphas = alphas / alphas.sum()

        weighted_sum_norms = (alphas * grad_norms).sum()
        grads = batch_product(
            grads, weighted_sum_norms / grad_norms.clamp_min(1e-15)
        )
        return grads


class GradDrop(MOOMethod):
    """Gradient Dropout (GradDrop) method for MOO.

    Args:
        leakage (List[float]): List of leakage rates for each task.

    Attributes:
        leakage (List[float]): List of leakage rates for each task.

    """

    requires_input: bool = True

    def __init__(self, leakage: List[float]):
        """
        Initialize GradDrop method.

        Args:
            leakage (List[float]): List of leakage rates for each task.

        Raises:
            AssertionError: If any leakage rate is not in the range [0, 1].

        """
        super(GradDrop, self).__init__()
        assert all(
            [0 <= x <= 1 for x in leakage]
        ), "All leakages should be in the range [0, 1]"
        self.leakage = leakage

    def forward(
        self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute new gradients using GradDrop method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.

        Raises:
            AssertionError: If the number of leakage parameters does not match the number of task gradients.

        """
        assert len(self.leakage) == len(
            grads
        ), "Leakage parameters should match the number of task gradients"
        sign_grads = [None for _ in range(len(grads))]
        for i in range(len(grads)):
            sign_grads[i] = inputs.sign() * grads[i]
            if len(grads[0].size()) > 1:  # It is batch-separated
                sign_grads[i] = grads[i].sum(dim=0, keepdim=True)

        odds = 0.5 * (
            1 + sum(sign_grads) / (sum(map(torch.abs, sign_grads)) + 1e-15)
        ).clamp(0, 1)
        assert odds.size() == sign_grads[0].size()  # pytype: disable=attribute-error

        new_grads = []
        samples = torch.rand(odds.size(), device=grads[0].device)
        for i in range(len(grads)):
            mask_i = torch.where(
                (odds > samples) & (sign_grads[i] > 0)  # pytype: disable=unsupported-operands
                | (odds < samples) & (sign_grads[i] < 0),  # pytype: disable=unsupported-operands
                torch.ones_like(odds),
                torch.zeros_like(odds),
            )
            mask_i = torch.lerp(
                mask_i, torch.ones_like(mask_i), self.leakage[i]
            )
            assert mask_i.size() == odds.size()
            new_grads.append(mask_i * grads[i])

        return torch.stack(new_grads, dim=0)


class GradNormBase(MOOMethod):
    """Base class for Gradient Normalization (GradNorm) method."""

    initial_values: torch.Tensor
    counter: torch.Tensor

    def __init__(self, num_tasks: int, alpha: float, update_at: int = 20):
        """
        Initialize GradNormBase method.

        Args:
            num_tasks (int): Number of tasks.
            alpha (float): Alpha parameter for GradNorm.
            update_at (int): Update interval.
        """
        super(GradNormBase, self).__init__()
        self.epsilon = 1e-5
        self.num_tasks = num_tasks
        self.weight_ = nn.Parameter(torch.ones([num_tasks]), requires_grad=True)
        self.alpha = alpha
        self.update_at = update_at
        self.register_buffer("initial_values", torch.ones(self.num_tasks))
        self.register_buffer("counter", torch.zeros([]))

    @property
    def weight(self) -> torch.Tensor:
        """
        Compute normalized weights.

        Returns:
            torch.Tensor: Normalized weights.
        """
        ws = self.weight_.exp().clamp(self.epsilon, float("inf"))
        norm_coef = self.num_tasks / ws.sum()
        return ws * norm_coef

    def _forward(self, grads: torch.Tensor, values: List[float]) -> torch.Tensor:
        """
        Compute new gradients using GradNorm method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            values (List[float]): Values for each task.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        if self.initial_values is None or self.counter == self.update_at:
            self.initial_values = torch.tensor(values)
        self.counter += 1

        with torch.enable_grad():
            grads_norm = grads.flatten(start_dim=1).norm(p=2, dim=1)
            mean_grad_norm = (
                torch.mean(batch_product(grads_norm, self.weight), dim=0)
                .detach()
                .clone()
            )

            values = [
                x / y.clamp_min(self.epsilon)
                for x, y in zip(values, self.initial_values)
            ]
            average_value = torch.mean(torch.stack(values))

            loss = grads.new_zeros([])
            for i, [grad, value] in enumerate(zip(grads_norm, values)):
                r_i = value / average_value.clamp_min(self.epsilon)
                loss += torch.abs(
                    grad * self.weight[i]
                    - mean_grad_norm * torch.pow(r_i, self.alpha)
                )
            loss.backward()

        with torch.no_grad():
            new_grads = batch_product(grads, self.weight.detach())
        return new_grads


class GradNorm(GradNormBase):
    """Gradient Normalization (GradNorm) method for MOO.

    Args:
        GradNormBase (class): Base class for GradNorm.

    Attributes:
        requires_input (bool): Flag indicating whether input is required.

    Methods:
        forward: Compute new gradients using GradNorm method.

    """

    requires_input: bool = False

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using GradNorm method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        return self._forward(grads, outputs)


class GradNormModified(GradNormBase):
    """
    Modified Gradient Normalization (GradNorm) method for MOO.

    Uses task-gradient convergence instead of task loss convergence.

    Attributes:
        requires_input (bool): Indicates whether the method requires input tensor.

    Methods:
        forward(grads, inputs, outputs): Compute new gradients using modified GradNorm method.

    """

    requires_input: bool = False

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using modified GradNorm method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        return self._forward(grads, grads.flatten(start_dim=1).norm(p=2, dim=1))


class PCGrad(MOOMethod):
    """Projected Conflicting Gradient (PCGrad) method for MOO.

    Attributes:
        requires_input (bool): Indicates whether the method requires input tensor.
    """

    requires_input: bool = False

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using PCGrad method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        size = grads.size()[1:]
        num_tasks = grads.size(0)
        grads_list = [g.flatten() for g in grads]

        new_grads = [None for _ in range(num_tasks)]
        for i in np.random.permutation(num_tasks):
            grad_i = grads_list[i]
            for j in np.random.permutation(num_tasks):
                if i == j:
                    continue
                grad_j = grads_list[j]
                if torch.cosine_similarity(grad_i, grad_j, dim=0) < 0:
                    grad_i = grad_i - projection(grad_i, grad_j)
                    assert id(grads_list[i]) != id(grad_i), "Aliasing!"
            new_grads[i] = grad_i.reshape(size)

        return torch.stack(new_grads, dim=0)


class GradVac(MOOMethod):
    """Gradient Vaccination (GradVac) method for MOO."""

    requires_input: bool = False

    def __init__(self, decay: float):
        """
        Initialize GradVac method.

        Args:
            decay: Decay rate for EMA.
        """
        super(GradVac, self).__init__()
        self.decay = decay

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using GradVac method.

        Args:
            grads: Gradients tensor.
            inputs: Input tensor.
            outputs: Output tensor.

        Returns:
            New gradients tensor.
        """

        def vac_projection(u: torch.Tensor, v: torch.Tensor, pre_ema: float, post_ema: float) -> torch.Tensor:
            norm_u = torch.dot(u, u).sqrt()
            norm_v = torch.dot(v, v).sqrt()

            numer = norm_u * (
                pre_ema * math.sqrt(1 - post_ema**2)
                - post_ema * math.sqrt(1 - pre_ema**2)
            )
            denom = norm_v * math.sqrt(1 - pre_ema**2)

            return numer / denom.clamp_min(1e-15) * v

        size = grads.size()[1:]
        num_tasks = grads.size(0)

        grads_list = [g.flatten() for g in grads]
        ema = [[0 for _ in range(num_tasks)] for _ in range(num_tasks)]

        new_grads = []
        for i in range(num_tasks):
            grad_i = grads_list[i]
            for j in np.random.permutation(num_tasks):
                if i == j:
                    continue
                grad_j = grads_list[j]
                cos_sim = torch.cosine_similarity(grad_i, grad_j, dim=0)
                if cos_sim < ema[i][j]:
                    grad_i = grad_i + vac_projection(
                        grad_i, grad_j, ema[i][j], cos_sim
                    )
                    assert id(grads_list[i]) != id(grad_i), "Aliasing!"
                ema[i][j] = (1 - self.decay) * ema[i][j] + self.decay * cos_sim
            new_grads.append(grad_i.reshape(size))

        return torch.stack(new_grads, dim=0)


class MinNormSolver:
    """Solver for finding the minimum norm solution in the convex hull of vectors."""

    MAX_ITER = 250
    STOP_CRIT = 1e-5

    @staticmethod
    def _min_norm_element_from2(v1v1: float, v1v2: float, v2v2: float) -> tuple:
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2.

        Args:
            v1v1: <x1, x1>.
            v1v2: <x1, x2>.
            v2v2: <x2, x2>.

        Returns:
            tuple: Coefficients and cost for the minimum norm element.
        """
        if v1v2 >= v1v1:
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs: list, dps: dict) -> tuple:
        """
        Find the minimum norm solution as a combination of two points in 2D.

        Args:
            vecs: List of vectors.
            dps: Dictionary to store dot products.

        Returns:
            tuple: Solution and updated dot products.
        """
        dmin = float("inf")
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = sum(
                        torch.dot(vecs[i][k], vecs[j][k]).item()
                        for k in range(len(vecs[i]))
                    )
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = sum(
                        torch.dot(vecs[i][k], vecs[i][k]).item()
                        for k in range(len(vecs[i]))
                    )
                if (j, j) not in dps:
                    dps[(j, j)] = sum(
                        torch.dot(vecs[j][k], vecs[j][k]).item()
                        for k in range(len(vecs[i]))
                    )
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y: np.ndarray) -> np.ndarray:
        """
        Project y onto the simplex.

        Args:
            y: Input array.

        Returns:
            Projected array.
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val: np.ndarray, grad: np.ndarray, n: int) -> np.ndarray:
        """
        Compute the next point for the projected gradient descent.

        Args:
            cur_val: Current value.
            grad: Gradient.
            n: Dimension of the problem.

        Returns:
            Next point.
        """
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / proj_grad[proj_grad > 0]

        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs: List) -> Tuple | None:
        """
        Find the minimum norm element in the convex hull of vectors.

        Args:
            vecs: List of vectors.

        Returns:
            Minimum norm element and its cost.
        """
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            v1v1 = sum(
                sol_vec[i] * sol_vec[j] * dps[(i, j)]
                for i in range(n)
                for j in range(n)
            )
            v1v2 = sum(
                sol_vec[i] * new_point[j] * dps[(i, j)]
                for i in range(n)
                for j in range(n)
            )
            v2v2 = sum(
                new_point[i] * new_point[j] * dps[(i, j)]
                for i in range(n)
                for j in range(n)
            )
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads: dict, losses: dict, normalization_type: str) -> dict:
    """
    Compute gradient normalizers based on the specified normalization type.

    Args:
        grads: A dictionary of gradients.
        losses: A dictionary of losses.
        normalization_type: The type of normalization ('l2', 'loss', 'loss+', 'none').

    Returns:
        A dictionary of gradient normalizers.
    """
    gn = {}
    if normalization_type == "l2":
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().item() for gr in grads[t]]))
    elif normalization_type == "loss":
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == "loss+":
        for t in grads:
            gn[t] = losses[t] * np.sqrt(
                np.sum([gr.pow(2).sum().item() for gr in grads[t]])
            )
    elif normalization_type == "none":
        for t in grads:
            gn[t] = 1.0
    else:
        print("ERROR: Invalid Normalization Type")
    return gn


class MGDAUB(MOOMethod):
    """MGDA-UB method for multiple objective optimization."""

    requires_input: bool = False

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using MGDA-UB method.

        Args:
            grads (torch.Tensor): Gradients tensor.
            inputs (torch.Tensor): Input tensor.
            outputs (torch.Tensor): Output tensor.

        Returns:
            torch.Tensor: New gradients tensor.
        """
        epsilon: float = 1e-3
        shape: Tuple[int] = grads.size()[1:]
        grads = grads.flatten(start_dim=1).unsqueeze(dim=1)

        weights, min_norm = MinNormSolver.find_min_norm_element(
            grads.unbind(dim=0)
        )
        weights = [min(w, epsilon) for w in weights]

        grads = torch.stack(
            [g.reshape(shape) * w for g, w in zip(grads, weights)], dim=0
        )
        return grads


class CAGrad(MOOMethod):
    """CAGrad method for multiple objective optimization."""

    requires_input: bool = False

    def __init__(self, alpha: float):
        """
        Initialize CAGrad method.

        Args:
            alpha: Alpha parameter for CAGrad.
        """
        super(CAGrad, self).__init__()
        self.alpha = alpha

    def forward(self, grads: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute new gradients using CAGrad method.

        Args:
            grads: Gradients tensor.
            inputs: Input tensor.
            outputs: Output tensor.

        Returns:
            New gradients tensor.
        """
        shape = grads.size()
        num_tasks = len(grads)
        grads = grads.flatten(start_dim=1).t()

        GG = grads.t().mm(grads).cpu()
        g0_norm = (GG.mean() + 1e-8).sqrt()

        x_start = np.ones(num_tasks) / num_tasks
        bnds = tuple((0, 1) for _ in x_start)
        cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}

        A = GG.numpy()
        b = x_start.copy()
        c = (self.alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (
                x.reshape(1, num_tasks).dot(A).dot(b.reshape(num_tasks, 1))
                + c
                * np.sqrt(
                    x.reshape(1, num_tasks).dot(A).dot(x.reshape(num_tasks, 1))
                    + 1e-8
                )
            ).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x

        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = (grads + lmbda * gw.unsqueeze(1)) / num_tasks

        g = g.t().reshape(shape)
        grads = g

        return grads


class MtlMethods(Enum):
    """Enumeration of available multi-task learning methods."""

    imtlg = IMTLG
    nsgd = NSGD
    gradnorm = GradNormModified
    pcgrad = PCGrad
    mgda_ub = MGDAUB
    identity = Identity
    cagrad = CAGrad
    graddrop = GradDrop
