"""
This script includes code adapted from the 'impartial-vaes' repository 
with minor modifications. The original code can be found at:
https://github.com/adrianjav/impartial-vaes

Credit to the original authors: Adrian Javaloy, Maryam Meghdadi, and Isabel Valera 
for their valuable work.
"""

import numpy as np
import torch
import torch.nn as nn


class MinNormLinearSolver(nn.Module):
    """Solves the min norm problem in case of 2 vectors (lies on a line)."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, v1v1, v1v2, v2v2):
        """
        Solver execution on scalar products of 2 vectors.

        Args:
            v1v1 (float): Scalar product <V1, V1>.
            v1v2 (float): Scalar product <V1, V2>.
            v2v2 (float): Scalar product <V2, V2>.

        Returns:
            tuple: A tuple containing:
                - gamma (float): Min-norm solution c = (gamma, 1. - gamma).
                - cost (float): The norm of min-norm point.
        """
        if v1v2 >= v1v1:
            return 1.0, v1v1
        if v1v2 >= v2v2:
            return 0.0, v2v2
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost


class MinNormPlanarSolver(nn.Module):
    """Solves the min norm problem in case the vectors lie on the same plane."""

    def __init__(self, n_tasks):
        """
        Initializes the MinNormPlanarSolver.

        Args:
            n_tasks (int): Number of tasks/vectors.
        """
        super().__init__()
        i_grid = torch.arange(n_tasks)
        j_grid = torch.arange(n_tasks)
        ii_grid, jj_grid = torch.meshgrid(i_grid, j_grid)
        i_triu, j_triu = np.triu_indices(n_tasks, 1)

        self.register_buffer("n", torch.tensor(n_tasks))
        self.register_buffer("i_triu", torch.from_numpy(i_triu))
        self.register_buffer("j_triu", torch.from_numpy(j_triu))
        self.register_buffer("ii_triu", ii_grid[i_triu, j_triu])
        self.register_buffer("jj_triu", jj_grid[i_triu, j_triu])
        self.register_buffer("one", torch.ones(self.ii_triu.shape))
        self.register_buffer("zero", torch.zeros(self.ii_triu.shape))

    @torch.no_grad()
    def line_solver_vectorized(self, v1v1, v1v2, v2v2):
        """
        Linear case solver, but for collection of vector pairs (Vi, Vj).

        Args:
            v1v1 (Tensor): Vector of scalar products <Vi, Vi>.
            v1v2 (Tensor): Vector of scalar products <Vi, Vj>.
            v2v2 (Tensor): Vector of scalar products <Vj, Vj>.

        Returns:
            tuple: A tuple containing:
                - gamma (Tensor): Vector of min-norm solution c = (gamma, 1. - gamma).
                - cost (Tensor): Vector of the norm of min-norm point.
        """
        gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2 + 1e-8)
        gamma = gamma.where(v1v2 < v2v2, self.zero)
        gamma = gamma.where(v1v2 < v1v1, self.one)

        cost = v2v2 + gamma * (v1v2 - v2v2)
        cost = cost.where(v1v2 < v2v2, v2v2)
        cost = cost.where(v1v2 < v1v1, v1v1)
        return gamma, cost

    @torch.no_grad()
    def forward(self, grammian):
        """
        Planar case solver, when Vi lies on the same plane.

        Args:
            grammian (Tensor): Grammian matrix G[i, j] = [<Vi, Vj>], G is a nxn tensor.

        Returns:
            Tensor: Coefficients c = [c1, ... cn] that solves the min-norm problem.
        """
        vivj = grammian[self.ii_triu, self.jj_triu]
        vivi = grammian[self.ii_triu, self.ii_triu]
        vjvj = grammian[self.jj_triu, self.jj_triu]

        gamma, cost = self.line_solver_vectorized(vivi, vivj, vjvj)
        offset = torch.argmin(cost)
        i_min, j_min = self.i_triu[offset], self.j_triu[offset]
        sol = torch.zeros(self.n, device=grammian.device)
        sol[i_min], sol[j_min] = gamma[offset], 1.0 - gamma[offset]
        return sol


class MinNormSolver(nn.Module):
    """Solves the min norm problem in the general case."""

    def __init__(self, n_tasks, max_iter=250, stop_crit=1e-6):
        """
        Initializes the MinNormSolver.

        Args:
            n_tasks (int): Number of tasks/vectors.
            max_iter (int, optional): Maximum number of iterations. Defaults to 250.
            stop_crit (float, optional): Stopping criterion. Defaults to 1e-6.
        """
        super().__init__()
        self.n = n_tasks
        self.linear_solver = MinNormLinearSolver()
        self.planar_solver = MinNormPlanarSolver(n_tasks)

        n_grid = torch.arange(n_tasks)
        i_grid = torch.arange(n_tasks, dtype=torch.float32) + 1
        ii_grid, jj_grid = torch.meshgrid(n_grid, n_grid)

        self.register_buffer("n_ts", torch.tensor(n_tasks))
        self.register_buffer("i_grid", i_grid)
        self.register_buffer("ii_grid", ii_grid)
        self.register_buffer("jj_grid", jj_grid)
        self.register_buffer("zero", torch.zeros(n_tasks))
        self.register_buffer("stop_crit", torch.tensor(stop_crit))

        self.max_iter = max_iter
        self.two_sol = nn.Parameter(torch.zeros(2))
        self.two_sol.require_grad = False

    @torch.no_grad()
    def projection_to_simplex(self, gamma):
        """
        Projects gamma to the simplex.

        Args:
            gamma (Tensor): The input tensor to project.

        Returns:
            Tensor: The projected tensor.
        """
        sorted_gamma, indices = torch.sort(gamma, descending=True)
        tmp_sum = torch.cumsum(sorted_gamma, 0)
        tmp_max = (tmp_sum - 1.0) / self.i_grid

        non_zeros = torch.nonzero(tmp_max[:-1] > sorted_gamma[1:])
        if non_zeros.shape[0] > 0:
            tmax_f = tmp_max[:-1][non_zeros[0][0]]
        else:
            tmax_f = tmp_max[-1]
        return torch.max(gamma - tmax_f, self.zero)

    @torch.no_grad()
    def next_point(self, cur_val, grad):
        """
        Computes the next point in the optimization.

        Args:
            cur_val (Tensor): Current value.
            grad (Tensor): Gradient.

        Returns:
            Tensor: The next point.
        """
        proj_grad = grad - (torch.sum(grad) / self.n_ts)
        lt_zero = torch.nonzero(proj_grad < 0)
        lt_zero = lt_zero.view(lt_zero.numel())
        gt_zero = torch.nonzero(proj_grad > 0)
        gt_zero = gt_zero.view(gt_zero.numel())
        tm1 = -cur_val[lt_zero] / proj_grad[lt_zero]
        tm2 = (1.0 - cur_val[gt_zero]) / proj_grad[gt_zero]

        t = torch.tensor(1.0, device=grad.device)
        tm1_gt_zero = torch.nonzero(tm1 > 1e-7)
        tm1_gt_zero = tm1_gt_zero.view(tm1_gt_zero.numel())
        if tm1_gt_zero.shape[0] > 0:
            t = torch.min(tm1[tm1_gt_zero])

        tm2_gt_zero = torch.nonzero(tm2 > 1e-7)
        tm2_gt_zero = tm2_gt_zero.view(tm2_gt_zero.numel())
        if tm2_gt_zero.shape[0] > 0:
            t = torch.min(t, torch.min(tm2[tm2_gt_zero]))

        next_point = proj_grad * t + cur_val
        next_point = self.projection_to_simplex(next_point)
        return next_point

    @torch.no_grad()
    def forward(self, vecs):
        """
        General case solver using simplex projection algorithm.

        Args:
            vecs (Tensor): 2D tensor V, where each row is a vector Vi.

        Returns:
            Tensor: Coefficients c = [c1, ... cn] that solves the min-norm problem.
        """
        if self.n == 1:
            return vecs[0]
        if self.n == 2:
            v1v1 = torch.dot(vecs[0], vecs[0])
            v1v2 = torch.dot(vecs[0], vecs[1])
            v2v2 = torch.dot(vecs[1], vecs[1])
            self.two_sol[0], cost = self.linear_solver(v1v1, v1v2, v2v2)
            self.two_sol[1] = 1.0 - self.two_sol[0]
            return self.two_sol.clone()

        grammian = torch.mm(vecs, vecs.t())
        sol_vec = self.planar_solver(grammian)

        ii, jj = self.ii_grid, self.jj_grid
        for iter_count in range(self.max_iter):
            grad_dir = -torch.mv(grammian, sol_vec)
            new_point = self.next_point(sol_vec, grad_dir)

            v1v1 = (sol_vec[ii] * sol_vec[jj] * grammian[ii, jj]).sum()
            v1v2 = (sol_vec[ii] * new_point[jj] * grammian[ii, jj]).sum()
            v2v2 = (new_point[ii] * new_point[jj] * grammian[ii, jj]).sum()

            gamma, cost = self.linear_solver(v1v1, v1v2, v2v2)
            new_sol_vec = gamma * sol_vec + (1 - gamma) * new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < self.stop_crit:
                return sol_vec
            sol_vec = new_sol_vec
        return sol_vec

