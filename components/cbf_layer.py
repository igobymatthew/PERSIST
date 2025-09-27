import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np

class CBFLayer(nn.Module):
    """
    A Control Barrier Function (CBF) layer that ensures system safety by
    modifying a desired control input to satisfy safety constraints.

    It solves a Quadratic Program (QP) to find a safe action `a_safe` that is
    as close as possible to the desired action `a_des` from the policy.

    The core of the CBF is the safety constraint:
    L_f h(x) + L_g h(x)u >= -alpha(h(x))

    where h(x) is the barrier function representing the safety margin,
    (L_f h, L_g h) are its Lie derivatives along the system dynamics, and alpha
    is a class-K function (here, a linear function alpha(h) = gamma * h).

    This implementation assumes the viability constraints are defined as g_i(x) <= 0.
    The barrier function for each constraint is h_i(x) = -g_i(x).
    """
    def __init__(self, x_dim, a_dim, constraints, relax_penalty=10.0, delta=1.0):
        """
        Initializes the CBF Layer.

        Args:
            x_dim (int): Dimensionality of the internal state space.
            a_dim (int): Dimensionality of the action space.
            constraints (list[callable]): A list of callables, where each
                                         callable takes an internal state `x`
                                         and returns the value of the constraint
                                         function g_i(x).
            relax_penalty (float): The penalty weight for the slack variable in the QP,
                                   allowing for soft constraint violations.
            delta (float): The coefficient for the class-K function (alpha(h) = delta * h),
                           controlling how strongly the system is pushed away from the
                           boundary.
        """
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.constraints = constraints
        self.relax_penalty = relax_penalty
        self.delta = delta

    def _solve_qp(self, a_des, Lf_h, Lg_h, h_x):
        """
        Solves the CBF-QP for a single state-action pair.

        min_{a, slack} ||a - a_des||^2 + p * slack^2
        s.t. Lg_h @ a >= -Lf_h - delta * h_x - slack
             slack >= 0
        """
        a_des = a_des.detach().cpu().numpy()
        Lf_h = Lf_h.detach().cpu().numpy()
        Lg_h = Lg_h.detach().cpu().numpy()
        h_x = h_x.detach().cpu().numpy()

        # Define the optimization variables
        a = cp.Variable(self.a_dim)
        slack = cp.Variable(len(self.constraints), nonneg=True)

        # Define the objective function
        objective = cp.Minimize(cp.sum_squares(a - a_des) + self.relax_penalty * cp.sum_squares(slack))

        # Define the constraints
        constraints = [Lg_h @ a >= -Lf_h - self.delta * h_x - slack]

        # Formulate and solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            # Use OSQP for speed and robustness
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            if a.value is not None:
                return torch.from_numpy(a.value).float()
            else:
                # If solver fails, return the original unsafe action
                return torch.from_numpy(a_des).float()
        except cp.error.SolverError:
            # Handle cases where the solver fails catastrophically
            return torch.from_numpy(a_des).float()


    def forward(self, a_des, x, linearized_dynamics):
        """
        Computes the safe action by solving the CBF-QP.

        Args:
            a_des (torch.Tensor): The desired action from the policy. Shape (batch_size, a_dim).
            x (torch.Tensor): The current internal state. Shape (batch_size, x_dim).
            linearized_dynamics (tuple[torch.Tensor, torch.Tensor]): A tuple (A, B)
                containing the linearized dynamics matrices from the DynamicsAdapter.

        Returns:
            torch.Tensor: The safe action. Shape (batch_size, a_dim).
        """
        A, B = linearized_dynamics
        a_safe_list = []

        if x.dim() == 1: # Handle non-batched input
            x = x.unsqueeze(0)
            a_des = a_des.unsqueeze(0)
            A = A.unsqueeze(0)
            B = B.unsqueeze(0)

        for i in range(x.shape[0]):
            x_i = x[i]
            a_des_i = a_des[i]
            A_i = A[i]
            B_i = B[i]

            # We need the gradient of each constraint function h_i(x) = -g_i(x)
            # We compute jacobian of g(x) and then negate it.
            # This requires the constraint functions to be defined in a way that
            # autograd can track.
            x_i_grad = x_i.detach().requires_grad_(True)

            g_x = torch.stack([c(x_i_grad) for c in self.constraints])

            # h(x) = -g(x). For constraint g(x) <= 0, we want h(x) >= 0.
            h_x = -g_x

            # Compute grad_h = -grad_g
            # We need to compute the jacobian of g with respect to x.
            grad_g = torch.autograd.functional.jacobian(lambda x: torch.stack([c(x) for c in self.constraints]), x_i_grad)
            grad_h = -grad_g

            # Lie derivatives
            # Lf_h = grad_h @ f(x) where f(x) = A@x
            # Lg_h = grad_h @ g(x) where g(x) = B
            Lf_h = (grad_h @ A_i @ x_i.detach()).squeeze()
            Lg_h = (grad_h @ B_i).squeeze()

            a_safe_i = self._solve_qp(a_des_i, Lf_h, Lg_h, h_x.detach())
            a_safe_list.append(a_safe_i)

        a_safe = torch.stack(a_safe_list).to(a_des.device)
        return a_safe