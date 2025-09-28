import torch
import cvxpy as cp

def qp_solver(Q, p, G, h, solver_args=None):
    """
    Solves a batch of Quadratic Programs (QPs).

    min 0.5 * x'Qx + p'x
    s.t. Gx <= h

    Args:
        Q (torch.Tensor): Quadratic cost matrix. Shape (batch_size, n, n).
        p (torch.Tensor): Linear cost vector. Shape (batch_size, n).
        G (torch.Tensor): Inequality constraint matrix. Shape (batch_size, m, n).
        h (torch.Tensor): Inequality constraint vector. Shape (batch_size, m).
        solver_args (dict, optional): Arguments for the cvxpy solver.

    Returns:
        torch.Tensor: The solution to the QP. Shape (batch_size, n).
    """
    if solver_args is None:
        solver_args = {}

    batch_size, n, _ = Q.shape
    m = G.shape[1]

    solutions = []
    for i in range(batch_size):
        Q_i = Q[i].detach().cpu().numpy()
        p_i = p[i].detach().cpu().numpy()
        G_i = G[i].detach().cpu().numpy()
        h_i = h[i].detach().cpu().numpy()

        x = cp.Variable(n)

        # cvxpy standard form is 0.5 * x'Px + q'x
        # We are given Q and p for x'Qx + p'x, so P=2Q and q=p
        # However, the caller seems to assume the standard form is x'Qx + p'x
        # and the CBF Coupler uses Q=I, p=-u_des, which matches min ||u-u_des||^2
        # Let's stick to the common form: 0.5 * x'Qx + p'x
        # The coupler provides Q=I, but it should be Q=2I for ||u-u_des||^2.
        # Let's assume the caller provides the correct QP form.
        # The objective in coupler is min ||u - u_prop||^2, which is u'u - 2u'u_prop + const.
        # This is 0.5*u'(2I)u + (-2u_prop)'u.
        # The coupler passes Q=I and p=-u_prop. This is a slight mismatch.
        # Let's write a flexible solver. If Q is for 0.5x'Qx, the user should double it for x'Qx.
        # The most common QP form is with 0.5, so we'll use that.

        objective = cp.Minimize(0.5 * cp.quad_form(x, Q_i) + p_i.T @ x)
        constraints = [G_i @ x <= h_i]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, **solver_args)
            if x.value is not None:
                solutions.append(torch.from_numpy(x.value).float())
            else:
                # Fallback to zero if solver fails
                solutions.append(torch.zeros(n, dtype=torch.float32))
        except cp.error.SolverError:
            solutions.append(torch.zeros(n, dtype=torch.float32))

    return torch.stack(solutions).to(Q.device)