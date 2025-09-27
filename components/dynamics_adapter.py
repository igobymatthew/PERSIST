import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

class DynamicsAdapter(nn.Module):
    """
    A component that linearizes a learned, non-linear dynamics model
    (like the InternalModel) around a given state-action pair.

    The linearization provides the A and B matrices for a local state-space
    representation: x_{t+1} approx= A * x_t + B * a_t.

    This is a crucial component for control techniques like Control Barrier
    Functions (CBFs) that rely on linearized dynamics.
    """
    def __init__(self, internal_model):
        """
        Initializes the DynamicsAdapter.

        Args:
            internal_model (nn.Module): The learned internal dynamics model to be linearized.
                                        It should have a `predict_next` method that takes
                                        (x, a, o) and returns x_next.
        """
        super().__init__()
        self.internal_model = internal_model

    def get_linearized_dynamics(self, x, a):
        """
        Computes the linearized dynamics (A and B matrices) around the given
        state (x) and action (a).

        The function f(x, a) -> x_next is linearized as:
        A = ∂f/∂x
        B = ∂f/∂a

        Args:
            x (torch.Tensor): The current internal state tensor. Shape (batch_size, x_dim).
            a (torch.Tensor): The current action tensor. Shape (batch_size, a_dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - A (torch.Tensor): The state transition matrix. Shape (batch_size, x_dim, x_dim).
                - B (torch.Tensor): The control matrix. Shape (batch_size, x_dim, a_dim).
        """
        # Ensure models are in eval mode for jacobian calculation.
        self.internal_model.eval()

        # Wrap the model's prediction logic for the jacobian function.
        def model_func(state, action):
            return self.internal_model.predict_next(state, action)

        # jacobian function doesn't support batching directly, so we iterate.
        if x.dim() == 1: # Handle non-batched input
            x = x.unsqueeze(0)
            a = a.unsqueeze(0)

        A_list = []
        B_list = []

        for i in range(x.shape[0]):
            def single_model_func(state, action):
                 return self.internal_model.predict_next(state.unsqueeze(0), action.unsqueeze(0)).squeeze(0)

            jacobians = jacobian(single_model_func, (x[i], a[i]), create_graph=False)
            A_i = jacobians[0] # ∂x_next/∂x
            B_i = jacobians[1] # ∂x_next/∂a
            A_list.append(A_i)
            B_list.append(B_i)

        A = torch.stack(A_list)
        B = torch.stack(B_list)

        return A, B

    def forward(self, x, a):
        """Convenience forward method."""
        return self.get_linearized_dynamics(x, a)