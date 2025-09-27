import torch
import torch.nn as nn

class StateEstimator(nn.Module):
    """
    A recurrent model (GRU) that estimates the internal state (e.g., energy, temp)
    from a sequence of external observations and actions.
    This is used when the environment has partial observability.
    """
    def __init__(self, obs_dim, act_dim, internal_dim, hidden_dim=128, n_layers=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.internal_dim = internal_dim
        self.hidden_dim = hidden_dim

        # The input to the GRU will be the concatenation of the observation and action
        self.gru = nn.GRU(
            input_size=obs_dim + act_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        # A linear layer to map the GRU's hidden state to the predicted internal state
        self.predictor = nn.Linear(hidden_dim, internal_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs_seq, act_seq, h_0=None):
        """
        Forward pass through the GRU.
        Args:
            obs_seq (Tensor): A sequence of observations, shape (batch, seq_len, obs_dim).
            act_seq (Tensor): A sequence of actions, shape (batch, seq_len, act_dim).
            h_0 (Tensor, optional): The initial hidden state. Defaults to None.
        Returns:
            predicted_state_seq (Tensor): Predicted internal states for the ENTIRE sequence.
            h_n (Tensor): The final hidden state.
        """
        # Concatenate observations and actions along the feature dimension
        x = torch.cat([obs_seq, act_seq], dim=-1)

        # Pass through GRU
        gru_out, h_n = self.gru(x, h_0)

        # Apply predictor to the entire sequence of GRU outputs
        # Reshape to (batch * seq_len, hidden_dim) to apply linear layer efficiently
        gru_out_reshaped = gru_out.contiguous().view(-1, self.hidden_dim)
        predicted_state_reshaped = self.predictor(gru_out_reshaped)

        # Reshape back to (batch, seq_len, internal_dim)
        predicted_state_seq = predicted_state_reshaped.view(obs_seq.size(0), obs_seq.size(1), self.internal_dim)

        return predicted_state_seq, h_n

    def train_estimator(self, obs_seq, act_seq, true_internal_state_seq):
        """
        A single training step for the estimator, now over the whole sequence.
        Args:
            obs_seq (Tensor): Shape (batch, seq_len, obs_dim).
            act_seq (Tensor): Shape (batch, seq_len, act_dim).
            true_internal_state_seq (Tensor): Shape (batch, seq_len, internal_dim).
        """
        self.optimizer.zero_grad()

        # Get predictions for the entire sequence
        predicted_state_seq, _ = self.forward(obs_seq, act_seq)

        # Calculate loss over the entire sequence
        loss = self.loss_fn(predicted_state_seq, true_internal_state_seq)
        loss.backward()
        self.optimizer.step()

        return loss.item()