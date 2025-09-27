import torch
import torch.nn as nn

class OODDetector(nn.Module):
    """
    Detects Out-of-Distribution (OOD) states using an energy-based method.

    This approach leverages the internal representation of a model (e.g., the
    ViabilityApproximator or a dedicated autoencoder) to calculate an "energy"
    score for a given state. States with high energy are considered OOD.

    The energy score is derived from the logits of a network. For a network
    f(x) that outputs logits for classification, the energy is -logsumexp(f(x)).
    In this implementation, we will use the output of the ViabilityApproximator's
    network as the basis for the energy score.
    """
    def __init__(self, model, threshold=-5.0):
        """
        Initializes the OODDetector.

        Args:
            model (nn.Module): The model to use for energy calculation. This should be
                               a model that processes the internal state `x` and has a
                               `forward` method returning logits. The ViabilityApproximator
                               is a good candidate.
            threshold (float): The energy threshold. States with an energy score
                               above this threshold will be flagged as OOD.
        """
        super().__init__()
        self.model = model
        self.threshold = threshold

    def get_energy(self, x):
        """
        Calculates the energy score for a given internal state.

        Args:
            x (torch.Tensor): The internal state tensor. Shape (batch_size, x_dim).

        Returns:
            torch.Tensor: The energy score for each state in the batch.
                          Shape (batch_size,).
        """
        # We need the logits from the model, not the final output (e.g., sigmoid)
        # Assuming the provided model's forward pass returns logits.
        logits = self.model(x)
        # Energy score is the negative log-sum-exp of the logits.
        energy = -torch.logsumexp(logits, dim=-1)
        return energy

    def is_ood(self, x):
        """
        Determines if a state is Out-of-Distribution.

        Args:
            x (torch.Tensor): The internal state tensor. Shape (batch_size, x_dim).

        Returns:
            torch.Tensor: A boolean tensor indicating whether each state is OOD.
                          Shape (batch_size,).
        """
        energy = self.get_energy(x)
        return energy > self.threshold

    def forward(self, x):
        """Convenience forward method to check for OOD."""
        return self.is_ood(x)