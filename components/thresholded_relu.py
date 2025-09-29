import torch
import torch.nn as nn
import torch.nn.functional as F


class ThresholdedReLU(nn.Module):
    """ReLU with a tunable threshold parameter.

    Standard ReLU(x) = max(0, x).
    Here: ReLU(x - threshold), so the neuron fires only if x > threshold.

    FireEvent can modify `threshold` dynamically to increase/decrease plasticity.
    """

    def __init__(self, threshold=0.0, learnable=False):
        super().__init__()
        if learnable:
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32))
        else:
            self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x):
        return F.relu(x - self.threshold)

    def extra_repr(self):
        is_learnable = isinstance(self.threshold, nn.Parameter)
        return f"threshold={self.threshold.item():.4f}, learnable={is_learnable}"
