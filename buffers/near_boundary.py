"""Utilities for collecting near-boundary states for viability learning."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class NearBoundaryConfig:
    """Configuration options for the near-boundary buffer."""

    capacity: int = 2048
    margin_low: float = 0.35
    margin_high: float = 0.65

    def __post_init__(self) -> None:
        if not 0.0 <= self.margin_low <= 1.0:
            raise ValueError("margin_low must be between 0 and 1.")
        if not 0.0 <= self.margin_high <= 1.0:
            raise ValueError("margin_high must be between 0 and 1.")
        if self.margin_low >= self.margin_high:
            raise ValueError("margin_low must be strictly less than margin_high.")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive.")


class NearBoundaryBuffer:
    """Stores internal states that lie close to the viability decision boundary."""

    def __init__(
        self,
        state_dim: int,
        *,
        device: torch.device,
        config: NearBoundaryConfig,
    ) -> None:
        self.state_dim = state_dim
        self.device = device
        self.config = config

        self.states = torch.zeros(
            (config.capacity, state_dim), dtype=torch.float32, device=self.device
        )
        self.labels = torch.zeros(config.capacity, dtype=torch.float32, device=self.device)
        self.margins = torch.zeros(config.capacity, dtype=torch.float32, device=self.device)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def consider(self, state: torch.Tensor | np.ndarray, label: float, margin: float) -> bool:
        """Store ``state`` if its predicted safety ``margin`` lies near the boundary."""
        margin_value = float(margin)
        if math.isnan(margin_value):
            return False
        if margin_value < self.config.margin_low or margin_value > self.config.margin_high:
            return False

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.ndim > 1:
            state_tensor = state_tensor.view(-1)
        if state_tensor.numel() != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, received {state_tensor.numel()}"
            )

        label_tensor = torch.as_tensor(label, dtype=torch.float32, device=self.device)
        margin_tensor = torch.as_tensor(margin_value, dtype=torch.float32, device=self.device)

        self.states[self._ptr] = state_tensor
        self.labels[self._ptr] = label_tensor
        self.margins[self._ptr] = margin_tensor

        self._ptr = (self._ptr + 1) % self.config.capacity
        self._size = min(self._size + 1, self.config.capacity)
        return True

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample stored near-boundary states and their labels."""
        if self._size == 0:
            empty = torch.empty(0, self.state_dim, dtype=torch.float32, device=self.device)
            return empty, torch.empty(0, dtype=torch.float32, device=self.device)

        replace = self._size < batch_size
        sample_size = batch_size if replace else min(batch_size, self._size)
        indices = np.random.choice(self._size, size=sample_size, replace=replace)
        idx_tensor = torch.as_tensor(indices, device=self.device)
        return self.states[idx_tensor], self.labels[idx_tensor]

    def to(self, device: torch.device) -> "NearBoundaryBuffer":
        """Move the buffer storage to ``device``."""
        self.device = device
        self.states = self.states.to(device)
        self.labels = self.labels.to(device)
        self.margins = self.margins.to(device)
        return self
