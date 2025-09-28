import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from buffers.near_boundary import NearBoundaryBuffer, NearBoundaryConfig


def test_near_boundary_buffer_filters_by_margin():
    config = NearBoundaryConfig(capacity=4, margin_low=0.4, margin_high=0.6)
    buffer = NearBoundaryBuffer(state_dim=3, device=torch.device("cpu"), config=config)

    state = torch.tensor([0.5, 0.5, 0.5])
    assert buffer.consider(state, label=1.0, margin=0.5)
    assert len(buffer) == 1

    # Outside lower bound -> ignored
    assert not buffer.consider(state, label=1.0, margin=0.2)
    # Outside upper bound -> ignored
    assert not buffer.consider(state, label=0.0, margin=0.9)
    assert len(buffer) == 1


def test_near_boundary_buffer_sampling_with_replacement():
    config = NearBoundaryConfig(capacity=2, margin_low=0.3, margin_high=0.7)
    buffer = NearBoundaryBuffer(state_dim=2, device=torch.device("cpu"), config=config)

    buffer.consider(torch.tensor([0.1, 0.2]), label=0.0, margin=0.35)
    samples, labels = buffer.sample(batch_size=4)

    assert samples.shape == (4, 2)
    assert labels.shape == (4,)
    assert torch.allclose(labels, torch.zeros_like(labels))
