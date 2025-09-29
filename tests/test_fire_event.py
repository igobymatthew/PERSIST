import torch
import torch.nn as nn

from components.fire_event import FireEvent


def test_fire_event_prunes_at_least_k_weights():
    layer = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [0.1, 0.2],
                    [0.2, 0.2],
                    [0.3, 0.4],
                ]
            )
        )

    prune_fraction = 0.5
    FireEvent.apply(layer, prune_fraction=prune_fraction, threshold_scale=1.0)

    zero_count = (layer.weight == 0).sum().item()
    expected_pruned = max(1, int(prune_fraction * layer.weight.numel()))

    assert zero_count >= expected_pruned
