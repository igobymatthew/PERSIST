import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.factory import ComponentFactory
from multiagent.resource_allocator import ResourceAllocator


def test_factory_creates_configured_resource_allocator():
    """Ensure the factory forwards allocator settings from the config file."""

    config_path = pathlib.Path(__file__).resolve().parents[1] / "config.yaml"
    factory = ComponentFactory(config_path=str(config_path))
    allocator_cfg = factory.config['resource_allocator']
    allocator_cfg.update({
        'mode': 'proportional',
        'alpha': 1.7,
        'fairness': 'alpha_fair',
        'tax': 0.05,
        'conflict_resolution': 'priority',
    })

    allocator = factory.create_resource_allocator()

    assert isinstance(allocator, ResourceAllocator)
    assert allocator.mode == allocator_cfg['mode']
    assert allocator.alpha == allocator_cfg['alpha']
    assert allocator.fairness == allocator_cfg['fairness']
    assert pytest.approx(allocator.tax) == allocator_cfg['tax']
    assert allocator.conflict_resolution == allocator_cfg['conflict_resolution']

    resources = allocator.get_all_resources()
    assert 'food' in resources

