import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.persist_agent import PersistAgent
from systems.persistence import PersistenceManager


def _assert_nested_equal(expected, actual):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(expected.keys()) == set(actual.keys())
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(expected) == len(actual)
        for expected_item, actual_item in zip(expected, actual):
            _assert_nested_equal(expected_item, actual_item)
    elif torch.is_tensor(expected):
        torch.testing.assert_close(expected, actual)
    else:
        assert expected == actual


def test_persist_agent_checkpoint_cycle(tmp_path):
    agent = PersistAgent(obs_dim=4, act_dim=2, act_limit=1.0)
    agent.to(torch.device("cpu"))

    persistence_manager = PersistenceManager(tmp_path)

    checkpoint = {
        'episode': 0,
        'total_steps': 1,
        'agent_state_dict': agent.get_state(),
        'optimizer_state_dict': agent.get_optimizer_state(),
    }

    expected_agent_state = {key: value.clone() for key, value in checkpoint['agent_state_dict'].items()}
    expected_optimizer_state = copy.deepcopy(checkpoint['optimizer_state_dict'])

    persistence_manager.save_checkpoint(checkpoint, checkpoint['total_steps'])
    assert persistence_manager.has_checkpoints()

    for param in agent.policy.actor.parameters():
        param.data.zero_()
    for optimizer in agent.get_optimizers().values():
        optimizer.state.clear()

    loaded = persistence_manager.load_latest_checkpoint()
    assert loaded is not None

    agent.load_state(loaded['agent_state_dict'])
    agent.load_optimizer_state(loaded['optimizer_state_dict'])

    reloaded_agent_state = agent.get_state()
    for key, expected_tensor in expected_agent_state.items():
        torch.testing.assert_close(expected_tensor, reloaded_agent_state[key])

    reloaded_optimizer_state = agent.get_optimizer_state()
    _assert_nested_equal(expected_optimizer_state, reloaded_optimizer_state)
