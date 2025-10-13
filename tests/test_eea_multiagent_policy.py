import numpy as np

from agents.eea_agent import EEAMultiAgentPolicy
from environments.multi_agent_gridlife import MultiAgentGridLifeEnv


def create_config(num_agents: int = 2):
    return {
        "seed": 123,
        "multiagent": {
            "num_agents": num_agents,
            "observation": {"view_radius": 2},
            "termination": {"max_steps": 25},
        },
        "agent_types": {
            "default": {
                "policy": "eea_agent",
                "homeostasis": {"mu": [0.7, 0.5, 0.9]},
            }
        },
        "resource_model": {
            "food": {"initial_density": 0.2},
        },
    }


def _flatten_obs(obs_dict):
    return np.concatenate(
        [
            obs_dict["vision"].flatten(),
            obs_dict["x"],
            obs_dict["neighbors"],
            obs_dict["time"],
        ]
    )


def test_eea_policy_produces_bounded_actions():
    config = create_config(num_agents=3)
    env = MultiAgentGridLifeEnv(config=config)
    policy = EEAMultiAgentPolicy(
        env, num_agents=config["multiagent"]["num_agents"], config=config
    )

    observations, _ = env.reset()
    for idx, agent_id in enumerate(sorted(observations.keys())):
        flat_obs = _flatten_obs(observations[agent_id])
        action = policy.get_action(flat_obs, agent_id=idx)
        assert action.shape == env.single_action_space.shape
        assert np.all(action <= 1.0) and np.all(action >= -1.0)


def test_eea_policy_rollout_runs_without_errors():
    config = create_config(num_agents=2)
    env = MultiAgentGridLifeEnv(config=config)
    policy = EEAMultiAgentPolicy(
        env, num_agents=config["multiagent"]["num_agents"], config=config
    )

    observations, _ = env.reset()
    for _ in range(8):
        actions = {}
        for idx, agent_id in enumerate(list(observations.keys())):
            flat_obs = _flatten_obs(observations[agent_id])
            actions[agent_id] = policy.get_action(flat_obs, agent_id=idx)
        observations, rewards, terminations, truncations, _ = env.step(actions)
        assert set(actions.keys()).issubset(env.agents)
        if terminations.get("__all__") or truncations.get("__all__"):
            break
