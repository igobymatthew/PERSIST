import torch

from components.dreamer_world_model import RSSM, _gaussian_kl, DreamerWorldModel
from components.internal_model import InternalModel
from components.viability_approximator import ViabilityApproximator


def test_rssm_forward_shapes():
    rssm = RSSM(action_dim=2, embed_dim=5, stoch_dim=4, deter_dim=6, hidden_dim=8)
    state = rssm.init_state(batch_size=3, device=torch.device("cpu"))
    action = torch.zeros(3, 2)
    next_state = rssm.img_step(state, action)
    assert next_state.stoch.shape == (3, 4)
    embed = torch.randn(3, 5)
    post, prior = rssm.obs_step(state, action, embed)
    assert post.stoch.shape == (3, 4)
    assert prior.stoch.shape == (3, 4)


def test_gaussian_kl_zero_when_equal():
    mean = torch.zeros(2, 3)
    std = torch.ones(2, 3)
    kl = _gaussian_kl(mean, std, mean, std)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_imagination_rollout_shapes():
    obs_dim = 10
    action_dim = 3
    internal_dim = 3
    model = DreamerWorldModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config={
            "embed_dim": 8,
            "deter_dim": 12,
            "stoch_dim": 6,
            "rssm_hidden": 10,
            "model_hidden": 16,
            "shield_horizon": 4,
        },
        device=torch.device("cpu"),
    )
    internal_model = InternalModel(internal_dim=internal_dim, act_dim=action_dim)
    viability = ViabilityApproximator(internal_dim=internal_dim)

    initial_obs = torch.randn(obs_dim)
    initial_internal = torch.zeros(internal_dim)
    action_sequences = torch.randn(5, 4, action_dim)

    rollout = model.rollout_viability(
        initial_obs=initial_obs,
        initial_internal_state=initial_internal,
        action_sequences=action_sequences,
        internal_model=internal_model,
        viability_approximator=viability,
    )

    assert rollout["predicted_observations"].shape == (5, 4, obs_dim)
    assert rollout["predicted_internal_states"].shape == (5, 4, internal_dim)
    assert rollout["margins"].shape == (5, 4)
