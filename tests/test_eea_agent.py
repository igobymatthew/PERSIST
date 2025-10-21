import numpy as np
import pytest
import torch.nn as nn

from agents.eea_agent import ContextWeights, EmotionalEquilibriumAgent
from components.fire_event import FireEvent
from components.replay_buffer import ReplayBuffer


class DummyStageProvider:
    def __init__(self) -> None:
        self.stage_index = 0
        self.stages = [
            {
                "name": "juvenile",
                "index": 0,
                "affect_targets": {
                    "ratio": (0.55, 0.65),
                    "happiness": (0.25, 0.85),
                    "fear": (0.05, 0.45),
                },
            },
            {
                "name": "adult",
                "index": 1,
                "affect_targets": {
                    "ratio": (0.7, 0.85),
                    "happiness": (0.4, 0.9),
                    "fear": (0.05, 0.35),
                },
            },
        ]

    def set_stage(self, index: int) -> None:
        self.stage_index = index

    def __call__(self):
        return self.stages[self.stage_index]


def test_stage_regularization_enforces_ratio_bounds():
    provider = DummyStageProvider()
    telemetry_payloads = []
    agent = EmotionalEquilibriumAgent(
        stage_provider=provider, telemetry_hook=telemetry_payloads.append
    )

    result = agent.evaluate(happiness=0.1, fear=0.9)

    bounds = provider().get("affect_targets").get("ratio")
    assert bounds[0] <= result["adjusted_ratio"] <= bounds[1]
    assert telemetry_payloads, "Telemetry hook should receive stage payloads"
    last_payload = telemetry_payloads[-1]
    assert last_payload["name"] == "juvenile"
    assert pytest.approx(result["adjusted_ratio"]) == pytest.approx(
        last_payload["affect_state"]["ratio"]
    )


def test_stage_transition_reseeds_entropy_and_prior():
    provider = DummyStageProvider()
    agent = EmotionalEquilibriumAgent(stage_provider=provider)
    weights = ContextWeights(environmental=2.0, social=1.0, internal=0.5)

    for _ in range(6):
        agent.evaluate(0.7, 0.2, context_weights=weights)

    assert agent.modulation.entropy_buffer.history, "Pre-stage history expected"

    provider.set_stage(1)
    result = agent.evaluate(0.6, 0.2, context_weights=weights)
    assert result["context"]["stage"] == "adult"

    expected_prior = sum(provider.stages[1]["affect_targets"]["ratio"]) / 2
    assert agent.meta.equilibrium_prior == pytest.approx(expected_prior, abs=0.01)
    assert agent.modulation.entropy_buffer.history, "History should be reseeded"


def test_fire_event_recovery_reseeds_stage_memory():
    FireEvent.clear_listeners()
    provider = DummyStageProvider()
    telemetry_payloads = []
    agent = EmotionalEquilibriumAgent(stage_provider=provider)
    agent.configure_stage_awareness(
        stage_provider=provider,
        telemetry_hook=telemetry_payloads.append,
        fire_event_register=FireEvent.register_listener,
    )
    weights = ContextWeights(environmental=1.5, social=0.5, internal=0.25)

    for _ in range(5):
        agent.evaluate(0.65, 0.25, context_weights=weights)

    agent.meta.equilibrium_prior = 0.2
    stage_info = provider()

    FireEvent.apply(
        nn.Linear(1, 1, bias=False),
        prune_fraction=0.0,
        threshold_scale=1.0,
        context={"reason": "unit_test", "stage": stage_info},
    )

    midpoint = sum(stage_info["affect_targets"]["ratio"]) / 2
    assert agent.meta.equilibrium_prior == pytest.approx(midpoint, abs=0.01)
    assert agent.modulation.entropy_buffer.history

    FireEvent.clear_listeners()


def test_replay_buffer_includes_stage_index():
    buffer = ReplayBuffer(
        capacity=8,
        obs_dim=4,
        action_dim=2,
        internal_dim=3,
        num_constraints=1,
        device="cpu",
    )

    obs = np.zeros(4, dtype=np.float32)
    action = np.zeros(2, dtype=np.float32)
    next_obs = np.ones(4, dtype=np.float32)
    internal = np.zeros(3, dtype=np.float32)
    next_internal = np.ones(3, dtype=np.float32)
    violations = np.zeros(1, dtype=np.float32)
    margins = np.ones(1, dtype=np.float32)

    buffer.store(
        obs,
        action,
        action,
        1.0,
        next_obs,
        0.0,
        internal,
        next_internal,
        1.0,
        violations,
        margins,
        life_stage_index=2,
    )

    batch = buffer.sample_batch(batch_size=1)
    assert batch["life_stage_index"].shape == (1,)
    assert batch["life_stage_index"][0].item() == pytest.approx(2.0)
