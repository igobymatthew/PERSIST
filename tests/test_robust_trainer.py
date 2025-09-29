import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.robust_trainer import RobustTrainer


class DummyActionSpace:
    def sample(self):
        return torch.zeros(1).numpy()


class DummyEnv:
    action_space = DummyActionSpace()
    internal_dim = 1
    action_dim = 1
    num_constraints = 1


class DummyReplayBuffer:
    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def sample_sequence_batch(self, batch_size, sequence_length):
        device = torch.device('cpu')
        obs_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        action_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        next_obs_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        internal_state_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        next_internal_state_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        unsafe_action_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        viability_label_seq = torch.zeros(batch_size, sequence_length, device=device)
        violations_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        constraint_margins_seq = torch.zeros(batch_size, sequence_length, 1, device=device)
        return {
            'obs_seq': obs_seq,
            'action_seq': action_seq,
            'next_obs_seq': next_obs_seq,
            'internal_state_seq': internal_state_seq,
            'next_internal_state_seq': next_internal_state_seq,
            'unsafe_action_seq': unsafe_action_seq,
            'viability_label_seq': viability_label_seq,
            'violations_seq': violations_seq,
            'constraint_margins_seq': constraint_margins_seq,
        }

    def sample_batch(self, batch_size):
        device = torch.device('cpu')
        zeros = torch.zeros(batch_size, 1, device=device)
        return {
            'obs': zeros,
            'next_obs': zeros,
            'internal_state': zeros,
            'next_internal_state': zeros,
            'action': zeros,
            'reward': torch.zeros(batch_size, device=device),
            'done': torch.zeros(batch_size, dtype=torch.bool, device=device),
        }


class DummyWorldModel:
    def train_model(self, *args, **kwargs):
        pass

    def encoder(self, obs):
        return torch.zeros(obs.shape[0], 1)

    def transition(self, z, action):
        return z


class DummyIntrinsicModule:
    def train_predictor(self, obs):
        pass


class DummyInternalModel:
    def train_model(self, *args, **kwargs):
        pass


class DummyViabilityApproximator:
    def train_on_demonstrations(self, *args, **kwargs):
        pass

    def train_model(self, *args, **kwargs):
        pass

    def get_margin(self, internal_states):
        return torch.zeros(internal_states.shape[0])


class DummySafetyNetwork:
    def train_network(self, *args, **kwargs):
        pass


class RecordingAgent:
    def __init__(self):
        self.learn_calls = 0
        self.last_adversary = None
        self.last_data = None

    def learn(self, data, ewc_penalty, adversary=None):
        self.learn_calls += 1
        self.last_adversary = adversary
        self.last_data = data
        return 0.0


class DummyNearBoundaryBuffer:
    def __len__(self):
        return 0


class DummyDemonstrationBuffer:
    def __len__(self):
        return 0


class DummyConstraintManager:
    def update(self, *args, **kwargs):
        pass


class DummySafetyProbe:
    def train_probe(self, *args, **kwargs):
        pass


def make_trainer(adversary):
    batch_size = 2
    sequence_length = 3
    components = {
        'config': {
            'training': {'batch_size': batch_size},
            'state_estimator': {'sequence_length': sequence_length},
            'rewards': {'intrinsic': 'rnd'},
            'env': {'partial_observability': False},
        },
        'device': torch.device('cpu'),
        'env': DummyEnv(),
        'replay_buffer': DummyReplayBuffer(batch_size, sequence_length),
        'world_model': DummyWorldModel(),
        'intrinsic_reward_module': DummyIntrinsicModule(),
        'internal_model': DummyInternalModel(),
        'viability_approximator': DummyViabilityApproximator(),
        'viability_ensemble': [],
        'safety_network': DummySafetyNetwork(),
        'agent': RecordingAgent(),
        'continual_learning_manager': None,
        'demonstration_buffer': DummyDemonstrationBuffer(),
        'near_boundary_buffer': DummyNearBoundaryBuffer(),
        'constraint_manager': None,
        'safety_probe': None,
        'adversary': adversary,
    }
    return RobustTrainer(components)


def test_update_models_passes_adversary_to_agent():
    adversary = object()
    trainer = make_trainer(adversary)

    trainer.update_models()

    assert trainer.agent.learn_calls == 1
    assert trainer.agent.last_adversary is adversary
