import numpy as np
import torch
from gymnasium.spaces import Dict as DictSpace

class ReplayMA:
    """
    A replay buffer for multi-agent settings (CTDE).
    It stores joint transitions and samples them for centralized training.
    This implementation uses a dictionary of numpy arrays for efficient storage.
    """
    def __init__(self, capacity, obs_space, act_space, num_agents, agent_ids, device):
        """
        Initializes the multi-agent replay buffer.

        Args:
            capacity (int): The maximum number of transitions to store.
            obs_space (gym.spaces.Dict): The observation space for a single agent.
            act_space (gym.spaces.Box): The action space for a single agent.
            num_agents (int): The number of agents.
            agent_ids (list[str]): The list of agent IDs.
            device (torch.device): The device to move tensors to.
        """
        self.capacity = capacity
        self.device = device
        self.agent_ids = agent_ids
        self.num_agents = num_agents

        self.buffers = {}
        for agent_id in self.agent_ids:
            self.buffers[agent_id] = {}
            # Create buffers for each component of the observation space
            for key, space in obs_space.spaces.items():
                self.buffers[agent_id][f'obs_{key}'] = np.zeros((capacity, *space.shape), dtype=space.dtype)
                self.buffers[agent_id][f'next_obs_{key}'] = np.zeros((capacity, *space.shape), dtype=space.dtype)

            self.buffers[agent_id]['action'] = np.zeros((capacity, *act_space.shape), dtype=act_space.dtype)
            self.buffers[agent_id]['reward'] = np.zeros(capacity, dtype=np.float32)
            self.buffers[agent_id]['done'] = np.zeros(capacity, dtype=np.float32)

        self.ptr, self.size = 0, 0
        print(f"✅ ReplayMA initialized with capacity {capacity} for {num_agents} agents.")

    def store(self, obs, act, rew, next_obs, done):
        """
        Stores a joint transition, consisting of dictionaries keyed by agent_id.
        The trainer is responsible for ensuring all dicts contain the same agent keys.
        """
        for agent_id in self.agent_ids:
            # Store observations by unpacking the observation dictionary
            for key in obs[agent_id].keys():
                self.buffers[agent_id][f'obs_{key}'][self.ptr] = obs[agent_id][key]
                self.buffers[agent_id][f'next_obs_{key}'][self.ptr] = next_obs[agent_id][key]

            # Store action, reward, and done
            self.buffers[agent_id]['action'][self.ptr] = act[agent_id]
            self.buffers[agent_id]['reward'][self.ptr] = rew[agent_id]
            self.buffers[agent_id]['done'][self.ptr] = done[agent_id]

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Samples a batch of joint transitions and collates them into a dictionary of tensors.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': {aid: {} for aid in self.agent_ids},
            'next_obs': {aid: {} for aid in self.agent_ids},
            'act': {},
            'rew': {},
            'done': {}
        }

        for agent_id in self.agent_ids:
            agent_buffer = self.buffers[agent_id]
            for key, buffer_arr in agent_buffer.items():
                sampled_data = torch.as_tensor(buffer_arr[idxs], device=self.device)

                if key.startswith('obs_'):
                    batch['obs'][agent_id][key.replace('obs_', '')] = sampled_data
                elif key.startswith('next_obs_'):
                    batch['next_obs'][agent_id][key.replace('next_obs_', '')] = sampled_data
                elif key == 'action':
                    batch['act'][agent_id] = sampled_data
                elif key == 'reward':
                    batch['rew'][agent_id] = sampled_data
                elif key == 'done':
                    batch['done'][agent_id] = sampled_data

        return batch

    def __len__(self):
        return self.size