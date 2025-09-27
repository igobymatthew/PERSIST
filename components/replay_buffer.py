import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, internal_dim, num_constraints, device):
        self.capacity = capacity
        self.device = device

        # Standard RL buffers
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.unsafe_action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        # Buffers for persistence components
        self.internal_state_buf = np.zeros((capacity, internal_dim), dtype=np.float32)
        self.next_internal_state_buf = np.zeros((capacity, internal_dim), dtype=np.float32)
        self.viability_label_buf = np.zeros(capacity, dtype=np.float32)
        self.violations_buf = np.zeros((capacity, internal_dim), dtype=np.float32)
        self.constraint_margins_buf = np.zeros((capacity, num_constraints), dtype=np.float32)


        self.ptr, self.size = 0, 0

    def store(self, obs, action, unsafe_action, reward, next_obs, done, internal_state, next_internal_state, viability_label, violations, constraint_margins):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.unsafe_action_buf[self.ptr] = unsafe_action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.internal_state_buf[self.ptr] = internal_state
        self.next_internal_state_buf[self.ptr] = next_internal_state
        self.viability_label_buf[self.ptr] = viability_label
        self.violations_buf[self.ptr] = violations
        self.constraint_margins_buf[self.ptr] = constraint_margins

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
            k: torch.as_tensor(v[idxs], dtype=torch.float32, device=self.device)
            for k, v in self._get_all_buffers().items()
        }
        return batch

    def sample_sequence_batch(self, batch_size, seq_len):
        start_indices = np.random.randint(0, self.size - seq_len + 1, size=batch_size)
        seq_indices = start_indices[:, np.newaxis] + np.arange(seq_len)

        is_valid = ~np.any(self.done_buf[seq_indices[:, :-1]], axis=1)

        while not np.all(is_valid):
            num_invalid = batch_size - np.sum(is_valid)
            new_starts = np.random.randint(0, self.size - seq_len + 1, size=num_invalid)
            new_seq_indices = new_starts[:, np.newaxis] + np.arange(seq_len)
            seq_indices[~is_valid] = new_seq_indices
            is_valid = ~np.any(self.done_buf[seq_indices[:, :-1]], axis=1)

        phys_indices = (self.ptr - self.size + seq_indices) % self.capacity

        # Explicitly create the batch dictionary to ensure all keys, including the
        # recently added 'violations_seq', are included.
        all_buffers = self._get_all_buffers()
        batch = {}
        for k, v in all_buffers.items():
            batch[f"{k}_seq"] = torch.as_tensor(v[phys_indices], dtype=torch.float32, device=self.device)

        return batch

    def _get_all_buffers(self):
        """Helper to return a dictionary of all buffer arrays."""
        return dict(
            obs=self.obs_buf,
            next_obs=self.next_obs_buf,
            action=self.action_buf,
            unsafe_action=self.unsafe_action_buf,
            reward=self.reward_buf,
            done=self.done_buf,
            internal_state=self.internal_state_buf,
            next_internal_state=self.next_internal_state_buf,
            viability_label=self.viability_label_buf,
            violations=self.violations_buf,
            constraint_margins=self.constraint_margins_buf
        )

    def __len__(self):
        return self.size