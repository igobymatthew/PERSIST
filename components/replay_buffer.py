import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, internal_dim):
        self.capacity = capacity
        # Standard RL buffers
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32) # This stores the SAFE action
        self.unsafe_action_buf = np.zeros((capacity, action_dim), dtype=np.float32) # For the safety network
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)

        # Buffers for persistence components
        self.internal_state_buf = np.zeros((capacity, internal_dim), dtype=np.float32)
        self.next_internal_state_buf = np.zeros((capacity, internal_dim), dtype=np.float32)
        self.viability_label_buf = np.zeros(capacity, dtype=np.float32)

        self.ptr, self.size = 0, 0

    def store(self, obs, action, unsafe_action, reward, next_obs, done, internal_state, next_internal_state, viability_label):
        # Store standard data
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action # Safe action
        self.unsafe_action_buf[self.ptr] = unsafe_action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        # Store persistence-related data
        self.internal_state_buf[self.ptr] = internal_state
        self.next_internal_state_buf[self.ptr] = next_internal_state
        self.viability_label_buf[self.ptr] = viability_label

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            action=self.action_buf[idxs], # Safe action
            unsafe_action=self.unsafe_action_buf[idxs],
            reward=self.reward_buf[idxs],
            done=self.done_buf[idxs],
            # Add persistence data to the batch
            internal_state=self.internal_state_buf[idxs],
            next_internal_state=self.next_internal_state_buf[idxs],
            viability_label=self.viability_label_buf[idxs]
        )
        return batch

    def __len__(self):
        return self.size