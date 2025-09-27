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

    def sample_sequence_batch(self, batch_size, seq_len):
        """
        Samples a batch of consecutive sequences for training recurrent models.
        This version is more robust and returns all necessary data fields.
        """
        start_indices = np.random.randint(0, self.size - seq_len, size=batch_size)

        # Create sequence indices for each start index
        # Shape: (batch_size, seq_len)
        seq_indices = start_indices[:, np.newaxis] + np.arange(seq_len)

        # Check for episode boundaries within each sequence (excluding the very last step)
        # A sequence is invalid if a 'done' flag appears in the first seq_len-1 steps
        is_valid = ~np.any(self.done_buf[seq_indices[:, :-1]], axis=1)

        # Keep resampling invalid sequences until all are valid
        while not np.all(is_valid):
            num_invalid = batch_size - np.sum(is_valid)
            new_starts = np.random.randint(0, self.size - seq_len, size=num_invalid)

            # Get new sequence indices for the invalid entries
            new_seq_indices = new_starts[:, np.newaxis] + np.arange(seq_len)

            # Replace the old invalid indices
            seq_indices[~is_valid] = new_seq_indices

            # Re-evaluate validity
            is_valid = ~np.any(self.done_buf[seq_indices[:, :-1]], axis=1)

        # Convert logical indices to physical buffer indices
        phys_indices = (self.ptr - self.size + seq_indices) % self.capacity

        # Retrieve all necessary data sequences
        batch = dict(
            obs_seq=self.obs_buf[phys_indices],
            act_seq=self.action_buf[phys_indices],
            reward_seq=self.reward_buf[phys_indices],
            next_obs_seq=self.next_obs_buf[phys_indices],
            done_seq=self.done_buf[phys_indices],
            internal_state_seq=self.internal_state_buf[phys_indices],
            next_internal_state_seq=self.next_internal_state_buf[phys_indices],
            unsafe_action_seq=self.unsafe_action_buf[phys_indices],
            viability_label_seq=self.viability_label_buf[phys_indices]
        )
        return batch

    def __len__(self):
        return self.size