import numpy as np
import torch

class MultiAgentTrainer:
    """
    Orchestrates the training loop for a multi-agent system using CTDE.
    """
    def __init__(self, components):
        print("--- Initializing MultiAgentTrainer ---")
        # Unpack all components into attributes
        for key, value in components.items():
            setattr(self, key, value)

        self.num_agents = self.config['multiagent']['num_agents']
        self.agent_ids = self.env.agents
        self.num_episodes = self.config.get('num_episodes', 1000)
        self.batch_size = self.config['training']['batch_size']
        self.update_every = self.config['training']['update_every'] # In steps

        # Assuming homogeneous agents for now, using one policy
        self.policy = self.policies['default']

        print("✅ MultiAgentTrainer initialized.")

    def run(self):
        """
        Executes the main training loop.
        """
        print("\n--- Starting Multi-Agent Training Loop ---")
        total_steps = 0

        for episode in range(self.num_episodes):
            obs, info = self.env.reset()
            done = False
            ep_len, ep_rew = 0, 0

            while not done:
                # --- 1. Get Proposals and Safety Projections ---
                if self.resource_allocator and obs:
                    current_internal_states = {aid: self.env.internal_states[aid] for aid in obs.keys()}
                    quotas = self.resource_allocator.get_quotas(current_internal_states)
                    # TODO: Use quotas in reward or agent logic.

                proposed_actions = {}
                with torch.no_grad():
                    for i, agent_id in enumerate(obs.keys()):
                        obs_flat = self._flatten_obs(obs[agent_id])
                        proposed_actions[agent_id] = self.policy.get_action(obs_flat, agent_id=i, deterministic=False)

                if self.cbf_coupler and obs:
                    proposed_actions_t = {aid: torch.as_tensor(act, dtype=torch.float32, device=self.device) for aid, act in proposed_actions.items()}
                    agent_positions_t = {aid: torch.as_tensor(self.env.agent_positions[aid], dtype=torch.float32, device=self.device) for aid in obs.keys()}
                    safe_actions_t = self.cbf_coupler.project_actions(agent_positions_t, proposed_actions_t)
                    final_actions = {aid: act.cpu().numpy() for aid, act in safe_actions_t.items()}
                else:
                    final_actions = proposed_actions

                # --- 2. Step the environment ---
                next_obs, rewards, terminations, truncations, infos = self.env.step(final_actions)

                # --- 3. Store experience (with padding for dead agents) ---
                if obs:
                    zero_obs = {key: np.zeros(space.shape, dtype=space.dtype) for key, space in self.env.single_observation_space.spaces.items()}
                    zero_action = np.zeros(self.env.single_action_space.shape, dtype=self.env.single_action_space.dtype)

                    obs_to_store, action_to_store, next_obs_to_store = {}, {}, {}
                    for agent_id in self.agent_ids:
                        obs_to_store[agent_id] = obs.get(agent_id, zero_obs)
                        action_to_store[agent_id] = final_actions.get(agent_id, zero_action)
                        next_obs_to_store[agent_id] = next_obs.get(agent_id, zero_obs)

                    self.replay_buffer.store(obs_to_store, action_to_store, rewards, next_obs_to_store, terminations)

                # --- 4. Update for next iteration ---
                obs = next_obs
                done = terminations.get("__all__", False) or truncations.get("__all__", False)

                total_steps += 1
                ep_len += 1
                if rewards: ep_rew += sum(rewards.values())

                if total_steps > self.batch_size and total_steps % self.update_every == 0:
                    self._update_models()

            print(f"Episode {episode}: Length={ep_len}, TotalReward={ep_rew:.2f}, Steps={total_steps}")

    def _flatten_obs(self, obs_dict):
        """Flattens a dictionary observation into a single numpy array."""
        # This needs to be consistent with how the agent expects the observation.
        # For now, we'll just concatenate them in a fixed order.
        vision = obs_dict['vision'].flatten()
        x = obs_dict['x']
        neighbors = obs_dict['neighbors']
        time = obs_dict['time']
        return np.concatenate([vision, x, neighbors, time])

    def _update_models(self):
        """
        Samples a batch from the replay buffer and updates the shared policy.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1. Sample a batch of joint transitions
        batch = self.replay_buffer.sample(self.batch_size)

        # 2. Process the batch for the shared agent
        # We need to flatten the agent-keyed dictionaries into large tensors.
        # Each row will correspond to one agent's experience in a transition.
        obs_list, act_list, rew_list, next_obs_list, done_list, agent_id_list = [], [], [], [], [], []

        for i in range(self.batch_size): # Iterate through transitions in the batch
            for agent_idx, agent_id in enumerate(self.agent_ids):
                # Flatten the observation dictionary for this agent
                obs_dict = {k: v[i] for k, v in batch['obs'][agent_id].items()}
                next_obs_dict = {k: v[i] for k, v in batch['next_obs'][agent_id].items()}

                obs_list.append(self._flatten_obs_tensor(obs_dict))
                next_obs_list.append(self._flatten_obs_tensor(next_obs_dict))

                # Append other data
                act_list.append(batch['act'][agent_id][i])
                rew_list.append(batch['rew'][agent_id][i])
                done_list.append(batch['done'][agent_id][i])
                agent_id_list.append(agent_idx) # Use the integer index for embedding

        # Stack everything into a single batch for the agent
        processed_batch = {
            'obs': torch.stack(obs_list),
            'act': torch.stack(act_list),
            'rew': torch.stack(rew_list),
            'next_obs': torch.stack(next_obs_list),
            'done': torch.stack(done_list),
            'agent_ids': torch.tensor(agent_id_list, dtype=torch.long)
        }

        # 3. Call the agent's update method
        self.policy.update(processed_batch)

    def _flatten_obs_tensor(self, obs_dict):
        """Flattens a dictionary of observation tensors into a single tensor."""
        # Ensure a consistent order
        tensors = [
            obs_dict['vision'].flatten(),
            obs_dict['x'],
            obs_dict['neighbors'],
            obs_dict['time']
        ]
        return torch.cat(tensors)