import torch
import numpy as np

from agents.mpc_agent import MPCAgent
from utils.trainer_utils import CurriculumScheduler, log_episode_data

class Trainer:
    def __init__(self, components):
        print("--- Initializing Trainer ---")
        # Unpack all components into attributes
        for key, value in components.items():
            setattr(self, key, value)

        # Training parameters from config
        train_config = self.config['train']
        self.batch_size = train_config['batch_size']
        self.update_every = train_config['update_every']
        self.amortize_after_steps = train_config.get('amortize_after_steps', 20000)
        self.num_episodes = self.config.get('num_episodes', 500)
        self.sequence_length = self.config.get('state_estimator', {}).get('sequence_length', 16)
        self.consolidate_every = self.config.get('continual', {}).get('consolidate_every', 5000)

        self.is_partially_observable = self.config.get('env', {}).get('partial_observability', False)

        self.scheduler = CurriculumScheduler(self.config)
        if self.scheduler.enabled:
            print("✅ Curriculum enabled for trainer.")
        else:
            print("ℹ️ Curriculum disabled, using fixed parameters.")

        print("✅ Trainer initialized.")

    def run(self):
        print("\n--- Starting Training Loop ---")
        total_steps = 0

        # Get initial lambda values
        lambda_intr = self.config['rewards']['lambda_intr']
        # lambda_homeo is now managed by ConstraintManager if enabled
        lambda_homeo = self.config['rewards']['lambda_homeo'] if self.constraint_manager is None else 0.0


        for episode in range(self.num_episodes):
            # Reset environment and estimator state
            external_obs = self.env.reset()

            if not self.is_partially_observable:
                true_internal_state = external_obs[-self.env.internal_dim:]
            else:
                true_internal_state = self.env.internal_state.copy()

            estimated_internal_state = torch.zeros(self.env.internal_dim, dtype=torch.float32, device=self.device)
            estimator_hidden_state = None

            done = False
            ep_len, total_task_reward, total_homeo_reward, total_intr_reward = 0, 0, 0, 0
            latest_policy_entropy = 0.0

            while not done and ep_len < self.config['env']['horizon']:
                # 1. Construct state
                if self.is_partially_observable:
                    obs_for_agent = np.concatenate([external_obs, estimated_internal_state.cpu().detach().numpy()])
                    state_for_components = estimated_internal_state.cpu().detach().numpy()
                else:
                    obs_for_agent = external_obs
                    state_for_components = true_internal_state

                # 2. OOD Check and Action Selection
                is_ood = False
                if self.ood_detector:
                    state_tensor = torch.as_tensor(state_for_components, dtype=torch.float32, device=self.device).unsqueeze(0)
                    if self.ood_detector.is_ood(state_tensor).any():
                        is_ood = True

                if is_ood:
                    # If OOD, use the fallback policy and bypass the agent and shield.
                    # The action is considered safe by definition.
                    safe_action = self.safe_fallback_policy.get_action().squeeze(0).cpu().numpy()
                    # For replay buffer consistency, we can set unsafe_action to the same.
                    unsafe_action = safe_action
                else:
                    # If not OOD, proceed with the normal agent and shield pipeline.
                    if total_steps > self.batch_size:
                        if isinstance(self.agent, MPCAgent):
                            agent_input = {'obs': external_obs, 'internal': state_for_components}
                            unsafe_action = self.agent.step(agent_input)
                        else:
                            unsafe_action = self.agent.step(obs_for_agent)
                    else:
                        unsafe_action = self.env.action_space.sample()

                    # 3. Shield projects action
                    shielded_action = self.shield.project(state_for_components, unsafe_action)

                    # 4. CBF layer provides final safety guarantee
                    if self.cbf_layer:
                        state_tensor = torch.as_tensor(state_for_components, dtype=torch.float32, device=self.device)
                        action_tensor = torch.as_tensor(shielded_action, dtype=torch.float32, device=self.device)

                        # Get linearized dynamics from the adapter
                        linearized_dynamics = self.dynamics_adapter.get_linearized_dynamics(state_tensor, action_tensor)

                        # Pass through CBF layer to get final action
                        cbf_safe_action_tensor = self.cbf_layer(action_tensor, state_tensor, linearized_dynamics)
                        safe_action = cbf_safe_action_tensor.cpu().detach().numpy()
                    else:
                        safe_action = shielded_action

                # 5. Environment steps
                next_external_obs, task_reward, done, info = self.env.step(safe_action)

                # 5. Get true next internal state
                if self.is_partially_observable:
                    true_next_internal_state = info['internal_state']
                else:
                    true_next_internal_state = next_external_obs[-self.env.internal_dim:]

                # 6. Estimate next internal state if needed
                if self.is_partially_observable:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(external_obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                        act_tensor = torch.as_tensor(safe_action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                        predicted_state, next_estimator_hidden_state = self.state_estimator(obs_tensor, act_tensor, estimator_hidden_state)
                        estimated_next_internal_state = predicted_state.squeeze(0).squeeze(0)
                else:
                    estimated_next_internal_state = torch.as_tensor(true_next_internal_state, dtype=torch.float32, device=self.device)
                    next_estimator_hidden_state = None

                # 7. Calculate rewards
                homeo_reward = self.homeostat.reward(estimated_next_internal_state.cpu().detach().numpy()) # For logging
                intr_reward = self._calculate_intrinsic_reward(external_obs, safe_action, next_external_obs)

                # The total reward for the agent's learning signal
                total_reward = task_reward + lambda_intr * intr_reward

                if self.constraint_manager:
                    # Use adaptive penalties from the manager instead of a fixed lambda
                    adaptive_penalty = self.constraint_manager.get_penalties(
                        estimated_next_internal_state,
                        torch.tensor(self.homeostat.mu, device=self.device),
                        torch.tensor(self.homeostat.w, device=self.device)
                    )
                    total_reward -= adaptive_penalty.item()
                    # For logging, we still want to know the "base" homeo reward
                    total_homeo_reward += homeo_reward
                else:
                    # Use the fixed lambda if manager is disabled
                    total_reward += lambda_homeo * homeo_reward
                    total_homeo_reward += homeo_reward


                if self.meta_learner:
                    self.meta_learner.step(total_reward, true_next_internal_state)

                # 8. Store experience
                viability_label = 1.0 if not (done and info.get('violation', False)) else 0.0
                violations = info.get('internal_state_violation', np.zeros(self.env.internal_dim))
                self.replay_buffer.store(external_obs, safe_action, unsafe_action, total_reward, next_external_obs, done, true_internal_state, true_next_internal_state, viability_label, violations)

                # Add state to rehearsal buffer for EWC
                if self.rehearsal_buffer:
                    self.rehearsal_buffer.add(obs_for_agent)

                total_task_reward += task_reward
                total_intr_reward += intr_reward

                # 9. Update states
                external_obs = next_external_obs
                true_internal_state = true_next_internal_state
                estimated_internal_state = estimated_next_internal_state
                estimator_hidden_state = next_estimator_hidden_state

                ep_len += 1
                total_steps += 1

                if self.scheduler.enabled:
                    current_params = self.scheduler.get_current_values(total_steps)
                    lambda_homeo = current_params['lambda_homeo']
                    lambda_intr = current_params['lambda_intr']
                    self.env.update_constraints(current_params['constraints'])

                if self.shield.mode == 'search' and total_steps >= self.amortize_after_steps:
                    print(f"\n--- Switching shield to AMORTIZED mode at step {total_steps} ---")
                    self.shield.mode = 'amortized'

                # 10. Update models and continual learning
                if total_steps % self.update_every == 0 and len(self.replay_buffer) > max(self.batch_size, self.sequence_length):
                    for _ in range(self.update_every):
                        latest_policy_entropy = self._update_models()

                # 11. Consolidate for EWC if enabled
                if self.continual_learning_manager and total_steps % self.consolidate_every == 0 and total_steps > 0:
                    self.continual_learning_manager.consolidate(self.rehearsal_buffer)

            # End of episode
            if self.meta_learner:
                self.meta_learner.episode_end()
                self.homeostat.mu = self.meta_learner.get_setpoints()

            log_episode_data(self.evaluator, episode, total_steps, ep_len, total_task_reward, total_homeo_reward, total_intr_reward, 0.0, self.env, done, info)

    def _calculate_intrinsic_reward(self, obs, act, next_obs):
        intrinsic_method = self.config['rewards']['intrinsic']
        if intrinsic_method == 'rnd':
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.intrinsic_reward_module.compute_intrinsic_reward(obs_tensor).item()
        elif intrinsic_method == 'empowerment':
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                z_t = self.world_model.encoder(obs_tensor)
                k = self.intrinsic_reward_module.k
                action_sequence = torch.stack([
                    torch.from_numpy(self.env.action_space.sample()) for _ in range(k)
                ]).float().unsqueeze(0).to(self.device)

                current_z = z_t
                for i in range(k):
                    action_for_rollout = action_sequence[:, i, :]
                    current_z = self.world_model.transition(current_z, action_for_rollout)
                z_t_plus_k = current_z

                return self.intrinsic_reward_module.compute_reward(z_t, action_sequence, z_t_plus_k).item()
        else: # 'surprise'
            return self.world_model.compute_surprise_reward(obs, act, next_obs)

    def _update_models(self):
        # 1. Sample batch
        seq_batch = self.replay_buffer.sample_sequence_batch(self.batch_size, self.sequence_length)

        # All data is already on the correct device from the replay buffer
        obs_seq = seq_batch['obs_seq']
        act_seq = seq_batch['act_seq']
        next_obs_seq = seq_batch['next_obs_seq']
        true_internal_state_seq = seq_batch['internal_state_seq']
        true_next_internal_state_seq = seq_batch['next_internal_state_seq']
        unsafe_act_seq = seq_batch['unsafe_action_seq']
        viability_label_seq = seq_batch['viability_label_seq']

        # 2. Update state estimator
        if self.is_partially_observable:
            act_seq_lagged = torch.cat([torch.zeros_like(act_seq[:, :1]), act_seq[:, :-1]], dim=1)
            with torch.no_grad():
                estimated_internal_state_seq, _ = self.state_estimator(obs_seq, act_seq_lagged)
            self.state_estimator.train_estimator(obs_seq, act_seq_lagged, true_internal_state_seq)
        else:
            estimated_internal_state_seq = true_internal_state_seq

        # 3. Update world model and intrinsic module
        self.world_model.train_model(obs_seq[:, 0], act_seq[:, 0], next_obs_seq[:, 0])
        intrinsic_method = self.config['rewards']['intrinsic']
        if intrinsic_method == 'rnd':
            self.intrinsic_reward_module.train_predictor(obs_seq[:, 0])
        elif intrinsic_method == 'empowerment':
            with torch.no_grad():
                z_batch = self.world_model.encoder(obs_seq[:, 0])
                k = self.intrinsic_reward_module.k
                action_sequences_batch = torch.stack([
                    torch.from_numpy(np.array([self.env.action_space.sample() for _ in range(k)]))
                    for _ in range(self.batch_size)
                ]).float().to(self.device)

                current_z_batch = z_batch
                for i in range(k):
                    actions_for_rollout = action_sequences_batch[:, i, :]
                    current_z_batch = self.world_model.transition(current_z_batch, actions_for_rollout)
                z_future_batch = current_z_batch
            self.intrinsic_reward_module.update(z_batch.detach(), action_sequences_batch, z_future_batch.detach())

        # 4. Update safety models
        flat_est_internal = estimated_internal_state_seq.reshape(-1, self.env.internal_dim)
        flat_true_next_internal = true_next_internal_state_seq.reshape(-1, self.env.internal_dim)
        flat_act = act_seq.reshape(-1, self.env.action_dim)
        flat_unsafe_act = unsafe_act_seq.reshape(-1, self.env.action_dim)
        flat_viability_label = viability_label_seq.reshape(-1)
        violations_seq = seq_batch['violations_seq']
        flat_violations = violations_seq.reshape(-1, self.env.internal_dim)


        self.internal_model.train_model(flat_est_internal, flat_act, flat_true_next_internal)

        if self.demonstration_buffer and len(self.demonstration_buffer) > 0:
            self.viability_approximator.train_on_demonstrations(self.demonstration_buffer, self.batch_size)
        self.viability_approximator.train_model(flat_est_internal, flat_viability_label)

        if self.constraint_manager:
            self.constraint_manager.update(flat_violations)

        self.safety_network.train_network(
            unsafe_actions=flat_unsafe_act,
            safe_actions=flat_act,
            internal_states=flat_est_internal
        )

        # 5. Update agent
        batch = self.replay_buffer.sample_batch(self.batch_size)

        # Calculate EWC penalty if continual learning is enabled
        ewc_penalty = 0.0
        if self.continual_learning_manager:
            ewc_penalty = self.continual_learning_manager.penalty()

        policy_entropy = self.agent.learn(data=batch, ewc_penalty=ewc_penalty)
        return policy_entropy