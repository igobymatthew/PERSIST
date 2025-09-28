import torch
import numpy as np

from agents.mpc_agent import MPCAgent
from utils.trainer_utils import CurriculumScheduler

class Trainer:
    """
    Handles model updates and action selection.
    """
    def __init__(self, components):
        print("--- Initializing Trainer ---")
        # Unpack all components into attributes
        for key, value in components.items():
            setattr(self, key, value)

        # Training parameters from config
        train_config = self.config['training']
        self.batch_size = train_config['batch_size']
        self.sequence_length = self.config.get('state_estimator', {}).get('sequence_length', 16)

        self.is_partially_observable = self.config.get('env', {}).get('partial_observability', False)

        print("✅ Trainer initialized.")

    def get_action(self, external_obs, obs_for_agent, state_for_components, total_steps):
        """
        Selects an action, applying OOD detection, shielding, and safety layers.
        """
        is_ood = False
        if self.ood_detector:
            state_tensor = torch.as_tensor(state_for_components, dtype=torch.float32, device=self.device).unsqueeze(0)
            if self.ood_detector.is_ood(state_tensor).any():
                is_ood = True

        if is_ood:
            safe_action = self.safe_fallback_policy.get_action().squeeze(0).cpu().numpy()
            unsafe_action = safe_action
        else:
            if total_steps > self.batch_size:
                if isinstance(self.agent, MPCAgent):
                    agent_input = {'obs': external_obs, 'internal': state_for_components}
                    unsafe_action = self.agent.step(agent_input)
                else:
                    unsafe_action = self.agent.step(obs_for_agent)
            else:
                unsafe_action = self.env.action_space.sample()

            shielded_action = self.shield.project(state_for_components, unsafe_action)

            if self.cbf_layer:
                state_tensor = torch.as_tensor(state_for_components, dtype=torch.float32, device=self.device)
                action_tensor = torch.as_tensor(shielded_action, dtype=torch.float32, device=self.device)
                linearized_dynamics = self.dynamics_adapter.get_linearized_dynamics(state_tensor, action_tensor)
                cbf_safe_action_tensor = self.cbf_layer(action_tensor, state_tensor, linearized_dynamics)
                safe_action = cbf_safe_action_tensor.cpu().detach().numpy()
            else:
                safe_action = shielded_action

        return safe_action, unsafe_action

    def calculate_intrinsic_reward(self, obs, act, next_obs):
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

    def update_models(self):
        # 1. Sample batch
        seq_batch = self.replay_buffer.sample_sequence_batch(self.batch_size, self.sequence_length)

        # All data is already on the correct device from the replay buffer
        obs_seq = seq_batch['obs_seq']
        act_seq = seq_batch['action_seq']
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
        flat_violations = violations_seq.reshape(-1, self.env.num_constraints)
        constraint_margins_seq = seq_batch['constraint_margins_seq']
        flat_constraint_margins = constraint_margins_seq.reshape(-1, self.env.num_constraints)


        self.internal_model.train_model(flat_est_internal, flat_act, flat_true_next_internal)

        # Train the safety probe
        if self.safety_probe:
            self.safety_probe.train_probe(flat_est_internal.detach(), flat_constraint_margins)

        if self.demonstration_buffer and len(self.demonstration_buffer) > 0:
            self.viability_approximator.train_on_demonstrations(self.demonstration_buffer, self.batch_size)
        self.viability_approximator.train_model(flat_est_internal, flat_viability_label)

        # Also train the ensemble models if they exist
        if self.viability_ensemble:
            for model in self.viability_ensemble:
                if self.demonstration_buffer and len(self.demonstration_buffer) > 0:
                    model.train_on_demonstrations(self.demonstration_buffer, self.batch_size)
                model.train_model(flat_est_internal, flat_viability_label)

        if self.constraint_manager:
            self.constraint_manager.update(flat_violations)

        self.safety_network.train_network(
            unsafe_actions=flat_unsafe_act,
            safe_actions=flat_act,
            internal_states=flat_est_internal
        )

        # 5. Update agent
        batch = self.replay_buffer.sample_batch(self.batch_size)

        agent_data = batch
        # If the environment is partially observable, the agent expects a concatenation
        # of the external observation and the internal state for its learning update.
        if self.is_partially_observable:
            obs = torch.cat([batch['obs'], batch['internal_state']], dim=1)
            next_obs = torch.cat([batch['next_obs'], batch['next_internal_state']], dim=1)

            # The agent's learn method expects a dictionary with specific keys.
            agent_data = {
                'obs': obs,
                'action': batch['action'],
                'reward': batch['reward'],
                'next_obs': next_obs,
                'done': batch['done']
            }

        # Calculate EWC penalty if continual learning is enabled
        ewc_penalty = 0.0
        if self.continual_learning_manager:
            ewc_penalty = self.continual_learning_manager.penalty()

        policy_entropy = self.agent.learn(data=agent_data, ewc_penalty=ewc_penalty)
        return policy_entropy