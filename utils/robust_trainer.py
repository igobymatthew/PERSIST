import numpy as np
import torch
from utils.trainer import Trainer

class RobustTrainer(Trainer):
    """
    A Trainer that incorporates adversarial training to make the agent more robust.
    """
    def __init__(self, components):
        super().__init__(components)
        if hasattr(self, 'adversary'):
            print("✅ RobustTrainer initialized with Adversary component.")
        else:
            print("⚠️ RobustTrainer initialized, but no Adversary component was found.")

    def update_models(self):
        """Run a full model update while injecting the adversarial learn step."""

        # 1. Sample sequence batch for model updates
        seq_batch = self.replay_buffer.sample_sequence_batch(self.batch_size, self.sequence_length)

        obs_seq = seq_batch['obs_seq']
        act_seq = seq_batch['action_seq']
        next_obs_seq = seq_batch['next_obs_seq']
        true_internal_state_seq = seq_batch['internal_state_seq']
        true_next_internal_state_seq = seq_batch['next_internal_state_seq']
        unsafe_act_seq = seq_batch['unsafe_action_seq']
        viability_label_seq = seq_batch['viability_label_seq']

        # 2. Update the state estimator if necessary
        if self.is_partially_observable:
            act_seq_lagged = torch.cat([torch.zeros_like(act_seq[:, :1]), act_seq[:, :-1]], dim=1)
            with torch.no_grad():
                estimated_internal_state_seq, _ = self.state_estimator(obs_seq, act_seq_lagged)
            self.state_estimator.train_estimator(obs_seq, act_seq_lagged, true_internal_state_seq)
        else:
            estimated_internal_state_seq = true_internal_state_seq

        # 3. Update world model and intrinsic reward module (mirrors base implementation)
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
            self.intrinsic_reward_module.update(
                z_batch.detach(), action_sequences_batch, z_future_batch.detach()
            )

        # 4. Update safety-related components
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

        if self.safety_probe:
            self.safety_probe.train_probe(flat_est_internal.detach(), flat_constraint_margins)

        if self.demonstration_buffer and len(self.demonstration_buffer) > 0:
            self.viability_approximator.train_on_demonstrations(self.demonstration_buffer, self.batch_size)
        self.viability_approximator.train_model(flat_est_internal, flat_viability_label)

        if self.viability_ensemble:
            for model in self.viability_ensemble:
                if self.demonstration_buffer and len(self.demonstration_buffer) > 0:
                    model.train_on_demonstrations(self.demonstration_buffer, self.batch_size)
                model.train_model(flat_est_internal, flat_viability_label)

        if self.near_boundary_buffer and len(self.near_boundary_buffer) > 0:
            nb_states, nb_labels = self.near_boundary_buffer.sample(self.batch_size)
            if nb_states.numel() > 0:
                self.viability_approximator.train_model(nb_states, nb_labels)
                if self.viability_ensemble:
                    for model in self.viability_ensemble:
                        model.train_model(nb_states, nb_labels)

        if self.constraint_manager:
            self.constraint_manager.update(flat_violations)

        self.safety_network.train_network(
            unsafe_actions=flat_unsafe_act,
            safe_actions=flat_act,
            internal_states=flat_est_internal
        )

        # 5. Prepare agent batch and include adversary in learning call
        batch = self.replay_buffer.sample_batch(self.batch_size)

        agent_data = batch
        if self.is_partially_observable:
            obs = torch.cat([batch['obs'], batch['internal_state']], dim=1)
            next_obs = torch.cat([batch['next_obs'], batch['next_internal_state']], dim=1)
            agent_data = {
                'obs': obs,
                'action': batch['action'],
                'reward': batch['reward'],
                'next_obs': next_obs,
                'done': batch['done']
            }

        ewc_penalty = 0.0
        if self.continual_learning_manager:
            ewc_penalty = self.continual_learning_manager.penalty()

        policy_entropy = self.agent.learn(
            data=agent_data,
            ewc_penalty=ewc_penalty,
            adversary=getattr(self, 'adversary', None)
        )

        if hasattr(self, 'telemetry_manager') and self.telemetry_manager:
            with torch.no_grad():
                margins = self.viability_approximator.get_margin(flat_est_internal)
                near_boundary_mask = (margins > 0.1) & (margins < 0.9)
                near_boundary_samples_count = near_boundary_mask.sum().item()

            self.telemetry_manager.update_on_model_update(
                policy_entropy=policy_entropy,
                near_boundary_samples_count=near_boundary_samples_count
            )

        return policy_entropy
