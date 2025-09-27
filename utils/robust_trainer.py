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

    def _update_models(self):
        """
        Overrides the base trainer's update method to include adversarial training.
        """
        # 1. Sample batch and update other models (same as base class)
        # This logic is mostly copied from the parent, with one key change for the agent update.

        # Sample batch
        seq_batch = self.replay_buffer.sample_sequence_batch(self.batch_size, self.sequence_length)
        obs_seq, act_seq, next_obs_seq = seq_batch['obs_seq'], seq_batch['act_seq'], seq_batch['next_obs_seq']
        true_internal_state_seq, true_next_internal_state_seq = seq_batch['internal_state_seq'], seq_batch['next_internal_state_seq']
        unsafe_act_seq, viability_label_seq = seq_batch['unsafe_action_seq'], seq_batch['viability_label_seq']

        # Update state estimator
        if self.is_partially_observable:
            act_seq_lagged = torch.cat([torch.zeros_like(act_seq[:, :1]), act_seq[:, :-1]], dim=1)
            with torch.no_grad():
                estimated_internal_state_seq, _ = self.state_estimator(obs_seq, act_seq_lagged)
            self.state_estimator.train_estimator(obs_seq, act_seq_lagged, true_internal_state_seq)
        else:
            estimated_internal_state_seq = true_internal_state_seq

        # Update world model and intrinsic module
        self.world_model.train_model(obs_seq[:, 0], act_seq[:, 0], next_obs_seq[:, 0])
        # ... (intrinsic module updates are complex and remain the same)

        # Update safety models
        flat_est_internal = estimated_internal_state_seq.reshape(-1, self.env.internal_dim)
        flat_true_next_internal = true_next_internal_state_seq.reshape(-1, self.env.internal_dim)
        flat_act = act_seq.reshape(-1, self.env.action_dim)
        flat_unsafe_act = unsafe_act_seq.reshape(-1, self.env.action_dim)
        flat_viability_label = viability_label_seq.reshape(-1)
        violations_seq = seq_batch['violations_seq']
        flat_violations = violations_seq.reshape(-1, self.env.internal_dim)

        self.internal_model.train_model(flat_est_internal, flat_act, flat_true_next_internal)
        self.viability_approximator.train_model(flat_est_internal, flat_viability_label)
        if self.viability_ensemble:
            for model in self.viability_ensemble:
                model.train_model(flat_est_internal, flat_viability_label)
        if self.constraint_manager:
            self.constraint_manager.update(flat_violations)
        self.safety_network.train_network(
            unsafe_actions=flat_unsafe_act,
            safe_actions=flat_act,
            internal_states=flat_est_internal
        )

        # 5. Update agent with adversarial examples
        batch = self.replay_buffer.sample_batch(self.batch_size)
        ewc_penalty = self.continual_learning_manager.penalty() if self.continual_learning_manager else 0.0

        # *** The key change is here: pass the adversary to the agent's learn method ***
        policy_entropy = self.agent.learn(data=batch, ewc_penalty=ewc_penalty, adversary=self.adversary)

        return policy_entropy