import torch
import numpy as np

from utils.trainer_utils import log_episode_data, CurriculumScheduler

class ExperimentCoordinator:
    """
    Orchestrates the training process, including the main training loop,
    and checkpointing.
    """
    def __init__(self, components, persistence_manager=None):
        """
        Initializes the ExperimentCoordinator.
        """
        print("--- Initializing Experiment Coordinator ---")
        for key, value in components.items():
            setattr(self, key, value)

        self.persistence_manager = persistence_manager
        self.start_episode = 0
        self.total_steps = 0

        # Unpack training parameters
        train_config = self.config['training']
        self.num_episodes = train_config.get('num_episodes', 500)
        self.checkpoint_every = train_config.get('checkpoint_every', 10000)
        self.update_every = train_config['update_every']
        self.amortize_after_steps = train_config.get('amortize_after_steps', 20000)
        self.consolidate_every = self.config.get('continual', {}).get('consolidate_every', 5000)

        self.is_partially_observable = self.config.get('env', {}).get('partial_observability', False)
        self.budget_decrement = self.config.get('budgets', {}).get('decrement_per_step', 0.01) if hasattr(self, 'budget_meter') and self.budget_meter else 0

        self.scheduler = CurriculumScheduler(self.config)
        if self.scheduler.enabled:
            print("✅ Curriculum enabled for coordinator.")
        else:
            print("ℹ️ Curriculum disabled, using fixed parameters.")

        print("✅ Experiment Coordinator initialized.")

    def run(self):
        """
        Starts the main training loop and handles experiment-level coordination.
        """
        print("\n--- Starting Training Loop ---")

        if self.persistence_manager and self.persistence_manager.has_checkpoints():
            self._load_checkpoint()

        if hasattr(self, 'telemetry_manager') and self.telemetry_manager:
            self.telemetry_manager.start_server()

        lambda_intr = self.config['rewards']['lambda_intr']
        lambda_homeo = self.config['rewards']['lambda_homeo'] if self.constraint_manager is None else 0.0

        for episode in range(self.start_episode, self.num_episodes):
            external_obs = self.env.reset()
            if self.budget_meter:
                self.budget_meter.reset()

            true_internal_state = external_obs[-self.env.internal_dim:] if not self.is_partially_observable else self.env.internal_state.copy()
            estimated_internal_state = torch.zeros(self.env.internal_dim, dtype=torch.float32, device=self.device)
            estimator_hidden_state = None

            done = False
            ep_len, total_task_reward, total_homeo_reward, total_intr_reward, ep_violations = 0, 0, 0, 0, 0
            ep_total_reward = 0.0

            while not done and ep_len < self.config['env']['horizon']:
                # print(f"[Coordinator] Step {ep_len}: Start.")
                obs_for_agent = np.concatenate([external_obs, estimated_internal_state.cpu().detach().numpy()]) if self.is_partially_observable else external_obs
                state_for_components = estimated_internal_state.cpu().detach().numpy() if self.is_partially_observable else true_internal_state

                # print(f"[Coordinator] Step {ep_len}: Getting action...")
                safe_action, unsafe_action, step_telemetry_info = self.trainer.get_action(external_obs, obs_for_agent, state_for_components, self.total_steps)
                # print(f"[Coordinator] Step {ep_len}: Got action. Stepping env...")

                next_external_obs, task_reward, done, info = self.env.step(safe_action)
                # print(f"[Coordinator] Step {ep_len}: Env stepped.")
                true_next_internal_state = info['internal_state'] if self.is_partially_observable else next_external_obs[-self.env.internal_dim:]

                if self.is_partially_observable:
                    with torch.no_grad():
                        obs_tensor = torch.as_tensor(external_obs, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                        act_tensor = torch.as_tensor(safe_action, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                        predicted_state, next_estimator_hidden_state = self.state_estimator(obs_tensor, act_tensor, estimator_hidden_state)
                        estimated_next_internal_state = predicted_state.squeeze(0).squeeze(0)
                else:
                    estimated_next_internal_state = torch.as_tensor(true_next_internal_state, dtype=torch.float32, device=self.device)
                    next_estimator_hidden_state = None

                budget_exhausted = False
                if self.budget_meter:
                    self.budget_meter.decrement(self.budget_decrement)
                    if self.budget_meter.is_exhausted():
                        budget_exhausted = True
                        done = True
                        info['budget_exhausted'] = True

                homeo_reward = self.homeostat.reward(estimated_next_internal_state.cpu().detach().numpy())
                intr_reward = self.trainer.calculate_intrinsic_reward(external_obs, safe_action, next_external_obs)
                total_reward = task_reward + lambda_intr * intr_reward

                if self.constraint_manager:
                    adaptive_penalty = self.constraint_manager.get_penalties(estimated_next_internal_state, torch.tensor(self.homeostat.mu, device=self.device), torch.tensor(self.homeostat.w, device=self.device))
                    total_reward -= adaptive_penalty.item()
                    total_homeo_reward += homeo_reward
                else:
                    total_reward += lambda_homeo * homeo_reward
                    total_homeo_reward += homeo_reward

                if self.budget_meter and budget_exhausted:
                    total_reward += self.budget_meter.get_penalty()

                ep_total_reward += total_reward

                if self.meta_learner:
                    self.meta_learner.step(total_reward, true_next_internal_state)

                if self.safety_reporter and self.safety_probe:
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(state_for_components, dtype=torch.float32, device=self.device)
                        predicted_margins = self.safety_probe(state_tensor)
                        self.safety_reporter.log_shield_decision(step=self.total_steps, internal_state=state_tensor, unsafe_action=torch.as_tensor(unsafe_action, dtype=torch.float32), safe_action=torch.as_tensor(safe_action, dtype=torch.float32), probe_margins=predicted_margins)

                viability_label = 1.0 if not (done and info.get('violation', False)) else 0.0
                violations = info.get('internal_state_violation', np.zeros(self.env.internal_dim))
                ep_violations += np.sum(violations > 0)
                constraint_margins = info.get('constraint_margins', np.zeros(self.env.num_constraints))
                self.replay_buffer.store(external_obs, safe_action, unsafe_action, total_reward, next_external_obs, done, true_internal_state, true_next_internal_state, viability_label, violations, constraint_margins)

                if self.near_boundary_buffer:
                    with torch.no_grad():
                        margin_tensor = self.viability_approximator.get_margin(estimated_next_internal_state)
                        margin_value = margin_tensor.item() if margin_tensor.ndim == 0 else margin_tensor.squeeze().item()
                    self.near_boundary_buffer.consider(
                        estimated_next_internal_state.detach(),
                        viability_label,
                        margin_value
                    )

                if self.rehearsal_buffer:
                    self.rehearsal_buffer.add(obs_for_agent)

                total_task_reward += task_reward
                total_intr_reward += intr_reward

                external_obs, true_internal_state, estimated_internal_state, estimator_hidden_state = next_external_obs, true_next_internal_state, estimated_next_internal_state, next_estimator_hidden_state
                ep_len += 1
                self.total_steps += 1

                if hasattr(self, 'telemetry_manager') and self.telemetry_manager:
                    self.telemetry_manager.update_on_step(step_telemetry_info)
                    self.telemetry_manager.update_sps(self.total_steps)

                if self.scheduler.enabled:
                    current_params = self.scheduler.get_current_values(self.total_steps)
                    lambda_homeo = current_params['lambda_homeo']
                    lambda_intr = current_params['lambda_intr']
                    self.env.update_constraints(current_params['constraints'])

                if self.shield.mode == 'search' and self.total_steps >= self.amortize_after_steps:
                    print(f"\n--- Switching shield to AMORTIZED mode at step {self.total_steps} ---")
                    self.shield.mode = 'amortized'

                if self.total_steps % self.update_every == 0 and len(self.replay_buffer) > max(self.trainer.batch_size, self.trainer.sequence_length):
                    for _ in range(self.update_every):
                        self.trainer.update_models()

                if self.continual_learning_manager and self.total_steps % self.consolidate_every == 0 and self.total_steps > 0:
                    self.continual_learning_manager.consolidate(self.rehearsal_buffer)

                if self.persistence_manager and self.total_steps % self.checkpoint_every == 0:
                    self._save_checkpoint(episode)

            if self.meta_learner:
                self.meta_learner.episode_end()
                self.homeostat.mu = self.meta_learner.get_setpoints()

            if hasattr(self, 'telemetry_manager') and self.telemetry_manager:
                self.telemetry_manager.update_on_episode_end(
                    episode_reward=ep_total_reward,
                    episode_violations=ep_violations
                )

            log_episode_data(self.evaluator, episode, self.total_steps, ep_len, total_task_reward, total_homeo_reward, total_intr_reward, 0.0, self.env, done, info)

        print("\n--- Training Finished ---")

    def _save_checkpoint(self, episode):
        """Saves the current state of the experiment."""
        # TODO: Add state dicts for other components like replay_buffer, schedulers, etc.
        if not self.persistence_manager: return

        state = {
            'episode': episode,
            'total_steps': self.total_steps,
            'agent_state_dict': self.agent.get_state(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
        }
        self.persistence_manager.save_checkpoint(state, self.total_steps)

    def _load_checkpoint(self):
        """Loads the latest checkpoint."""
        if not self.persistence_manager: return None

        state = self.persistence_manager.load_latest_checkpoint()
        if state:
            self.start_episode = state.get('episode', 0) + 1
            self.total_steps = state.get('total_steps', 0)
            self.agent.load_state(state.get('agent_state_dict'))
            if hasattr(self.agent, 'optimizer') and 'optimizer_state_dict' in state:
                self.agent.optimizer.load_state_dict(state['optimizer_state_dict'])
            print(f"--- Resumed from checkpoint at episode {self.start_episode}, step {self.total_steps} ---")
        return state