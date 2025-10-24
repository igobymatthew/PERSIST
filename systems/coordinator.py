import inspect
from typing import Dict, Optional

import torch
import numpy as np

from components.fire_event import FireEvent
from utils.trainer_utils import log_episode_data, CurriculumScheduler
from multiagent.lineage import (
    BlendConfig,
    LineageArchive,
    LineageBlender,
    LineageMetadata,
    estimate_actor_fisher,
    estimate_viability_fisher,
)


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
        train_config = self.config["training"]
        self.num_episodes = train_config.get("num_episodes", 500)
        self.checkpoint_every = train_config.get("checkpoint_every", 10000)
        self.update_every = train_config["update_every"]
        self.amortize_after_steps = train_config.get("amortize_after_steps", 20000)
        self.consolidate_every = self.config.get("continual", {}).get(
            "consolidate_every", 5000
        )

        self.is_partially_observable = self.config.get("env", {}).get(
            "partial_observability", False
        )
        self.budget_decrement = (
            self.config.get("budgets", {}).get("decrement_per_step", 0.01)
            if hasattr(self, "budget_meter") and self.budget_meter
            else 0
        )

        self.scheduler = CurriculumScheduler(self.config)
        if self.scheduler.enabled:
            print("✅ Curriculum enabled for coordinator.")
        else:
            print("ℹ️ Curriculum disabled, using fixed parameters.")

        self._configure_life_stage_observers()

        self.lineage_archive: Optional[LineageArchive] = getattr(
            self, "lineage_archive", None
        )
        self.lineage_blender: Optional[LineageBlender] = getattr(
            self, "lineage_blender", None
        )
        self.lineage_config: Dict[str, object] = (
            getattr(self, "lineage_config", {}) or {}
        )
        self._blend_config = self._build_lineage_blend_config()
        self._lineage_species = self.lineage_config.get("species_id")

        print("✅ Experiment Coordinator initialized.")

    def _configure_life_stage_observers(self) -> None:
        """Wire life-stage providers, telemetry, and fire events into agents."""

        manager = getattr(self, "life_stage_manager", None)
        telemetry = getattr(self, "telemetry_manager", None)

        stage_provider = None
        if manager is not None and hasattr(manager, "current_stage_summary"):
            stage_provider = manager.current_stage_summary

        telemetry_hook = None
        if telemetry is not None and hasattr(telemetry, "update_life_stage"):
            telemetry_hook = telemetry.update_life_stage

        fire_event_register = None
        if manager is not None:
            fire_event_register = FireEvent.register_listener

        def _configure_stage_target(target) -> None:
            if target is None or not hasattr(target, "configure_stage_awareness"):
                return

            configure = getattr(target, "configure_stage_awareness")
            try:
                signature = inspect.signature(configure)
            except (TypeError, ValueError):
                signature = None

            kwargs = {}
            if signature is not None:
                params = signature.parameters
                if "stage_provider" in params and stage_provider is not None:
                    kwargs["stage_provider"] = stage_provider
                if "telemetry_hook" in params and telemetry_hook is not None:
                    kwargs["telemetry_hook"] = telemetry_hook
                if "fire_event_register" in params and fire_event_register is not None:
                    kwargs["fire_event_register"] = fire_event_register

            configure(**kwargs)

        _configure_stage_target(getattr(self, "agent", None))

        trainer = getattr(self, "trainer", None)
        _configure_stage_target(trainer)

        if trainer is not None:
            policies = getattr(trainer, "policies", None)
            if isinstance(policies, dict):
                for policy in policies.values():
                    _configure_stage_target(policy)

    def run(self):
        """
        Starts the main training loop and handles experiment-level coordination.
        """
        print("\n--- Starting Training Loop ---")

        if self.persistence_manager and self.persistence_manager.has_checkpoints():
            self._load_checkpoint()

        if hasattr(self, "telemetry_manager") and self.telemetry_manager:
            self.telemetry_manager.start_server()

        lambda_intr = self.config["rewards"]["lambda_intr"]
        lambda_homeo = (
            self.config["rewards"]["lambda_homeo"]
            if self.constraint_manager is None
            else 0.0
        )

        for episode in range(self.start_episode, self.num_episodes):
            external_obs = self.env.reset()
            if self.budget_meter:
                self.budget_meter.reset()

            if getattr(self, "life_stage_manager", None):
                stage_reset = self.life_stage_manager.reset()
                if stage_reset:
                    self.env.update_constraints(stage_reset.constraints_to_apply)
                    self._announce_life_stage_transition(stage_reset, initial=True)
                self._update_life_stage_metrics(0)

            self._initialize_lineage_for_episode(episode)

            true_internal_state = (
                external_obs[-self.env.internal_dim :]
                if not self.is_partially_observable
                else self.env.internal_state.copy()
            )
            estimated_internal_state = torch.zeros(
                self.env.internal_dim, dtype=torch.float32, device=self.device
            )
            estimator_hidden_state = None

            done = False
            (
                ep_len,
                total_task_reward,
                total_homeo_reward,
                total_intr_reward,
                ep_violations,
            ) = (0, 0, 0, 0, 0)
            ep_total_reward = 0.0

            while not done and ep_len < self.config["env"]["horizon"]:
                if getattr(self, "life_stage_manager", None):
                    self._maybe_transition_life_stage(ep_len)

                # print(f"[Coordinator] Step {ep_len}: Start.")
                obs_for_agent = (
                    np.concatenate(
                        [external_obs, estimated_internal_state.cpu().detach().numpy()]
                    )
                    if self.is_partially_observable
                    else external_obs
                )
                state_for_components = (
                    estimated_internal_state.cpu().detach().numpy()
                    if self.is_partially_observable
                    else true_internal_state
                )

                # print(f"[Coordinator] Step {ep_len}: Getting action...")
                safe_action, unsafe_action, step_telemetry_info = (
                    self.trainer.get_action(
                        external_obs,
                        obs_for_agent,
                        state_for_components,
                        self.total_steps,
                    )
                )
                # print(f"[Coordinator] Step {ep_len}: Got action. Stepping env...")

                next_external_obs, task_reward, done, info = self.env.step(safe_action)
                # print(f"[Coordinator] Step {ep_len}: Env stepped.")
                true_next_internal_state = (
                    info["internal_state"]
                    if self.is_partially_observable
                    else next_external_obs[-self.env.internal_dim :]
                )

                if info.get("fire_triggered"):
                    actor_model = getattr(self.agent, "actor", None)
                    if actor_model is None and hasattr(self.agent, "policy"):
                        actor_model = getattr(self.agent.policy, "actor", None)
                    fire_context = {"reason": "environment_signal"}
                    manager = getattr(self, "life_stage_manager", None)
                    if manager is not None:
                        summary = manager.current_stage_summary()
                        if summary is not None:
                            fire_context["stage"] = summary
                    FireEvent.apply(actor_model, context=fire_context)
                    if self.continual_learning_manager and self.rehearsal_buffer:
                        self.continual_learning_manager.consolidate(
                            self.rehearsal_buffer
                        )

                if self.is_partially_observable:
                    with torch.no_grad():
                        obs_tensor = (
                            torch.as_tensor(
                                external_obs, dtype=torch.float32, device=self.device
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        act_tensor = (
                            torch.as_tensor(
                                safe_action, dtype=torch.float32, device=self.device
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        predicted_state, next_estimator_hidden_state = (
                            self.state_estimator(
                                obs_tensor, act_tensor, estimator_hidden_state
                            )
                        )
                        estimated_next_internal_state = predicted_state.squeeze(
                            0
                        ).squeeze(0)
                else:
                    estimated_next_internal_state = torch.as_tensor(
                        true_next_internal_state,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    next_estimator_hidden_state = None

                budget_exhausted = False
                if self.budget_meter:
                    self.budget_meter.decrement(self.budget_decrement)
                    if self.budget_meter.is_exhausted():
                        budget_exhausted = True
                        done = True
                        info["budget_exhausted"] = True

                homeo_reward = self.homeostat.reward(
                    estimated_next_internal_state.cpu().detach().numpy()
                )
                intr_reward = self.trainer.calculate_intrinsic_reward(
                    external_obs, safe_action, next_external_obs
                )
                total_reward = task_reward + lambda_intr * intr_reward

                if self.constraint_manager:
                    adaptive_penalty = self.constraint_manager.get_penalties(
                        estimated_next_internal_state,
                        torch.tensor(self.homeostat.mu, device=self.device),
                        torch.tensor(self.homeostat.w, device=self.device),
                    )
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
                        state_tensor = torch.as_tensor(
                            state_for_components,
                            dtype=torch.float32,
                            device=self.device,
                        )
                        predicted_margins = self.safety_probe(state_tensor)
                        self.safety_reporter.log_shield_decision(
                            step=self.total_steps,
                            internal_state=state_tensor,
                            unsafe_action=torch.as_tensor(
                                unsafe_action, dtype=torch.float32
                            ),
                            safe_action=torch.as_tensor(
                                safe_action, dtype=torch.float32
                            ),
                            probe_margins=predicted_margins,
                        )

                viability_label = (
                    1.0 if not (done and info.get("violation", False)) else 0.0
                )
                violations = info.get(
                    "internal_state_violation", np.zeros(self.env.internal_dim)
                )
                ep_violations += np.sum(violations > 0)
                constraint_margins = info.get(
                    "constraint_margins", np.zeros(self.env.num_constraints)
                )
                life_stage_index = -1.0
                if getattr(self, "life_stage_manager", None):
                    metrics = self.life_stage_manager.metrics(ep_len)
                    if metrics is not None:
                        life_stage_index = float(metrics.index)

                self.replay_buffer.store(
                    external_obs,
                    safe_action,
                    unsafe_action,
                    total_reward,
                    next_external_obs,
                    done,
                    true_internal_state,
                    true_next_internal_state,
                    viability_label,
                    violations,
                    constraint_margins,
                    life_stage_index,
                )

                if self.near_boundary_buffer:
                    with torch.no_grad():
                        margin_tensor = self.viability_approximator.get_margin(
                            estimated_next_internal_state
                        )
                        margin_value = (
                            margin_tensor.item()
                            if margin_tensor.ndim == 0
                            else margin_tensor.squeeze().item()
                        )
                    self.near_boundary_buffer.consider(
                        estimated_next_internal_state.detach(),
                        viability_label,
                        margin_value,
                    )

                if self.rehearsal_buffer:
                    self.rehearsal_buffer.add(obs_for_agent)

                total_task_reward += task_reward
                total_intr_reward += intr_reward

                (
                    external_obs,
                    true_internal_state,
                    estimated_internal_state,
                    estimator_hidden_state,
                ) = (
                    next_external_obs,
                    true_next_internal_state,
                    estimated_next_internal_state,
                    next_estimator_hidden_state,
                )
                ep_len += 1
                self.total_steps += 1

                if getattr(self, "life_stage_manager", None):
                    self._update_life_stage_metrics(ep_len)

                if hasattr(self, "telemetry_manager") and self.telemetry_manager:
                    self.telemetry_manager.update_on_step(step_telemetry_info)
                    self.telemetry_manager.update_sps(self.total_steps)

                if self.scheduler.enabled:
                    current_params = self.scheduler.get_current_values(self.total_steps)
                    lambda_homeo = current_params["lambda_homeo"]
                    lambda_intr = current_params["lambda_intr"]
                    self.env.update_constraints(current_params["constraints"])

                if (
                    self.shield.mode == "search"
                    and self.total_steps >= self.amortize_after_steps
                ):
                    print(
                        f"\n--- Switching shield to AMORTIZED mode at step {self.total_steps} ---"
                    )
                    self.shield.mode = "amortized"

                if self.total_steps % self.update_every == 0 and len(
                    self.replay_buffer
                ) > max(self.trainer.batch_size, self.trainer.sequence_length):
                    for _ in range(self.update_every):
                        self.trainer.update_models()

                if (
                    self.continual_learning_manager
                    and self.total_steps % self.consolidate_every == 0
                    and self.total_steps > 0
                ):
                    self.continual_learning_manager.consolidate(self.rehearsal_buffer)

                if (
                    self.persistence_manager
                    and self.total_steps % self.checkpoint_every == 0
                ):
                    self._save_checkpoint(episode)

            if self.meta_learner:
                self.meta_learner.episode_end()
                self.homeostat.mu = self.meta_learner.get_setpoints()

            if hasattr(self, "telemetry_manager") and self.telemetry_manager:
                self.telemetry_manager.update_on_episode_end(
                    episode_reward=ep_total_reward, episode_violations=ep_violations
                )

            log_episode_data(
                self.evaluator,
                episode,
                self.total_steps,
                ep_len,
                total_task_reward,
                total_homeo_reward,
                total_intr_reward,
                0.0,
                self.env,
                done,
                info,
            )

            final_stage_name = None
            final_stage_index = None
            metrics_dict: Dict[str, object] = {}
            manager = getattr(self, "life_stage_manager", None)
            if manager is not None:
                metrics = manager.metrics(ep_len)
                if metrics is not None:
                    final_stage_name = metrics.name
                    final_stage_index = metrics.index
                    metrics_dict = metrics.as_dict()
                    affect_targets = metrics_dict.get("affect_targets") or {}
                    metrics_dict["affect_targets"] = {
                        key: list(value) for key, value in affect_targets.items()
                    }

            termination_reason = (
                "budget_exhausted"
                if info.get("budget_exhausted")
                else ("violation" if info.get("violation") else "episode_end")
            )
            self._archive_lineage_snapshot(
                event="death" if info.get("violation") else "lifespan_end",
                stage_name=final_stage_name,
                stage_index=final_stage_index,
                reason=termination_reason,
                episode=episode,
                metrics=metrics_dict,
            )

        print("\n--- Training Finished ---")

    def _save_checkpoint(self, episode):
        """Saves the current state of the experiment."""
        # TODO: Add state dicts for other components like replay_buffer, schedulers, etc.
        if not self.persistence_manager:
            return

        state = {
            "episode": episode,
            "total_steps": self.total_steps,
            "agent_state_dict": self.agent.get_state(),
            "optimizer_state_dict": self.agent.get_optimizer_state(),
        }
        self.persistence_manager.save_checkpoint(state, self.total_steps)

    def _load_checkpoint(self):
        """Loads the latest checkpoint."""
        if not self.persistence_manager:
            return None

        state = self.persistence_manager.load_latest_checkpoint()
        if state:
            self.start_episode = state.get("episode", 0) + 1
            self.total_steps = state.get("total_steps", 0)
            self.agent.load_state(state.get("agent_state_dict"))
            if "optimizer_state_dict" in state:
                self.agent.load_optimizer_state(state["optimizer_state_dict"])
            print(
                f"--- Resumed from checkpoint at episode {self.start_episode}, step {self.total_steps} ---"
            )
        return state

    def _maybe_transition_life_stage(self, episode_step: int) -> None:
        manager = getattr(self, "life_stage_manager", None)
        if not manager:
            return
        transition = manager.advance(episode_step)
        if transition:
            self.env.update_constraints(transition.constraints_to_apply)
            self._announce_life_stage_transition(transition, initial=False)
        self._update_life_stage_metrics(episode_step)

    def _update_life_stage_metrics(self, episode_step: int) -> None:
        manager = getattr(self, "life_stage_manager", None)
        if not manager:
            return
        metrics = manager.metrics(episode_step)
        if metrics and hasattr(self, "telemetry_manager") and self.telemetry_manager:
            self.telemetry_manager.update_life_stage(metrics)

    def _announce_life_stage_transition(self, transition, *, initial: bool) -> None:
        manager = getattr(self, "life_stage_manager", None)
        if not manager:
            return
        stage_name = transition.new_stage.name
        if initial:
            print(f"🍼 Life stage initialized → {stage_name}")
            return

        previous_stage = (
            transition.previous_stage.name if transition.previous_stage else None
        )
        previous_index = (
            transition.index - 1
            if transition.previous_stage and transition.index is not None
            else None
        )
        if previous_stage is not None:
            self._archive_lineage_snapshot(
                event="stage_complete",
                stage_name=previous_stage,
                stage_index=previous_index,
                reason="life_stage_transition",
                metrics={"stage_duration": transition.previous_stage.duration},
            )

        print(f"🧬 Life stage transition → {stage_name} at step {self.total_steps}")
        actor_model = getattr(self.agent, "actor", None)
        if actor_model is None and hasattr(self.agent, "policy"):
            actor_model = getattr(self.agent.policy, "actor", None)
        if actor_model is not None:
            fire_context = {"reason": "life_stage_transition"}
            summary = manager.current_stage_summary()
            if summary is not None:
                fire_context["stage"] = summary
            FireEvent.apply(actor_model, context=fire_context)

    # ------------------------------------------------------------------
    # Lineage helpers
    # ------------------------------------------------------------------
    def _build_lineage_blend_config(self) -> BlendConfig:
        cfg = self.lineage_config or {}
        return BlendConfig(
            alpha=float(cfg.get("blend_alpha", 0.45)),
            lora_rank=int(cfg.get("lora_rank", 8)),
            max_ancestors=int(cfg.get("max_ancestors", 3)),
        )

    def _lineage_enabled(self) -> bool:
        return bool(
            self.lineage_archive
            and self.lineage_blender
            and self.lineage_config.get("enabled", False)
        )

    def _initialize_lineage_for_episode(self, episode: int) -> None:
        if not self._lineage_enabled():
            return
        if not self.lineage_archive or not self.lineage_archive.has_records():
            return

        manager = getattr(self, "life_stage_manager", None)
        stage_name = manager.current_stage_name() if manager else None
        limit = int(
            self.lineage_config.get("max_ancestors", self._blend_config.max_ancestors)
        )
        ancestors = list(
            self.lineage_archive.iter_records(
                species=self._lineage_species,
                stage=stage_name,
                limit=limit,
            )
        )
        if not ancestors:
            ancestors = list(
                self.lineage_archive.iter_records(
                    species=self._lineage_species,
                    limit=limit,
                )
            )
        if not ancestors:
            return

        self.lineage_blender.blend_agent(
            agent=self.agent,
            viability_model=getattr(self, "viability_approximator", None),
            safety_network=getattr(self, "safety_network", None),
            ancestors=ancestors,
            config=self._blend_config,
        )

    def _archive_lineage_snapshot(
        self,
        *,
        event: str,
        stage_name: Optional[str],
        stage_index: Optional[int] = None,
        reason: Optional[str] = None,
        episode: Optional[int] = None,
        metrics: Optional[Dict[str, object]] = None,
    ) -> None:
        if not self._lineage_enabled():
            return
        if not self.lineage_archive:
            return

        fisher = self._estimate_fisher_masks()
        metadata = LineageMetadata(
            species=self._lineage_species,
            stage=stage_name,
            stage_index=stage_index,
            event=event,
            reason=reason,
            episode=episode,
            total_steps=self.total_steps,
            environment_seed=self.config.get("seed"),
            metrics=metrics or {},
        )
        safety_state = (
            self.safety_network.state_dict()
            if getattr(self, "safety_network", None)
            else None
        )
        affect_state = self._export_affect_state()
        try:
            self.lineage_archive.record_snapshot(
                metadata=metadata,
                policy_state=(
                    self.agent.get_state() if hasattr(self.agent, "get_state") else None
                ),
                viability_state=(
                    self.viability_approximator.state_dict()
                    if getattr(self, "viability_approximator", None)
                    else None
                ),
                safety_state=safety_state,
                affect_state=affect_state,
                fisher_mask=fisher if fisher else None,
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - archival should not interrupt training
            print(f"⚠️ Failed to archive lineage snapshot: {exc}")

    def _export_affect_state(self) -> Optional[Dict[str, torch.Tensor]]:
        buffer = getattr(self.agent, "affect_buffer", None)
        if buffer is None:
            return None
        if hasattr(buffer, "state_dict"):
            state = buffer.state_dict()
        elif hasattr(buffer, "serialize"):
            state = buffer.serialize()
        elif hasattr(buffer, "__getstate__"):
            state = buffer.__getstate__()
        else:
            return None
        if isinstance(state, dict):
            processed: Dict[str, object] = {}
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    processed[key] = value.detach().cpu()
                else:
                    processed[key] = value
            return processed
        return None

    def _resolve_actor_model(self) -> Optional[torch.nn.Module]:
        actor = getattr(self.agent, "actor", None)
        if actor is not None:
            return actor
        policy = getattr(self.agent, "policy", None)
        if policy is not None and hasattr(policy, "actor"):
            return policy.actor
        return None

    def _estimate_fisher_masks(self) -> Dict[str, torch.Tensor]:
        if not self.lineage_config.get("estimate_fisher", True):
            return {}
        if not hasattr(self, "replay_buffer") or len(self.replay_buffer) == 0:
            return {}

        min_samples = int(self.lineage_config.get("fisher_min_samples", 256))
        if len(self.replay_buffer) < min_samples:
            return {}

        batch_size = int(self.lineage_config.get("fisher_batch_size", 64))
        num_batches = int(self.lineage_config.get("fisher_batches", 4))

        actor_batches = []
        viability_batches = []
        for _ in range(num_batches):
            batch = self.replay_buffer.sample_batch(batch_size=batch_size)
            obs = batch.get("obs")
            if obs is not None:
                actor_batches.append(obs)
            internal = batch.get("internal_state")
            labels = batch.get("viability_label")
            if internal is not None and labels is not None:
                viability_batches.append((internal, labels.unsqueeze(-1)))

        fisher: Dict[str, torch.Tensor] = {}
        actor = self._resolve_actor_model()
        if actor is not None and actor_batches:
            try:
                actor_fisher = estimate_actor_fisher(
                    actor, actor_batches, device=self.device
                )
                fisher.update(
                    {f"actor.{name}": tensor for name, tensor in actor_fisher.items()}
                )
            except Exception as exc:  # pragma: no cover - diagnostics only
                print(f"⚠️ Actor Fisher estimation failed: {exc}")

        viability_model = getattr(self, "viability_approximator", None)
        if viability_model is not None and viability_batches:
            try:
                viability_fisher = estimate_viability_fisher(
                    viability_model, viability_batches, device=self.device
                )
                fisher.update(viability_fisher)
            except Exception as exc:  # pragma: no cover
                print(f"⚠️ Viability Fisher estimation failed: {exc}")

        return {key: value.cpu() for key, value in fisher.items()}
