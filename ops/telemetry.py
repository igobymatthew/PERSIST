import time
from prometheus_client import start_http_server, Gauge, Counter

class TelemetryManager:
    """
    Manages and exposes operational metrics for Prometheus.
    """
    def __init__(self, config):
        """
        Initializes the TelemetryManager and defines Prometheus metrics.
        """
        self.config = config.get('telemetry', {})
        self.enabled = self.config.get('enabled', False)
        if not self.enabled:
            return

        self.port = self.config.get('port', 8000)
        self._total_steps_in_episode = 0
        self._shield_triggers_in_episode = 0
        self._cbf_interventions_in_episode = 0
        self._ood_detections_in_episode = 0


        # --- Define Prometheus Metrics ---

        # Gauges (value can go up or down)
        self.survival_steps_gauge = Gauge('persist_survival_steps', 'Number of steps the agent survived in the last episode.')
        self.episode_reward_gauge = Gauge('persist_episode_reward', 'Total reward achieved in the last episode.')
        self.policy_entropy_gauge = Gauge('persist_policy_entropy', 'Entropy of the policy distribution.')
        self.steps_per_second_gauge = Gauge('persist_steps_per_second', 'Training steps per second.')
        self.near_boundary_density_gauge = Gauge('persist_near_boundary_density', 'Number of samples collected near the viability boundary in the last batch.')
        self.shield_trigger_rate_gauge = Gauge('persist_shield_trigger_rate', 'Rate at which the safety shield was triggered in the last episode.')

        # Counters (value only goes up)
        self.episodes_total_counter = Counter('persist_episodes_total', 'Total number of episodes trained.')
        self.constraint_violations_counter = Counter('persist_constraint_violations_total', 'Total number of constraint violations across all episodes.')

        self.last_sps_update_time = time.time()
        self.last_sps_step_count = 0

    def start_server(self):
        """
        Starts the Prometheus HTTP server in a background thread.
        """
        if not self.enabled:
            return
        # This will start a daemon thread
        start_http_server(self.port)
        print(f"📈 Prometheus metrics server started on port {self.port}")

    def update_on_step(self, info):
        """
        Updates metrics that are tracked at each environment step.
        'info' is a dictionary that might contain:
        - 'shield_triggered': boolean
        - 'cbf_intervened': boolean
        - 'ood_detected': boolean
        """
        if not self.enabled:
            return

        self._total_steps_in_episode += 1
        if info.get('shield_triggered', False):
            self._shield_triggers_in_episode += 1
        if info.get('cbf_intervened', False):
            self._cbf_interventions_in_episode += 1
        if info.get('ood_detected', False):
            self._ood_detections_in_episode += 1


    def update_on_episode_end(self, episode_reward, episode_violations):
        """
        Updates metrics that are tracked at the end of an episode.
        """
        if not self.enabled:
            return

        self.episodes_total_counter.inc()
        self.survival_steps_gauge.set(self._total_steps_in_episode)
        self.episode_reward_gauge.set(episode_reward)
        self.constraint_violations_counter.inc(episode_violations)

        if self._total_steps_in_episode > 0:
            shield_rate = self._shield_triggers_in_episode / self._total_steps_in_episode
            self.shield_trigger_rate_gauge.set(shield_rate)

        # Reset per-episode counters
        self._total_steps_in_episode = 0
        self._shield_triggers_in_episode = 0
        self._cbf_interventions_in_episode = 0
        self._ood_detections_in_episode = 0

    def update_on_model_update(self, policy_entropy, near_boundary_samples_count):
        """
        Updates metrics after a model training step.
        """
        if not self.enabled:
            return

        self.policy_entropy_gauge.set(policy_entropy)
        self.near_boundary_density_gauge.set(near_boundary_samples_count)


    def update_sps(self, total_env_steps):
        """
        Updates the steps-per-second metric periodically.
        """
        if not self.enabled:
            return

        current_time = time.time()
        delta_time = current_time - self.last_sps_update_time
        delta_steps = total_env_steps - self.last_sps_step_count

        if delta_time > 2: # Update every 2 seconds to smooth the value
            sps = delta_steps / delta_time
            self.steps_per_second_gauge.set(sps)
            self.last_sps_update_time = current_time
            self.last_sps_step_count = total_env_steps