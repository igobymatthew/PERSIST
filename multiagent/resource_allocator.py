import threading
import numpy as np

class ResourceAllocator:
    """
    Manages a pool of shared resources for multiple agents.

    This class is designed to be thread-safe to handle requests from
    multiple agent processes or threads that might exist in a more complex
    multi-agent setup.
    """
    def __init__(self, resource_config, allocator_config):
        """
        Initializes the ResourceAllocator.

        Args:
            resource_config (dict): A dictionary defining the available resources
                                    and their initial quantities.
                                    Example: {'food': 10}
            allocator_config (dict): Configuration for the allocation mode.
        """
        self._resources = resource_config.copy()
        self.mode = allocator_config.get('mode', 'none')
        self.alpha = allocator_config.get('alpha', 1.0) # For alpha-fairness
        self._lock = threading.Lock()
        print(f"✅ ResourceAllocator initialized in '{self.mode}' mode with resources: {self._resources}")

    def get_quotas(self, agent_states):
        """
        Calculates resource quotas for all agents based on the configured mode.

        Args:
            agent_states (dict): A dictionary mapping agent_id to their internal state.

        Returns:
            dict: A dictionary mapping agent_id to their resource quotas.
                  Example: {'agent_0': {'food': 1}, 'agent_1': {'food': 0}}
        """
        if self.mode == 'proportional':
            return self._get_proportional_quotas(agent_states)
        else: # 'none' or other modes not implemented
            return {agent_id: {} for agent_id in agent_states}

    def _get_proportional_quotas(self, agent_states):
        """
        Calculates quotas based on agent needs (e.g., lower energy gets higher priority for food).
        This implements a form of weighted proportional allocation.
        """
        with self._lock:
            quotas = {agent_id: {'food': 0} for agent_id in agent_states}
            available_food = self._resources.get('food', 0)
            if available_food <= 0 or not agent_states:
                return quotas

            # Calculate weights based on need (inverse of energy)
            weights = {}
            total_weight = 0
            for agent_id, state in agent_states.items():
                # state[0] is energy. Add a small epsilon to avoid division by zero.
                energy = state[0]
                # Alpha-fairness: weight = (1 - energy)^alpha
                weight = (1.0 - energy + 1e-6)**self.alpha
                weights[agent_id] = weight
                total_weight += weight

            if total_weight > 0:
                # Allocate integer units of food proportionally
                for agent_id in sorted(weights.keys()): # Sort for deterministic allocation
                    proportion = weights[agent_id] / total_weight
                    claim = int(np.floor(proportion * available_food))
                    quotas[agent_id]['food'] = claim

            return quotas

    def check_availability(self, resource, amount=1):
        """
        Checks if a certain amount of a resource is available.
        """
        with self._lock:
            return self._resources.get(resource, 0) >= amount

    def claim(self, resource, amount=1):
        """
        Claims a certain amount of a resource.
        This is an atomic operation.
        """
        with self._lock:
            if self._resources.get(resource, 0) >= amount:
                self._resources[resource] -= amount
                return True
            else:
                return False

    def release(self, resource, amount=1):
        """
        Releases a certain amount of a resource back into the pool.
        """
        with self._lock:
            self._resources[resource] = self._resources.get(resource, 0) + amount

    def get_all_resources(self):
        """
        Returns a copy of the current state of all resources.
        """
        with self._lock:
            return self._resources.copy()