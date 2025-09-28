import threading
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        logger.info(f"ResourceAllocator initialized in '{self.mode}' mode with resources: {self._resources}")

    def get_quotas(self, agent_states):
        """
        Calculates resource quotas for all agents based on the configured mode.

        Args:
            agent_states (dict): A dictionary mapping agent_id to their internal state.

        Returns:
            dict: A dictionary mapping agent_id to their resource quotas.
                  Example: {'agent_0': {'food': 1}, 'agent_1': {'food': 0}}
        """
        logger.debug(f"Calculating quotas for {len(agent_states)} agents in '{self.mode}' mode.")
        if self.mode == 'proportional':
            return self._get_proportional_quotas(agent_states)
        else: # 'none' or other modes not implemented
            logger.debug("Allocator mode is 'none', returning empty quotas.")
            return {agent_id: {} for agent_id in agent_states}

    def _get_proportional_quotas(self, agent_states):
        """
        Calculates quotas based on agent needs (e.g., lower energy gets higher priority for food).
        This implements a form of weighted proportional allocation.
        """
        with self._lock:
            quotas = {agent_id: {'food': 0} for agent_id in agent_states}
            available_food = self._resources.get('food', 0)
            logger.debug(f"Available food for allocation: {available_food}")

            if available_food <= 0 or not agent_states:
                return quotas

            weights, total_weight = {}, 0
            for agent_id, state in agent_states.items():
                energy = state[0]
                weight = (1.0 - energy + 1e-6)**self.alpha
                weights[agent_id] = weight
                total_weight += weight

            logger.debug(f"Calculated agent weights: {weights}")

            if total_weight > 0:
                for agent_id in sorted(weights.keys()):
                    proportion = weights[agent_id] / total_weight
                    claim = int(np.floor(proportion * available_food))
                    quotas[agent_id]['food'] = claim

            logger.info(f"Calculated quotas: {quotas}")
            return quotas

    def check_availability(self, resource, amount=1):
        """
        Checks if a certain amount of a resource is available.
        """
        with self._lock:
            return self._resources.get(resource, 0) >= amount

    def claim(self, resource, amount=1, agent_id="Unknown"):
        """
        Claims a certain amount of a resource for a specific agent.
        This is an atomic operation.
        """
        with self._lock:
            if self._resources.get(resource, 0) >= amount:
                self._resources[resource] -= amount
                logger.debug(f"Agent '{agent_id}' claimed {amount} of {resource}. Remaining: {self._resources[resource]}")
                return True
            else:
                logger.warning(f"Agent '{agent_id}' failed to claim {amount} of {resource}. "
                               f"Required: {amount}, Available: {self._resources.get(resource, 0)}")
                return False

    def release(self, resource, amount=1, agent_id="Unknown"):
        """
        Releases a certain amount of a resource back into the pool.
        """
        with self._lock:
            self._resources[resource] = self._resources.get(resource, 0) + amount
            logger.debug(f"Agent '{agent_id}' released {amount} of {resource}. New total: {self._resources[resource]}")

    def get_all_resources(self):
        """
        Returns a copy of the current state of all resources.
        """
        with self._lock:
            return self._resources.copy()