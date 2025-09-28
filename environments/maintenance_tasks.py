import numpy as np

class MaintenanceManager:
    """
    Manages the state and logic for self-maintenance tasks in the environment,
    such as refueling, cooling down, and repairing.
    """
    def __init__(self, config, grid_size):
        """
        Initializes the MaintenanceManager.

        Args:
            config (dict): The 'maintenance' section of the main config.
            grid_size (tuple): The (width, height) of the environment grid.
        """
        self.config = config
        self.grid_size = grid_size
        self.enabled = self.config.get('enabled', False)

        if not self.enabled:
            return

        self.stations = {}
        self.station_types = self.config.get('tasks', [])
        self.penalties = self.config.get('penalty_costs', {})

        for station_type in self.station_types:
            self._add_station(station_type)

    def _place_randomly(self):
        """Returns a random position within the grid."""
        return np.random.randint(0, self.grid_size[0], size=2)

    def _add_station(self, station_type):
        """Adds a new maintenance station of a given type."""
        self.stations[station_type] = {
            'pos': self._place_randomly(),
            'penalty': self.penalties.get(station_type, 0.1)
        }

    def reset(self):
        """Resets the positions of all maintenance stations."""
        if not self.enabled:
            return
        for station_type in self.stations:
            self.stations[station_type]['pos'] = self._place_randomly()

    def get_station_positions(self):
        """Returns a dictionary of station types to their current positions."""
        if not self.enabled:
            return {}
        return {stype: s['pos'] for stype, s in self.stations.items()}

    def apply_maintenance(self, agent_pos, internal_state, mu_homeo):
        """
        Checks if the agent is at a maintenance station and applies the
        corresponding effect to the internal state.

        Args:
            agent_pos (np.ndarray): The agent's current position.
            internal_state (np.ndarray): The agent's internal state.
            mu_homeo (np.ndarray): The homeostatic setpoints.

        Returns:
            tuple: A tuple containing the updated internal_state and the task reward penalty.
        """
        if not self.enabled:
            return internal_state, 0.0

        reward_penalty = 0.0

        for station_type, station_data in self.stations.items():
            if np.array_equal(agent_pos, station_data['pos']):
                if station_type == 'refuel':
                    internal_state[0] = 1.0  # Reset energy to full
                elif station_type == 'cool_down':
                    internal_state[1] = mu_homeo[1] # Reset temp to setpoint
                elif station_type == 'repair':
                    internal_state[2] = 1.0  # Reset integrity to full

                reward_penalty = station_data['penalty']
                # Relocate the station after use
                station_data['pos'] = self._place_randomly()
                # Agent can only use one station per step
                break

        return internal_state, reward_penalty