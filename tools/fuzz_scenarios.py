import yaml
import numpy as np
import os
from datetime import datetime

class ScenarioFuzzer:
    """
    Generates fuzzed scenarios to test agent robustness against unexpected
    environmental conditions.
    """
    def __init__(self, base_config, fuzz_params):
        """
        Args:
            base_config (dict): The baseline configuration for the environment.
            fuzz_params (dict): Parameters controlling the fuzzing process.
        """
        self.base_config = base_config
        self.fuzz_params = fuzz_params

    def fuzz_scenario(self, seed=None):
        """
        Generates a single fuzzed scenario configuration.

        Args:
            seed (int, optional): A seed for the random number generator.

        Returns:
            dict: A fuzzed scenario configuration.
        """
        if seed is not None:
            np.random.seed(seed)

        fuzzed_config = self.base_config.copy()

        # Fuzz internal state parameters
        if "internal_state" in self.fuzz_params:
            for key, bounds in self.fuzz_params["internal_state"].items():
                if key in fuzzed_config["internal_state"]:
                    fuzzed_config["internal_state"][key] = self._fuzz_value(
                        self.base_config["internal_state"][key], bounds
                    )

        # Fuzz viability constraints
        if "viability" in self.fuzz_params:
             for i, constraint in enumerate(fuzzed_config["viability"]["constraints"]):
                # This is a simplified example; a real implementation would
                # need to parse and intelligently modify the constraint strings.
                pass

        return fuzzed_config

    def _fuzz_value(self, base_value, bounds):
        """Fuzzes a single value or list of values within given bounds."""
        if isinstance(base_value, list):
            return [self._fuzz_single_item(v, bounds) for v in base_value]
        return self._fuzz_single_item(base_value, bounds)

    def _fuzz_single_item(self, item, bounds):
        min_val, max_val = bounds
        fuzzed_item = item + np.random.uniform(min_val, max_val)
        return np.clip(fuzzed_item, 0, None) # Ensure values don't go below zero

    def generate_fuzz_suite(self, num_scenarios, output_dir):
        """
        Generates and saves a suite of fuzzed scenarios.

        Args:
            num_scenarios (int): The number of scenarios to generate.
            output_dir (str): The directory to save the YAML files.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_scenarios):
            fuzzed_scenario = self.fuzz_scenario(seed=i)
            filename = os.path.join(output_dir, f"fuzzed_scenario_{i:03d}.yaml")
            with open(filename, "w") as f:
                yaml.dump(fuzzed_scenario, f, default_flow_style=False)
        print(f"Generated {num_scenarios} fuzzed scenarios in {output_dir}")


if __name__ == "__main__":
    # Example usage:
    base_config = {
        "env": {"name": "GridLife-v0", "horizon": 512},
        "internal_state": {
            "dims": ["energy", "temp", "integrity"],
            "mu": [0.7, 0.5, 0.9],
            "w": [1.0, 0.5, 1.5],
        },
        "viability": {
            "constraints": [
                "energy >= 0.2",
                "temp in [0.3, 0.7]",
                "integrity >= 0.6",
            ]
        },
    }

    fuzz_params = {
        "internal_state": {
            "mu": [-0.2, 0.2],  # Fuzz setpoints by +/- 0.2
            "w": [-0.5, 0.5],   # Fuzz weights by +/- 0.5
        }
    }

    fuzzer = ScenarioFuzzer(base_config, fuzz_params)
    output_directory = os.path.join("benchmarks", "fuzz_suite")
    fuzzer.generate_fuzz_suite(20, output_directory)