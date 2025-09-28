import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import yaml

# Add the project root to the path to allow imports from `main`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main

class TestMainFlow(unittest.TestCase):

    @patch('main.questionary.select')
    @patch('main.questionary.confirm')
    @patch('main.ExperimentCoordinator')
    @patch('main.ComponentFactory')
    def test_yaml_config_loading_is_efficient(self, mock_component_factory, mock_coordinator, mock_confirm, mock_select):
        """
        Verifies that running from a config file instantiates the ComponentFactory only once.
        This test passes if the fix is correctly applied.
        """
        # --- Setup Mocks ---
        # 1. Simulate user selecting the 'from config.yaml' option.
        mock_select.return_value.ask.return_value = "Run directly from `config.yaml` (original behavior)"

        # 2. Simulate user confirming to proceed with the experiment.
        mock_confirm.return_value.ask.return_value = True

        # 3. Create a mock factory instance.
        mock_factory_instance = MagicMock()

        # 4. Create a dummy config that satisfies the Trainer's constructor.
        dummy_config = {
            'training': {'batch_size': 32},
            'state_estimator': {'sequence_length': 16},
            'env': {'partial_observability': False},
            'multiagent': {'enabled': False},
            'adversarial': {'enabled': False}
        }
        mock_factory_instance.config = dummy_config

        # 5. The factory's get_all_components method must return a dictionary.
        mock_factory_instance.get_all_components.return_value = {}

        mock_component_factory.return_value = mock_factory_instance

        # --- Run main ---
        # Mock os.path.exists to prevent a FileNotFoundError for config.yaml.
        # CRITICAL FIX: Provide valid YAML to mock_open so `config` is not None.
        valid_yaml_content = yaml.dump(dummy_config)

        with patch('os.path.exists', return_value=True):
             with patch('builtins.open', unittest.mock.mock_open(read_data=valid_yaml_content)):
                main.main()

        # --- Assert ---
        # With the fix, ComponentFactory should only be called ONCE inside run_experiment.
        self.assertEqual(mock_component_factory.call_count, 1,
                         "ComponentFactory should only be called once.")

if __name__ == '__main__':
    unittest.main()