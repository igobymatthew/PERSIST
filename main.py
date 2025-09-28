from utils.factory import ComponentFactory
from systems.coordinator import ExperimentCoordinator
from systems.persistence import PersistenceManager
from utils.trainer import Trainer
from utils.robust_trainer import RobustTrainer
from utils.multi_agent_trainer import MultiAgentTrainer
import os

def main():
    """
    Main entry point for the training script.
    """
    print("--- Starting PERSIST Framework ---")

    # 1. Initialize the factory
    factory = ComponentFactory(config_path="config.yaml")

    # 2. Set up persistence
    config = factory.config
    log_dir = config.get('logging', {}).get('log_dir', 'logs')
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    persistence_manager = PersistenceManager(checkpoint_dir)

    # 3. Create all components
    components = factory.get_all_components()
    components['config'] = config

    # 4. Initialize the appropriate trainer
    if config.get('multiagent', {}).get('enabled', False):
        trainer = MultiAgentTrainer(components)
    elif config.get('adversarial', {}).get('enabled', False):
        trainer = RobustTrainer(components)
    else:
        trainer = Trainer(components)
    components['trainer'] = trainer

    # 5. Initialize the coordinator
    coordinator = ExperimentCoordinator(components, persistence_manager)

    # 6. Start the training process
    try:
        coordinator.run()
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
        raise

if __name__ == "__main__":
    main()