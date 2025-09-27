from utils.factory import ComponentFactory
from utils.trainer import Trainer
from utils.robust_trainer import RobustTrainer
from utils.multi_agent_trainer import MultiAgentTrainer

def main():
    """
    Main entry point for the training script.

    This function orchestrates the training process by:
    1. Initializing the ComponentFactory, which handles component creation.
    2. Creating all components based on the config.
    3. Initializing the appropriate Trainer (single-agent, robust, or multi-agent).
    4. Starting the training loop.
    """
    print("--- Starting PERSIST Framework Training ---")

    # 1. Initialize the factory.
    factory = ComponentFactory(config_path="config.yaml")

    # 2. Create all components.
    components = factory.get_all_components()

    # 3. Initialize the appropriate trainer.
    config = components['config']
    if config.get('multiagent', {}).get('enabled', False):
        print("--- Initializing Multi-Agent Trainer ---")
        trainer = MultiAgentTrainer(components)
    elif config.get('adversarial', {}).get('enabled', False):
        print("--- Initializing Robust Single-Agent Trainer ---")
        trainer = RobustTrainer(components)
    else:
        print("--- Initializing Standard Single-Agent Trainer ---")
        trainer = Trainer(components)

    # 4. Start the training process.
    try:
        trainer.run()
        print("\n--- Training Finished ---")
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
        raise

if __name__ == "__main__":
    main()