from utils.factory import ComponentFactory
from utils.trainer import Trainer

def main():
    """
    Main entry point for the training script.

    This function orchestrates the training process by:
    1. Initializing the ComponentFactory, which handles the creation of all
       necessary components based on the `config.yaml` file.
    2. Using the factory to get a dictionary of all initialized components
       (environment, agent, models, etc.).
    3. Initializing the Trainer with these components.
    4. Starting the training loop by calling the Trainer's `run` method.
    """
    print("--- Starting PERSIST Framework Training ---")

    # 1. Initialize the factory. It loads the config and sets up the device.
    factory = ComponentFactory(config_path="config.yaml")

    # 2. Create all components.
    components = factory.get_all_components()

    # 3. Initialize the trainer with all the components.
    trainer = Trainer(components)

    # 4. Start the training process.
    try:
        trainer.run()
        print("\n--- Training Finished ---")
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
        # Optionally, re-raise the exception if you want to see the full traceback
        raise

if __name__ == "__main__":
    main()