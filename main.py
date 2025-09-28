import sys
import os
import yaml
import questionary
from copy import deepcopy

from utils.factory import ComponentFactory
from systems.coordinator import ExperimentCoordinator
from systems.persistence import PersistenceManager
from utils.trainer import Trainer
from utils.robust_trainer import RobustTrainer
from utils.multi_agent_trainer import MultiAgentTrainer

def get_base_config():
    """Loads the default config.yaml to be used as a template."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def generate_standard_config():
    """Generates a basic, safe single-agent configuration."""
    print("--- Generating standard single-agent configuration ---")
    config = get_base_config()

    # Disable most advanced features for a clean run
    config['multiagent']['enabled'] = False
    config['adversarial']['enabled'] = False
    config['risk_sensitive']['enabled'] = False
    config['meta_learning']['enabled'] = False
    config['mpc']['enabled'] = False
    config['safety']['cbf']['enabled'] = False
    config['safety']['chance_constraint']['enabled'] = False
    config['ood']['enabled'] = False
    config['continual']['enabled'] = False
    config['budgets']['enabled'] = False
    config['population']['ensemble_shield']['enabled'] = False
    config['maintenance']['enabled'] = True # Keep this for core behavior
    config['safety_probe']['enabled'] = True
    config['rewards']['intrinsic'] = 'surprise' # A simple but effective default

    return config

def generate_multi_agent_config():
    """Generates a configuration for multi-agent training."""
    print("--- Generating multi-agent configuration ---")
    config = get_base_config()
    config['multiagent']['enabled'] = True
    return config

def generate_robust_config():
    """Generates a configuration for a robust agent with advanced safety."""
    print("--- Generating robust agent configuration ---")
    config = get_base_config()

    config['multiagent']['enabled'] = False
    config['adversarial']['enabled'] = True
    config['safety']['cbf']['enabled'] = True
    config['ood']['enabled'] = True
    config['continual']['enabled'] = True

    return config

def generate_custom_config():
    """Guides the user through a series of questions to build a custom config."""
    print("--- Generating custom configuration ---")
    config = get_base_config()

    # --- Top-level choices ---
    is_multiagent = questionary.confirm("Run a multi-agent experiment?", default=False).ask()
    config['multiagent']['enabled'] = is_multiagent

    if not is_multiagent:
        # --- Single-agent specific questions ---
        intrinsic_method = questionary.select(
            "Choose the intrinsic reward method:",
            choices=["surprise", "rnd", "empowerment"],
            default="surprise"
        ).ask()
        config['rewards']['intrinsic'] = intrinsic_method

        use_curriculum = questionary.confirm("Use curriculum learning?", default=True).ask()
        config['curriculum']['enabled'] = use_curriculum

        use_adversarial = questionary.confirm("Enable adversarial training for robustness?", default=False).ask()
        config['adversarial']['enabled'] = use_adversarial

        use_cbf = questionary.confirm("Use Control Barrier Functions (CBF) for safety?", default=False).ask()
        config['safety']['cbf']['enabled'] = use_cbf

    # --- Common questions ---
    use_telemetry = questionary.confirm("Enable Prometheus telemetry?", default=True).ask()
    config['telemetry']['enabled'] = use_telemetry

    print("\nCustom configuration generated based on your selections.")
    return config


def run_experiment(config):
    """
    Initializes components and runs the experiment based on the given config.
    """
    # 1. Initialize the factory with the generated config
    factory = ComponentFactory(config=config)

    # 2. Set up persistence
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
        print("\n--- Starting PERSIST Framework ---")
        coordinator.run()
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
        raise

def main():
    """
    Main entry point with an interactive CLI.
    """
    print("Welcome to the PERSIST Framework!")

    if not os.path.exists("config.yaml"):
        print("Error: `config.yaml` not found. Please ensure it exists in the root directory.")
        return

    choice = questionary.select(
        "What would you like to do?",
        choices=[
            "Run a standard single-agent experiment (Recommended for beginners)",
            "Run a multi-agent experiment",
            "Run a robust agent experiment (with adversarial training)",
            "Run a custom experiment (you will be asked a few questions)",
            "Run directly from `config.yaml` (original behavior)",
            "Exit"
        ]).ask()

    config = None
    if choice is None or choice == "Exit":
        print("Exiting.")
        return

    if "standard single-agent" in choice:
        config = generate_standard_config()
    elif "multi-agent" in choice:
        config = generate_multi_agent_config()
    elif "robust agent" in choice:
        config = generate_robust_config()
    elif "custom experiment" in choice:
        config = generate_custom_config()
    elif "from `config.yaml`" in choice:
        print("--- Running directly from `config.yaml` ---")
        # Load the config directly to avoid instantiating the factory twice.
        # The factory will be created once inside run_experiment.
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)

    if config:
        # Pretty print the final config for user confirmation
        print("\n--- Final Configuration ---")
        print(yaml.dump(config, default_flow_style=False, indent=2))
        print("---------------------------\n")

        if questionary.confirm("Proceed with this configuration?").ask():
            run_experiment(config)
        else:
            print("Operation cancelled by user.")

if __name__ == "__main__":
    main()