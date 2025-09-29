import sys
import os
import yaml
import questionary
from typing import Callable, Optional, TypeVar

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

Number = TypeVar("Number", int, float)


def _prompt_numeric(
    message: str,
    default: Number,
    caster: Callable[[str], Number],
    *,
    min_value: Optional[Number] = None,
    max_value: Optional[Number] = None,
) -> Number:
    """Prompt the user for a numeric value while enforcing optional bounds."""

    default_str = str(default)

    def _validate(text: str):
        if text.strip() == "":
            return True
        try:
            value = caster(text)
        except (TypeError, ValueError):
            return "Please enter a valid number."

        if min_value is not None and value < min_value:
            return f"Value must be ≥ {min_value}."
        if max_value is not None and value > max_value:
            return f"Value must be ≤ {max_value}."
        return True

    answer = questionary.text(
        f"{message} (default: {default_str})",
        default=default_str,
        validate=_validate,
    ).ask()

    if answer is None or answer.strip() == "":
        return default
    return caster(answer)


def _maybe_customize_training(config: dict) -> None:
    """Offer the user a chance to fine-tune common training hyperparameters."""

    training_cfg = config.get("training")
    if not training_cfg:
        return

    print("\n--- Core Training Hyperparameters ---")
    if not questionary.confirm(
        "Review or modify epochs, step schedules, learning rates, and decay settings?",
        default=True,
    ).ask():
        print("Keeping training defaults.")
        return

    # Ensure we have sensible defaults to surface to the user.
    training_cfg.setdefault("epochs", 200)
    training_cfg.setdefault("max_steps", 500_000)
    training_cfg.setdefault("checkpoint_every", 10_000)

    training_cfg["epochs"] = _prompt_numeric(
        "Number of training epochs", training_cfg["epochs"], int, min_value=1
    )
    training_cfg["max_steps"] = _prompt_numeric(
        "Maximum environment interaction steps", training_cfg["max_steps"], int, min_value=1
    )
    training_cfg["update_every"] = _prompt_numeric(
        "Gradient updates every N environment steps",
        training_cfg.get("update_every", 64),
        int,
        min_value=1,
    )
    training_cfg["model_rollouts"] = _prompt_numeric(
        "Model-based rollouts per update",
        training_cfg.get("model_rollouts", 5),
        int,
        min_value=0,
    )
    training_cfg["amortize_after_steps"] = _prompt_numeric(
        "Switch shield to amortized mode after N steps",
        training_cfg.get("amortize_after_steps", 20_000),
        int,
        min_value=1,
    )
    training_cfg["batch_size"] = _prompt_numeric(
        "Batch size",
        training_cfg.get("batch_size", 1024),
        int,
        min_value=1,
    )
    training_cfg["gamma"] = _prompt_numeric(
        "Discount factor (gamma)",
        training_cfg.get("gamma", 0.99),
        float,
        min_value=0.0,
        max_value=1.0,
    )
    training_cfg["tau"] = _prompt_numeric(
        "Target network decay (tau)",
        training_cfg.get("tau", 0.005),
        float,
        min_value=0.0,
        max_value=1.0,
    )
    training_cfg["actor_lr"] = _prompt_numeric(
        "Actor learning rate",
        training_cfg.get("actor_lr", 3e-4),
        float,
        min_value=0.0,
    )
    training_cfg["critic_lr"] = _prompt_numeric(
        "Critic learning rate",
        training_cfg.get("critic_lr", 3e-4),
        float,
        min_value=0.0,
    )
    training_cfg["checkpoint_every"] = _prompt_numeric(
        "Checkpoint every N steps",
        training_cfg["checkpoint_every"],
        int,
        min_value=1,
    )


def _maybe_customize_model_capacity(config: dict) -> None:
    """Allow the user to adjust common hidden dimensions for learned modules."""

    print("\n--- Model Capacity ---")
    if not questionary.confirm(
        "Adjust hidden dimensions for estimator, empowerment, and safety networks?",
        default=False,
    ).ask():
        print("Keeping network dimensions at their defaults.")
        return

    state_estimator_cfg = config.setdefault("state_estimator", {})
    empowerment_cfg = config.setdefault("empowerment", {})
    safety_net_cfg = config.setdefault("safety_network", {})

    state_estimator_cfg["hidden_dim"] = _prompt_numeric(
        "State estimator hidden dimension",
        state_estimator_cfg.get("hidden_dim", 128),
        int,
        min_value=1,
    )
    state_estimator_cfg["n_layers"] = _prompt_numeric(
        "State estimator number of layers",
        state_estimator_cfg.get("n_layers", 2),
        int,
        min_value=1,
    )
    empowerment_cfg["hidden_dim"] = _prompt_numeric(
        "Empowerment model hidden dimension",
        empowerment_cfg.get("hidden_dim", 256),
        int,
        min_value=1,
    )
    empowerment_cfg["k"] = _prompt_numeric(
        "Empowerment rollout depth (k)",
        empowerment_cfg.get("k", 4),
        int,
        min_value=1,
    )
    safety_net_cfg["hidden_dim"] = _prompt_numeric(
        "Safety network hidden dimension",
        safety_net_cfg.get("hidden_dim", 128),
        int,
        min_value=1,
    )
    safety_net_cfg["lr"] = _prompt_numeric(
        "Safety network learning rate",
        safety_net_cfg.get("lr", 1e-4),
        float,
        min_value=0.0,
    )


def _maybe_customize_curriculum(config: dict) -> None:
    """Offer curriculum schedule refinement when enabled."""

    curriculum_cfg = config.get("curriculum")
    if not curriculum_cfg or not curriculum_cfg.get("enabled", False):
        return

    print("\n--- Curriculum Schedule ---")
    if not questionary.confirm(
        "Tweak curriculum steps or annealing targets?",
        default=False,
    ).ask():
        print("Keeping curriculum schedule unchanged.")
        return

    curriculum_cfg["steps"] = _prompt_numeric(
        "Curriculum duration in steps",
        curriculum_cfg.get("steps", 50_000),
        int,
        min_value=1,
    )

    for key in ["lambda_homeo", "lambda_intr"]:
        if key in curriculum_cfg:
            start, end = curriculum_cfg[key]["start"], curriculum_cfg[key]["end"]
            curriculum_cfg[key]["start"] = _prompt_numeric(
                f"{key} start value",
                start,
                float,
            )
            curriculum_cfg[key]["end"] = _prompt_numeric(
                f"{key} end value",
                end,
                float,
            )

    for constraint in curriculum_cfg.get("viability_constraints", []):
        start = _prompt_numeric(
            f"{constraint['dim_name']} curriculum start",
            constraint.get("start", 0.0),
            float,
        )
        end = _prompt_numeric(
            f"{constraint['dim_name']} curriculum end",
            constraint.get("end", 0.0),
            float,
        )
        constraint["start"], constraint["end"] = start, end


def _run_common_walkthroughs(config: dict) -> dict:
    """Run the shared customization flows before returning a config."""

    _maybe_customize_training(config)
    _maybe_customize_model_capacity(config)
    _maybe_customize_curriculum(config)
    return config


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

    return _run_common_walkthroughs(config)

def generate_multi_agent_config():
    """Generates a configuration for multi-agent training."""
    print("--- Generating multi-agent configuration ---")
    config = get_base_config()
    config['multiagent']['enabled'] = True
    return _run_common_walkthroughs(config)

def generate_robust_config():
    """Generates a configuration for a robust agent with advanced safety."""
    print("--- Generating robust agent configuration ---")
    config = get_base_config()

    config['multiagent']['enabled'] = False
    config['adversarial']['enabled'] = True
    config['safety']['cbf']['enabled'] = True
    config['ood']['enabled'] = True
    config['continual']['enabled'] = True

    return _run_common_walkthroughs(config)

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
    return _run_common_walkthroughs(config)


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
