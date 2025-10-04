import os
import yaml
import questionary
import math
import random
import importlib
from typing import Callable, Optional, TypeVar, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from utils.factory import ComponentFactory
from systems.coordinator import ExperimentCoordinator
from systems.persistence import PersistenceManager
from utils.trainer import Trainer
from utils.robust_trainer import RobustTrainer
from utils.multi_agent_trainer import MultiAgentTrainer
from evolution.ga_core import GA, Individual
from evolution.operators import uniform_crossover, gaussian_mutation
from evolution.nsga2 import nsga2_select


def get_base_config():
    """Loads the default config.yaml to be used as a template."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

Number = TypeVar("Number", int, float)


console = Console()


class StepProgressTracker:
    """Render a color-coded step indicator for the CLI walkthrough."""

    def __init__(self, steps: Sequence[str]):
        self.steps = list(steps)
        self.total = len(self.steps)
        self.current = 0

    def advance(self, title: str) -> None:
        """Advance to the next step and print the indicator."""

        if not self.steps:
            return

        self.current = min(self.current + 1, self.total)
        text = Text()

        for idx, step in enumerate(self.steps, start=1):
            if idx < self.current:
                marker, style = "✔", "green"
            elif idx == self.current:
                marker, style = "➤", "yellow"
            else:
                marker, style = "●", "grey70"

            step_text = f" {marker} {step} "
            text.append(step_text, style=style)

            if idx != self.total:
                text.append(" › ", style="grey50")

        console.print(
            Panel(
                text,
                title=f"Step {self.current}/{self.total}: {title}",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )


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

    label = message.split("(")[0].strip()

    if answer is None or answer.strip() == "":
        console.print(f":white_check_mark: Using default {label} → {default}", style="green")
        return default

    value = caster(answer)
    console.print(f":white_check_mark: Set {label} → {value}", style="green")
    return value


def _render_summary_card(title: str, payload: dict, *, subtitle: Optional[str] = None) -> None:
    """Display a bordered summary card for the provided section details."""

    body = Text()

    if not payload:
        body.append("No changes made.", style="grey70")
    else:
        for key, value in payload.items():
            body.append(f"{key}: ", style="bold white")
            body.append(f"{value}\n", style="white")

    console.print(
        Panel(
            body,
            title=title,
            subtitle=subtitle,
            border_style="magenta",
            box=box.DOUBLE,
        )
    )


def _render_top_level_summary(config: dict) -> None:
    """Summarize the primary experiment toggles for quick review."""

    toggles = {
        "multiagent": config.get("multiagent", {}).get("enabled", False),
        "adversarial": config.get("adversarial", {}).get("enabled", False),
        "curriculum": config.get("curriculum", {}).get("enabled", False),
        "telemetry": config.get("telemetry", {}).get("enabled", False),
        "risk_sensitive": config.get("risk_sensitive", {}).get("enabled", False),
        "evolution": config.get("evolution", {}).get("enabled", False),
    }

    _render_summary_card(
        "Top-Level Configuration",
        toggles,
        subtitle="Primary feature switches",
    )


def _maybe_customize_training(config: dict, progress: Optional[StepProgressTracker] = None) -> None:
    """Offer the user a chance to fine-tune common training hyperparameters."""

    training_cfg = config.get("training")
    if not training_cfg:
        return

    if progress:
        progress.advance("Core Training Hyperparameters")

    console.print("\n[bold cyan]Core Training Hyperparameters[/bold cyan]")
    if not questionary.confirm(
        "Review or modify epochs, step schedules, learning rates, and decay settings?",
        default=True,
    ).ask():
        console.print("[yellow]Keeping training defaults.[/]")
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

    summary_payload = {
        "epochs": training_cfg["epochs"],
        "max_steps": training_cfg["max_steps"],
        "update_every": training_cfg["update_every"],
        "batch_size": training_cfg["batch_size"],
        "actor_lr": training_cfg["actor_lr"],
        "critic_lr": training_cfg["critic_lr"],
    }
    _render_summary_card("Training Summary", summary_payload, subtitle="Review these values before continuing")


def _maybe_customize_model_capacity(
    config: dict, progress: Optional[StepProgressTracker] = None
) -> None:
    """Allow the user to adjust common hidden dimensions for learned modules."""

    if progress:
        progress.advance("Model Capacity")

    console.print("\n[bold cyan]Model Capacity[/bold cyan]")
    if not questionary.confirm(
        "Adjust hidden dimensions for estimator, empowerment, and safety networks?",
        default=False,
    ).ask():
        console.print("[yellow]Keeping network dimensions at their defaults.[/]")
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

    summary_payload = {
        "state_estimator": f"{state_estimator_cfg['n_layers']}×{state_estimator_cfg['hidden_dim']}",
        "empowerment_hidden": empowerment_cfg["hidden_dim"],
        "empowerment_k": empowerment_cfg["k"],
        "safety_hidden": safety_net_cfg["hidden_dim"],
        "safety_lr": safety_net_cfg["lr"],
    }
    _render_summary_card("Model Capacity Summary", summary_payload)


def _maybe_customize_curriculum(
    config: dict, progress: Optional[StepProgressTracker] = None
) -> None:
    """Offer curriculum schedule refinement when enabled."""

    curriculum_cfg = config.get("curriculum")
    if not curriculum_cfg or not curriculum_cfg.get("enabled", False):
        return

    if progress:
        progress.advance("Curriculum Schedule")

    console.print("\n[bold cyan]Curriculum Schedule[/bold cyan]")
    if not questionary.confirm(
        "Tweak curriculum steps or annealing targets?",
        default=False,
    ).ask():
        console.print("[yellow]Keeping curriculum schedule unchanged.[/]")
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

    summary_payload = {
        "steps": curriculum_cfg["steps"],
        "lambda_homeo": curriculum_cfg.get("lambda_homeo"),
        "lambda_intr": curriculum_cfg.get("lambda_intr"),
        "viability_constraints": len(curriculum_cfg.get("viability_constraints", [])),
    }
    _render_summary_card("Curriculum Summary", summary_payload)


def _run_common_walkthroughs(config: dict) -> dict:
    """Run the shared customization flows before returning a config."""

    steps = ["Training", "Model Capacity"]
    if config.get("curriculum", {}).get("enabled", False):
        steps.append("Curriculum")
    progress = StepProgressTracker(steps)

    _maybe_customize_training(config, progress)
    _maybe_customize_model_capacity(config, progress)
    _maybe_customize_curriculum(config, progress)

    if steps:
        console.print(
            Panel(
                "Configuration walkthrough complete! Review the final summary below.",
                border_style="green",
                box=box.HEAVY,
            )
        )
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

def generate_ga_config():
    """Generates a configuration for a GA-based experiment."""
    print("--- Generating GA-based experiment configuration ---")
    config = get_base_config()

    # Disable other modes
    config['multiagent']['enabled'] = False
    config['adversarial']['enabled'] = False
    config['risk_sensitive']['enabled'] = False

    # Enable and configure evolution
    config['evolution'] = {
        'enabled': True,
        'algorithm': 'nsga2',
        'pop_size': 50,
        'generations': 10,
        'elitism': 2,
        'mutation_rate': 0.1,
        'viability': 'constraints.viability_policy',
        'evaluator': 'evaluators.rl_metaeval.eval_rl_individual',
        'search_space': 'search_spaces.ppo_small_net_v1',
        'fire': {
            'trigger': 'plateau:3',
            'intensity': 1.6,
            'ewc_mask': True,
        }
    }
    console.print(
        Panel(
            "GA experiment configured. The GA will optimize hyperparameters using a dummy evaluator.",
            border_style="green",
            box=box.HEAVY,
        )
    )
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
    return _run_common_walkthroughs(config)


def run_ga_experiment(config):
    """Runs a GA-based experiment using the 'evolution' config."""
    evo_config = config['evolution']

    # 1. Dynamically load the evaluator function
    eval_path = evo_config['evaluator']
    eval_module_name, eval_func_name = eval_path.rsplit('.', 1)
    eval_module = importlib.import_module(f"evolution.{eval_module_name}")
    fitness_fn = getattr(eval_module, eval_func_name)

    # 2. Dynamically load the search space
    space_path = evo_config['search_space']
    space_module_name, space_func_name = space_path.rsplit('.', 1)
    space_module = importlib.import_module(f"evolution.{space_module_name}")
    search_space_fn = getattr(space_module, space_func_name)
    search_space = search_space_fn()

    # 3. Define the init function based on the search space
    def init_fn() -> Individual:
        genes = {}
        for param, spec in search_space.items():
            if spec[0] == 'choice':
                genes[param] = random.choice(spec[1])
            elif spec[0] == 'uniform':
                genes[param] = random.uniform(spec[1], spec[2])
            elif spec[0] == 'log_uniform':
                genes[param] = 10**random.uniform(math.log10(spec[1]), math.log10(spec[2]))
        return {"genes": genes}

    # 4. Dynamically load the viability function
    viability_path = evo_config['viability']
    module_name, func_name = viability_path.rsplit('.', 1)
    viability_module = importlib.import_module(f"evolution.{module_name}")
    viability_fn = getattr(viability_module, func_name)

    # 5. Setup the GA
    ga = GA(
        init_fn=init_fn,
        fitness_fn=fitness_fn,
        select_fn=nsga2_select,  # Using nsga2_select as a placeholder
        crossover_fn=uniform_crossover,
        mutate_fn=gaussian_mutation,
        viability_fn=viability_fn,
        pop_size=evo_config['pop_size'],
        elitism=evo_config['elitism'],
        mutation_rate=evo_config['mutation_rate'],
        max_generations=evo_config['generations'],
        seed=config.get('seed')
    )

    # 6. Run the GA
    console.print("\n[bold cyan]Starting Genetic Algorithm[/bold cyan]")
    final_pop, final_fits = ga.run()
    console.print("\n[bold green]Genetic Algorithm Finished[/bold green]")

    # 7. Print results
    if final_fits:
        key = list(final_fits[0].keys())[0]
        ranked_pop = [p for _, p in sorted(zip(final_fits, final_pop), key=lambda x: x[0][key])]
        best_ind = ranked_pop[0]

        summary_payload = {
            "best_fitness": sorted(final_fits, key=lambda x: x[key])[0],
            "best_genes": yaml.dump(best_ind['genes'], indent=2),
        }
        _render_summary_card("GA Run Summary", summary_payload, subtitle="Best individual found")


def run_experiment(config):
    """
    Initializes components and runs the experiment based on the given config.
    """
    if config.get('evolution', {}).get('enabled', False):
        run_ga_experiment(config)
        return

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
            "Run a GA-based experiment (meta-optimization)",
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
    elif "GA-based" in choice:
        config = generate_ga_config()
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
        console.print("\n[bold underline]Final Configuration Overview[/bold underline]")
        _render_top_level_summary(config)
        print(yaml.dump(config, default_flow_style=False, indent=2))
        print("---------------------------\n")

        if questionary.confirm("Proceed with this configuration?").ask():
            run_experiment(config)
        else:
            print("Operation cancelled by user.")

if __name__ == "__main__":
    main()