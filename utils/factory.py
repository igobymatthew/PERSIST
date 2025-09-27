import torch
import yaml

from environments.grid_life import GridLifeEnv
from agents.persist_agent import PersistAgent
from agents.mpc_agent import MPCAgent
from components.homeostat import Homeostat
from components.replay_buffer import ReplayBuffer
from components.latent_world_model import LatentWorldModel
from components.internal_model import InternalModel
from components.viability_approximator import ViabilityApproximator
from components.rnd import RND
from components.empowerment import Empowerment
from components.shield import Shield
from components.safety_network import SafetyNetwork
from components.demonstration_buffer import DemonstrationBuffer
from components.state_estimator import StateEstimator
from components.meta_learner import MetaLearner
from utils.evaluation import Evaluator

class ComponentFactory:
    def __init__(self, config_path="config.yaml"):
        print("--- Initializing ComponentFactory ---")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def create_env(self):
        print("Initializing environment...")
        env = GridLifeEnv(self.config)
        print("✅ Environment initialized.")
        return env

    def create_agent(self, env, world_model, internal_model, viability_approximator):
        print("Initializing agent...")
        mpc_config = self.config.get('mpc', {})
        if mpc_config.get('enabled', False):
            agent = MPCAgent(
                latent_world_model=world_model,
                internal_model=internal_model,
                viability_approximator=viability_approximator,
                action_space=env.action_space,
                mpc_config=mpc_config
            )
        else:
            agent_obs_dim = env.external_obs_dim + env.internal_dim
            agent = PersistAgent(
                obs_dim=agent_obs_dim,
                act_dim=env.action_dim,
                act_limit=env.act_limit,
                risk_sensitive_config=self.config.get('risk_sensitive')
            )
        agent.to(self.device)
        print("✅ Agent initialized.")
        return agent

    def create_homeostat(self, meta_learner=None):
        print("Initializing homeostat...")
        initial_mu = meta_learner.get_setpoints() if meta_learner else self.config['internal_state']['mu']
        homeostat = Homeostat(
            mu=initial_mu,
            w=self.config['internal_state']['w']
        )
        print("✅ Homeostat initialized.")
        return homeostat

    def create_world_model(self, env):
        print("Initializing latent world model...")
        world_model = LatentWorldModel(
            obs_dim=env.observation_space_dim,
            act_dim=env.action_dim
        ).to(self.device)
        print("✅ Latent world model initialized.")
        return world_model

    def create_internal_model(self, env):
        print("Initializing internal model...")
        internal_model = InternalModel(
            internal_dim=env.internal_dim,
            act_dim=env.action_dim
        ).to(self.device)
        print("✅ Internal model initialized.")
        return internal_model

    def create_viability_approximator(self, env):
        print("Initializing viability approximator...")
        viability_approximator = ViabilityApproximator(
            internal_dim=env.internal_dim
        ).to(self.device)
        print("✅ Viability approximator initialized.")
        return viability_approximator

    def create_intrinsic_reward_module(self, env, world_model):
        intrinsic_method = self.config['rewards']['intrinsic']
        if intrinsic_method == 'rnd':
            print("Initializing RND module...")
            module = RND(obs_dim=env.observation_space_dim).to(self.device)
            print("✅ RND module initialized.")
        elif intrinsic_method == 'empowerment':
            print("Initializing Empowerment module...")
            emp_config = self.config.get('empowerment', {})
            module = Empowerment(
                state_dim=world_model.latent_dim,
                action_dim=env.action_dim,
                k=emp_config.get('k', 4),
                hidden_dim=emp_config.get('hidden_dim', 256),
                lr=emp_config.get('lr', 1e-4)
            ).to(self.device)
            print("✅ Empowerment module initialized.")
        elif intrinsic_method == 'surprise':
            print("Using world model surprise as intrinsic reward.")
            module = None # Surprise is calculated directly from the world model
        else:
            raise ValueError(f"Unknown intrinsic reward method: {intrinsic_method}")
        return module

    def create_safety_network(self, env):
        print("Initializing safety network...")
        safety_net_config = self.config.get('safety_network', {})
        safety_network = SafetyNetwork(
            internal_dim=env.internal_dim,
            action_dim=env.action_dim,
            hidden_dim=safety_net_config.get('hidden_dim', 128)
        ).to(self.device)
        print("✅ Safety network initialized.")
        return safety_network

    def create_shield(self, env, internal_model, viability_approximator, safety_network):
        print("Initializing safety shield...")
        shield = Shield(
            internal_model=internal_model,
            viability_approximator=viability_approximator,
            action_space=env.action_space,
            conf=self.config['viability']['shield']['conf'],
            safety_network=safety_network,
            mode='search'
        )
        print("✅ Safety shield initialized.")
        return shield

    def create_replay_buffer(self, env):
        print("Initializing replay buffer...")
        buffer = ReplayBuffer(
            capacity=10000,
            obs_dim=env.observation_space_dim,
            action_dim=env.action_dim,
            internal_dim=env.internal_dim,
            device=self.device
        )
        print("✅ Replay buffer initialized.")
        return buffer

    def create_demonstration_buffer(self):
        if 'demonstrations' in self.config and self.config['demonstrations'].get('filepath'):
            print("Initializing demonstration buffer...")
            buffer = DemonstrationBuffer(filepath=self.config['demonstrations']['filepath'])
            if len(buffer) == 0:
                print("⚠️ Demonstration buffer is empty.")
            else:
                print("✅ Demonstration buffer initialized.")
            return buffer
        return None

    def create_state_estimator(self, env):
        if self.config.get('env', {}).get('partial_observability', False):
            print("Initializing state estimator...")
            estimator = StateEstimator(
                obs_dim=env.external_obs_dim,
                act_dim=env.action_dim,
                internal_dim=env.internal_dim
            ).to(self.device)
            print("✅ State estimator initialized.")
            return estimator
        return None

    def create_meta_learner(self):
        meta_config = self.config.get('meta_learning', {})
        if meta_config.get('enabled', False):
            print("Initializing MetaLearner...")
            learner = MetaLearner(
                initial_mu=self.config['internal_state']['mu'],
                learning_rate=meta_config.get('learning_rate', 0.01),
                update_frequency=meta_config.get('update_frequency', 100)
            )
            print("✅ MetaLearner initialized.")
            return learner
        return None

    def create_evaluator(self):
        print("Initializing evaluator...")
        evaluator = Evaluator(log_file='training.log')
        print("✅ Evaluator initialized.")
        return evaluator

    def get_all_components(self):
        """A helper method to create and return all components in a structured way."""
        env = self.create_env()

        # Models that other components depend on
        world_model = self.create_world_model(env)
        internal_model = self.create_internal_model(env)
        viability_approximator = self.create_viability_approximator(env)
        safety_network = self.create_safety_network(env)

        # Agent and other primary components
        agent = self.create_agent(env, world_model, internal_model, viability_approximator)
        shield = self.create_shield(env, internal_model, viability_approximator, safety_network)

        # Learning and data components
        meta_learner = self.create_meta_learner()
        homeostat = self.create_homeostat(meta_learner)
        intrinsic_module = self.create_intrinsic_reward_module(env, world_model)
        state_estimator = self.create_state_estimator(env)

        # Buffers and logging
        replay_buffer = self.create_replay_buffer(env)
        demo_buffer = self.create_demonstration_buffer()
        evaluator = self.create_evaluator()

        components = {
            'env': env,
            'agent': agent,
            'homeostat': homeostat,
            'world_model': world_model,
            'internal_model': internal_model,
            'viability_approximator': viability_approximator,
            'intrinsic_reward_module': intrinsic_module,
            'safety_network': safety_network,
            'shield': shield,
            'replay_buffer': replay_buffer,
            'demonstration_buffer': demo_buffer,
            'state_estimator': state_estimator,
            'meta_learner': meta_learner,
            'evaluator': evaluator,
            'device': self.device,
            'config': self.config
        }

        return components