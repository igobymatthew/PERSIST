import torch
import yaml
import re

from environments.grid_life import GridLifeEnv
from environments.multi_agent_gridlife import MultiAgentGridLifeEnv
from agents.persist_agent import PersistAgent
from agents.mpc_agent import MPCAgent
from agents.shared_sac import SharedSAC
from components.homeostat import Homeostat
from components.replay_buffer import ReplayBuffer
from buffers.replay_ma import ReplayMA
from multiagent.resource_allocator import ResourceAllocator
from multiagent.cbf_coupler import CBFCoupler
from components.latent_world_model import LatentWorldModel
from components.internal_model import InternalModel
from components.viability_approximator import ViabilityApproximator
from components.rnd import RND
from components.empowerment import Empowerment
from components.shield import Shield
from population.ensemble_shield import EnsembleShield
from components.safety_network import SafetyNetwork
from components.demonstration_buffer import DemonstrationBuffer
from components.state_estimator import StateEstimator
from components.meta_learner import MetaLearner
from utils.evaluation import Evaluator
from components.constraint_manager import ConstraintManager
from components.ood_detector import OODDetector
from policies.safe_fallback import SafeFallbackPolicy
from components.dynamics_adapter import DynamicsAdapter
from components.cbf_layer import CBFLayer
from buffers.rehearsal import RehearsalBuffer
from components.continual import ContinualLearningManager
from components.budget_meter import BudgetMeter
from components.adversary import Adversary
from components.safety_probe import SafetyProbe
from utils.reporting import SafetyReporter

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

    def create_viability_ensemble(self, env):
        population_config = self.config.get('population', {})
        ensemble_config = population_config.get('ensemble_shield', {})
        if not ensemble_config.get('enabled', False):
            return None

        ensemble_size = ensemble_config.get('ensemble_size', 3)
        print(f"Initializing viability ensemble of size {ensemble_size}...")
        ensemble = [
            ViabilityApproximator(internal_dim=env.internal_dim).to(self.device)
            for _ in range(ensemble_size)
        ]
        print(f"✅ Viability ensemble of size {ensemble_size} initialized.")
        return ensemble

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

    def create_shield(self, env, internal_model, viability_approximator, safety_network, viability_ensemble=None):
        population_config = self.config.get('population', {})
        ensemble_config = population_config.get('ensemble_shield', {})

        if ensemble_config.get('enabled', False) and viability_ensemble:
            print("Initializing Ensemble Safety Shield...")
            shield = EnsembleShield(
                viability_models=viability_ensemble,
                internal_model=internal_model,
                action_space=env.action_space,
                vote_method=ensemble_config.get('vote_method', 'veto_if_any_unsafe')
            )
            print("✅ Ensemble Safety Shield initialized.")
        else:
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

    def create_constraint_manager(self, env):
        print("Initializing constraint manager...")
        safety_config = self.config.get('safety', {})
        chance_config = safety_config.get('chance_constraint', {})
        if not (safety_config and chance_config and chance_config.get('enabled', False)):
            print("Constraint manager is disabled in config.")
            return None

        manager = ConstraintManager(
            num_constraints=env.internal_dim,
            target_violation_rate=chance_config.get('target_violation_rate', 0.01),
            dual_lr=chance_config.get('dual_lr', 5e-4),
            device=self.device
        )
        print("✅ Constraint manager initialized.")
        return manager

    def create_ood_detector(self, viability_approximator):
        print("Initializing OOD detector...")
        ood_config = self.config.get('ood', {})
        if not ood_config.get('enabled', False):
            print("OOD detector is disabled in config.")
            return None

        detector = OODDetector(
            model=viability_approximator,
            threshold=ood_config.get('threshold', -5.0)
        ).to(self.device)
        print("✅ OOD detector initialized.")
        return detector

    def create_safe_fallback_policy(self, env):
        print("Initializing safe fallback policy...")
        policy = SafeFallbackPolicy(
            action_dim=env.action_dim,
            device=self.device
        )
        print("✅ Safe fallback policy initialized.")
        return policy

    def _parse_constraints(self):
        constraints_config = self.config['viability']['constraints']
        dims = self.config['internal_state']['dims']
        dim_map = {name: i for i, name in enumerate(dims)}
        parsed_constraints = []
        for constr_str in constraints_config:
            match_simple = re.match(r"(\w+)\s*([<>]=)\s*(-?[\d.]+)", constr_str)
            match_interval = re.match(r"(\w+)\s+in\s+\[(-?[\d.]+),\s*(-?[\d.]+)\]", constr_str)
            if match_simple:
                var, op, val_str = match_simple.groups()
                val = float(val_str)
                idx = dim_map[var]
                if op == '>=':
                    parsed_constraints.append(lambda x, _idx=idx, _val=val: _val - x[_idx])
                elif op == '<=':
                    parsed_constraints.append(lambda x, _idx=idx, _val=val: x[_idx] - _val)
            elif match_interval:
                var, min_val_str, max_val_str = match_interval.groups()
                min_val, max_val = float(min_val_str), float(max_val_str)
                idx = dim_map[var]
                parsed_constraints.append(lambda x, _idx=idx, _val=min_val: _val - x[_idx])
                parsed_constraints.append(lambda x, _idx=idx, _val=max_val: x[_idx] - _val)
            else:
                raise ValueError(f"Could not parse constraint string: {constr_str}")
        print(f"✅ Parsed {len(parsed_constraints)} CBF constraints.")
        return parsed_constraints

    def create_dynamics_adapter(self, internal_model):
        print("Initializing dynamics adapter...")
        adapter = DynamicsAdapter(internal_model=internal_model)
        print("✅ Dynamics adapter initialized.")
        return adapter

    def create_cbf_layer(self, env):
        print("Initializing CBF layer...")
        safety_config = self.config.get('safety', {})
        cbf_config = safety_config.get('cbf', {})
        if not cbf_config.get('enabled', False):
            print("CBF layer is disabled in config.")
            return None
        constraints = self._parse_constraints()
        cbf_layer = CBFLayer(
            x_dim=env.internal_dim, a_dim=env.action_dim, constraints=constraints,
            relax_penalty=cbf_config.get('relax_penalty', 10.0),
            delta=cbf_config.get('delta', 1.0)
        )
        print("✅ CBF layer initialized.")
        return cbf_layer

    def create_rehearsal_buffer(self):
        continual_config = self.config.get('continual', {})
        if not continual_config.get('enabled', False):
            return None
        print("Initializing rehearsal buffer...")
        buffer = RehearsalBuffer(
            capacity=continual_config.get('rehearsal_capacity', 2000),
            device=self.device
        )
        print("✅ Rehearsal buffer initialized.")
        return buffer

    def create_continual_learning_manager(self, agent):
        continual_config = self.config.get('continual', {})
        if not continual_config.get('enabled', False):
            return None
        print("Initializing ContinualLearningManager...")
        manager = ContinualLearningManager(
            agent=agent,
            ewc_lambda=continual_config.get('ewc_lambda', 1.0),
            device=self.device
        )
        print("✅ ContinualLearningManager initialized.")
        return manager

    def create_budget_meter(self):
        budget_config = self.config.get('budgets', {})
        if not budget_config.get('enabled', False):
            return None
        print("Initializing BudgetMeter...")
        meter = BudgetMeter(
            budget_config=budget_config
        )
        print("✅ BudgetMeter initialized.")
        return meter

    def create_adversary(self):
        adversarial_config = self.config.get('adversarial', {})
        if not adversarial_config.get('enabled', False):
            return None
        print("Initializing Adversary...")
        adversary = Adversary(
            epsilon=adversarial_config.get('epsilon', 0.05),
            alpha=adversarial_config.get('alpha', 0.01),
            num_iter=adversarial_config.get('num_iter', 10)
        )
        print("✅ Adversary initialized.")
        return adversary

    def create_safety_probe(self, env):
        probe_config = self.config.get('safety_probe', {})
        if not probe_config.get('enabled', False):
            return None
        print("Initializing SafetyProbe...")
        probe = SafetyProbe(
            internal_dim=env.internal_dim,
            num_constraints=env.num_constraints,
            hidden_sizes=probe_config.get('hidden_sizes', (64, 64)),
            lr=probe_config.get('lr', 1e-3)
        ).to(self.device)
        return probe

    def create_safety_reporter(self, env):
        probe_config = self.config.get('safety_probe', {})
        if not probe_config.get('enabled', False):
            return None
        print("Initializing SafetyReporter...")
        reporter = SafetyReporter(
            log_path=probe_config.get('log_path', 'safety_reports.log'),
            constraint_names=env.constraint_names
        )
        return reporter

    def create_multi_agent_env(self):
        print("Initializing multi-agent environment...")
        env = MultiAgentGridLifeEnv(self.config)
        print("✅ Multi-agent environment initialized.")
        return env

    def create_ma_replay_buffer(self, env):
        print("Initializing multi-agent replay buffer...")
        training_cfg = self.config['training']
        buffer = ReplayMA(
            capacity=training_cfg['buffer']['capacity'],
            obs_space=env.single_observation_space,
            act_space=env.single_action_space,
            num_agents=self.config['multiagent']['num_agents'],
            agent_ids=env.agents,
            device=self.device
        )
        print("✅ Multi-agent replay buffer initialized.")
        return buffer

    def create_ma_policies(self, env):
        print("Initializing multi-agent policies...")
        agent_types_cfg = self.config['agent_types']
        policies = {}

        default_cfg = agent_types_cfg['default']
        if default_cfg['policy'] == 'shared_sac':
            obs_dim = sum(np.prod(space.shape) for space in env.single_observation_space.spaces.values())

            policy = SharedSAC(
                obs_dim=obs_dim,
                act_dim=env.single_action_space.shape[0],
                num_agents=self.config['multiagent']['num_agents'],
                role_embedding_dim=default_cfg['role_embedding_dim'],
                config=self.config
            ).to(self.device)
            policies['default'] = policy
        else:
            raise ValueError(f"Unsupported multi-agent policy type: {default_cfg['policy']}")

        print("✅ Multi-agent policies initialized.")
        return policies

    def create_resource_allocator(self):
        print("Initializing resource allocator...")
        allocator_config = self.config.get('resource_allocator', {})
        if allocator_config.get('mode', 'none') == 'none':
            return None

        resource_config = {
            'food': self.config['resource_model']['food']['initial_density'] * 10 * 10
        }

        allocator = ResourceAllocator(resource_config)
        print("✅ Resource allocator initialized.")
        return allocator

    def create_cbf_coupler(self, env):
        print("Initializing CBF Coupler...")
        # Using agent_types.default for now, as heterogeneity is a future step
        agent_config = self.config.get('agent_types', {}).get('default', {})
        cbf_config = agent_config.get('cbf', {})

        if not cbf_config.get('enabled', False):
            print("CBF Coupler is disabled in config.")
            return None

        coupler = CBFCoupler(
            agent_ids=env.agents,
            action_dim=env.single_action_space.shape[0],
            min_safe_distance=1.0, # Corresponds to single-cell occupancy
            gamma=cbf_config.get('gamma', 0.5)
        )
        print("✅ CBF Coupler initialized.")
        return coupler

    def get_all_components(self):
        is_multi_agent = self.config.get('multiagent', {}).get('enabled', False)

        if is_multi_agent:
            print("\n--- Building components for MULTI-AGENT training ---")
            env = self.create_multi_agent_env()
            policies = self.create_ma_policies(env)
            replay_buffer = self.create_ma_replay_buffer(env)
            resource_allocator = self.create_resource_allocator()
            cbf_coupler = self.create_cbf_coupler(env)

            components = {
                'env': env,
                'policies': policies,
                'replay_buffer': replay_buffer,
                'resource_allocator': resource_allocator,
                'cbf_coupler': cbf_coupler,
                'device': self.device,
                'config': self.config
            }
        else:
            print("\n--- Building components for SINGLE-AGENT training ---")
            env = self.create_env()
            world_model = self.create_world_model(env)
            internal_model = self.create_internal_model(env)
            viability_approximator = self.create_viability_approximator(env)
            viability_ensemble = self.create_viability_ensemble(env)
            safety_network = self.create_safety_network(env)
            dynamics_adapter = self.create_dynamics_adapter(internal_model)
            agent = self.create_agent(env, world_model, internal_model, viability_approximator)
            continual_learning_manager = self.create_continual_learning_manager(agent)
            shield = self.create_shield(env, internal_model, viability_approximator, safety_network, viability_ensemble)
            safe_fallback_policy = self.create_safe_fallback_policy(env)
            cbf_layer = self.create_cbf_layer(env)
            meta_learner = self.create_meta_learner()
            constraint_manager = self.create_constraint_manager(env)
            ood_detector = self.create_ood_detector(viability_approximator)
            homeostat = self.create_homeostat(meta_learner)
            intrinsic_module = self.create_intrinsic_reward_module(env, world_model)
            state_estimator = self.create_state_estimator(env)
            replay_buffer = self.create_replay_buffer(env)
            rehearsal_buffer = self.create_rehearsal_buffer()
            demo_buffer = self.create_demonstration_buffer()
            evaluator = self.create_evaluator()
            budget_meter = self.create_budget_meter()
            adversary = self.create_adversary()
            safety_probe = self.create_safety_probe(env)
            safety_reporter = self.create_safety_reporter(env)

            components = {
                'env': env, 'agent': agent, 'homeostat': homeostat, 'budget_meter': budget_meter,
                'world_model': world_model, 'internal_model': internal_model,
                'viability_approximator': viability_approximator,
                'viability_ensemble': viability_ensemble,
                'intrinsic_reward_module': intrinsic_module,
                'safety_network': safety_network, 'shield': shield,
                'replay_buffer': replay_buffer, 'demonstration_buffer': demo_buffer,
                'rehearsal_buffer': rehearsal_buffer,
                'state_estimator': state_estimator, 'meta_learner': meta_learner,
                'constraint_manager': constraint_manager, 'ood_detector': ood_detector,
                'continual_learning_manager': continual_learning_manager,
                'safe_fallback_policy': safe_fallback_policy,
                'dynamics_adapter': dynamics_adapter, 'cbf_layer': cbf_layer,
                'evaluator': evaluator, 'device': self.device, 'config': self.config,
                'adversary': adversary, 'safety_probe': safety_probe, 'safety_reporter': safety_reporter
            }

        return components