import torch
import numpy as np

class CurriculumScheduler:
    def __init__(self, config):
        self.config = config.get('curriculum', {})
        if not self.config.get('enabled', False):
            self.enabled = False
            return

        self.enabled = True
        self.total_steps = self.config['steps']
        self.lambdas = {}
        for key in ['lambda_homeo', 'lambda_intr']:
            if key in self.config:
                self.lambdas[key] = (self.config[key]['start'], self.config[key]['end'])

        self.constraints = {}
        for const_conf in self.config.get('viability_constraints', []):
            self.constraints[const_conf['dim_name']] = (const_conf['start'], const_conf['end'])

    def _interpolate(self, start, end, progress):
        return start + (end - start) * progress

    def get_current_values(self, step):
        if not self.enabled:
            return {'lambda_homeo': 0, 'lambda_intr': 0, 'constraints': {}}

        progress = min(1.0, step / self.total_steps)

        current_values = {}
        for key, (start, end) in self.lambdas.items():
            current_values[key] = self._interpolate(start, end, progress)

        current_constraints = {}
        for key, (start, end) in self.constraints.items():
            current_constraints[key] = self._interpolate(start, end, progress)

        current_values['constraints'] = current_constraints

        return current_values

def log_episode_data(evaluator, episode, total_steps, ep_len, task_reward, homeo_reward, intr_reward, policy_entropy, env, done, info):
    final_internal_state = env.internal_state
    violation_occurred = done and info.get('violation', False)

    constraint_satisfaction = {}
    for c in env.constraints:
        val = final_internal_state[c['dim_idx']]
        op = c['op']
        threshold = c['val']
        satisfied = (op == '>=' and val >= threshold) or \
                    (op == '<=' and val <= threshold)
        constraint_satisfaction[c['name']] = {
            'satisfied': bool(satisfied),
            'value': float(val),
            'threshold': float(threshold)
        }

    episode_data = {
        'episode': episode + 1,
        'total_steps': total_steps,
        'survival_steps': ep_len,
        'task_reward': task_reward,
        'homeo_reward': homeo_reward,
        'intr_reward': intr_reward,
        'policy_entropy': policy_entropy,
        'constraint_satisfaction': constraint_satisfaction,
        'violation': violation_occurred
    }
    evaluator.log_episode(episode_data)

    print(f"Episode {episode + 1}: Steps = {ep_len}, Task Reward = {task_reward:.2f}, "
          f"Homeo Reward = {homeo_reward:.2f}, Intr Reward = {intr_reward:.2f}, "
          f"Entropy = {policy_entropy:.2f}, Violation = {violation_occurred}")