import json
import time
from datetime import datetime

class Evaluator:
    def __init__(self, log_file='training.log'):
        self.log_file = log_file

    def log_episode(self, episode_data):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'episode': episode_data.get('episode'),
            'total_steps': episode_data.get('total_steps'),
            'survival_steps': episode_data.get('survival_steps'),
            'task_reward': episode_data.get('task_reward'),
            'homeo_reward': episode_data.get('homeo_reward'),
            'intr_reward': episode_data.get('intr_reward'),
            'policy_entropy': episode_data.get('policy_entropy'),
            'constraint_satisfaction': episode_data.get('constraint_satisfaction', {}),
            'violation': episode_data.get('violation', False)
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')