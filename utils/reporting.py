import json
import torch
from datetime import datetime

class SafetyReporter:
    """
    Handles the logging of safety-related events, providing interpretability
    for the Safety Shield's decisions.
    """
    def __init__(self, log_path="safety_reports.log", constraint_names=None):
        """
        Initializes the SafetyReporter.

        Args:
            log_path (str): The file path for saving the reports.
            constraint_names (list[str]): A list of human-readable names for the constraints.
        """
        self.log_path = log_path
        self.constraint_names = constraint_names if constraint_names else []
        print(f"✅ SafetyReporter initialized. Reports will be saved to: {self.log_path}")

    def log_shield_decision(self, step, internal_state, unsafe_action, safe_action, probe_margins):
        """
        Logs a detailed report when the shield makes a decision.

        Args:
            step (int): The current training step.
            internal_state (torch.Tensor): The agent's internal state.
            unsafe_action (torch.Tensor): The original action proposed by the policy.
            safe_action (torch.Tensor): The action after being processed by the shield.
            probe_margins (torch.Tensor): The predicted constraint margins from the SafetyProbe.
        """
        if torch.equal(unsafe_action, safe_action):
            # If the action was not changed, there's no need to log a report.
            return

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'step': step,
            'reason': "Shield projection modified the policy's proposed action.",
            'internal_state': internal_state.cpu().numpy().tolist(),
            'proposed_action': unsafe_action.cpu().numpy().tolist(),
            'corrected_action': safe_action.cpu().numpy().tolist(),
            'predicted_margins': {}
        }

        if self.constraint_names:
            for i, name in enumerate(self.constraint_names):
                report['predicted_margins'][name] = probe_margins[i].item()
        else:
            report['predicted_margins'] = probe_margins.cpu().numpy().tolist()

        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(report) + '\n')
        except IOError as e:
            print(f"Error writing safety report: {e}")