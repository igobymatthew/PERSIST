import torch
import itertools
from utils.solvers import qp_solver

class CBFCoupler:
    """
    Couples the safety constraints of multiple agents using a joint
    Control Barrier Function (CBF) formulation, solved via a single
    Quadratic Program (QP), specifically for collision avoidance.
    """
    def __init__(self, agent_ids, action_dim, min_safe_distance, gamma=0.5):
        """
        Initializes the CBFCoupler.

        Args:
            agent_ids (list[str]): A list of agent IDs to manage.
            action_dim (int): The action dimension for a single agent (e.g., 2 for position).
            min_safe_distance (float): The minimum allowable distance between any two agents.
            gamma (float): The CBF class-K function parameter (controls how aggressively to enforce the barrier).
        """
        self.agent_ids = sorted(agent_ids) # Sort for deterministic order
        self.agent_id_to_idx = {aid: i for i, aid in enumerate(self.agent_ids)}
        self.num_agents = len(self.agent_ids)
        self.action_dim = action_dim
        self.total_action_dim = self.num_agents * self.action_dim
        self.min_safe_distance_sq = min_safe_distance**2
        self.gamma = gamma
        print(f"✅ CBFCoupler initialized for {self.num_agents} agents with min_dist={min_safe_distance}.")

    def project_actions(self, agent_positions, proposed_actions):
        """
        Projects a joint proposed action to a safe joint action using a coupled QP.

        Args:
            agent_positions (dict): A dictionary mapping agent_id to their position tensor.
            proposed_actions (dict): A dictionary mapping agent_id to their proposed action tensor.

        Returns:
            dict: A dictionary of safe actions for each agent.
        """
        if self.num_agents < 2:
            return proposed_actions

        device = list(agent_positions.values())[0].device

        # 1. Construct QP matrices for: min ||u - u_proposed||^2
        Q = torch.eye(self.total_action_dim, device=device)
        p_list = [proposed_actions[aid] for aid in self.agent_ids]
        p = -torch.cat(p_list, dim=0)

        # 2. Build CBF constraints for all unique agent pairs
        G_list, h_list = [], []
        agent_pairs = list(itertools.combinations(self.agent_ids, 2))

        for aid1, aid2 in agent_pairs:
            idx1 = self.agent_id_to_idx[aid1]
            idx2 = self.agent_id_to_idx[aid2]

            pos1 = agent_positions[aid1]
            pos2 = agent_positions[aid2]

            # Barrier function: h(x) = ||pos1 - pos2||^2 - d_min^2 >= 0
            dist_vec = pos1 - pos2
            dist_sq = torch.sum(dist_vec**2)
            h_val = dist_sq - self.min_safe_distance_sq

            # Constraint is: L_f h + L_g h * u >= -gamma * h(x)
            # With integrator dynamics (pos_dot = u), L_f h = 0.
            # L_g h * u = 2 * (pos1 - pos2) * (u1 - u2)
            # So, the constraint for the QP is: -2 * (pos1-pos2) * u1 + 2 * (pos1-pos2) * u2 <= gamma * h(x)

            G_row = torch.zeros(self.total_action_dim, device=device)
            grad_h_part = -2 * dist_vec

            G_row[idx1 * self.action_dim : (idx1 + 1) * self.action_dim] = grad_h_part
            G_row[idx2 * self.action_dim : (idx2 + 1) * self.action_dim] = -grad_h_part

            G_list.append(G_row)
            h_list.append(self.gamma * h_val)

        G = torch.stack(G_list, dim=0)
        h = torch.stack(h_list, dim=0)

        # 3. Solve the joint QP
        # The qp_solver expects batched inputs, so we add a batch dimension.
        try:
            safe_actions_vec_b = qp_solver(Q.unsqueeze(0), p.unsqueeze(0), G.unsqueeze(0), h.unsqueeze(0), solver_args={'check_Q_spd': False})
            safe_actions_vec = safe_actions_vec_b.squeeze(0)
        except Exception:
            # If the QP fails (e.g., infeasible), fall back to a safe but suboptimal action (e.g., zero velocity).
            # This is a critical fallback for robustness.
            print("Warning: QP solver failed. Falling back to zero-velocity actions.")
            safe_actions_vec = torch.zeros_like(p)


        # 4. Split the joint safe action back into a dictionary
        safe_actions_split = torch.split(safe_actions_vec, self.action_dim, dim=0)
        safe_actions_dict = {aid: action for aid, action in zip(self.agent_ids, safe_actions_split)}

        return safe_actions_dict