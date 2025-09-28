import numpy as np
import os
import itertools
import yaml
import argparse

def get_env_dynamics(config):
    """
    Extracts a simplified, deterministic version of the environment dynamics
    from the config for the purpose of reachability analysis.
    """
    # For this analysis, we assume a simplified dynamics model.
    # The agent can choose to move towards/away from hot/hazard tiles,
    # which we model as a discrete set of actions on the internal state.
    actions = [
        np.array([0.0, 0.1]),   # Move towards heat
        np.array([0.0, -0.05]), # Move away from heat
        np.array([-0.01, 0.0]), # Normal energy decay
    ]
    return actions

def get_viability_constraints(config):
    """
    Parses viability constraints from the structured config.
    """
    constraints = []
    dim_map = {'energy': 0, 'temp': 1, 'integrity': 2}

    for c_obj in config['viability']['constraints']:
        variable = c_obj['variable']
        if variable not in dim_map:
            continue # Only consider energy and temp for this 2D analysis

        dim_idx = dim_map[variable]
        op = c_obj['operator']
        threshold = c_obj['threshold']

        if op == 'in':
            min_val, max_val = threshold[0], threshold[1]
            constraints.append({'dim': dim_idx, 'type': 'min', 'val': min_val})
            constraints.append({'dim': dim_idx, 'type': 'max', 'val': max_val})
        elif op == '>=':
            constraints.append({'dim': dim_idx, 'type': 'min', 'val': threshold})
        elif op == '<=':
            constraints.append({'dim': dim_idx, 'type': 'max', 'val': threshold})

    return constraints


def compute_and_save_viable_set(
    config,
    output_dir="data",
    grid_resolution=50,
    horizon=10,
):
    """
    Computes the viable set for a 2D system (energy, temp) using grid-based
    backward reachability analysis (value iteration).

    Args:
        config (dict): The agent's configuration dictionary.
        output_dir (str): Directory to save the viable set data.
        grid_resolution (int): Resolution of the state space grid.
        horizon (int): The time horizon for the reachability analysis.
    """
    print("Running grid-based backward reachability analysis...")

    # 1. Create a discrete grid for the internal state space (energy, temp)
    energy_levels = np.linspace(0, 1, grid_resolution)
    temp_levels = np.linspace(0, 1, grid_resolution)
    grid_shape = (grid_resolution, grid_resolution)

    # State vectors for each grid point
    states = np.array(list(itertools.product(energy_levels, temp_levels)))

    # 2. Identify the initial set of "unsafe" states based on constraints
    constraints = get_viability_constraints(config)
    unsafe_mask = np.zeros(grid_shape, dtype=bool)
    for c in constraints:
        dim, c_type, val = c['dim'], c['type'], c['val']
        if dim == 0: # Energy
            if c_type == 'min': unsafe_mask[energy_levels < val, :] = True
            if c_type == 'max': unsafe_mask[energy_levels > val, :] = True
        elif dim == 1: # Temperature
            if c_type == 'min': unsafe_mask[:, temp_levels < val] = True
            if c_type == 'max': unsafe_mask[:, temp_levels > val] = True

    # This is V_0, the set of states to avoid at all costs.
    V = unsafe_mask.copy()

    # 3. Get the simplified dynamics
    actions = get_env_dynamics(config)

    # 4. Perform backward reachability (value iteration)
    print(f"Performing backward reachability for a horizon of {horizon} steps...")
    for t in range(horizon):
        V_prev = V.copy()
        # Iterate over all states in the grid
        for i, j in itertools.product(range(grid_resolution), range(grid_resolution)):
            # If the state is already known to be unsafe, skip it
            if V[i, j]:
                continue

            # For a state to be "safe" at this step, there must exist at least one
            # action that keeps it in the "safe" set V_prev.
            # We check the inverse: is it true that for ALL actions, the next state
            # is unsafe? If so, this state becomes unsafe.
            all_actions_lead_to_unsafe = True
            for action in actions:
                current_state = np.array([energy_levels[i], temp_levels[j]])
                next_state = np.clip(current_state + action, 0, 1)

                # Find the grid cell corresponding to the next state
                next_idx_i = np.argmin(np.abs(energy_levels - next_state[0]))
                next_idx_j = np.argmin(np.abs(temp_levels - next_state[1]))

                # If this action leads to a state that was safe in the previous step,
                # then this current state is not "all actions lead to unsafe".
                if not V_prev[next_idx_i, next_idx_j]:
                    all_actions_lead_to_unsafe = False
                    break # Found a safe action, no need to check others

            if all_actions_lead_to_unsafe:
                V[i, j] = True # Mark current state as unsafe

        print(f"  Step {t+1}/{horizon}: Found {V.sum()} unsafe states.")


    # The viable mask is the inverse of the final unsafe set V
    viable_mask = ~V

    # 5. Save the results
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "viable_set.npz")
    np.savez(
        output_path,
        states=states,
        viable_mask=viable_mask.ravel(),
        grid_resolution=grid_resolution
    )

    print(f"\nViable set computation complete.")
    print(f"  - Grid resolution: {grid_resolution}x{grid_resolution}")
    print(f"  - Total states: {grid_resolution**2}")
    print(f"  - Viable states: {viable_mask.sum()} ({viable_mask.sum() / (grid_resolution**2) * 100:.2f}%)")
    print(f"Saved computed viable set to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the viable set using grid-based reachability.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution for state space.")
    parser.add_argument("--horizon", type=int, default=15, help="Time horizon for backward reachability.")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save the result.")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)

    compute_and_save_viable_set(
        config=config_data,
        output_dir=args.output_dir,
        grid_resolution=args.resolution,
        horizon=args.horizon,
    )