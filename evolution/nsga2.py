from typing import List, Dict, Any, Tuple

Individual = Dict[str, Any]

def fast_non_dominated_sort(
    fits: List[Dict[str, float]]
) -> List[List[int]]:
    """
    Performs a fast non-dominated sort on a list of fitness dictionaries.
    Returns a list of fronts, where each front is a list of individual indices.
    """
    pop_size = len(fits)
    fronts = [[]]
    domination_counts = [0] * pop_size
    dominated_solutions = [[] for _ in range(pop_size)]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            fit_i = list(fits[i].values())
            fit_j = list(fits[j].values())

            # Check for domination
            dominates_j = all(x <= y for x, y in zip(fit_i, fit_j)) and any(x < y for x, y in zip(fit_i, fit_j))
            dominates_i = all(y <= x for x, y in zip(fit_i, fit_j)) and any(y < x for x, y in zip(fit_i, fit_j))

            if dominates_j:
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif dominates_i:
                dominated_solutions[j].append(i)
                domination_counts[i] += 1

        if domination_counts[i] == 0:
            fronts[0].append(i)

    front_idx = 0
    while len(fronts[front_idx]) > 0:
        next_front = []
        for i in fronts[front_idx]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        front_idx += 1
        fronts.append(next_front)

    return fronts[:-1] # The last front is always empty


def crowding_distance_assignment(
    fits: List[Dict[str, float]],
    front_indices: List[int]
) -> Dict[int, float]:
    """
    Calculates the crowding distance for each individual in a front.
    """
    if not front_indices:
        return {}

    distances = {i: 0.0 for i in front_indices}
    num_objectives = len(fits[0])

    for obj_idx in range(num_objectives):
        # Sort by the current objective
        sorted_front = sorted(front_indices, key=lambda i: list(fits[i].values())[obj_idx])

        # Assign infinite distance to boundary solutions
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')

        obj_values = [list(fits[i].values())[obj_idx] for i in sorted_front]
        min_obj, max_obj = obj_values[0], obj_values[-1]

        if max_obj == min_obj:
            continue

        # Normalize and add distances for intermediate solutions
        for i in range(1, len(sorted_front) - 1):
            distances[sorted_front[i]] += (obj_values[i+1] - obj_values[i-1]) / (max_obj - min_obj)

    return distances


def nsga2_select(
    pop: List[Individual],
    fits: List[Dict[str, float]],
    k: int
) -> List[Individual]:
    """
    Selects k individuals from the population using NSGA-II.
    """
    fronts = fast_non_dominated_sort(fits)

    next_pop_indices = []
    front_idx = 0

    # Add fronts until the population size would be exceeded
    while front_idx < len(fronts) and len(next_pop_indices) + len(fronts[front_idx]) <= k:
        next_pop_indices.extend(fronts[front_idx])
        front_idx += 1

    # If not all fronts fit, use crowding distance to select from the last front
    if len(next_pop_indices) < k and front_idx < len(fronts):
        remaining_needed = k - len(next_pop_indices)
        last_front = fronts[front_idx]

        distances = crowding_distance_assignment(fits, last_front)

        # Sort by crowding distance (descending)
        sorted_by_crowding = sorted(last_front, key=lambda i: distances[i], reverse=True)

        next_pop_indices.extend(sorted_by_crowding[:remaining_needed])

    return [pop[i] for i in next_pop_indices]