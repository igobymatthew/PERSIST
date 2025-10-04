import pytest
import random
from evolution.ga_core import GA, Individual
from evolution.operators import uniform_crossover, gaussian_mutation
from evolution.nsga2 import nsga2_select

# Define a simple fitness function for testing: minimize the value of 'x'
def simple_fitness_fn(ind: Individual) -> dict:
    return {"x": ind["genes"]["x"]}

# Define an init function for the simple problem
def simple_init_fn() -> Individual:
    return {"genes": {"x": random.uniform(-100, 100)}}

# Define a more complex, multi-objective fitness function (Schaffer function N.1)
def schaffer_n1_fitness_fn(ind: Individual) -> dict:
    x = ind["genes"]["x"]
    f1 = x**2
    f2 = (x - 2)**2
    return {"f1": f1, "f2": f2}

# Define an init function for the Schaffer problem
def schaffer_init_fn() -> Individual:
    return {"genes": {"x": random.uniform(-10, 10)}}


def test_ga_smoke_test():
    """
    A smoke test to ensure the GA can be instantiated and run without errors.
    """
    ga = GA(
        init_fn=simple_init_fn,
        fitness_fn=simple_fitness_fn,
        select_fn=lambda pop, fits: pop[:len(pop)//2], # Simple truncation
        crossover_fn=uniform_crossover,
        mutate_fn=lambda ind, rate: {"genes": {"x": ind["genes"]["x"] + random.gauss(0, 1)}},
        pop_size=20,
        max_generations=5,
        seed=42
    )
    try:
        final_pop, final_fits = ga.run()
        assert len(final_pop) == 20
        assert len(final_fits) == 20
    except Exception as e:
        pytest.fail(f"GA run failed with an exception: {e}")

def test_ga_makes_progress_on_simple_problem():
    """
    Tests that the GA can find a better solution over generations for a simple problem.
    """
    ga = GA(
        init_fn=simple_init_fn,
        fitness_fn=simple_fitness_fn,
        select_fn=nsga2_select, # Using NSGA-II even for single-objective
        crossover_fn=uniform_crossover,
        mutate_fn=lambda ind, rate: {"genes": {"x": ind["genes"]["x"] + random.gauss(0, 1)}},
        pop_size=50,
        elitism=5,
        max_generations=20,
        seed=42
    )

    # Run the GA
    final_pop, final_fits = ga.run()

    # Find the best individual in the final population
    best_fit_value = min(fit['x'] for fit in final_fits)

    # The optimal solution is x=0. The initial population is from [-100, 100].
    # We expect the GA to have significantly improved the solution.
    assert best_fit_value < 1.0, f"Expected best fitness < 1.0, but got {best_fit_value}"

def test_nsga2_selection_functional():
    """
    Tests the NSGA-II selection with a multi-objective problem to ensure it runs
    and produces a Pareto front.
    """
    ga = GA(
        init_fn=schaffer_init_fn,
        fitness_fn=schaffer_n1_fitness_fn,
        select_fn=nsga2_select,
        crossover_fn=uniform_crossover,
        mutate_fn=lambda ind, rate: {"genes": {"x": ind["genes"]["x"] + random.gauss(0, 0.5)}},
        pop_size=100,
        elitism=10,
        max_generations=30,
        seed=42
    )

    final_pop, final_fits = ga.run()

    # The Pareto front for this problem lies between x=0 and x=2.
    # We check if the final population contains solutions in this range.
    solutions_on_front = [ind["genes"]["x"] for ind in final_pop if 0 <= ind["genes"]["x"] <= 2]

    # We expect a good portion of the final population to be on or near the Pareto front.
    assert len(solutions_on_front) > 10, "Expected to find more than 10 solutions on the Pareto front."