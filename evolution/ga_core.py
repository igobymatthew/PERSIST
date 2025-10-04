# persist/evolution/ga_core.py
from typing import Callable, Dict, Any, List, Tuple
import random

Individual = Dict[str, Any]  # {"genes": ..., "meta": ...}

class GA:
    def __init__(
        self,
        init_fn: Callable[[], Individual],
        fitness_fn: Callable[[Individual], Dict[str, float]],  # can return multiple objectives
        select_fn: Callable[[List[Individual], List[Dict[str,float]]], List[Individual]],
        crossover_fn: Callable[[Individual, Individual], Individual],
        mutate_fn: Callable[[Individual, float], Individual],
        viability_fn: Callable[[Individual], bool] = lambda ind: True,
        pop_size: int = 50,
        elitism: int = 2,
        mutation_rate: float = 0.1,
        max_generations: int = 50,
        seed: int | None = None,
    ):
        self.__dict__.update(locals())
        if seed is not None:
            random.seed(seed)

    def run(self):
        pop = [self.init_fn() for _ in range(self.pop_size)]
        for gen in range(self.max_generations):
            fits = [self.fitness_fn(ind) if self.viability_fn(ind) else {"violate": 1e9} for ind in pop]
            # sort by primary objective if single; or pass to NSGA-II if multi
            pop = self._evolve(pop, fits)
        return pop, fits

    def _evolve(self, pop, fits):
        # user can swap this for NSGA-II; default: sort by first key ascending
        key = list(fits[0].keys())[0]
        ranked = [ind for _, ind in sorted(zip(fits, pop), key=lambda x: x[0][key])]
        next_pop = ranked[:self.elitism]
        while len(next_pop) < self.pop_size:
            p1, p2 = random.sample(ranked[: len(ranked)//2], 2)
            child = self.crossover_fn(p1, p2)
            if random.random() < self.mutation_rate:
                child = self.mutate_fn(child, self.mutation_rate)
            next_pop.append(child)
        return next_pop