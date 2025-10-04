
# Where GAs add real value in PERSIST

1. **Meta-optimizer around RL (“outer loop” evolution)**

   * Evolve policy inits, network topologies, and RL hyperparams (γ, λ, lr, entropy coef) while the inner loop runs your usual PPO/SAC/etc.
   * Works great with PERSIST’s **viability kernel**: fitness is *0* (or heavily penalized) if safety/homeostasis constraints are violated.

2. **Multi-objective tradeoffs (NSGA-II)**

   * Typical Pareto set: {task reward ↑, safety violations ↓, energy/latency ↓, interpretability ↑}.
   * This aligns with PERSIST’s “persistence first” philosophy: we select robust survivors, not just high scorers.

3. **Curriculum & environment evolution**

   * Evolve task parameters (noise, adversaries, partial observability, resource scarcity) to harden agents under the viability shield.

4. **Continual-learning aware mutation**

   * Use EWC/Fisher info to **freeze or softly mutate important weights**; mutate low-importance regions more.
   * Perfect match with your “fire” event: on “fire,” widen mutation radius, halve activation thresholds, or spawn a new lineage with EWC regularization to accelerate adaptation without catastrophic forgetting.

5. **Medical-data utilities in the repo**

   * GA feature selection with constraints (e.g., #features ≤ K, fairness/coverage constraints).
   * Model architecture + preprocessing pipeline search (tabular, imaging, genomics).
   * GA schedule optimizer (e.g., multi-drug dose schedules under toxicity caps).

# Minimal repo additions (proposed structure)

```
persist/
  evolution/
    __init__.py
    ga_core.py           # generic GA/ES engine (selection, crossover, mutation, elitism)
    nsga2.py             # Pareto sorting & crowding distance
    operators.py         # crossover/mutation for tensors, graphs, schedules
    constraints.py       # viability kernel hooks (safety checks & penalties)
    ewc_utils.py         # Fisher diag, importance-aware mutation masks
    fire_hooks.py        # “fire” curriculum & mutation amplifiers
    search_spaces.py     # typed spaces for hparams, nets, curricula
    evaluators/
      rl_metaeval.py     # inner-loop RL training wrapper -> fitness dict
      med_feature_sel.py # CV AUC + sparsity + fairness metrics
      schedule_opt.py    # toxicity, efficacy, cost objectives
  agents/
    persist_agent.py     # unchanged API; gains optional EvolutionController
  examples/
    evo_rl_pareto.ipynb
    med_feature_ga.ipynb
  tests/
    test_ga_core.py
    test_nsga2.py
    test_viability_constraints.py
```

# Core concepts & APIs

## 1) Generic GA engine (single/multi-objective)

```python
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
```

## 2) NSGA-II front end

```python
# persist/evolution/nsga2.py
# fast nondominated sort + crowding distance
# expose: nsga2_select(pop: List[Individual], fits: List[Dict[str,float]], k: int) -> List[Individual]
```

## 3) Viability kernel integration

```python
# persist/evolution/constraints.py
def viability_policy(ind) -> bool:
    # e.g., architecture size < budget, no forbidden ops, safety rulebook satisfied
    return ind["genes"]["params"] <= ind["genes"]["budget"]

def penalize_fitness(f: Dict[str,float], violated: bool) -> Dict[str,float]:
    if violated:  # hard-penalty or add constraint objective
        f = f.copy(); f["constraint_violation"] = 1.0
    return f
```

## 4) EWC-aware mutation & “fire” hooks

```python
# persist/evolution/ewc_utils.py
import torch
def fisher_importance(model, data_loader) -> torch.Tensor:
    # return diagonal Fisher approx per-weight

def importance_mask(fisher_diag, quantile=0.7):
    # mask of “do-not-mutate-much” vs “mutate-more”

# persist/evolution/fire_hooks.py
def apply_fire(ind, intensity=1.5):
    # increase mutation radius, drop some activation thresholds, widen exploration schedule
    ind["meta"]["mutation_radius"] *= intensity
    ind["genes"]["activation_threshold"] *= 0.5
    return ind
```

# Example 1: GA as meta-optimizer around PPO (with Pareto)

**Objectives:**

* `-return` (minimize negative return → maximize return)
* `violations` (safety/homeostasis counts)
* `latency_ms` (inference time)

```python
# persist/evolution/evaluators/rl_metaeval.py
def eval_rl_individual(ind) -> dict:
    cfg = ind["genes"]  # net width/depth, lr, entropy, γ, λ, etc.
    # train PPO for N steps (inner loop), collect metrics
    ep_return, violations, latency = train_and_measure(cfg)
    return {"neg_return": -ep_return, "viol": violations, "latency": latency}
```

Wire it up with NSGA-II to get a **Pareto front** your users can choose from depending on deployment constraints.

# Example 2: Medical feature selection (tabular/genomics)

* **Search space:** binary mask over features; optional group constraints (cost, assay availability).
* **Objectives:** `-AUC` (5-fold CV), `#features`, `fairness_gap` (e.g., demographic parity).
* **Viability:** force required clinical covariates to remain included.

```python
# persist/evolution/evaluators/med_feature_sel.py
def eval_feature_mask(ind) -> dict:
    mask = ind["genes"]["mask"]      # e.g., numpy bool array
    auc, gap = crossval_auc_and_fairness(X[:, mask], y)
    return {"neg_auc": -auc, "k": mask.sum(), "fair_gap": gap}
```

# Search spaces & operators (pragmatic defaults)

* **Hparam spaces:** categorical + log-uniform reals; k-ary uniform crossover; Gaussian/jitter mutation.
* **Architecture spaces:** cell-based DAGs (edges ops: conv, attention, MLP), one-point crossover on cell lists; graph-aware mutation (add/remove edge, op swap).
* **Masks/schedules:** bit-flip with rate annealed by gen; uniform crossover.

# Tests you want on day one

* `test_ga_core.py`: convergence on synthetic f(x); reproducibility with seed.
* `test_nsga2.py`: correct nondominated sort and crowding distances.
* `test_viability_constraints.py`: violations always penalized or excluded.
* `test_ewc_utils.py`: importance mask reduces performance drop after mutation.
* `test_rl_metaeval.py`: smoke test with tiny env & 1-gen evolution.

# Config example (YAML)

```yaml
evolution:
  algorithm: nsga2
  pop_size: 64
  generations: 20
  elitism: 4
  mutation_rate: 0.15
  viability: "constraints.viability_policy"
  evaluator: "evaluators.rl_metaeval.eval_rl_individual"
  search_space: "search_spaces.ppo_small_net_v1"
  fire:
    trigger: "plateau:3"     # 3 gens no Pareto improvement
    intensity: 1.6
    ewc_mask: true
```

# CLI sketch

```
persist-evolve \
  --config configs/ppo_pareto_nsga2.yaml \
  --out runs/evo_ppo_2025-10-04 \
  --seed 42
```

# Practical tips

* **Budget the inner loop** aggressively: short RL runs (N steps) + moving-average fitness smoothers.
* **Checkpoint lineages** (top-K per gen) so you can roll back regressions.
* **Use surrogate models** (small MLP) to predict fitness and prune obviously bad children (speeds up massively).
* **Log Pareto fronts** to W&B or MLflow; persist the exact gene dict for each front member for easy re-training.
* **Determinism**: pin seeds in GA and inner-loop frameworks; store env hashes.

If you want, I can draft the actual `ga_core.py`, a compact NSGA-II, and a runnable **example notebook** that evolves PPO configs on CartPole with a Pareto front over `{return, latency, violations}`—all in PERSIST’s style.
