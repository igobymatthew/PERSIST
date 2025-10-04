# persist/evolution/operators.py
import random

def uniform_crossover(p1, p2):
    """
    Performs uniform crossover on two individuals.
    Assumes genes are dicts with the same keys.
    """
    child_genes = {}
    for key in p1["genes"]:
        child_genes[key] = random.choice([p1["genes"][key], p2["genes"][key]])
    return {"genes": child_genes}

def gaussian_mutation(ind, mutation_rate):
    """
    Applies Gaussian mutation to an individual's genes.
    Assumes genes are numeric values.
    """
    mutated_genes = {}
    for key, val in ind["genes"].items():
        if random.random() < mutation_rate:
            mutated_genes[key] = val + random.gauss(0, 0.1)
        else:
            mutated_genes[key] = val
    return {"genes": mutated_genes}