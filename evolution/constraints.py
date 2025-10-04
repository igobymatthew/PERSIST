# persist/evolution/constraints.py
from typing import Dict

def viability_policy(ind) -> bool:
    # e.g., architecture size < budget, no forbidden ops, safety rulebook satisfied
    if "budget" in ind["genes"] and "params" in ind["genes"]:
        return ind["genes"]["params"] <= ind["genes"]["budget"]
    return True

def penalize_fitness(f: Dict[str,float], violated: bool) -> Dict[str,float]:
    if violated:  # hard-penalty or add constraint objective
        f = f.copy()
        f["constraint_violation"] = 1.0
    return f