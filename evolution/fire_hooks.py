# persist/evolution/fire_hooks.py

def apply_fire(ind, intensity=1.5):
    """
    Applies a "fire" event to an individual, typically by increasing mutation rates.
    """
    # Example: increase a meta-parameter for mutation radius
    if "meta" not in ind:
        ind["meta"] = {}

    ind["meta"]["mutation_radius"] = ind["meta"].get("mutation_radius", 1.0) * intensity

    # Example: decrease an activation threshold
    if "genes" in ind and "activation_threshold" in ind["genes"]:
        ind["genes"]["activation_threshold"] *= 0.5

    return ind