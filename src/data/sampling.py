"""Draws the mixture parameters for one trial, with the two difficulty regimes of B12 (decisions-m0.md 2026-08-13).

A parameter is either a fixed number or a [low, high] range drawn uniformly. One regime is drawn per trial, and a regime only narrows some of the default ranges.
"""


# this draws one value. A fixed number uses no randomness, and a range uses exactly one uniform draw
def draw(rng, spec):
    if isinstance(spec, list):
        return float(rng.uniform(spec[0], spec[1]))
    return float(spec)


# this draws the trial's regime once, or returns None for a split without regimes
def draw_regime(rng, regimes):
    """The names are sorted so the draw does not depend on the order of the YAML file."""
    if not regimes:
        return None
    weights = regimes["weights"]
    names = sorted(weights)
    weight_values = []
    for name in names:
        weight_values.append(float(weights[name]))
    draw_point = rng.uniform() * sum(weight_values)
    cumulative = 0.0
    for name in names:
        cumulative += float(weights[name])
        if draw_point < cumulative:
            return name
    # Only float rounding can reach this line.
    return names[-1]


# this merges a split's own settings over the defaults, and keeps its regimes block apart so it is never mistaken for a parameter
def split_config(config, split):
    """The eval splits set regimes to null, so every parameter is drawn from the wide default ranges."""
    settings = dict(config["defaults"])
    regimes = None
    for key, value in config["splits"][split].items():
        if key == "regimes":
            regimes = value
        else:
            settings[key] = value
    return settings, regimes


# this gives the parameter ranges for one trial: the split's settings, narrowed by the regime
def resolve(settings, regimes, regime):
    """The hard regime has no block of its own, so it keeps the full ranges."""
    params = dict(settings)
    if regime is not None and regime in regimes:
        for key, spec in regimes[regime].items():
            params[key] = spec
    return params
