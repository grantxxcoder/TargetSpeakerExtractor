"""Parameter sampling for the mixture generator: the two regimes and the
distribution shapes decided in B12.

Implements docs/decisions/decisions.md, "2026-08-13 — B12 architecture: two
regimes, a sampler layer, no relational constraints". Band values live in
docs/data/difficulty-dial.md §2.

Nothing here is wired into scripts/build_manifest.py yet — that is PR2, which
must reproduce the current manifest byte-identically. This module is tested on
its own because every number in every later experiment comes out of it.

    regime = draw_regime(rng, cfg)      # once per trial
    params = resolve(cfg, regime)       # that trial's parameter bands
    sir    = draw(rng, params["sir_db"])

Spec forms accepted by draw():

    3.0                                  fixed, consumes no randomness
    [lo, hi]                             uniform  (the existing config syntax)
    {dist: fixed,    value: x}
    {dist: uniform,  lo: a, hi: b}
    {dist: truncnorm, lo: a, hi: b, mu: m, sigma: s}
"""

from __future__ import annotations

from scipy.stats import norm

# The six parameters a regime may narrow. Everything else is global: either a
# deliberate worst case or an experimental variable B13 stratifies on
# independently, so it must not move with difficulty.
REGIME_SCOPED = (
    "sir_db", "snr_db", "overlap_ratio",
    "t60_s", "source_distance_m", "target_activity_ratio",
)


def draw(rng, spec):
    """One value from a parameter spec. Consumes at most one uniform."""
    if isinstance(spec, (int, float)):
        return float(spec)

    if isinstance(spec, list):
        if len(spec) != 2:
            raise ValueError(f"list spec must be [lo, hi], got {spec!r}")
        return float(rng.uniform(*spec))

    if not isinstance(spec, dict):
        raise TypeError(f"unsupported spec {spec!r}")

    dist = spec.get("dist")
    if dist == "fixed":
        return float(spec["value"])
    if dist == "uniform":
        return float(rng.uniform(spec["lo"], spec["hi"]))
    if dist == "truncnorm":
        return _truncnorm(rng, spec["lo"], spec["hi"], spec["mu"], spec["sigma"])
    raise ValueError(f"unknown dist {dist!r} in {spec!r}")


def _truncnorm(rng, lo, hi, mu, sigma):
    """Truncated normal by inverse CDF over a single rng.uniform draw.

    Deliberately not scipy.stats.truncnorm.rvs(random_state=rng): rvs uses
    rejection sampling, so the number of uniforms consumed per sample can
    change with a scipy version. That would silently change what `seed: 42`
    means and make the logged seed worthless. This consumes exactly one.
    """
    if not lo < hi:
        raise ValueError(f"truncnorm needs lo < hi, got [{lo}, {hi}]")
    if sigma <= 0:
        raise ValueError(f"truncnorm needs sigma > 0, got {sigma}")
    a, b = norm.cdf((lo - mu) / sigma), norm.cdf((hi - mu) / sigma)
    if b - a <= 0:
        raise ValueError(f"truncnorm window [{lo}, {hi}] has no mass under "
                         f"mu={mu}, sigma={sigma}")
    x = mu + sigma * norm.ppf(a + rng.uniform() * (b - a))
    return float(min(max(x, lo), hi))  # ppf can overshoot by a float epsilon


def draw_regime(rng, cfg):
    """The trial's regime, drawn once. None when cfg declares no regimes.

    One regime per trial, not one per parameter: with six parameters chosen
    independently the all-base case would occur in ~1.6 % of trials, so the
    base condition would barely exist. Drawn once it is exactly its weight.
    """
    regimes = cfg.get("regimes")
    if not regimes:
        return None
    weights = regimes["weights"]
    names = sorted(weights)  # sorted, so the draw does not depend on YAML order
    total = sum(float(weights[n]) for n in names)
    if total <= 0:
        raise ValueError(f"regime weights must sum to > 0, got {weights!r}")
    u = rng.uniform() * total
    acc = 0.0
    for name in names:
        acc += float(weights[name])
        if u < acc:
            return name
    return names[-1]  # float rounding only


def split_config(config, split):
    """(sampling config, flat cfg) for one split of the generator config.

    A split inherits the top-level `regimes:` block unless it declares its own;
    `regimes: null` opts out, which is how the eval splits draw every parameter
    independently from the wide ranges (decisions.md 2026-08-13). `regimes` is
    popped rather than merged, so it can never be mistaken for a parameter.
    """
    split_cfg = dict(config["splits"][split])
    regimes = split_cfg.pop("regimes", config.get("regimes"))
    cfg = {**config["defaults"], **split_cfg}
    return {"defaults": cfg, "regimes": regimes}, cfg


def resolve(cfg, regime):
    """The parameter bands for one trial: defaults, overlaid by the regime.

    `hard` inherits defaults wholesale and so needs no override block; `base`
    is therefore a sub-range of `hard`, which is why the recorded `regime`
    column is provenance and not difficulty, and must not be a B13 reporting
    stratum. A regime may only narrow the six REGIME_SCOPED parameters.
    """
    params = dict(cfg["defaults"])
    if regime is None:
        return params

    regimes = cfg.get("regimes") or {}
    if regime not in regimes.get("weights", {}):
        raise KeyError(f"unknown regime {regime!r}")

    for key, spec in (regimes.get(regime) or {}).items():
        if key not in REGIME_SCOPED:
            raise KeyError(f"regime {regime!r} overrides {key!r}, which is "
                           f"global; only {list(REGIME_SCOPED)} are scoped")
        if key not in params:
            raise KeyError(f"regime {regime!r} overrides {key!r}, absent from "
                           f"defaults")
        params[key] = spec
    return params
