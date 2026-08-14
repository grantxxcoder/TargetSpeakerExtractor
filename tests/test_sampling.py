"""Unit tests for src/data/sampling.py (B12, PR1).

Two properties matter more than the rest and are tested hardest:

  1. The existing [lo, hi] syntax draws exactly what build_manifest.py draws
     today, so PR2's byte-identical manifest test can pass.
  2. Every spec consumes a fixed, known number of uniforms, so the logged seed
     keeps meaning the same thing across library versions.
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.data.sampling import REGIME_SCOPED, draw, draw_regime, resolve

SEED = 42


def rng():
    return np.random.default_rng(SEED)


def consumed(spec, n=1):
    """How many uniforms draw() takes off the stream."""
    r = rng()
    for _ in range(n):
        draw(r, spec)
    probe = r.uniform()
    ref = rng()
    for k in range(64):
        if ref.uniform() == probe:
            return k
    raise AssertionError("probe draw not found in the first 64 uniforms")


# --- draw: uniform is unchanged from today --------------------------------

def test_list_spec_matches_the_current_implementation():
    # scripts/build_manifest.py:56 — float(rng.uniform(*value))
    assert draw(rng(), [-5.0, 15.0]) == float(rng().uniform(-5.0, 15.0))


def test_dict_uniform_matches_list_form():
    assert draw(rng(), {"dist": "uniform", "lo": 0.15, "hi": 0.6}) == \
           draw(rng(), [0.15, 0.6])


def test_uniform_consumes_one_uniform():
    assert consumed([0.0, 20.0]) == 1


def test_scalar_and_fixed_consume_nothing():
    assert draw(rng(), 0.75) == 0.75
    assert draw(rng(), {"dist": "fixed", "value": 0.75}) == 0.75
    assert consumed(0.75) == 0
    assert consumed({"dist": "fixed", "value": 0.75}) == 0


def test_uniform_stays_in_band():
    r = rng()
    vals = [draw(r, [0.66, 2.0]) for _ in range(2000)]
    assert min(vals) >= 0.66 and max(vals) <= 2.0


# --- draw: truncnorm ------------------------------------------------------

TN = {"dist": "truncnorm", "lo": 0.25, "hi": 0.5, "mu": 0.35, "sigma": 0.1}


def test_truncnorm_consumes_exactly_one_uniform():
    # The reason the inverse-CDF form was mandated: rvs() rejection sampling
    # consumes a variable number, so a scipy upgrade would move the stream.
    assert consumed(TN) == 1
    assert consumed(TN, n=5) == 5


def test_truncnorm_is_inverse_cdf_of_that_uniform():
    u = rng().uniform()
    a, b = norm.cdf((0.25 - 0.35) / 0.1), norm.cdf((0.5 - 0.35) / 0.1)
    assert draw(rng(), TN) == pytest.approx(0.35 + 0.1 * norm.ppf(a + u * (b - a)))


def test_truncnorm_stays_in_band():
    r = rng()
    vals = [draw(r, TN) for _ in range(5000)]
    assert min(vals) >= 0.25 and max(vals) <= 0.5


def test_truncnorm_concentrates_near_mu():
    r = rng()
    sym = {"dist": "truncnorm", "lo": 0.0, "hi": 1.0, "mu": 0.5, "sigma": 0.15}
    vals = [draw(r, sym) for _ in range(20000)]
    assert np.mean(vals) == pytest.approx(0.5, abs=0.01)
    assert np.std(vals) < 0.15  # truncation narrows it; uniform would be 0.289


def test_truncnorm_is_reproducible():
    assert draw(rng(), TN) == draw(rng(), TN)


@pytest.mark.parametrize("bad", [
    {"dist": "truncnorm", "lo": 0.5, "hi": 0.5, "mu": 0.5, "sigma": 0.1},
    {"dist": "truncnorm", "lo": 0.5, "hi": 0.2, "mu": 0.3, "sigma": 0.1},
    {"dist": "truncnorm", "lo": 0.0, "hi": 1.0, "mu": 0.5, "sigma": 0.0},
])
def test_truncnorm_rejects_impossible_specs(bad):
    with pytest.raises(ValueError):
        draw(rng(), bad)


# --- draw: malformed specs fail loudly ------------------------------------

@pytest.mark.parametrize("bad", [
    [1.0], [1.0, 2.0, 3.0],
    {"dist": "beta", "a": 2, "b": 5},   # dropped in B12
    {"lo": 0.0, "hi": 1.0},             # no dist
])
def test_bad_specs_raise(bad):
    with pytest.raises((ValueError, TypeError)):
        draw(rng(), bad)


def test_string_spec_raises():
    with pytest.raises(TypeError):
        draw(rng(), "0.5")


# --- draw_regime ----------------------------------------------------------

CFG = {
    "defaults": {
        "sir_db": [-5.0, 15.0],
        "snr_db": [0.0, 20.0],
        "overlap_ratio": [0.2, 0.7],
        "t60_s": [0.15, 0.6],
        "source_distance_m": [0.66, 2.0],
        "target_activity_ratio": 0.75,
        "same_gender_fraction": 0.5,
        "enrollment_length_s": 5.0,
        "mixture_length_s": [15.0, 20.0],
    },
    "regimes": {
        "weights": {"base": 0.6, "hard": 0.4},
        "base": {
            "sir_db": [0.0, 12.0],
            "snr_db": [8.0, 20.0],
            "overlap_ratio": [0.1, 0.45],
            "t60_s": [0.25, 0.5],
            "source_distance_m": [0.66, 1.4],
        },
    },
}


def test_regime_weights_are_honoured():
    r = rng()
    names = [draw_regime(r, CFG) for _ in range(20000)]
    assert names.count("base") / len(names) == pytest.approx(0.6, abs=0.015)
    assert set(names) == {"base", "hard"}


def test_regime_consumes_one_uniform():
    r = rng()
    draw_regime(r, CFG)
    probe = r.uniform()
    ref = rng()
    ref.uniform()
    assert probe == ref.uniform()


def test_regime_is_none_when_config_declares_none():
    assert draw_regime(rng(), {"defaults": CFG["defaults"]}) is None


def test_regime_draw_ignores_yaml_key_order():
    flipped = {**CFG, "regimes": {**CFG["regimes"],
                                  "weights": {"hard": 0.4, "base": 0.6}}}
    assert draw_regime(rng(), CFG) == draw_regime(rng(), flipped)


def test_zero_weights_raise():
    bad = {**CFG, "regimes": {"weights": {"base": 0.0, "hard": 0.0}}}
    with pytest.raises(ValueError):
        draw_regime(rng(), bad)


# --- resolve --------------------------------------------------------------

def test_hard_inherits_defaults_wholesale():
    assert resolve(CFG, "hard") == CFG["defaults"]


def test_no_regime_is_a_no_op():
    # The path PR2's byte-identical acceptance test runs through.
    assert resolve({"defaults": CFG["defaults"]}, None) == CFG["defaults"]


def test_base_overrides_exactly_its_declared_keys():
    base, defaults = resolve(CFG, "base"), CFG["defaults"]
    changed = {k for k in defaults if base[k] != defaults[k]}
    assert changed == set(CFG["regimes"]["base"])
    assert changed <= set(REGIME_SCOPED)


def test_base_is_a_subrange_of_hard_where_it_only_narrows():
    # Why `regime` records provenance, not difficulty: a hard draw can land
    # anywhere inside base's bands. Not enforced by resolve() — see the
    # overlap_ratio test below for the case where it is broken on purpose.
    base, hard = resolve(CFG, "base"), resolve(CFG, "hard")
    for key in ("sir_db", "snr_db", "t60_s", "source_distance_m"):
        assert base[key][0] >= hard[key][0] and base[key][1] <= hard[key][1]


def test_overlap_ratio_base_band_reaches_below_the_default_floor():
    # Recorded, not asserted away. decisions.md says base is a sub-range of
    # hard; difficulty-dial.md §2 proposes overlap_ratio base [0.1, 0.45]
    # against a default floor of 0.2, because that floor is itself the B9 bug
    # (no present trial can have zero overlap, so silent-target trials are
    # detectable at AUC 1.000). resolve() therefore must not enforce
    # containment. Whichever document is wrong, it is a PR2/PR3 decision.
    assert resolve(CFG, "base")["overlap_ratio"][0] < \
           resolve(CFG, "hard")["overlap_ratio"][0]


def test_resolve_does_not_mutate_the_config():
    resolve(CFG, "base")["sir_db"] = "clobbered"
    assert CFG["defaults"]["sir_db"] == [-5.0, 15.0]


def test_global_parameters_are_untouched_by_either_regime():
    for regime in ("base", "hard"):
        params = resolve(CFG, regime)
        for key in ("same_gender_fraction", "enrollment_length_s",
                    "mixture_length_s"):
            assert params[key] == CFG["defaults"][key]


def test_unknown_regime_raises():
    with pytest.raises(KeyError):
        resolve(CFG, "medium")


def test_overriding_a_global_parameter_raises():
    bad = {**CFG, "regimes": {**CFG["regimes"],
                              "base": {"enrollment_length_s": 10.0}}}
    with pytest.raises(KeyError):
        resolve(bad, "base")


def test_overriding_a_key_absent_from_defaults_raises():
    # Catches a typo'd or renamed parameter rather than silently adding it.
    defaults = {k: v for k, v in CFG["defaults"].items() if k != "t60_s"}
    bad = {"defaults": defaults, "regimes": CFG["regimes"]}
    with pytest.raises(KeyError):
        resolve(bad, "base")


# --- the two together -----------------------------------------------------

def test_every_parameter_of_a_trial_comes_from_one_regime():
    r = rng()
    for _ in range(500):
        params = resolve(CFG, draw_regime(r, CFG))
        assert draw(r, params["sir_db"]) >= -5.0
        assert 0.15 <= draw(r, params["t60_s"]) <= 0.6
