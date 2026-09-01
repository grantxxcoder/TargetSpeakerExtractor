"""Signal-domain separation scores: SDR, SIR and SAR.

Splits what an estimate contains beyond the target into two piles that mean
opposite things -- interference the model failed to remove, and artefacts the
model invented. Borrowed from the BSS_EVAL decomposition of Vincent et al.
(2006); the scale-invariant framing follows Le Roux et al. (2019).

Why this exists: SI-SDR collects residual interferer, residual noise and
processing artefacts into one denominator, where they count identically per unit
of energy but are not equally damaging. See the objective section of the
methodology chapter, which states that gap as a caveat.

ARTEFACT IS DEFINED BY ELIMINATION. The estimate is projected onto the span of
the true sources; whatever cannot be explained by any scaled combination of them
was not in the microphone signal, so the model created it. This is not a
subtraction of the sources -- an estimate that attenuates the interferer to 30 %
has that counted as interference, which plain subtraction would miscount.

Requires all three clean sources, so it is computable on constructed mixtures
only, never on real recordings.
"""

from dataclasses import dataclass

import numpy as np

# Floors the denominator at a fraction of the numerator, which caps every score
# at 10*log10(1/TAU) = +30 dB and keeps them scale-invariant. An absolute
# epsilon does neither: when the artefact part is essentially zero the score
# becomes a function of the estimate's gain rather than of its content, which a
# test caught at a 17 dB swing under a x7.5 gain. Same value and same reasoning
# as `tau_pres` in the training objective, where it floors L_pres at -30 dB.
TAU = 1e-3
CEILING_DB = 30.0


def _project_onto(signal, sources):
    basis = np.asarray(sources, dtype=np.float64).T
    coefficients, *_ = np.linalg.lstsq(basis, signal, rcond=None)
    return basis @ coefficients


def _energy(signal):
    return float(np.sum(signal ** 2))


def _ratio_db(numerator_signal, denominator_signal):
    numerator = _energy(numerator_signal)
    denominator = _energy(denominator_signal)
    return 10.0 * np.log10(numerator / (denominator + TAU * numerator))


@dataclass
class SeparationScores:
    signal_to_distortion_db: float = None
    signal_to_interference_db: float = None
    signal_to_artefact_db: float = None

    def __str__(self):
        return (f"SDR {self.signal_to_distortion_db:+.2f} dB  "
                f"SIR {self.signal_to_interference_db:+.2f} dB  "
                f"SAR {self.signal_to_artefact_db:+.2f} dB")


def decompose(estimate, target, interferer, noise):
    """Split `estimate` into target, interference and artefact parts.

    All four arguments are 1-D time-domain signals. They are truncated to the
    shortest common length, because the renderer pads the tail by the room's
    decay time and a estimate written whole-clip can differ by a few samples.

    Scale-invariant by construction: the projections absorb any global gain, so
    scaling the estimate leaves all three scores unchanged.
    """
    length = min(len(estimate), len(target), len(interferer), len(noise))
    estimate = np.asarray(estimate, dtype=np.float64)[:length]
    sources = np.stack([np.asarray(target, dtype=np.float64)[:length],
                        np.asarray(interferer, dtype=np.float64)[:length],
                        np.asarray(noise, dtype=np.float64)[:length]])

    target_part = _project_onto(estimate, sources[:1])
    explainable_part = _project_onto(estimate, sources)
    interference_part = explainable_part - target_part
    artefact_part = estimate - explainable_part

    return SeparationScores(
        signal_to_distortion_db=_ratio_db(target_part,
                                          interference_part + artefact_part),
        signal_to_interference_db=_ratio_db(target_part, interference_part),
        signal_to_artefact_db=_ratio_db(explainable_part, artefact_part),
    )


def improvement_over_mixture(estimate, mixture, target, interferer, noise):
    """How much each score moved relative to doing nothing.

    Reported rather than the absolute scores because an absolute SIR depends on
    the trial's signal-to-interference ratio, which varies by construction. What
    a system can be judged on is what it added.
    """
    after = decompose(estimate, target, interferer, noise)
    before = decompose(mixture, target, interferer, noise)
    return SeparationScores(
        signal_to_distortion_db=after.signal_to_distortion_db
                                 - before.signal_to_distortion_db,
        signal_to_interference_db=after.signal_to_interference_db
                                   - before.signal_to_interference_db,
        signal_to_artefact_db=after.signal_to_artefact_db
                               - before.signal_to_artefact_db,
    )
