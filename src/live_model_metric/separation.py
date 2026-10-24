"""Signal-domain separation scores: SDR, SIR and SAR.

The estimate is split into the part explained by the target, the part explained by the interferer and noise, and an artefact part that no scaled mix of the true sources explains, so the model made it.
This is the BSS_EVAL decomposition of Vincent et al. (2006), with the scale-invariant framing of Le Roux et al. (2019). It needs all three clean sources, so it only works on constructed mixtures.
"""

import numpy as np

# Floors each denominator at a fraction of its numerator, which caps every score at 10 log10(1 / TAU) = +30 dB and keeps them scale-invariant.
# The same value and reasoning as the floor on L_pres in the training objective.
TAU = 1e-3


# this projects a signal onto the space spanned by the given sources, by least squares
def project_onto(signal, sources):
    basis = np.asarray(sources, dtype=np.float64).T
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(basis, signal, rcond=None)
    return basis @ coefficients


# this gives a signal's energy: the sum of its squared samples
def energy(signal):
    return float(np.sum(signal ** 2))


# this gives the floored energy ratio of two signals in dB
def ratio_db(numerator_signal, denominator_signal):
    numerator = energy(numerator_signal)
    denominator = energy(denominator_signal)
    return 10.0 * np.log10(numerator / (denominator + TAU * numerator))


# this splits an estimate into target, interference and artefact parts and returns SDR, SIR and SAR in dB
def decompose(estimate, target, interferer, noise):
    """All four inputs are 1-D signals. They are cut to the shortest length, because the renderer pads the tail by the room's decay time.

    Scaling the estimate leaves all three scores unchanged, because the projections absorb any gain.
    """
    length = min(len(estimate), len(target), len(interferer), len(noise))
    estimate = np.asarray(estimate, dtype=np.float64)[:length]
    sources = np.stack([np.asarray(target, dtype=np.float64)[:length], np.asarray(interferer, dtype=np.float64)[:length], np.asarray(noise, dtype=np.float64)[:length]])

    target_part = project_onto(estimate, sources[:1])
    explainable_part = project_onto(estimate, sources)
    interference_part = explainable_part - target_part
    artefact_part = estimate - explainable_part
    return {"signal_to_distortion_db": ratio_db(target_part, interference_part + artefact_part), "signal_to_interference_db": ratio_db(target_part, interference_part), "signal_to_artefact_db": ratio_db(explainable_part, artefact_part)}
