"""Turn an AMI headset into an approximate training target for the table mic.

AMI has no clean target. The speaker's own headset (IHM) is the nearest thing,
but used raw it is wrong three ways (decisions-m8.md 2026-10-12):

  * TIMING. The headset hears the speaker a few ms before the table mic. A model
    with zero lookahead cannot produce words before they reach its input, and
    SI-SDR punishes even a few ms of shift.  -> align
  * LEVEL. The headset is far louder than the speaker is at the table mic, so
    the volume-matching term would teach a boost.  -> rescale
  * BLEED. The other speaker leaks into the headset ~21 dB down, and Gemini hears
    52 % of their words in the raw headset.  -> gate

Gating removes bleed wherever only the other person talks. Bleed under the
speaker's own voice stays. The result is an APPROXIMATE target, never ground
truth. All three steps are ours (Claude-originated, declared in decisions-m8.md);
only the delay estimator is borrowed, from Knapp & Carter (1976).
"""

import numpy as np

SAMPLE_RATE = 16000


def activity_mask(intervals, start_s, n, pad_s=0.0, merge_gap_s=0.0, sr=SAMPLE_RATE):
    """Boolean mask over n samples from start_s: True inside any interval.

    intervals: (start, end) seconds in meeting time. Each is widened by pad_s on
    both sides; gaps shorter than merge_gap_s between them are filled.
    """
    spans = sorted((a - pad_s, b + pad_s) for a, b in intervals)
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    mask = np.zeros(n, dtype=bool)
    for a, b in merged:
        lo = max(0, int(round((a - start_s) * sr)))
        hi = min(n, int(round((b - start_s) * sr)))
        if hi > lo:
            mask[lo:hi] = True
    return mask


def faded(mask, fade):
    """Float gain from a boolean mask, with raised-cosine ramps of `fade` samples
    INSIDE each kept span, so gated samples stay exactly zero."""
    gain = mask.astype(np.float32)
    if fade <= 0:
        return gain
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        k = min(fade, (b - a) // 2)        # a span shorter than two fades gets shorter ramps
        if k:
            ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)).astype(np.float32)
            gain[a:a + k] *= ramp
            gain[b - k:b] *= ramp[::-1]
    return gain


def gcc_phat_lag(reference, signal, max_lag):
    """Delay of `reference` behind `signal`, in samples, by GCC-PHAT.

    Knapp & Carter (1976), "The generalized correlation method for estimation of
    time delay", IEEE Trans. ASSP 24(4). Phase transform: whiten the
    cross-spectrum so the peak is sharp and reverb-robust, then take the lag of
    its maximum within +-max_lag. Positive = the reference (table mic) hears it
    LATER than the signal (headset). Returns (lag, peak height).
    """
    n = 1 << int(np.ceil(np.log2(len(reference) + len(signal))))
    cross = np.fft.rfft(reference, n) * np.conj(np.fft.rfft(signal, n))
    cross /= np.abs(cross) + 1e-12
    r = np.fft.irfft(cross, n)
    candidates = np.concatenate([r[-max_lag:], r[:max_lag + 1]])   # lags -max..+max
    i = int(np.argmax(candidates))
    return i - max_lag, float(candidates[i])


def _fit_mask(n, own, other, cfg, sr=SAMPLE_RATE):
    """Samples to fit delay and gain on: this speaker alone, if there is at least
    `min_solo_s` of it in the window, else the whole window."""
    solo = activity_mask(own, 0.0, n) & ~activity_mask(other, 0.0, n)
    use_solo = solo.sum() >= cfg["align"]["min_solo_s"] * sr
    return (solo if use_solo else np.ones(n, dtype=bool)), bool(use_solo), float(solo.sum()) / sr


def estimate_lag(mixture, headset_ext, margin, own, other, cfg, sr=SAMPLE_RATE):
    """(lag in samples, GCC-PHAT peak) of the table mic behind the headset,
    fitted on this speaker's solo stretches in the window."""
    n = len(mixture)
    max_lag = int(round(cfg["align"]["max_lag_ms"] * sr / 1000))
    assert max_lag < margin, "the fetch margin must exceed the lag search"
    fit, _, _ = _fit_mask(n, own, other, cfg, sr)
    headset = headset_ext[margin:margin + n]
    return gcc_phat_lag(mixture * fit, headset * fit, max_lag)


def process_headset(mixture, headset_ext, margin, own, other, cfg, lag, sr=SAMPLE_RATE):
    """Aligned, rescaled, gated headset, the same length as `mixture`.

    mixture:     table mic over the window, n samples.
    headset_ext: the headset over the window plus `margin` samples each side,
                 so it can be shifted by up to `margin` without running out.
    own, other:  activity intervals (seconds, window-relative) of this speaker
                 and the other one; own also carries its pad for the gate.
    cfg:         the `target:` block of the data config.
    lag:         the delay to apply, in samples. Chosen by the caller: the
                 window's own estimate_lag() when confident, else the speaker's
                 meeting-level value (scripts/build_ami_train.py).
    """
    n = len(mixture)
    assert abs(lag) < margin, "the fetch margin must exceed the applied lag"
    fit, use_solo, solo_s = _fit_mask(n, own, other, cfg, sr)

    # 1. Align: shift the headset LATER by the lag.
    aligned = headset_ext[margin - lag:margin - lag + n]

    # 2. Rescale: least-squares gain onto the table mic over the same samples.
    num = float(np.dot(mixture[fit], aligned[fit]))
    den = float(np.dot(aligned[fit], aligned[fit])) + 1e-12
    gain = num / den

    # 3. Gate: zero outside this speaker's own (padded, merged) activity.
    keep = activity_mask(own, 0.0, n, pad_s=cfg["gate"]["pad_s"],
                         merge_gap_s=cfg["gate"]["merge_gap_s"])
    fade = int(round(cfg["gate"]["fade_ms"] * sr / 1000))
    out = (gain * aligned * faded(keep, fade)).astype(np.float32)

    info = {"lag_ms": round(1000 * lag / sr, 3), "fit_on_solo": use_solo,
            "solo_s": round(solo_s, 3),
            "gain_db": round(20 * np.log10(abs(gain) + 1e-12), 2), "gain_sign": int(np.sign(gain)),
            "gated_frac": round(1.0 - float(keep.mean()), 4)}
    return out, info
