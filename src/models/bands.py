"""Band plans: how many frequency bins of the STFT go into each band.

This is plain arithmetic, with no torch and no state. Both final models use one plan, wesep_16k, so it is the only plan this file supports. The other candidate plans were removed on 2026-10-24.

Band splitting is from Luo & Yu, "Music Source Separation with Band-Split RNN", IEEE/ACM TASLP 2023, and the speech version from
Yu et al., "High Fidelity Speech Enhancement with Band-split RNN", Interspeech 2023. Neither publishes a 16 kHz plan (decisions-m1.md 2026-08-18).

This file was debugged by Claude Code and errors fixed.
"""

import math

# The 16 kHz plan inherited from the WeSep / REAL-TSE baseline. It is their own unpublished choice, not from Yu et al., whose 48 kHz model uses 33 bands (decisions-m1.md).
WESEP_16K_SEGMENTS = [
    {"bandwidth_hz": 100, "count": 15},
    {"bandwidth_hz": 200, "count": 10},
    {"bandwidth_hz": 500, "count": 5},
    {"bandwidth_hz": 2000, "count": 1},
]


# this is the one place the model asks for its band plan. I just used weseps plan at the end, so the other plans were removed
def band_plan(sample_rate, n_fft):
    """Return the number of bins in each band of the wesep_16k plan. Whatever is left over at the top becomes one last band, so the counts add up to n_fft // 2 + 1.

    At 16 kHz with n_fft 512 this gives 32 bands: 15 of 3 bins, 10 of 6, 5 of 16, 1 of 64 and a last band of 8.
    """
    num_bins = n_fft // 2 + 1
    hz_per_bin = sample_rate / n_fft
    widths = []
    for segment in WESEP_16K_SEGMENTS:
        width = int(math.floor(segment["bandwidth_hz"] / hz_per_bin))
        if width < 1:
            raise ValueError(f"A {segment['bandwidth_hz']} Hz band is narrower than one bin ({hz_per_bin:.2f} Hz).")
        widths.extend([width] * segment["count"])
    remainder = num_bins - sum(widths)
    if remainder < 1:
        raise ValueError(f"The bands use {sum(widths)} of the {num_bins} bins, which leaves nothing for the last band.")
    widths.append(remainder)
    return widths
