"""Band plan tables: how many FFT bins go into each band.

Pure arithmetic -- no torch, no state, no I/O. Kept that way deliberately so the
plan can be swapped from YAML, tested in isolation, and regenerated at any n_fft.
The wesep reference computes it inline in the BSRNN constructor, which is what
makes the band-plan ablation impossible there.

Band splitting is from Luo & Yu, "Music Source Separation with Band-Split RNN",
IEEE/ACM TASLP 2023; the speech adaptation is Yu et al., "High Fidelity Speech
Enhancement with Band-split RNN", Interspeech 2023. Neither publishes a 16 kHz
table -- see docs/decisions/decisions-m1.md 2026-08-18.
"""

import math

BAND_PRESETS = {
    # Inherited 16 kHz plan. NOT from Yu et al. (Interspeech 2023) -- that paper
    # specifies 33 subbands (20x200 + 6x500 + 7x2k Hz) for its 48 kHz model.
    # This is the wesep / REAL-TSE baseline's own 16 kHz adaptation, unpublished
    # and unjustified anywhere in the literature. See decisions-m1.md.
    "wesep_16k": [
        {"bandwidth_hz":  100, "count": 15},
        {"bandwidth_hz":  200, "count": 10},
        {"bandwidth_hz":  500, "count":  5},
        {"bandwidth_hz": 2000, "count":  1},
    ],  
     
    # --- control --------------------------------------------------------
    # Equal-width bands. If this matches wesep_16k, non-uniform splitting is
    # doing no work and a large part of the architecture's story is wrong.
    "uniform_32": {"uniform": 32},
    
    # --- the paper's own shape, truncated to our bandwidth ---------------
    # Yu et al. (Interspeech 2023) sec 4.2 specify 20x200 + 6x500 + 7x2k Hz
    # for 48 kHz. Applying the same progression to our 0-8 kHz gives this.
    # The closest thing to a published plan for this architecture.
    "yu_truncated": {"segments": [
        {"bandwidth_hz":  200, "count": 20},   # 0 - 4 kHz
        {"bandwidth_hz":  500, "count":  6},   # 4 - 7 kHz
    ]},                                        # remainder -> 7 - 8 kHz
    
    # --- perceptually motivated -----------------------------------------
    # Bark critical bands (Zwicker, 1961). Band edges are where the auditory
    # system integrates energy and where masking occurs -- a real reason for
    # boundaries to sit where they do, which no other plan here has.
    "bark": {"edges_hz": [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080,
                        1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700,
                        4400, 5300, 6400, 7700, 8000]},
                        
    # --- matched to the downstream model --------------------------------
    # Mel spacing, the filterbank every ASR front-end and our proxy losses
    # actually consume. Argument is project-specific: our objective is what a
    # live model understands, so aligning band structure with the
    # representation that model listens through is a motivation nobody else
    # in this literature has, because nobody else has our metric.
    "mel_32": {"mel": 32},

    # --- task-motivated --------------------------------------------------
    # Maximum resolution across F0 and the first two formants (80-1000 Hz),
    # coarse above. Two-speaker separation leans on resolving harmonic
    # structure; this spends the budget there deliberately.
    "f0_focused": {"segments": [
        {"bandwidth_hz":  100, "count": 10},   # 0 - 1 kHz, finest available
        {"bandwidth_hz":  250, "count": 12},   # 1 - 4 kHz
        {"bandwidth_hz": 1000, "count":  3},   # 4 - 7 kHz
    ]}, 


} 




def uniform_plan(n_fft, n_bands):
    """Equal-width bands -- the control that shows non-uniformity does any work."""
    n_bins = n_fft // 2 + 1
    base, extra = divmod(n_bins, n_bands)
    return [base + 1] * extra + [base] * (n_bands - extra)

def _from_segments(sample_rate, n_fft, segments):
    n_bins, hz = n_fft // 2 + 1, sample_rate / n_fft
    widths = []
    for seg in segments:
        w = int(math.floor(seg["bandwidth_hz"] / hz))
        if w < 1:
            raise ValueError(f"{seg['bandwidth_hz']} Hz < one bin ({hz:.2f} Hz)")
        widths.extend([w] * int(seg["count"]))
    remainder = n_bins - sum(widths)
    if remainder < 1:
        raise ValueError(f"segments consume {sum(widths)} of {n_bins} bins")
    widths.append(remainder)
    return widths
    
def _from_edges(sample_rate, n_fft, edges_hz): 
    """Band edges in Hz -> bin counts. Natural form for perceptual scales."""
    n_bins, hz = n_fft // 2 + 1, sample_rate / n_fft
    idx = [int(round(e / hz)) for e in edges_hz]
    idx[0], idx[-1] = 0, n_bins                  # snap to full coverage
    return [b - a for a, b in zip(idx, idx[1:])]

   
def _mel_edges(n_bands, sample_rate):
    to_mel   = lambda f: 2595 * math.log10(1 + f / 700)
    from_mel = lambda m: 700 * (10 ** (m / 2595) - 1)
    top = to_mel(sample_rate / 2)
    return [from_mel(top * i / n_bands) for i in range(n_bands + 1)]


def band_plan(sample_rate, n_fft, spec):
    """Bin counts per band, summing to n_fft // 2 + 1.

    Luo & Yu (TASLP 2023) for band splitting; Yu et al. (Interspeech 2023) for
    the speech adaptation. Neither publishes a 16 kHz table -- decisions-m1.md.
    """
    if isinstance(spec, str):
        spec = BAND_PRESETS[spec]
    if isinstance(spec, list):                   # bare segments list
        spec = {"segments": spec}
    
    if   "segments" in spec: widths = _from_segments(sample_rate, n_fft, spec["segments"])
    elif "edges_hz" in spec: widths = _from_edges(sample_rate, n_fft, spec["edges_hz"])
    elif "uniform"  in spec: widths = uniform_plan(n_fft, spec["uniform"])
    elif "mel"      in spec: widths = _from_edges(sample_rate, n_fft, _mel_edges(spec["mel"], sample_rate))
    else: raise ValueError(f"unknown band spec kind: {list(spec)}")
    
    n_bins = n_fft // 2 + 1
    assert sum(widths) == n_bins, f"{sum(widths)} != {n_bins}"
    assert min(widths) >= 1, f"empty band in {widths}"
    return widths


def describe_plan(widths, sample_rate, n_fft):
    hz, lo = sample_rate / n_fft, 0
    for i, w in enumerate(widths):
        print(f"band {i:2d}  bins {lo:3d}-{lo+w-1:3d} ({w:3d})  "
            f"{lo*hz:7.1f} - {(lo+w)*hz:7.1f} Hz")
        lo += w
        
# widths = band_plan(16000, 512, "wesep_16k")
# print(len(widths), "bands, sum =", sum(widths))     # expect 32 bands, sum 257
# describe_plan(widths, 16000, 512)


