Plain-English scratchpad: things explained to myself in my own words. Formal versions are in `docs/decisions/decisions-m0.md`, glossary terms in `docs/data/definitions.md`. Nothing here is authoritative.

**What is a "Sabine floor", and where did the term come from?** 
 Meaning: the shortest possible reverberation time for a room of that size, even with perfectly absorbing walls. Sabine's equation relates reverb time to room volume and surface area:

 $$T_{60} = \frac{0.161,V}{S\alpha}$$

 where $V$ is volume, $S$ is total wall/floor/ceiling area and $\alpha$ is average absorption. Absorption cannot exceed 1, so $\alpha = 1$ gives the minimum T60 for that shape: 0.161 * vol / surf. Bigger rooms have a longer minimum, since sound travels further between bounces. Origin: not a literature term. It entered the repo in commit 9bb7ded (yesterday's exploration session), notebook cell 43. The only other trace of Sabine is pra.inverse_sabine(t60, dims) at build_manifest.py:143, where pyroomacoustics does the same calculation and raises ValueError for an impossible T60. The equation is Sabine (1922), Collected Papers on Acoustics, now cited at both sites (2026-08-12). "Sabine floor" is my own label for $\alpha = 1$; define it on first use in the thesis, as a reader won't recognise it.
---

**What does FFT stand for, and what does it actually do?**
 Fast Fourier Transform. It gives how much of each frequency is in a short slice of audio (amplitudes over time). "Fast" refers only to the algorithm: the same answer as the Discrete Fourier Transform in $O(N \log N)$ instead of $O(N^2)$, not an approximation.

**What is a "bin"?**
 The FFT returns a fixed number of evenly spaced frequency buckets, not a smooth curve; each bucket is a bin. Bin $k$ sits at $k \times \frac{\text{sample rate}}{n_{\text{fft}}}$. For us, $16000 / 512 = 31.25$ Hz per bin: bin 0 is 0 Hz, bin 1 is 31.25 Hz, bin 2 is 62.5 Hz, bin 256 is 8000 Hz. The transform fixes the width.

**Why 257 bins and not 512?**
 A 512-sample FFT returns 512 complex numbers. For real-valued audio, the upper half is the complex conjugate (mirror image) of the lower half and carries no new information, so we keep $n_{\text{fft}}/2 + 1 = 257$. The $+1$ is because both endpoints are kept: 0 Hz (DC) and 8000 Hz (Nyquist).

**My audio is 16 kHz, so why do the bins stop at 8 kHz?**
 16 kHz is the *sample rate* (16,000 amplitude measurements per second), not the highest frequency. By the Nyquist limit, a signal sampled at $f_s$ can only represent frequencies up to $f_s/2$. Faster ones alias to lower ones, so they are filtered out before the file is written. So the spectrum is 0–8 kHz in 257 steps of 31.25 Hz, and the band plan is about the 8 kHz, not the 16 kHz.

**Bins versus bands: what is the difference?**
 Bins come from the FFT: 257 of them, all 31.25 Hz wide, set by `n_fft`. Bands are my design choice: 32 contiguous groups of bins, deliberately of different sizes. **Bins are physics; bands are architecture.** Nothing forces the band boundaries, which is why nobody in the literature has justified theirs.

**Reading the band plan table (the 2000 Hz is not a typo).**
 The plan is `[3]*15 + [6]*10 + [16]*5 + [64] + [8]`, which sums to 257.

 | how many bands | bins each | width of each | covers |
 | --- | --- | --- | --- |
 | 15 | 3 | ~94 Hz | 0 – 1.4 kHz |
 | 10 | 6 | ~188 Hz | 1.4 – 3.3 kHz |
 | 5 | 16 | 500 Hz | 3.3 – 5.8 kHz |
 | 1 | 64 | 2000 Hz | 5.8 – 7.8 kHz |
 | 1 | 8 | 250 Hz | 7.8 – 8 kHz |

 The 2000 Hz row is *one* band of 64 bins ($64 \times 31.25 = 2000$ Hz), so it is 2 kHz wide. The idea: many narrow bands low down, one wide band at the top.

**Why unequal bands at all?**
 Two reasons; the second is easy to miss.
 (1) *Information density.* F0 and the first two formants, which distinguish one voice from another, live below about 1 kHz. Above 4 kHz is mostly fricative energy and spectral tilt, with far less structure per Hz. So resolution goes where the information is.
 (2) *Normalisation.* Speech energy falls roughly 6 dB per octave, so the lowest bins are orders of magnitude louder than the highest. Normalised together, the high bands would be numerically invisible and add almost nothing to the gradient. Splitting first gives every band its own normalisation and its own projection to 128 dimensions, so a quiet 6 kHz band gets the same representational budget as a loud 200 Hz one. This is a large part of *why* BSRNN works, and a better viva answer than "it splits the spectrum".
---




**Energy ratio:** The energy of a signal is the sum of its squared samples. So if we have a sample rate of 64000, then the energy is the sum of each component squared ie ||a||^2 = a[0]^2 + a[1]^2+....+a[63999]^2
