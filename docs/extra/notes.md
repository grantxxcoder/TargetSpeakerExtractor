Plain-English scratchpad — things explained to myself in my own words.
Formal versions live in `docs/decisions/decisions-m0.md`; glossary terms live in
`docs/data/definitions.md`. Nothing here is authoritative.

**What is a "Sabine floor", and where did the term come from?** 
 Meaning: the shortest reverberation time a room of that size can physically have, even if you made every wall perfectly absorbing. Sabine's equation relates reverb time to room volume and surface area:

 $$T_{60} = \frac{0.161,V}{S\alpha}$$

 where $V$ is volume, $S$ is total wall/floor/ceiling area, and $\alpha$ is average absorption. Absorption cannot exceed 1 (a wall cannot swallow more than all the sound hitting it), so setting $\alpha = 1$ gives the minimum possible T60 for that shape — 0.161 * vol / surf. Bigger room, longer minimum, because sound travels further between bounces. Where it came from: the term is not the literature's. It entered the repo in commit 9bb7ded (yesterday's exploration session), in notebook cell 43. The only other trace of Sabine in the project is pra.inverse_sabine(t60, dims) at build_manifest.py:143, which is pyroomacoustics doing this same calculation and raising ValueError when you ask for an impossible T60. The equation is Sabine (1922), Collected Papers on Acoustics — now cited at both sites (2026-08-12). "Sabine floor" is a convenient label I coined for $\alpha = 1$; if you use it in the thesis, define it on first use, because a reader won't recognise it as standard.
---

**What does FFT stand for, and what does it actually do?**
 Fast Fourier Transform. Give it a short slice of audio — a list of amplitudes over time — and it tells you how much of each frequency is present in that slice. "Fast" refers only to the algorithm: it returns the same answer as the Discrete Fourier Transform but in $O(N \log N)$ instead of $O(N^2)$. Nothing about the result is approximate.

**What is a "bin"?**
 The FFT does not hand back a smooth curve over frequency. It hands back a fixed number of evenly-spaced buckets, and each bucket is a bin. Bin $k$ sits at frequency $k \times \frac{\text{sample rate}}{n_{\text{fft}}}$. For us: $16000 / 512 = 31.25$ Hz per bin. So bin 0 is 0 Hz, bin 1 is 31.25 Hz, bin 2 is 62.5 Hz, and bin 256 is 8000 Hz. All bins are the same width — that is fixed by the transform, not a choice.

**Why 257 bins and not 512?**
 An FFT of 512 samples returns 512 complex numbers. But our audio is real-valued (a waveform, not a complex signal), and for real input the upper half of the output is the mirror image — the complex conjugate — of the lower half. It carries no new information, so we throw it away and keep $n_{\text{fft}}/2 + 1 = 257$. The $+1$ is because we keep both endpoints: 0 Hz (DC) and 8000 Hz (Nyquist).

**My audio is 16 kHz, so why do the bins stop at 8 kHz?**
 Because 16 kHz is the *sample rate*, not the highest frequency. It means 16,000 amplitude measurements per second. The Nyquist limit says a signal sampled at rate $f_s$ can only represent frequencies up to $f_s/2$ — sample any faster-wiggling frequency and it comes back disguised as a lower one (aliasing), so it is filtered out before the file is written. So 16 kHz sampling gives an 8 kHz ceiling, and the entire spectrum I have to work with is 0–8 kHz in 257 steps of 31.25 Hz. Worth not confusing these two numbers: everything in the band plan is about the 8 kHz, not the 16 kHz.

**Bins versus bands — what is the difference?**
 Bins are given to me by the FFT: 257 of them, all 31.25 Hz wide, decided entirely by `n_fft`. Bands are my design choice: contiguous groups of bins, 32 of them, deliberately different sizes. So 257 equal bins get grouped into 32 unequal bands. **Bins are physics; bands are architecture.** Nothing forces the band boundaries — which is exactly why nobody in the literature has justified theirs.

**Reading the band plan table (and no, the 2000 Hz is not a typo).**
 The plan is `[3]*15 + [6]*10 + [16]*5 + [64] + [8]`, which sums to 257.

 | how many bands | bins each | width of each | covers |
 | --- | --- | --- | --- |
 | 15 | 3 | ~94 Hz | 0 – 1.4 kHz |
 | 10 | 6 | ~188 Hz | 1.4 – 3.3 kHz |
 | 5 | 16 | 500 Hz | 3.3 – 5.8 kHz |
 | 1 | 64 | 2000 Hz | 5.8 – 7.8 kHz |
 | 1 | 8 | 250 Hz | 7.8 – 8 kHz |

 The 2000 Hz row is *one* band containing 64 bins: $64 \times 31.25 = 2000$ Hz, so that single band is 2 kHz wide. The rows above have many bands, each narrow. That is the whole shape of the idea — lots of fine bands down low, one fat band up top.

**Why unequal bands at all?**
 Two reasons, and the second is the one that is easy to miss.
 (1) *Information density.* F0 and the first two formants — what actually distinguishes one voice from another — live below about 1 kHz. Above 4 kHz there is mostly fricative energy and spectral tilt, far less structure per Hz. So spend resolution where the information is.
 (2) *Normalisation.* Speech energy falls roughly 6 dB per octave, so the lowest bins are orders of magnitude louder than the highest. If all 257 bins were normalised together, the high bands would be numerically invisible and contribute almost nothing to the gradient. Splitting first means every band gets its own normalisation and its own projection to 128 dimensions — so a quiet 6 kHz band gets exactly the same representational budget as a loud 200 Hz one. This is a large part of *why* BSRNN works, and a better answer in a viva than "it splits the spectrum".
---




**Energy ratio:** The energy of a signal is the sum of its squared samples. So if we have a sample rate of 64000, then the energy is the sum of each component squared ie ||a||^2 = a[0]^2 + a[1]^2+....+a[63999]^2


