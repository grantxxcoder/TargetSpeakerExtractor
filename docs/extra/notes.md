Plain-English scratchpad — things explained to myself in my own words.
Formal versions live in `docs/decisions/decisions-m0.md`; glossary terms live in
`docs/data/definitions.md`. Nothing here is authoritative.

**What is a "Sabine floor", and where did the term come from?** 
 Meaning: the shortest reverberation time a room of that size can physically have, even if you made every wall perfectly absorbing. Sabine's equation relates reverb time to room volume and surface area:

 $$T_{60} = \frac{0.161,V}{S\alpha}$$

 where $V$ is volume, $S$ is total wall/floor/ceiling area, and $\alpha$ is average absorption. Absorption cannot exceed 1 (a wall cannot swallow more than all the sound hitting it), so setting $\alpha = 1$ gives the minimum possible T60 for that shape — 0.161 * vol / surf. Bigger room, longer minimum, because sound travels further between bounces. Where it came from: the term is not the literature's. It entered the repo in commit 9bb7ded (yesterday's exploration session), in notebook cell 43. The only other trace of Sabine in the project is pra.inverse_sabine(t60, dims) at build_manifest.py:143, which is pyroomacoustics doing this same calculation and raising ValueError when you ask for an impossible T60. The equation is Sabine (1922), Collected Papers on Acoustics — now cited at both sites (2026-08-12). "Sabine floor" is a convenient label I coined for $\alpha = 1$; if you use it in the thesis, define it on first use, because a reader won't recognise it as standard.