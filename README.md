# TargetSpeakerExtractor

Stellenbosch University Machine Learning and AI masters project focusing on
live target speaker extraction modelling.

## What this project is

A streaming target speaker extraction (TSE) model, optimised for how
accurately a **live speech-to-speech model** recovers what the target speaker
said — not for the signal or perceptual quality of the separated audio. The
primary contribution is a defined, gaming-resistant metric for that
(`docs/metric-definitions.md`) plus the harness that computes it.

The extractor outputs **audio**. The live model also accepts text, and that
path is measured as a benchmark reference condition, but it is not the build
target — see `docs/decisions.md`.

Start with `docs/specification.md` (the brief), then `docs/research-plan.md`.

## External dependencies

Nothing is vendored yet — this list is the intended dependency set, to be
pinned to exact commits as each is actually brought in.

| Purpose | Source | Status |
|---|---|---|
| Constructed trial + training data | LibriSpeech, WHAM! noise, WHAMR!-style RIRs | not yet built |
| Real-audio transfer set | AMI corpus (CC BY 4.0, direct download) | not yet built |
| Conventional metrics | SI-SDR, DNSMOS-P808, an offline ASR for WER | not yet pinned |
| Judge (primary) | a closed live speech-to-speech API — exact model ID pinned per run | not yet chosen |
| Judge (reproducibility anchor) | an open-weight speech-to-speech model | not yet chosen |
| Front-end ASR, text reference condition | an off-the-shelf streaming ASR | not yet chosen |

**Note on what is no longer a dependency.** Earlier versions of this README
listed the REAL-TSE Challenge baseline repo and its official scoring pipeline.
Replicating the challenge baselines and using its eval pipeline were both
dropped in the 2026-08-07 re-scope (spec note 8). We still borrow ideas,
data-construction methods and metric-design lessons from the challenge, and
cite it as the anchor benchmark — but our numbers are never comparable to
published REAL-TSE results. See `docs/decisions.md`.
