# TargetSpeakerExtractor

Stellenbosch University Machine Learning and AI masters project focusing on
live target speaker extraction modelling.

## What this project is

A streaming target speaker extraction (TSE) model, optimised for how
accurately a **live speech-to-speech model** recovers what the target speaker
said — not for the signal or perceptual quality of the separated audio. The
primary contribution is a defined, gaming-resistant metric for that
(`docs/data/metric-definitions.md`) plus the harness that computes it.

The extractor outputs **audio**. The live model also accepts text, and that
path is measured as a benchmark reference condition, but it is not the build
target — see `docs/decisions/decisions.md`.

Start with `docs/decisions/specification.md` (the brief), then `docs/decisions/research-plan.md`.

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

The REAL-TSE Challenge is cited as the anchor benchmark for real
conversational TSE, and we borrow its data-construction methods and its
lessons about metric gaming — but we replicate neither its baselines nor its
eval pipeline, so our numbers are never comparable to published REAL-TSE
results. See `docs/decisions/decisions.md`.
