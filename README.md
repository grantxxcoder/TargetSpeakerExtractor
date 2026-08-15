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
| Voice-activity detection | Silero VAD (MIT), `silero-vad` 6.2.1 from PyPI | **pinned, in use** |
| Conventional metrics | SI-SDR, DNSMOS-P808, an offline ASR for WER | not yet pinned |
| Judge (primary) | a closed live speech-to-speech API — exact model ID pinned per run | not yet chosen |
| Judge (reproducibility anchor) | an open-weight speech-to-speech model | not yet chosen |
| Front-end ASR, text reference condition | an off-the-shelf streaming ASR | not yet chosen |

The REAL-TSE Challenge is cited as the anchor benchmark for real
conversational TSE, and we borrow its data-construction methods and its
lessons about metric gaming — but we replicate neither its baselines nor its
eval pipeline, so our numbers are never comparable to published REAL-TSE
results. See `docs/decisions/decisions.md`.

## The data pipeline, file by file

Run in this order. Each step caches its output, so re-running is cheap.

| # | Command | Reads | Writes | Cost |
|---|---|---|---|---|
| 1 | `scripts/make_splits.py` | LibriSpeech `SPEAKERS.TXT` | `experiments/configs/splits.yaml` | seconds |
| 2 | `scripts/build_vad_index.py` | LibriSpeech audio | `data/index/vad_segments.csv` | **~2.2 h, once** |
| 3 | `scripts/build_manifest.py --split X` | the two indexes above | `data/manifests/X.csv` | ~1 min per split |
| 4 | the renderer | manifests + audio | trial audio on disk | not built yet |

`data/` is not in git (`.gitignore:/data/`), so every generated file carries a
`.meta.yaml` sidecar recording the config, its md5, the git commit, the seed and
the date. **Those sidecars are the only travelling record of how the data was
made** — when reproducing a result, check them first.

### What each source file is for

| File | What it does |
|---|---|
| `src/data/sampling.py` | Every random draw for a trial: the two difficulty regimes, the distribution shapes, which parameters a regime may narrow (B12) |
| `src/data/vad.py` | Where speech actually is inside a recording, and the interval arithmetic built on that — overlap, activity, interruption (B2) |
| `scripts/make_splits.py` | Speaker-disjoint train/val/eval splits, stratified by sex and enrollment-guard tier (B10) |
| `scripts/build_vad_index.py` | One cached pass of the detector over all 137,876 indexed utterances |
| `scripts/build_manifest.py` | One row per trial: who speaks, when, how loud, in what room. Reads file headers only, never audio |
| `scripts/check_manifest_parity.py` | Proves a refactor changed no draw, by rebuilding and diffing against the previous manifest |
| `scripts/measure_vad_impact.py` | The measurement behind the B2 decision — re-runnable, writes to `experiments/results/` |

### Why there is a voice-activity pass at all

A LibriSpeech utterance is someone reading a sentence, and the file is trimmed
loosely around them: **86 % of a file is speech**, with a near-constant 0.331 s of
silence before the first word and 0.129 s after the last.

The generator used to treat each file as speech end to end, because the duration
was all it had. That overstated overlap by **~25 %**, and by a different amount in
every trial (mean 0.071, max 0.274) — so it could not be corrected with a
multiplier, and it sorted trials into the wrong overlap buckets. Those buckets are
the per-condition results table (B13), which is the thesis's central artefact.

Step 2 fixes the measurement. **It does not change the audio** — mixtures still
contain the pauses, because that is what speech sounds like. Full evidence,
including the settings sweep that chose 250 ms, is in
`experiments/results/2026-08-15-vad-impact/` and `docs/decisions/decisions.md`
(2026-08-15).

### Running any of it

The environment is a virtualenv beside the repo, not inside it:

```bash
python3 -m venv ../tse_venv
../tse_venv/bin/pip install -r requirements.txt

../tse_venv/bin/python scripts/build_vad_index.py
../tse_venv/bin/python -m pytest tests/ -q
```

Versions in `requirements.txt` are pinned exactly, because several of them
define the data rather than merely produce it — the VAD weights decide what
"overlap" means, and `pyroomacoustics` and `pyloudnorm` decide what the rendered
audio is. A silent minor bump would move results without appearing in any diff.
