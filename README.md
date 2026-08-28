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
target — see `docs/decisions/decisions-m0.md`.

Start with `docs/decisions/specification.md` (the brief), then `docs/decisions/research-plan.md`.

## Where it stands

**The data exists. The model does not.** As of 2026-08-17 all 21,208 trials are
rendered to disk — 63,624 files, 27 GB, 105.4 h of audio, 0 render failures — and
40 sampled trials have been checked by ear.

`src/models/`, `src/eval/` and `src/live_model_metric/` are still empty
directories. Neither the extractor nor the metric that is the primary
contribution has been implemented; those are two separate build efforts and
neither has begun.

Outstanding on M0: floor and ceiling WER calibration (the blocker, and what C2
needs), the per-parameter EDA, and revising the exploratory notebook. Full status,
including the per-decision detail: `docs/reports/m0-status.html`; the checklist
itself is `docs/decisions/milestones.md`.

## External dependencies

Corpora are downloaded, not vendored. Each row below is pinned as it is actually
brought in; the unpinned rows are the intended set, not choices already made.

| Purpose | Source | Status |
|---|---|---|
| Constructed trial + training data | LibriSpeech, WHAM! noise, RIRs simulated with `pyroomacoustics` (WHAMR!-style, not WHAMR!'s files) | **built — 21,208 trials, 27 GB** |
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
results. See `docs/decisions/decisions-m0.md`.

## The data pipeline, file by file

Run in this order. Each step caches its output, so re-running is cheap.

| # | Command | Reads | Writes | Cost |
|---|---|---|---|---|
| 1 | `scripts/make_splits.py` | LibriSpeech `SPEAKERS.TXT` | `experiments/configs/splits.yaml` | seconds |
| 2 | `scripts/build_vad_index.py` | LibriSpeech audio | `data/index/vad_segments.csv` | **~2.2 h, once** |
| 3 | `scripts/screen_noise_speech.py` | WHAM! noise | `data/index/noise_speech_{tr,cv,tt}.csv` | ~25 min, once |
| 4 | `scripts/build_manifest.py --split X` | the indexes above | `data/manifests/X.csv` | ~1 min per split |
| 5 | `scripts/render_trials.py --split X` | manifest + corpora | `data/rendered/X/` | **3.2 h for all six splits** |

Step 4 requests a trial count and may deliver fewer: a draw whose constraints
cannot be satisfied is dropped after 20 attempts. `train` is **19,938 of the
20,000 requested**, and the 62 shortfalls are named in `data/manifests/train.failed.txt`.
That file records *manifest* failures, not render failures — every rendered split
reports `n_failed: 0`.

Step 5 writes one directory per trial, holding three stems and the render record:

```
data/rendered/X/<trial_id>/
    mixture.wav      what the model hears
    target.wav       the reference: the target through its own room, alone (A1)
    enrollment.wav   who to listen for -- dry, no room (A4)
    meta.json        gains, clip guard, RIR lengths, both transcripts
data/rendered/X/render.meta.yaml            the split's render record
```

`meta.json` records what the renderer *did*; the manifest row records what was
*asked for* (SIR, SNR, room, positions, overlap, condition, regime). Per-condition
analysis needs both, joined on `trial_id`.

Each stage is a **separate script** on purpose: they have wildly different costs
(seconds, hours, minutes, minutes, hours) and different failure modes, and you
re-run them at different times. Nothing is chained automatically.

`data/` is not in git (`.gitignore:/data/`), so every generated file carries a
`.meta.yaml` sidecar recording the config, its md5, the git commit, the seed and
the date. **Those sidecars are the only travelling record of how the data was
made** — when reproducing a result, check them first.

### Which stage invalidates which

Each stage depends on the one above it. **Changing a stage invalidates everything
below it**, and nothing warns you automatically — the sidecars are what let you
check.

```
splits.yaml            change it -> rebuild the VAD index (new speakers), manifests, audio
   |
vad_segments.csv       change the vad: config -> rebuild every manifest, re-render all audio
   |
manifests/X.csv        rebuild it -> RE-RENDER THAT SPLIT'S AUDIO. Always.
   |
rendered/X/            the training and eval data
```

**A manifest rebuild always means re-rendering that split's audio.** The manifest
decides who speaks, when, how loud and in what room; the audio is that decision
made real. Rebuild one without the other and your audio no longer matches its own
labels, and *every downstream number is quietly wrong* — nothing crashes.

This is why the render step goes **last, and only once the manifests are settled**.
Rendering 21,208 trials is a multi-hour job; doing it before a known-pending
rebuild throws that time away. B2's rebuild was one such, and it will not be the
last.

Held to in practice: the render ran on 2026-08-16/17, *after* B2's rebuild, and
took **3.2 h** — against ~83 min projected from a 100-trial sample, so treat that
sample as too small to extrapolate from rather than as a measurement. The cost of
getting the ordering wrong is now concrete: any change that invalidates the
manifests buys a 3.2 h re-render and 27 GB rewritten.

**How to tell if your audio is stale.** The renderer copies its source manifest's
identity into its own sidecar precisely so this is checkable rather than assumed.
Note the field names differ across the two files — the rendered side prefixes them
`manifest_`, because it also records its *own* config and commit:

| Rendered: `data/rendered/X/render.meta.yaml` | must equal | Manifest: `data/manifests/X.meta.yaml` |
|---|---|---|
| `manifest_config_md5` | = | `config_md5` |
| `manifest_git_commit` | = | `git_commit` |

If they differ, the audio was rendered from a different manifest than the one now
on disk, and every downstream number is quietly wrong. All six splits:

```bash
for s in train val eval_public eval_private smoke_train smoke_val; do
  m=$(awk '/^config_md5:/{print $2}' "data/manifests/$s.meta.yaml")
  r=$(awk '/^manifest_config_md5:/{print $2}' "data/rendered/$s/render.meta.yaml")
  [ "$m" = "$r" ] && echo "$s  ok" || echo "$s  STALE  manifest=$m rendered=$r"
done
```

Also worth checking in the same sidecar: `partial: true` means the render was
interrupted, and `n_skipped` counts trials already on disk that were left
untouched — a resumed run reports `n_rendered: 0` with everything skipped, which
is success, not a no-op failure.

### What each source file is for

| File | What it does |
|---|---|
| `src/data/sampling.py` | Every random draw for a trial: the two difficulty regimes, the distribution shapes, which parameters a regime may narrow (B12) |
| `src/data/vad.py` | Where speech actually is inside a recording, and the interval arithmetic built on that — overlap, activity, interruption (B2) |
| `src/data/render.py` | One manifest row to three stems: room simulation, the level chain (A3), the clip guard (A6), the enrollment EQ. Pure — no disk writes, no RNG beyond the trial-seeded EQ |
| `scripts/make_splits.py` | Speaker-disjoint train/val/eval splits, stratified by sex and enrollment-guard tier (B10) |
| `scripts/build_vad_index.py` | One cached pass of the detector over all 137,876 indexed utterances |
| `scripts/build_manifest.py` | One row per trial: who speaks, when, how loud, in what room. Reads file headers only, never audio |
| `scripts/render_trials.py` | Drives `render.py` across a split in parallel. Resumable, writes atomically via a temp dir, and `--trials <ids>` renders named cases for listening |
| `scripts/check_manifest_parity.py` | Proves a refactor changed no draw, by rebuilding and diffing against the previous manifest |
| `scripts/screen_noise_speech.py` | Detects speech hiding in the WHAM! noise beds and measures what rejecting it would cost. Measures only — the rule is chosen from its report |
| `scripts/measure_vad_impact.py` | The measurement behind the B2 decision — re-runnable, writes to `experiments/results/` |
| `src/run_log.py` | Appends each slow job's wall time to `docs/run_times.md` |

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
`experiments/results/2026-08-15-vad-impact/` and `docs/decisions/decisions-m0.md`
(2026-08-15).

### Running any of it

The environment is a virtualenv beside the repo, not inside it:

```bash
python3 -m venv ../tse_venv
../tse_venv/bin/pip install -r requirements.txt

../tse_venv/bin/python scripts/build_vad_index.py
../tse_venv/bin/python -m pytest tests/ -q

# render: --limit N to time it first, --trials <ids> for single cases to listen to
../tse_venv/bin/python scripts/render_trials.py --split smoke_val
../tse_venv/bin/python scripts/render_trials.py --split train --workers 8
```

### Training

Run from the repo root. `epochs`, `patience`, `lr` and the loss weights all come
from the config — no training hyperparameter is a command-line flag.

```bash
# smoke first: 50 train / 20 val trials, proves the wiring before the real run
../tse_venv/bin/python scripts/train.py --split smoke --epochs 5

# the real run
../tse_venv/bin/python scripts/train.py --split full

# continue a stopped run; refuses if the checkpoint's config differs
../tse_venv/bin/python scripts/train.py --split full --resume
```

Writes three things: `models/model_<split>.pt` (best validation epoch only, with
the optimiser and scheduler state so `--resume` continues rather than restarts),
a wall-time row in `docs/run_times.md`, and
`experiments/results/<date>-train-<split>/{meta.yaml,history.csv}`.

**Not on this laptop for `--split full`.** 15.7 GB RAM with VSCode open is not
enough — `systemd-oomd` killed the editor and a terminal on 2026-08-24, before
training had even been started. Close VSCode, or use server-class compute
(`docs/decisions/specification.md`). `requirements.txt` pins a CPU torch and
`docs/run_times.md` records no usable GPU here. The measured rows are all CPU
and smoke-sized — best figure 277 s/epoch at batch 3 over 50 trials, from the
30-epoch run (2.3 h); the 243 s/epoch row is a single epoch and includes
startup. Wiring timings, not a basis for projecting the full split.

**Judging the first curve.** The do-nothing anchor — emitting the mixture
unchanged — scores `total = -2.24` at the config's `w` and `w_m`
(`experiments/results/2026-08-20-loss-anchor/`). A model that settles above that
is losing to a passthrough, which is a wiring or learning-rate problem, not a
slow start.

### Listening to one example

What the model actually did to a single validation trial. A listening check, not
a measurement — interpretability metrics are a later, separate script.

```bash
# first val trial, against models/model_smoke.pt
../tse_venv/bin/python scripts/pass_a_test_case_through.py --split smoke --index 0

# a named trial, and a checkpoint that is not the default
../tse_venv/bin/python scripts/pass_a_test_case_through.py --split smoke \
    --trial-id smoke_val-42-000013 --checkpoint models/model_full.pt
```

`--index` and `--trial-id` are mutually exclusive; the checkpoint defaults to
`models/model_<split>.pt` and is required — an untrained forward pass tells you
nothing. Writes one directory per trial under
`experiments/results/<date>-passthrough-<split>/<trial_id>/`, holding **two files**:

```
estimate.wav     what came out — the whole clip, the only signal that exists nowhere else
meta.yaml        which trial, which 4 s window the loss covers, checkpoint epoch,
                 the trial's SIR/SNR/overlap/regime, peak and RMS dBFS, both loss dicts
```

**No chunking and no stitching.** Rendered clips are 15.7–20.4 s; the estimate is
the whole one, from a single forward pass. The model is causal
(`causal: true`, `lookahead_frames: 0`), so appending later audio cannot change
earlier output — which makes one full-length pass exactly what streaming emits.
Measured on `smoke_val-42-000000`:

| | max abs diff | |
|---|---|---|
| full 17.7 s pass vs a 4 s pass, both from sample 0 | `1.68e-08` | causality holds |
| 4 independent 4 s chunks concatenated vs one full pass | `4.37e-03`, rel L2 `1.04e-02` | one seam artefact per join |

Stitching is not merely unnecessary but harmful: each seam reinjects the
incomplete-overlap-add tail, the last `n_fft - hop` = 384 samples (23.4 ms).
`--crop-only` writes just the 4 s crop if you want the smaller file.

**Causal is not context-free**, and this is the trap. A crop taken from mid-clip
starts the LSTM and cLN state *cold*; the same window inside a full pass has state
warmed by everything before it. For the crop at sample 19611 (1.23 s in) those
differ by `5.60e-03` max, rel L2 `3.06e-01` — 14 % of the estimate's peak. So the
audio written here is the **warm-state** version, i.e. what deployment produces,
*not* the cold-start crop the trainer saw. The loss is still reported on the
cold-start crop, because that is the only number comparable to `history.csv`;
over the same window the two score `-2.4197` (cold) against `-2.4150` (warm), so
quoting one beside the other is safe by measurement rather than by the causality
argument.

**The mixture, target and enrollment are deliberately not copied.** They are
already in `data/rendered/<split>/<trial_id>/`, so copying them would add
812 KiB per inspected trial against a 27 GB dataset and go stale the moment a
manifest is rebuilt. The meta records `source_dir` and `crop_window_s` instead —
listen to the stems in place, at the window the estimate corresponds to:

```
source_dir: data/rendered/smoke_val/smoke_val-42-000000
crop_window_s: [1.226, 5.226]
```

`estimate.wav` is **float32 and unnormalised**, 256 KiB for a 4 s crop.
Normalising would hide the gain error `L_MR` exists to catch, since `L_pres` is
scale-invariant and cannot see it; `PCM_16` would clip an estimate above 1.0 and
make a gain bug sound like a model artefact. The peak is reported instead,
flagged `<-- CLIPS` past 1.0.

The terminal prints the loss terms beside the **pass-through anchor for that same
crop**, because a single crop's loss alone does not say whether the model did
anything:

```
  crop branch : PRESENT
  term             model    pass-through     delta
  L_pres         -6.3977         -6.9036   +0.5058
  L_MR            0.2010          0.1947   +0.0063
  L_abs             n/a             n/a        n/a
  total          -2.4197         -2.7266   +0.3069

  WORSE than emitting the mixture unchanged, by 0.3069
```

Two traps in reading that. **The single-crop `total` is not comparable to
`val_total` in `history.csv`**: the objective is
`(1-w)*mean_present[...] + w*mean_absent[...]`, and one crop is either present or
absent, so the other half contributes 0 instead of its mean — hence the `n/a`
row. And **this anchor is for this crop only**, not the 300-crop median of
`-2.24` above; a single crop's anchor moves by many dB with SIR and target
activity.

Versions in `requirements.txt` are pinned exactly, because several of them
define the data rather than merely produce it — the VAD weights decide what
"overlap" means, and `pyroomacoustics` and `pyloudnorm` decide what the rendered
audio is. A silent minor bump would move results without appearing in any diff.
