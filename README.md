# TargetSpeakerExtractor

Stellenbosch University Machine Learning and AI masters project: a **streaming
target speaker extractor** built so that a live speech-to-speech model (Gemini)
hears what the target speaker said, and not the other speaker.

It is optimised for how much of the target's *content* Gemini recovers, not for
how good the audio sounds. The primary contribution is the metric for that
(`docs/data/metric-definitions.md`) and the harness that computes it.

**Scope, on every claim:** *optimised for Gemini* (`gemini-3.7-flash`), on
*two-speaker mixtures* (target + at most one other speaker + noise). Our numbers
are **not comparable to published REAL-TSE results**: different data, metric and
protocol.

Start with `docs/decisions/specification.md` (the brief), then
`docs/decisions/milestones.md`.

## Where it stands — 2026-09-29

Milestone 5. Experiment freeze 2026-10-14, submission 2026-11-05. The thesis is
in `report/` (`report/report.pdf`).

**In plain words.** With no processing, Gemini gets 63 % of the target's words
wrong on overlapping two-speaker clips. After our best streaming model it gets
40 % wrong. That closes **37 %** of the gap to hearing the target alone (1 %).
The model runs in real time on a laptop CPU. An off-the-shelf model (WeSep)
does better, at 26 %, but it cannot stream: it needs the whole clip and runs
2.9× slower than real time.

**Words Gemini got wrong** (LCF-WER). Lower is better. `sir0_val`, clips where
both people speak, n = 103, 3 judge runs each.

| system | words wrong (%) | ± SD (pts) | other speaker's words let through (%) | invented words per clip | can stream? |
|---|---|---|---|---|---|
| No processing | 62.94 | 0.37 | 63.90 | 1.27 | — |
| Baseline | 55.18 | 3.22 | 48.62 | 1.91 | yes |
| **Extension (our best)** | **39.59** | 0.34 | 23.08 | 2.20 | **yes** |
| WeSep (borrowed, reference) | 25.80 | 0.56 | 13.04 | 1.80 | **no** |
| Target alone (perfect) | 1.19 | 0.18 | 0.00 | 0.22 | — |

One row: "after the extension, Gemini got 39.59 % of the target's words wrong,
averaged over 3 runs; the 3 run scores differ by an SD of 0.34 points."
± is the SD of the 3 whole-run scores. Judge `gemini-3.7-flash`, audio in /
text out, prompt `src/live_model_metric/judge_prompt.txt` (sha256[:12]
`d118b7d3bf30`), runs 2026-09-25 to 2026-09-27 (`decisions-m4.md` 2026-09-27).

**Read these three caveats with the table:**

- **The extension invents more words than doing nothing.** 2.20 invented words
  per clip against 1.27. Part of the gain comes from cutting hard, and Gemini
  hears some of the cuts as words. Clips with 2+ invented words: 46.9 % against
  30.7 %. A paired test puts that gap outside the noise (`decisions-m4.md`
  2026-09-27).
- **39.59 % is optimistic.** The extension's epoch was picked on these same 103
  clips. The first selection-free result is on `eval_public`, clips where only
  the other speaker talks (n = 123). There the extension lets through 57 % of
  the stranger's words, against WeSep's 70 % and the baseline's 92 %. The gap
  to WeSep is outside the noise. But the extension is **all-or-nothing**: it
  goes fully silent on 56 of 123 clips, and on the rest Gemini hears nearly
  everything the other speaker said (`decisions-m4.md` 2026-09-29).
- **The extension is three changes at once.** They cannot be separated: a
  richer voice cue, a frozen speaker encoder, and picking checkpoints by word
  error rate plus longer training (`decisions-m2.md` 2026-09-27).

**Speed** (`decisions-m3.md` 2026-09-28, idle re-run). Both our models process
80 ms of audio in about 55 ms (real-time factor 0.68–0.69; below 1 keeps up).
Mean delay is 175 ms, inside the 200–300 ms budget. Each figure is the median
of 3 runs on an i5-1135G7, 4 threads. The two models cannot be told apart on
speed.

### The systems

| name in the report | checkpoint (`models/`, gitignored) | what it is |
|---|---|---|
| Baseline | `model_sir0_10000-e6.pt` | Causal band-split RNN (Luo & Yu 2023) with a spectral voice cue from the enrolment (TF-Map, Zhang et al. 2025). 7.19 M parameters, 9,955 training trials, epoch 6 of 16 |
| Extension | `model_sir0_cuecontext-wer-e21.pt` | Baseline + the cue split into parts (item 1a) + a frozen ECAPA speaker embedding (Desplanques et al. 2020) fused into the separator (item 1c). 7.28 M trainable. ECAPA runs once before the stream, so it adds no per-chunk cost. Config `experiments/configs/bsrnn_cue_context.yaml` |
| WeSep | `../wesep_pretrained/tfmap_context_causal_100` | Borrowed pretrained model (Wang et al. 2024). An outside reference only. Not causal, so it cannot run live |

`models/README.md` says why each checkpoint is kept. The selection of e21 is
recorded in `decisions-m2.md` 2026-09-26.

## Repo map

| path | what it holds |
|---|---|
| `src/data/` | Trial sampling, voice-activity detection, rendering, per-frame speaker-state labels |
| `src/models/` | The extractor (`bsrnn.py`), voice cue (`conditioning.py`), speaker encoder (`context_encoder.py`), losses, stateful streaming (`streaming.py`) |
| `src/live_model_metric/` | **The metric.** Words wrong, other-speaker leakage (ICR), invented words (FR), the Gemini judge, speech gate, SDR/SIR/SAR, DNSMOS |
| `src/estimates/` | The shared runner that writes `estimate.wav` for any system, ours or WeSep |
| `src/demo/` | The live demo: server, mixer, web page |
| `scripts/` | Every runnable step. The main ones are listed below |
| `experiments/configs/` | Every YAML config. No hyperparameter lives in source |
| `experiments/results/` | One directory per run. Mostly gitignored, so **the decision logs are the tracked record** |
| `docs/decisions/` | Spec, milestones, one decision log per milestone (`decisions-m0.md` … `-m4.md`), `decisions-pending.md` |
| `docs/data/` | Data construction, metric definitions, glossary |
| `docs/run_times.md` | Measured wall time of every job over a minute. Check it before planning a run |
| `docs/weak-points-register.md` | The standing audit. Read it before reopening evaluation or the objective |
| `report/`, `presentations/` | Thesis LaTeX and slides |
| `tests/` | pytest suite: 515 tests, 11 min on the laptop (2026-09-21) |

## Setup

Virtualenvs and model snapshots sit **beside** the repo, not inside it.

```bash
python3 -m venv ../tse_venv
../tse_venv/bin/pip install -r requirements.txt
# judge and demo; installed, not yet pinned in requirements.txt
../tse_venv/bin/pip install google-genai==2.21.0 python-dotenv==1.2.3
```

| sibling path | needed for |
|---|---|
| `../tse_venv/` | everything |
| `../ecapa_pretrained/` | the extension. SpeechBrain ECAPA VoxCeleb snapshot, hashes recorded per run |
| `../wesep_venv/`, `../wesep_pretrained/` | WeSep estimates only. WeSep needs torch 2.7.1 / numpy 1.26.4, so it cannot share our venv |

Put `GEMINI_API_KEY` in `.env` (gitignored). A judge run aborts if it is
missing, instead of scoring silence.

Versions are pinned exactly because some of them *define* the data. The VAD
weights decide what "overlap" means, and `pyroomacoustics`/`pyloudnorm` decide
what the audio is.

## Running it

All commands run from the repo root with `../tse_venv/bin/python`. Times are
measured (`docs/run_times.md`), on the laptop unless stated.

### 1. Data

Each step caches its output. Nothing is chained automatically.

| # | command | writes | measured |
|---|---|---|---|
| 1 | `scripts/make_splits.py` | `experiments/configs/splits.yaml` | seconds |
| 2 | `scripts/build_vad_index.py` | `data/index/vad_segments.csv` | 2.2 h, once |
| 3 | `scripts/screen_noise_speech.py` | `data/index/noise_speech_{tr,cv,tt}.csv` | 25 min, once |
| 4 | `scripts/build_manifest.py --split X` | `data/manifests/X.csv` | ~1 min per split |
| 5 | `scripts/render_trials.py --split X` | `data/rendered/X/<trial_id>/` | 1.1 h for 4,979 `sir0_train` trials |

Each trial directory holds `mixture.wav` (what the model hears), `target.wav`
(the target alone, in its room), `enrollment.wav` (a dry sample of the target's
voice) and `meta.json`. The models are trained and scored on the `sir0_*` splits,
where both voices are equally loud on average.

**Rebuilding a manifest means re-rendering that split's audio. Always.**
Otherwise the audio no longer matches its labels and every number downstream is
quietly wrong. `data/` is not in git, so the `.meta.yaml` sidecar beside each
file is the only record of how it was made. Check for stale audio:

```bash
for s in $(ls data/rendered); do
  m=$(awk '/^config_md5:/{print $2}' "data/manifests/$s.meta.yaml")
  r=$(awk '/^manifest_config_md5:/{print $2}' "data/rendered/$s/render.meta.yaml")
  [ "$m" = "$r" ] && echo "$s  ok" || echo "$s  STALE  manifest=$m rendered=$r"
done
```

Recipes for changing the data: `docs/data/changing-the-data.md`.

### 2. Train (Kaggle GPU)

The laptop has no usable GPU, and 15.7 GB RAM is not enough for a full run. Train on Kaggle:

```bash
scripts/make_kaggle_bundle.py --split sir0 --code-only   # code zip; drop --code-only to rebuild the data zip
scripts/make_kaggle_notebook.py                          # writes notebooks/kaggle_train_mid.ipynb
scripts/preflight_kaggle.py                              # which config the uploaded bundle will ACTUALLY train
```

What the notebook runs:

```bash
scripts/train.py --split sir0 --config experiments/configs/bsrnn_cue_context.yaml
```

The extension took two 14-epoch sessions of 11.2 h each on a Tesla T4. All
hyperparameters come from the config. `--resume` continues a run, and refuses
if the config changed. `train.py` refuses to point its in-loop word-error probe,
which picks checkpoints, at `sir0_privval` or `eval_private`.

### 3. Evaluate

```bash
# one checkpoint -> estimate.wav per trial (14 min for 123 trials)
scripts/make_estimates.py --split sir0 --condition both \
    --checkpoint models/model_sir0_cuecontext-wer-e21.pt \
    --config experiments/configs/bsrnn_cue_context.yaml --out experiments/results/<date>-est-<tag>

# score it: Gemini judge (primary), then the offline ASR used in the report
scripts/evaluate.py --split sir0_val --est <est dir> --metrics content --listener judge
scripts/evaluate.py --config experiments/configs/eval_offline_asr_turbo.yaml --split sir0_val --est <est dir>

# every trial case separately (both / target only / other speaker only / noise only)
scripts/eval_by_case.py --est <est dir> --split sir0_val --listener judge

# the whole battery in the right order; --dry-run first
scripts/run_eval_suite.py --tag e21 --checkpoint models/model_sir0_cuecontext-wer-e21.pt \
    --config experiments/configs/bsrnn_cue_context.yaml --dry-run

# speed: 80 ms chunks, CPU
scripts/measure_rtf.py --checkpoint models/model_sir0_cuecontext-wer-e21.pt \
    --config experiments/configs/bsrnn_cue_context.yaml --chunk-ms 80 --threads 4 --device cpu
```

WeSep estimates: run `scripts/make_estimates_wesep.py` under `../wesep_venv`.
For listening to one trial, use `scripts/pass_a_test_case_through.py`.

**Rules the results depend on:**

- Every judge result records model ID, exact prompt, modality and date.
  Training-time Gemini calls do the same.
- `sir0_privval` and `eval_private` are never scored, filtered or selected on
  during training. `eval_public` has now been looked at, so nothing may be tuned
  on it.
- The report's offline ASR is `faster-whisper large-v3-turbo`. The in-loop probe
  and the `evaluate.py` default are `small.en`. Never compare numbers from the
  two.

### 4. Live demo

```bash
scripts/demo_live.py        # then open the printed URL (127.0.0.1:8765)
```

You record or upload a target voice, an enrolment and optionally another
speaker. The demo builds a mixture the way trials are built and streams it
through the extractor 80 ms at a time. It then sends the mixture, the extracted
audio and the clean target to Gemini with the benchmark prompt. Settings are in
`experiments/configs/demo_live.yaml`; the checkpoint is set there (currently
`cuecontext-wer-e13`). Each run is logged to `experiments/results/demo-live/`.

### Tests

```bash
../tse_venv/bin/python -m pytest tests/ -q
```

## External dependencies

| purpose | source |
|---|---|
| Speech | LibriSpeech |
| Noise | WHAM! noise, screened for hidden speech |
| Rooms | Simulated with `pyroomacoustics` (WHAMR!-style, not WHAMR!'s files) |
| Voice activity | Silero VAD 6.2.1 |
| Speaker encoder | SpeechBrain ECAPA-TDNN, VoxCeleb, frozen |
| Judge | `gemini-3.7-flash`, audio in / text out |
| Offline ASR | `faster-whisper` 1.2.1: `large-v3-turbo` (report), `small.en` (training loop) |
| Perceptual quality | DNSMOS, ONNX snapshot in `src/live_model_metric/dnsmos_onnx/` |
| External reference model | WeSep `tfmap_context_causal_100` |

The REAL-TSE Challenge is the anchor benchmark for real conversational TSE. We
borrow its data-construction methods and its lessons about metric gaming. We
replicate neither its baselines nor its eval pipeline.
