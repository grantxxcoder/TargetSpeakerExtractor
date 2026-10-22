# Data acquisition runbook (M0)

**Written 2026-08-10.** Companion to `docs/decisions/milestones.md` §M0 and `docs/decisions/decisions-m0.md` (2026-08-07, "Data: constructed primary, AMI secondary").

Covers **acquiring and preparing source material only**. The mixture generator is the next job.

**How to use this document.** Every command is one line with no inline comments, safe to paste one at a time. Each section names the directory to `cd` into first, and paths are relative to it, so nothing depends on an environment variable.

Directory layout this runbook assumes, all under the repo root:

```
data/raw/              downloads and extraction, emptied as you go
data/wham_noise_16k/   converted noise, kept
data/librispeech/      LibriSpeech, kept
LibriMix/              read-only reference clone
```

`data/`, `reference/` and `LibriMix/` are gitignored. Keep it that way.

## What we are building, in one paragraph

One generator, one consumer. **Every split is rendered to disk from the same code** (LibriSpeech speech + WHAM! noise + a seeded synthetic RIR) and frozen as fixed audio files. Eval must be byte-stable for the judge; training is rendered too, since with B7 off its draws are fixed anyway.

> **Changed 2026-08-15.** Training mixtures were previously synthesised on the fly, on the grounds that the set would "cost no disk and never repeat". B7 turned repetition off the same day, so on-the-fly regeneration produced byte-identical audio every epoch while risking a CPU-bound dataloader starving the GPU on a 4-vCPU Kaggle instance. It also moved *more* data: a rendered trial needs no corpora on the training machine. See `docs/decisions/decisions-m0.md` 2026-08-15. Byte-stability is *implied* by the pin-and-date rule at `docs/data/metric-definitions.md:186` and the private-split requirement at `:186`, but never stated; add it to `metric-definitions.md` §4. Everything is speaker-disjoint.

We are **not** following CARTSE's or PS4's data recipes. Both build real conversational corpora with pseudo-labels, which cannot supply the clean target signal or the exact verbatim `t` and `d` our metric needs. We borrow two cheap, separable components from CARTSE (target-absent examples, channel-gap enrollment EQ) and REAL-T's construction method for the AMI set. See `docs/decisions/decisions-m0.md`.

## Before you start

- [ ] Branch: `git checkout -b m0-data-setup`. Never commit to main.
- [ ] Confirm `data/`, `reference/` and `LibriMix/` are in `.gitignore`.
- [ ] **≥70 GB free** on the volume holding the repo. The global peak is ~58 GB if you follow the step order; steady state is ~36 GB.
- [ ] Have `wget`, `md5sum`, `unzip` and `ffmpeg` installed.

**Order matters.** WHAM! goes first: it has the largest transient (52 GB) and shrinks the most (to ~6 GB), so the LibriSpeech extraction never overlaps with it.

---

## Step 1 — WHAM! noise

The 17 GB archive unpacks to 35 GB of **32-bit float stereo** WAV at 16 kHz, 4× more bytes than our 16-bit mono can use. Convert on ingest and delete the originals, the biggest storage saving available.

**Why 16 kHz.** Its 8 kHz Nyquist bandwidth covers almost all content that determines word identity; only the upper energy of sibilant fricatives sits above it. More importantly, every frozen model in our loop (ASR/SSL proxy encoders, speaker encoder, VAD) is a 16 kHz model, and the live judge takes raw 16-bit PCM at 16 kHz. Any other rate puts an uncontrolled resample between extractor and evaluator, a confound in a benchmark measuring what the judge hears. See `docs/data/definitions.md:60`.

The 4× saving is **not** from the sample rate (WHAM! is already 16 kHz, so `-ar 16000` below is a no-op guard). It comes entirely from `-ac 1` (stereo to mono) and `-sample_fmt s16` (32-bit float to 16-bit).

### 1a. Download — from `data/raw/`

```
wget -c https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
```

### 1b. Verify — from `data/raw/`

```
md5sum wham_noise.zip
```

The output must be `d5af15645d521d3920e01954c6cd7594`. Otherwise stop: a truncated download often still extracts without error and surfaces much later as silent data loss.

### 1c. Extract, then delete the archive — from `data/raw/`

```
unzip -q wham_noise.zip -d .
```

```
rm wham_noise.zip
```

Delete the zip before conversion to hold the peak at ~52 GB instead of ~58 GB.

### 1d. Create the output directory — from `data/`

```
mkdir -p wham_noise_16k
```

### 1e. Convert to 16 kHz mono FLAC — from `data/raw/wham_noise/`

```
find . -name '*.wav' -print0 | while IFS= read -r -d '' f; do out="../../wham_noise_16k/${f#./}"; mkdir -p "$(dirname "$out")"; ffmpeg -nostdin -loglevel error -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "${out%.wav}.flac"; done
```

This takes a while (28,000 files). `-nostdin` is required: without it `ffmpeg` swallows the `find` pipe and the loop silently exits after one file.

### 1f. Check the count — from `data/wham_noise_16k/`

```
find . -name '*.flac' | wc -l
```

Expect `28000`, split 20,000 `tr` / 5,000 `cv` / 3,000 `tt`.

### 1g. Reclaim the space — from `data/raw/`

```
rm -rf wham_noise
```

Ends at ~6 GB. Map WHAM!'s own tr/cv/tt split onto our train/val/eval splits; mixing noise across that boundary leaks.


## Step 2 — WHAMR! scripts (not the data)

4.8 MB. We want the room-parameter distributions and reverb recipe, then synthesise our own RIRs with `pyroomacoustics`.

### From `data/raw/`

```
wget -c https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/whamr_scripts.tar.gz
```

```
md5sum whamr_scripts.tar.gz
```

Must be `11a2384408bab4b7f3c64f171a593c70`.

```
tar -xzf whamr_scripts.tar.gz
```

```
rm whamr_scripts.tar.gz
```

**Do not** build the WHAM!/WHAMR! mixtures. They need wsj0-2mix and so the paid LDC WSJ0 licence, which we do not have. We need neither: the noise archive is standalone and we generate reverb ourselves.

Read `whamr_scripts/` for the room-size, absorption and source/mic placement ranges. Cite Maciejewski et al. 2020, and note in a code comment that the RIRs are ours, generated to their published distributions, not their released RIRs.

## Step 3 — LibriSpeech

Download all four archives, verify in one pass, then extract and delete one at a time so no tar coexists with its own extract.

### 3a. Create the directory — from `data/`

```
mkdir -p librispeech
```

### 3b. Download — from `data/librispeech/`

```
wget -c https://openslr.trmal.net/resources/12/train-clean-100.tar.gz
```

```
wget -c https://openslr.trmal.net/resources/12/train-clean-360.tar.gz
```

```
wget -c https://openslr.trmal.net/resources/12/dev-clean.tar.gz
```

```
wget -c https://openslr.trmal.net/resources/12/test-clean.tar.gz
```

Sizes: 6.3 GB, 23 GB, 337 MB, 346 MB.

### 3c. Verify all four at once — from `data/librispeech/`

```
wget -c https://openslr.trmal.net/resources/12/md5sum.txt
```

```
md5sum -c md5sum.txt --ignore-missing
```

`--ignore-missing` checks only the archives you downloaded. Every line must end in `OK`; otherwise stop.

### 3d. Extract and delete, one at a time — from `data/librispeech/`

**Be in `data/librispeech/`, not one level deeper.** Each tarball has its own top-level `LibriSpeech/` directory, so all four extract into a shared `data/librispeech/LibriSpeech/`. Extracting from inside it gives `data/librispeech/LibriSpeech/train-clean-360/LibriSpeech/train-clean-360/…`, a nested duplicate that wastes tens of GB and hides the real speakers from `make_splits.py`. Check `pwd` before each `tar`.

If it has already happened, do not delete either tree. Merge the nested one upward, as it may hold speakers the outer one lacks:

```
rsync -a --remove-source-files data/librispeech/LibriSpeech/train-clean-360/LibriSpeech/train-clean-360/ data/librispeech/LibriSpeech/train-clean-360/
```

```
rm -rf data/librispeech/LibriSpeech/train-clean-360/LibriSpeech
```


```
tar -xzf train-clean-100.tar.gz
```

```
rm train-clean-100.tar.gz
```

```
tar -xzf train-clean-360.tar.gz
```

```
rm train-clean-360.tar.gz
```

```
tar -xzf dev-clean.tar.gz
```

```
rm dev-clean.tar.gz
```

```
tar -xzf test-clean.tar.gz
```

```
rm test-clean.tar.gz
```

LibriSpeech takes **~30 GB, peaking at ~58 GB during the 360 extraction** (52 GB of LibriSpeech plus ~6 GB of converted WHAM!). Deleting each tar before the next extraction keeps it there.

Do **not** download `train-other-500` (30 GB): "other" is lower-quality audio, and the ceiling condition needs clean targets. Skip `original-mp3` (87 GB) and `intro-disclaimers` too.

The per-chapter `.trans.txt` files are your verbatim `t` and `d`, uppercase with no punctuation. The judge outputs punctuation and casing, so decide now whether your metric's text normaliser makes the comparison fair.

## Step 4 — Python environment

### From the repo root

```
python -m venv .venv
```

```
source .venv/bin/activate
```

```
pip install numpy scipy soundfile pyroomacoustics tqdm pyyaml
```

`pyroomacoustics` is the RIR synthesiser. `soundfile` reads FLAC directly, so nothing is decoded to WAV on disk.

## Step 5 — LibriMix, reference only

### From the repo root

```
git clone https://github.com/JorisCos/LibriMix
```

(If already cloned to the repo root, leave it there; `LibriMix/` is gitignored.)

**Do not run `generate_librimix.sh`.** It emits ~430 GB for Libri2Mix alone and produces the wrong artefact: 100% overlap, no partial overlap, no target-absent examples, no enrollment segment, no reverberation. Our metric needs each of these controllable per trial.

Read it for two things: the LUFS loudness-normalisation logic, and the metadata/CSV schema, a sensible starting shape for our manifest. The real normalisation (pyloudnorm, target loudness, gain computation) is in `scripts/create_librimix_metadata.py`; `create_librimix_from_metadata.py` only applies those gains. Cite Cosentino et al. 2020 for anything borrowed.

One nuance: "100% overlap" is exact in LibriMix `min` mode; `max` mode zero-pads the shorter source, leaving a non-overlapping tail. Neither gives a *controllable* overlap ratio, which is what we need.

## Step 6 — Freeze the split assignment BEFORE generating anything

**What this means.** Decide now which LibriSpeech *speakers* (not folders or files) belong to training and which to evaluation, and write it to a file everything downstream reads. If speaker 1272 is in both training and eval, the model has heard that voice before and your eval number is silently inflated.

This is cheap now and unrecoverable later.

`scripts/make_splits.py` does it. Run once, from the repo root:

```
python scripts/make_splits.py
```

It reads `SPEAKERS.TXT` (ships with LibriSpeech), assigns every speaker, asserts the splits are disjoint, and writes `experiments/configs/splits.yaml` with the seed, git commit and a checksum of its input.

The assignment it makes:

| Split | Source | Speakers | Purpose |
|---|---|---|---|
| `train` | train-clean-100 + train-clean-360 | 1172 | training |
| `val` | dev-clean | 40 | tuning, early stopping |
| `eval_public` | test-clean, half | 20 | publishable trials |
| `eval_private` | test-clean, other half | 20 | headline numbers, held back |
| `smoke_train` | subset of train | 20 | laptop prototyping |
| `smoke_val` | subset of val | 5 | laptop prototyping |

`eval_public` / `eval_private` exists because `docs/data/metric-definitions.md:186` requires a split held back from publication, so headline numbers can't be overfitted to trials anyone can inspect. The halves are stratified by sex so a difference between them means something.

Then:

```
git add experiments/configs/splits.yaml
```

```
git commit -m "Pin speaker-disjoint splits (seed 42)"
```

**Commit it before generating any data.** Once mixtures exist, re-running with a different seed silently redefines "eval" and invalidates every number already produced.

## Step 7 — Prototype locally, train elsewhere

**Your laptop cannot train this model.** An i5 with Iris Xe integrated graphics and 16 GB RAM has no NVIDIA GPU, so no CUDA. Intel's path exists (`torch.xpu` in PyTorch since 2.5, plus Intel Extension for PyTorch), but Gen12 *integrated* Iris Xe is not on Intel's supported hardware list, shares the 16 GB of system RAM, and gives on the order of 1–2 TFLOPS FP32: enough for smoke-test inference, not training. (DirectML is Windows-only; OpenVINO is inference-only but could speed up local listening tests.) A causal BSRNN at ~25 M parameters is a 40–70 hour job on a proper GPU; here it would never finish.

**On the laptop: everything except the training run.** Downloading and preparing corpora. Writing and debugging the generator. Generating a handful of mixtures and *listening to them*, the highest-value debugging step in the project (no GPU needed). Building the manifest. Writing the metric harness. Running the judge (an API call, run on Google's servers). Every unit test.

Use `smoke_train` / `smoke_val` for this: 25 speakers total, a few hundred mixtures, a model shrunk to a couple of layers. The goal is a pipeline that runs end to end, not a result. If the smoke config trains for 50 steps without crashing and the loss moves, the code is right and only compute is missing.

**On the HPC or GCP: the training run only.** Same code, different config file.

### Getting the data onto the cluster

Do **not** re-run this runbook there or upload 36 GB from home broadband. Two better options, in order:

1. **Re-download on the cluster.** LibriSpeech and WHAM! are public URLs, and university-datacentre bandwidth likely beats your upload, for the cost of a job script. Copy `docs/data/data-setup.md` steps 1–3 into a shell script and run it once there. Many HPC sites already host LibriSpeech in a shared read-only dataset directory, so ask first.
2. **`rsync` what can't be re-downloaded**: `experiments/configs/`, your code, and the frozen val/eval audio (~2–5 GB). The eval set must be byte-identical everywhere, or numbers are not comparable across machines.

Exclude the re-downloadable corpora and the local virtualenv, but **keep** the frozen audio in `data/eval/`:

```
rsync -avP --exclude .venv/ --exclude LibriMix/ --exclude reference/ --exclude data/raw/ --exclude data/librispeech/ --exclude data/wham_noise_16k/ ~/Documents/University/Masters/Project/TargetSpeakerExtractor/ user@cluster:~/tse/
```

A bare `--exclude data/` would also drop the frozen eval set, which cannot be regenerated identically, so exclude the corpus subdirectories by name. `.venv/` is excluded because a laptop-built virtualenv breaks anywhere else; rebuild it there.

**Since 2026-08-15, move the rendered training audio and leave the corpora behind.** A rendered trial is self-contained, so ship ~26 GB of rendered audio instead of ~36 GB of corpora. Exclude `data/librispeech/`, `data/wham_noise_16k/` and `data/raw/` by name; include `data/rendered/` and `data/manifests/`.

### The one thing that makes this painless

Never hardcode a path. Every script takes the corpus root as an argument or from the config, so laptop and cluster differ by one YAML line. `make_splits.py` already does this (`--librispeech-root`); keep that discipline in the generator.

Google Cloud machines are ephemeral: keep the corpora on a persistent disk or GCS bucket, not the VM's boot disk, or you will pay to re-download them every session.

## Step 8 — AMI (defer to a gap week)

Off the critical path: `docs/decisions/milestones.md:20` says start it whenever there is a gap. Zero GPU cost, slow download.

### Annotations first — from `data/raw/`

```
wget -c https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
```

22 MB, NXT XML, CC BY 4.0. Carries the segment and speaker annotations for REAL-T-style trial construction (Li et al., Interspeech 2025).

### Signals

Use the chooser at <https://groups.inf.ed.ac.uk/ami/download/>, which emits a `wget` script. Select **Individual headsets** (~120 MB per meeting, our approximate ceiling) and **Microphone array** (~360 MB per meeting, our mixture). The full corpus of ~170 meetings is ~80 GB, so take 20–30 meetings first and expand only if the transfer check looks informative. Skip all video streams.

Mandatory caveat (`docs/decisions/decisions-m0.md:62`): the AMI ceiling is **approximate**, computed from IHM, which has cross-talk bleed and a different channel response from the distant mic. Never label it ground truth.

Before writing an NXT XML parser, check whether a community repo already packages AMI's diarisation annotations usably, starting with the pyannote AMI diarisation setup. *(Unverified as of 2026-08-10.)*

---

## Storage summary

| Point in the runbook | Disk |
|---|---|
| Peak, step 1c (zip + extracted, before `rm`) | ~52 GB |
| After step 1g | ~6 GB |
| Peak, step 3d (WHAM 6 + 360 tar + extract) | ~58 GB |
| **Steady state, sources ready** | **~36 GB** |
| Frozen val + eval audio, added later | ~2–5 GB |
| Training mixtures, rendered once | ~26 GB (decisions-m0.md 2026-08-15) |

Global peak ~58 GB during setup, then ~67 GB steady state once training audio is rendered. **Budget ~90 GB free**, not the 70 GB assumed before the 2026-08-15 decision to render training audio to disk.

## Do not delete the source corpora afterwards

`CLAUDE.md` requires that a config plus a seed reproduces a run, and rendered audio is downstream of the corpora, not a replacement: **any manifest rebuild invalidates every rendered file and needs the sources to re-render.** B2's rebuild is one such case, and will not be the last.

Safe to delete: every archive after checksum verification, and the 32-bit float WHAM! originals after conversion. Nothing else.

## Verification before you call M0 sources done

- [X] Every checksum matched. Checked, not assumed.
- [X] `data/wham_noise_16k/` holds 28,000 FLAC files across tr/cv/tt.
- [ ] LibriSpeech speaker counts: 251 (train-clean-100), 921 (train-clean-360), 40 (dev-clean), 40 (test-clean).
- [ ] Spot-check ten converted noise files: 16 kHz, 1 channel, audibly identical to source.
- [ ] `experiments/configs/splits.yaml` committed, with disjointness asserted in code rather than eyeballed.
- [X] WHAM! CC BY-NC licence note added to `docs/decisions/decisions-m0.md`.
- [ ] `git status` shows no audio staged.
- [ ] PR opened. Small and single-purpose.

## Sources

- LibriSpeech (CC BY 4.0): <https://www.openslr.org/12>
- WHAM! / WHAMR! (CC BY-NC 4.0): <http://wham.whisper.ai/>
- WHAMR! generation docs: <https://wham.whisper.ai/WHAMR_README.html>
- LibriMix: <https://github.com/JorisCos/LibriMix>
- AMI corpus (CC BY 4.0): <https://groups.inf.ed.ac.uk/ami/download/>
- AMI licence: <https://groups.inf.ed.ac.uk/ami/corpus/license.shtml>
