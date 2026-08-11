# Data acquisition runbook (M0)

**Written 2026-08-10.** Companion to `docs/milestones.md` §M0 and
`docs/decisions.md` (2026-08-07, "Data: constructed primary, AMI secondary").

Covers **acquiring and preparing source material only**. Writing the mixture
generator is the next job and is not in this document.

**How to use this document.** Every command is a single line with no inline
comments, so it is safe to paste one line at a time. Each section states which
directory you must be in first — `cd` there yourself before pasting. Paths are
relative to that directory, so nothing depends on an environment variable.

Directory layout this runbook assumes, all under the repo root:

```
data/raw/              downloads and extraction, emptied as you go
data/wham_noise_16k/   converted noise, kept
data/librispeech/      LibriSpeech, kept
LibriMix/              read-only reference clone
```

`data/`, `reference/` and `LibriMix/` are gitignored. Keep it that way.

## What we are building, in one paragraph

One generator, two consumers. Training mixtures are synthesised **on the fly**
from LibriSpeech speech + WHAM! noise + a seeded synthetic RIR, so the training
set costs no disk and never repeats. Val and eval are generated once from the
same code and **frozen to disk as fixed audio files**, because the eval set must
be byte-stable for the judge. Byte-stability is *implied* by the pin-and-date
rule at `docs/metric-definitions.md:209` and the private-split requirement at
`:198-200`, but is nowhere stated outright — add it explicitly to
`metric-definitions.md` §4. Everything is speaker-disjoint.

We are **not** following CARTSE's or PS4's data recipes. Both build corpora of
real conversational audio with pseudo-labels, which cannot supply the clean
target signal or the exact verbatim `t` and `d` our metric requires. We borrow
two cheap, separable components from CARTSE (target-absent examples,
channel-gap enrollment EQ) and REAL-T's construction method for the AMI set.
See `docs/decisions.md`.

## Before you start

- [ ] Branch: `git checkout -b m0-data-setup`. Never commit to main.
- [ ] Confirm `data/`, `reference/` and `LibriMix/` are in `.gitignore`.
- [ ] **≥70 GB free** on the volume holding the repo. Global peak is ~58 GB if
      you follow the step order; steady state is ~36 GB.
- [ ] Have `wget`, `md5sum`, `unzip` and `ffmpeg` installed.

**Order matters.** WHAM! goes first because it has the largest transient
(52 GB) and shrinks the most (to ~6 GB). Doing it first means the LibriSpeech
extraction never overlaps with it.

---

## Step 1 — WHAM! noise

The 17 GB archive unpacks to 35 GB of **32-bit float stereo** WAV at 16 kHz.
We work in 16-bit mono, so that is 4× more bytes than we can use. Convert on
ingest and delete the originals — the single biggest storage win available.

**Why 16 kHz.** Nyquist gives 8 kHz of bandwidth, which covers essentially all
of the content that determines word identity; only the upper energy of sibilant
fricatives sits above it. More decisively, every frozen model in our loop —
ASR/SSL proxy encoders, speaker encoder, VAD — is a 16 kHz model, and the live
judge takes raw 16-bit PCM at 16 kHz. Any other rate means an uncontrolled
resample sits between the extractor and the evaluator, which is a confound in a
benchmark measuring what the judge hears. See `docs/definitions.md:60`.

Note that the 4× saving here is **not** from the sample rate: WHAM! noise is
already 16 kHz, so `-ar 16000` below is a no-op guard. The reduction comes
entirely from `-ac 1` (stereo → mono) and `-sample_fmt s16` (32-bit float →
16-bit).

### 1a. Download — from `data/raw/`

```
wget -c https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip
```

### 1b. Verify — from `data/raw/`

```
md5sum wham_noise.zip
```

The output must be `d5af15645d521d3920e01954c6cd7594`. Do not continue if it
differs. A truncated download often still extracts without error and shows up
much later as silent data loss.

### 1c. Extract, then delete the archive — from `data/raw/`

```
unzip -q wham_noise.zip -d .
```

```
rm wham_noise.zip
```

Delete the zip now rather than after conversion. That is what holds the peak at
~52 GB instead of ~58 GB.

### 1d. Create the output directory — from `data/`

```
mkdir -p wham_noise_16k
```

### 1e. Convert to 16 kHz mono FLAC — from `data/raw/wham_noise/`

```
find . -name '*.wav' -print0 | while IFS= read -r -d '' f; do out="../../wham_noise_16k/${f#./}"; mkdir -p "$(dirname "$out")"; ffmpeg -nostdin -loglevel error -i "$f" -ac 1 -ar 16000 -sample_fmt s16 "${out%.wav}.flac"; done
```

This takes a while — 28,000 files. `-nostdin` is load-bearing: without it
`ffmpeg` swallows the `find` pipe, the loop exits after one file, and you get a
single output with no error message.

### 1f. Check the count — from `data/wham_noise_16k/`

```
find . -name '*.flac' | wc -l
```

Expect `28000`, split 20,000 `tr` / 5,000 `cv` / 3,000 `tt`.

### 1g. Reclaim the space — from `data/raw/`

```
rm -rf wham_noise
```

Ends at ~6 GB. Keep WHAM!'s own tr/cv/tt split and map it onto our
train/val/eval splits — mixing noise across that boundary leaks.


## Step 2 — WHAMR! scripts (not the data)

4.8 MB. We want the room-parameter distributions and reverb recipe, then
synthesise our own RIRs with `pyroomacoustics`.

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

**Do not** build the actual WHAM!/WHAMR! mixtures. They require wsj0-2mix,
which needs the LDC WSJ0 licence — paid, and we do not have it. We need none of
it: the noise archive is standalone and we generate reverb ourselves.

Read `whamr_scripts/` for the room-size, absorption and source/mic placement
ranges. Cite Maciejewski et al. 2020, and note in a code comment that the RIRs
are ours, generated to their published distributions, not their released RIRs.

## Step 3 — LibriSpeech

Download all four archives, verify in one pass, then extract and delete one at
a time so the tars never coexist with their own extracts.

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

`--ignore-missing` checks only the archives you actually downloaded. Every line
printed must end in `OK`. Do not continue otherwise.

### 3d. Extract and delete, one at a time — from `data/librispeech/`

**Be in `data/librispeech/`, not one level deeper.** Each tarball contains its
own top-level `LibriSpeech/` directory, so all four extract into a shared
`data/librispeech/LibriSpeech/`. If you `cd` into that directory first and
extract there, you get `data/librispeech/LibriSpeech/train-clean-360/LibriSpeech/train-clean-360/…`
— a nested duplicate that wastes tens of GB and leaves the real speakers
somewhere `make_splits.py` won't find them. Check with `pwd` before each `tar`.

If it has already happened, don't delete either tree — merge the nested one
upward, because it may hold speakers the outer one lacks:

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

All four extract into a shared `LibriSpeech/` subdirectory. **~30 GB. Peak
~58 GB during the 360 extraction** — 52 GB of LibriSpeech plus the ~6 GB of
converted WHAM! already on disk. Deleting each tar before extracting the next
is what keeps it there.

Do **not** download `train-other-500` (30 GB): "other" is lower-quality audio
and we need clean targets for the ceiling condition. Nor `original-mp3` (87 GB)
or `intro-disclaimers`.

The per-chapter `.trans.txt` files are your verbatim `t` and `d`. They are
uppercase with no punctuation. Decide now whether your metric's text normaliser
matches that convention, because the judge outputs punctuation and casing and
the comparison has to be fair.

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

`pyroomacoustics` is the RIR synthesiser. `soundfile` reads FLAC directly, so
nothing needs decoding to WAV on disk.

## Step 5 — LibriMix, reference only

### From the repo root

```
git clone https://github.com/JorisCos/LibriMix
```

(If you already cloned it to the repo root, that's where it is — `LibriMix/`
is gitignored, so it won't be committed either way.)

**Do not run `generate_librimix.sh`.** It emits ~430 GB for Libri2Mix alone,
and produces the wrong artefact: 100% overlap, no partial overlap, no
target-absent examples, no enrollment segment, no reverberation. Every one of
those is a variable our metric needs controllable per trial.

Read it for two things: the LUFS loudness-normalisation logic, and the
metadata/CSV schema, which is a sane starting shape for our manifest. The real
normalisation (pyloudnorm, target loudness, gain computation) lives in
`scripts/create_librimix_metadata.py` — `create_librimix_from_metadata.py` only
applies gains already computed there. Cite Cosentino et al. 2020 for anything
borrowed.

One nuance: "100% overlap" is exact in LibriMix `min` mode; in `max` mode the
shorter source is zero-padded, giving a non-overlapping tail. Either way there
is no *controllable* overlap ratio, which is the property we need.

## Step 6 — Freeze the split assignment BEFORE generating anything

**What this means.** Decide, right now, which LibriSpeech *speakers* belong to
training and which to evaluation — and write that decision to a file that
everything downstream reads. Not folders, not files: speakers. If speaker 1272
appears in both training and eval, the model has heard that voice before and
your eval number is inflated. It will not error, it will not look wrong, and
you will not find out.

This is cheap now and unrecoverable later, which is why it happens before the
generator exists rather than after.

`scripts/make_splits.py` does it. Run once, from the repo root:

```
python scripts/make_splits.py
```

It reads `SPEAKERS.TXT` (ships with LibriSpeech), assigns every speaker, asserts
the splits are disjoint, and writes `experiments/configs/splits.yaml` with the
seed, the git commit and a checksum of its input.

The assignment it makes:

| Split | Source | Speakers | Purpose |
|---|---|---|---|
| `train` | train-clean-100 + train-clean-360 | 1172 | training |
| `val` | dev-clean | 40 | tuning, early stopping |
| `eval_public` | test-clean, half | 20 | publishable trials |
| `eval_private` | test-clean, other half | 20 | headline numbers, held back |
| `smoke_train` | subset of train | 20 | laptop prototyping |
| `smoke_val` | subset of val | 5 | laptop prototyping |

`eval_public` / `eval_private` exists because
`docs/metric-definitions.md:198-200` requires holding a split back from
publication, so headline numbers can't be overfitted to trials anyone can
inspect. The two halves are stratified by sex so a difference between them
means something.

Then:

```
git add experiments/configs/splits.yaml
```

```
git commit -m "Pin speaker-disjoint splits (seed 42)"
```

**Commit it before generating any data.** After mixtures exist, re-running with
a different seed silently redefines what "eval" means and invalidates every
number you have already produced.

## Step 7 — Prototype locally, train elsewhere

**Your laptop cannot train this model, and that is fine.** An i5 with Iris Xe
integrated graphics and 16 GB RAM has no NVIDIA GPU, so no CUDA. Intel's own
path is real — `torch.xpu` has been in PyTorch since 2.5, plus Intel Extension
for PyTorch — but Gen12 *integrated* Iris Xe is not on Intel's supported
hardware list, it shares your 16 GB of system RAM rather than having its own,
and it delivers on the order of 1–2 TFLOPS FP32. Usable for smoke-test
inference; not for a training run. (DirectML is Windows-only, so irrelevant on
Linux; OpenVINO is inference-only but could speed up local listening tests.) A
causal BSRNN at ~25 M parameters is a 40–70 hour job on a proper GPU; here it
is not hours-slower, it is not-going-to-finish slower.

So split the work by what each machine is actually good at:

**On the laptop — everything except the training run.** Downloading and
preparing corpora. Writing and debugging the generator. Generating a handful of
mixtures and *listening to them*, which is the single highest-value debugging
step in this whole project and needs no GPU. Building the manifest. Writing the
metric harness. Running the judge (it's an API call — the work happens on
Google's servers, not yours). Every unit test.

Use `smoke_train` / `smoke_val` for this: 25 speakers total, a few hundred
mixtures, a model shrunk to a couple of layers. The goal is proving the pipeline
runs end to end, not producing a result. If the smoke config trains for 50 steps
without crashing and the loss moves, the code is right and the only thing
missing is compute.

**On the HPC or GCP — the training run only.** Same code, different config file.

### Getting the data onto the cluster

You do **not** re-run this runbook there, and you do not upload 36 GB from home
broadband. Two better options, in order:

1. **Re-download on the cluster.** LibriSpeech and WHAM! come from public URLs
   at university-datacentre bandwidth — likely faster than your upload speed,
   and it costs you nothing but a job script. Copy `docs/data-setup.md` steps
   1–3 into a shell script and run it there once. Many HPC sites also already
   host LibriSpeech in a shared read-only dataset directory; ask before
   downloading anything.
2. **`rsync` what can't be re-downloaded.** That is `experiments/configs/`,
   your code, and the frozen val/eval audio (~2–5 GB) — the eval set must be
   byte-identical everywhere or your numbers aren't comparable across machines.

Exclude the re-downloadable corpora and the local virtualenv, but **keep**
`data/eval/` — that is the frozen audio that must be byte-identical on both
machines:

```
rsync -avP --exclude .venv/ --exclude LibriMix/ --exclude reference/ --exclude data/raw/ --exclude data/librispeech/ --exclude data/wham_noise_16k/ ~/Documents/University/Masters/Project/TargetSpeakerExtractor/ user@cluster:~/tse/
```

A bare `--exclude data/` would drop the frozen eval set along with the
corpora, which is exactly the file you cannot regenerate identically. Exclude
the corpus subdirectories by name instead. `.venv/` is excluded because a
virtualenv built on your laptop is broken anywhere else — rebuild it there.

This is exactly why training mixtures are generated on the fly: there is no
70 GB training set to move, only ~36 GB of public corpora that either machine
can fetch for itself.

### The one thing that makes this painless

Never hardcode a path. Every script takes the corpus root as an argument or
reads it from the config, so laptop and cluster differ by one line in a YAML
file and nothing else. `make_splits.py` already works this way
(`--librispeech-root`). Keep that discipline in the generator.

Google Cloud adds one wrinkle: machines are ephemeral, so put the corpora in a
persistent disk or GCS bucket rather than the VM's boot disk, or you will
re-download them every session and pay for it each time.

## Step 8 — AMI (defer to a gap week)

Off the critical path — `docs/milestones.md:30` says start it whenever there is
a gap. Zero GPU cost, slow download.

### Annotations first — from `data/raw/`

```
wget -c https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
```

22 MB, NXT XML, CC BY 4.0. Carries the segment and speaker annotations that
drive REAL-T-style trial construction (Li et al., Interspeech 2025).

### Signals

From the chooser at <https://groups.inf.ed.ac.uk/ami/download/>, which emits a
`wget` script for your selection. Select **Individual headsets** (~120 MB per
meeting, our approximate ceiling) and **Microphone array** (~360 MB per
meeting, our mixture). At ~170 meetings that is ~80 GB for the full corpus, so
take 20–30 meetings first and expand only if the transfer check looks
informative. Skip all video streams.

Remember the mandatory caveat (`docs/decisions.md:139`): the AMI ceiling is
**approximate**, computed from IHM, which has cross-talk bleed and a different
channel response from the distant mic. Never label it ground truth.

Before writing an NXT XML parser, check whether a community repo already
packages AMI's diarisation annotations usably — the pyannote AMI diarisation
setup is the one to look at first. *(Unverified as of 2026-08-10.)*

---

## Storage summary

| Point in the runbook | Disk |
|---|---|
| Peak, step 1c (zip + extracted, before `rm`) | ~52 GB |
| After step 1g | ~6 GB |
| Peak, step 3d (WHAM 6 + 360 tar + extract) | ~58 GB |
| **Steady state, sources ready** | **~36 GB** |
| Frozen val + eval audio, added later | ~2–5 GB |
| Training mixtures | 0 GB, generated on the fly |

Global peak ~58 GB, so 70 GB free leaves ~12 GB of headroom. Enough, but do not
run this on a nearly full disk. The earlier 150 GB estimate assumed archives
were kept and training mixtures materialised; neither is necessary.

## Do not delete the source corpora afterwards

Tempting, but it breaks the project. On-the-fly training generation means the
sources *are* the training set — and `CLAUDE.md` requires that a config plus a
seed reproduces a run, which is only true while LibriSpeech and WHAM! still
exist. Delete them and the seed reconstructs nothing.

Safe to delete: every archive after checksum verification, and the 32-bit float
WHAM! originals after conversion. Nothing else.

## Verification before you call M0 sources done

- [X] Every checksum matched. Checked, not assumed.
- [X] `data/wham_noise_16k/` holds 28,000 FLAC files across tr/cv/tt.
- [ ] LibriSpeech speaker counts: 251 (train-clean-100), 921 (train-clean-360),
      40 (dev-clean), 40 (test-clean).
- [ ] Spot-check ten converted noise files: 16 kHz, 1 channel, audibly
      identical to source.
- [ ] `experiments/configs/splits.yaml` committed, disjointness asserted in
      code rather than eyeballed.
- [X] WHAM! CC BY-NC licence note added to `docs/decisions.md`.
- [ ] `git status` shows no audio staged.
- [ ] PR opened. Small and single-purpose.

## Sources

- LibriSpeech (CC BY 4.0) — <https://www.openslr.org/12>
- WHAM! / WHAMR! (CC BY-NC 4.0) — <http://wham.whisper.ai/>
- WHAMR! generation docs — <https://wham.whisper.ai/WHAMR_README.html>
- LibriMix — <https://github.com/JorisCos/LibriMix>
- AMI corpus (CC BY 4.0) — <https://groups.inf.ed.ac.uk/ami/download/>
- AMI licence — <https://groups.inf.ed.ac.uk/ami/corpus/license.shtml>
