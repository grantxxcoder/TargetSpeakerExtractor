# What each dataset is for

**Memory aid.** Written 2026-08-10, for when you forget why WHAM! is on your disk.

## One-line version

LibriSpeech is the **voices**. WHAM! is the **background noise**. WHAMR! gives the **room**. AMI is the **reality check** at the very end.

## The pieces

### LibriSpeech — the speech

Read audiobooks: one speaker per recording, clean, 16 kHz, exact transcripts. Per trial it supplies:

- the **target speaker's** utterance
- the **interferer's** utterance (a different speaker)
- the **enrollment** clip (≥5 s of the target, from a *different* recording)
- the **verbatim text** `t` and `d` for both, from the `.trans.txt` files

The primary set is constructed because signal losses need a *clean* target and content scoring needs *exact* text. Real meeting corpora have neither.

### WHAM! noise — the background

Real (not synthetic) recordings from cafés, bars, restaurants and parks around the San Francisco Bay Area, mixed under the speakers at a controlled SNR. This makes the task realistically hard rather than white-noise hard.

Its tr/cv/tt split maps onto our train/val/eval. Never mix noise across that boundary: a noise clip heard in training must not appear in eval.

### WHAMR! scripts — the room

**We use no WHAMR! audio.** We only read their scripts for the room-size, absorption and mic/source placement distributions, then generate our own impulse responses with `pyroomacoustics`.

Convolving a dry utterance with an RIR adds the room's reflections and reverb tail. Without it everything sounds like a studio, and the model never learns real rooms.

Cite Maciejewski et al. 2020 anyway: the distributions are theirs.

### LibriMix — reference only, never run

A cloned repo, read for its loudness-normalisation logic and manifest schema. Its data is wrong for us (100% overlap, no enrollment, no reverb, no target-absent), and running it takes ~430 GB. Do not run it.

### AMI — the reality check

Real meetings, recorded on headset mics *and* distant mics at the same time.

Used for secondary evaluation (`ami_eval`) and, since 2026-10-12, for M8 fine-tuning (`ami_train`, 20 other scenario groups, no shared speakers; `decisions-m8.md`). The distant mic is the mixture. The headset is the approximate ceiling and, gated, the approximate training target. AMI answers the one question the constructed set cannot: does any of this transfer to real audio? For M8 models it is in-domain, not a transfer test.

The headset picks up bleed from other speakers, so it is not a clean target. Never call it ground truth.

## How they combine into one trial

```
target utterance   (LibriSpeech, speaker A)  ─┐
                          × RIR (pyroomacoustics, WHAMR! params)
interferer utterance (LibriSpeech, speaker B) ─┤
                          × RIR                ├─ mix at chosen SNR ─→  x   (mixture)
background noise   (WHAM!)                    ─┘

clean target (speaker A × RIR, no interferer, no noise)  ─→  ceiling reference
enrollment   (LibriSpeech, speaker A, different recording, ≥5 s, EQ-augmented)  ─→  e
transcripts  (LibriSpeech .trans.txt)  ─→  t (target), d (interferer)
```

Every trial ships all five (`x`, clean target, `e`, `t`, `d`) plus the experimental variables as metadata: overlap ratio, SNR, enrollment-device condition.

## What is used where

| | Train | Val | Primary eval | Secondary eval |
|---|---|---|---|---|
| LibriSpeech | yes | yes | yes | — |
| WHAM! noise | yes | yes | yes | — |
| Generated RIRs | yes | yes | yes | — |
| AMI | `ami_train` (M8 only) | — | — | `ami_eval` |

Speakers differ in every column, enforced by `experiments/configs/splits.yaml`.

## Things that are NOT in the data pipeline

- **The judge** (Gemini Live or similar). It never touches training, never filters data, never appears as a proxy. Held-out evaluator only.
- **wsj0 / wsj0-2mix.** Needs a paid LDC licence, which we neither have nor need.
- **`train-other-500`.** Lower-quality audio; we need clean targets.
- **Real conversational corpora for training.** No clean target, no verbatim text. See `docs/decisions/decisions-m0.md` 2026-08-07.

## Storage, roughly

| | Size | Kept? |
|---|---|---|
| LibriSpeech | ~30 GB | yes, permanently |
| WHAM! noise, converted | ~6 GB | yes, permanently |
| Training mixtures | ~26 GB | yes, rendered once (decisions-m0.md 2026-08-15) |
| Val + eval mixtures | ~2–5 GB | yes, frozen |
| AMI subset | ~10–80 GB | later, optional |

**Changed 2026-08-15**: training mixtures were previously "0 GB, generated on the fly, never stored". They are now rendered to disk like every other split. On-the-fly generation rested on the training set "never repeating", which B7 turned off the same day. With fixed draws it repeats byte-for-byte, so regenerating each epoch bought nothing and risked a CPU-bound dataloader starving the GPU. See `docs/decisions/decisions-m0.md` 2026-08-15.

Keep LibriSpeech and WHAM! after rendering. Re-rendering after a manifest rebuild needs them, and a logged seed only reproduces a run while they exist.

## See also

- `docs/data/data-setup.md`: how to download and prepare all of it
- `docs/data/data-licences.md`: what you may do with it
- `docs/data/changing-the-data.md`: **how to change a distribution and rebuild**
- `docs/data/data-construction-parameters.md`: what each generator parameter is
- `docs/data/difficulty-dial.md`: which parameters to narrow first, and T60 in room terms
- `docs/decisions/decisions-m0.md`: why these choices, dated
