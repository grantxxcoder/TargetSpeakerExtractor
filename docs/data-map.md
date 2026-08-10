# What each dataset is for

**Memory aid.** Written 2026-08-10. If you come back in three weeks and
can't remember why WHAM! is on your disk, read this.

## One-line version

LibriSpeech is the **voices**. WHAM! is the **background noise**. WHAMR!
gives the **room**. AMI is the **reality check** at the very end.

## The pieces

### LibriSpeech — the speech

Read audiobooks, one speaker per recording, clean, 16 kHz, with exact
transcripts.

Supplies four things per trial:

- the **target speaker's** utterance
- the **interferer's** utterance (a different speaker)
- the **enrollment** clip (≥5 s of the target, from a *different* recording)
- the **verbatim text** `t` and `d` for both, from the `.trans.txt` files

Why this and not real conversation: we need a *clean* target signal to
compute signal losses against, and *exact* text to score content fidelity.
Real meeting corpora have neither. That is the whole reason the primary set
is constructed.

### WHAM! noise — the background

Real recordings from cafés, bars, restaurants and parks around the San
Francisco Bay Area. Not synthetic noise.

Mixed in underneath the speakers at a controlled SNR. This is what makes
the task hard in a realistic way rather than a white-noise way.

Its own tr/cv/tt split maps onto our train/val/eval. Never mix noise across
that boundary — a noise clip heard in training must not appear in eval.

### WHAMR! scripts — the room

**We use no WHAMR! audio.** We only read their scripts to learn the
room-size, absorption and mic/source placement distributions, then generate
our own impulse responses with `pyroomacoustics`.

An RIR convolved with a dry utterance makes it sound like it was recorded
in a room — reflections, reverb tail. Without this, everything sounds like
a studio and the model never learns to cope with a real room.

Cite Maciejewski et al. 2020 anyway: the distributions are theirs.

### LibriMix — reference only, never run

A cloned repo we read for its loudness-normalisation logic and manifest
schema. It generates the wrong kind of data for us (100% overlap, no
enrollment, no reverb, no target-absent). ~430 GB if you run it. Don't.

### AMI — the reality check

Real recorded meetings, with headset mics *and* distant mics capturing the
same speech simultaneously.

Used **only for secondary evaluation**, never training. Distant mic is the
mixture; headset is the approximate ceiling. It answers the one question
the constructed set structurally cannot: does any of this transfer to real
audio?

"Approximate" is load-bearing — the headset channel still picks up bleed
from other speakers, so it is not a clean target. Never call it ground
truth.

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

Every trial ships all five: `x`, clean target, `e`, `t`, `d`. Plus the
metadata — overlap ratio, SNR, enrollment-device condition — because those
are the experimental variables.

## What is used where

| | Train | Val | Primary eval | Secondary eval |
|---|---|---|---|---|
| LibriSpeech | yes | yes | yes | — |
| WHAM! noise | yes | yes | yes | — |
| Generated RIRs | yes | yes | yes | — |
| AMI | **never** | **never** | — | yes |

Different speakers in every column. Speaker-disjoint, enforced by
`experiments/configs/splits.yaml`.

## Things that are NOT in the data pipeline

- **The judge** (Gemini Live or similar). It never touches training, never
  filters data, never appears as a proxy. Held-out evaluator only.
- **wsj0 / wsj0-2mix.** Needs a paid LDC licence. We don't have it and
  don't need it.
- **`train-other-500`.** Lower-quality audio; we need clean targets.
- **Real conversational corpora for training.** No clean target, no
  verbatim text. See `docs/decisions.md` 2026-08-07.

## Storage, roughly

| | Size | Kept? |
|---|---|---|
| LibriSpeech | ~30 GB | yes, permanently |
| WHAM! noise, converted | ~6 GB | yes, permanently |
| Training mixtures | 0 GB | generated on the fly, never stored |
| Val + eval mixtures | ~2–5 GB | yes, frozen |
| AMI subset | ~10–80 GB | later, optional |

Do not delete LibriSpeech or WHAM! after generating. On-the-fly training
means those corpora *are* the training set, and a logged seed only
reproduces a run while they still exist.

## See also

- `docs/data-setup.md` — how to download and prepare all of it
- `docs/data-licences.md` — what you're allowed to do with it
- `docs/decisions.md` — why these choices, dated
