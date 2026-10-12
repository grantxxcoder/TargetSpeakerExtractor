# Decision Log — M8 (real-audio fine-tune on AMI)

Covers fine-tuning e21 on real AMI meeting audio: training split, headset-mic training target, two training arms, epoch selection and the single final test on `ami_eval`. All M8 decisions go here. See `milestones.md` M8.

Created 2026-10-12, after the report's first draft. The 2026-10-14 experiment freeze was lifted the same day.

**Standing constraint.** `ami_eval`, `sir0_privval` and `eval_private` are never trained, filtered or selected on. Every Gemini call, training-time or test, records model ID, prompt sha, modality and date, as a judge call does (`decisions-m4.md` 2026-09-15).

---

## 2026-10-12 — Fine-tune e21 on real AMI recordings

**Why.** On real meetings e21 mutes the person it should keep. It misses 45.3 % of the target's words (doing nothing misses 3.5 %), and 110/300 outputs sit ~26 dB below their input (`decisions-m4.md` 2026-10-06). AMI table-mic speech is ~−48.6 dB against −29.2 dB in training, so the model has never heard input this quiet. Training on real recordings at their real level is the direct fix.

**Splits: two, by AMI scenario group.**
- `ami_train`: every usable window in the 20 scenario groups `ami_eval` does not use (ES2004/06/08/09/10/11/12/13/14/16, IS1002/05/06/07/08, TS3005–09), 78 meetings, same window rules as `ami_eval`.
- `ami_eval`: unchanged, 300 trials, the only held-out number.
- No person is in both: 0 of 140 scenario speakers recur across groups (counted from `meetings.xml`). Enrolment comes from a sibling meeting in the same group, so it never crosses the split.
- EN/IB/IN meetings left out for simplicity. 6 `ami_eval` speakers recur in IN meetings, so IN would need a speaker filter.
- No separate val split: with only 467 windows, all of it trains.

**Training target: the gated headset.** AMI has no clean target, and the raw headset is a bad one:
- **Bleed.** The other speaker sits ~21 dB down. Gemini catches 52 % of their words in the raw headset, double e21's 25 % leakage. A perfect copy of the raw headset scores 72.5 % LCF-WER.
- **Timing.** The headset hears the target a few ms before the table mic. With zero lookahead the model cannot follow, and SI-SDR punishes the shift.
- **Level.** The headset is far louder, so the volume-matching term (`L_gain`) would teach a boost.

So each headset is processed (`src/data/ami_targets.py`):
1. **Aligned** to the table mic by GCC-PHAT (Knapp & Carter, 1976).
2. **Rescaled** by a least-squares fit to the table mic.
3. **Gated**: zeroed outside the speaker's own word and vocal-sound times, padded ±0.2 s, with 10 ms fades.

Gating removes bleed where only the other person talks; bleed under the target's own voice remains, ~21 dB down. **The target is still approximate, never ground truth.** Rejected: running WeSep over the headset, as the model might learn WeSep's consistent artefacts.

**Arms: from e21, 3 epochs each, every epoch kept.**
- A, AMI only: `ami_train` rows ×4.
- B, 50/50: the same AMI rows ×4 plus an equal seeded sample of `sir0_train` rows, as replay against forgetting.
- One epoch sees every AMI row 4 times, each with a different random 4 s crop, so ×4 covers each 10–20 s window about once.
- Settings are those of the control fine-tune `bsrnn_interf_ft_control.yaml` (lr 1.25e-4, `w_interf` 1.0, fresh optimiser). Only the data differs.
- Remix augmentation and its noise identity assume mixture = sum of stems, which is false across microphones, so remix is off for AMI rows (`remixable = 0`).
- The in-loop val pass and Whisper probe stay on `sir0_val` as a forgetting monitor.

**Epoch selection: Gemini on the training windows, unprocessed.**
- Each of the 6 checkpoints renders every `ami_train` window: raw table mic, whole window, the trial's designated target. Gemini scores it against the human transcript, as the final test will hear it.
- Lowest LCF-WER wins; ties go to fewer invented words per trial. e21 is scored as the reference row, not a candidate.
- **Caveat for every write-up:** this scores clips the model trained on, so it measures fit, not generalisation. It cannot detect overfitting and will likely favour the last epoch. Selection never touches `ami_eval`, so the final number stays unbiased.

**Final test.** The winner, once, on `ami_eval`, beside e21 and WeSep, with a meeting-clustered paired interval against e21. Optionally add the control checkpoint at the matching epoch, to rule out "it just trained longer". On `sir0_val` six control epochs changed nothing (−0.99, CI [−5.08, +2.89]).

**Claim wording.** "Unseen people and meetings from the same corpus and rooms it was tuned on." Never "transfers to real audio": `ami_eval` is now in-domain.

**Supersedes** `decisions-m0.md` 2026-08-07 "AMI eval only" for the 20 groups above. That entry's reason stands: there is no clean target.

**AI-use (Stellenbosch declaration).** Claude-originated: the gating, alignment and rescale of the headset; ×4 repeat epochs; the 50/50 replay design; the `sir0_val` forgetting monitor; reusing the control fine-tune as a free control.

---

## 2026-10-12 — `ami_train` built: 451 trials, 1.91 h

**Built.** 451 two-speaker windows from the 20 groups: 1.91 h, 78 speakers, 74 meetings (Edinburgh 259, TNO 110, IDIAP 82; 352 same-sex). 15 windows dropped because a speaker had no solo stretch in a sibling meeting to enrol from. Nothing shared with `ami_eval`: 0 people, meetings (enrolment meetings included) or groups (`tests/test_ami_splits.py`).

**Differs from the plan.**
- A separate builder, `scripts/build_ami_train.py`. `build_ami_trials.py` (ami_eval) is untouched apart from a guard refusing any non-ami_eval config, so `ami_eval` cannot change.
- Audio fetched by HTTP byte range (`src/data/ami_fetch.py`): 0.83 GB instead of ~22 GB of whole meetings. Every piece is logged with URL, byte range and sha256 (`data/raw/ami/segments/index.csv`); a fetched piece was checked sample-identical to the whole local file.

**Found while building.**
- IS1007d has no table mic on the server (404): no trials from it; its headsets are enrolment sources only (like IS1003b for `ami_eval`).
- ES2010d is stored stereo with bit-identical channels: read as mono, left = right checked on every piece.
- The mirror answers 403 under ~12+ parallel requests: 8 workers, with backoff.
- One IS1002c window's raw table mic has 7 full-scale samples (a clipped transient): kept as recorded, counted in `mixture_full_scale_samples`.

**Delay rule, added after measuring.** Per-window delay estimates with a GCC-PHAT peak ≥ 0.2 agree within one speaker in one meeting to 0.5 ms (median range), and 3/480 are implausible. Below 0.1, 55/134 are implausible (negative, or > 8 ms). People mostly stay seated, so a window uses its own delay only if its peak is ≥ 0.2; otherwise it takes that speaker's median confident delay in the meeting, or the median of all their windows if none is confident.
- Applied: 480 own, 307 speaker-meeting (confident), 115 speaker-meeting (all).
- Implausible applied delays fell from 82 to 20 of 902. 17 of those cluster at ~10 ms and include confident estimates (a speaker ~3.5 m away), so they are kept. 3 negative remain (0.3 %), accepted.
- Typical delay 2.4 ms (IQR 2.1–2.9), about 85 cm of extra path to the table mic.

**What the targets look like.** At the table mic each speaker is a median 20 dB quieter than in their own headset (5–95 %: 10 to 43 dB). The gate silences a median 59 % of each target. The two speakers are equally loud on average at the table mic (median 0.2 dB apart; 5–95 %: ±19 dB). Median overlap per window is 5.5 %.

**Manifests** (`scripts/make_ami_manifests.py`): `amift_train` 1,804 rows (451 × 4). `amimix_train` 3,608 rows: the same, plus 1,804 `sir0_train` rows stratified by condition (both 893, target-only 461, interferer-only 353, noise-only 97).

**AI-use.** Byte-range fetching and the delay confidence rule are Claude-originated.

**Still to check before upload:** listen to 5 trials, and the Gemini anchor pass on `ami_train` (`milestones.md` M8).

---

## 2026-10-12 — Gemini check of the gated-headset targets: they pass

`gemini-3.7-flash`, audio-in / text-out, aistudio, prompt `d118b7d3bf30`, run 2026-10-12, one run, speech gate on. 150 `ami_train` trials, a seeded site-proportional subset (`data/manifests/ami_train_check.csv`; Edinburgh 86, TNO 37, IDIAP 27). Anchors only, 290 calls, 48 min. `experiments/results/2026-10-12-evaluate-ami_train_check/`.

| | No processing | Target alone (gated headset) |
|---|---|---|
| LCF-WER | 97.2 % | **12.6 %** |
| target words missed | 2.6 % | 3.8 % |
| other speaker passed (mean leak) | 82.0 % | **1.1 %** |
| invented / trial | 1.52 | 0.59 |

- **Gating works.** On the same trials Gemini hears 82 % of the other speaker in the raw table mic and 1.1 % in the gated target (the raw headset on `ami_eval`, different trials: 52 %).
- **It costs the target little:** 3.8 % of target words missed.
- **12.6 % is the ceiling on these clips**, not ~1 % as on `sir0_val`: AMI transcripts are not verbatim and the speech is conversational. Read every epoch result against it.
- Scoring training data is allowed (`decisions-m4.md` 2026-09-15); no trial here is in a holdout. The answers are cached and reused at epoch selection.
