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

---

## 2026-10-12 — Both arms trained; selection and test plan fixed before any score

**Trained.** Both arms ran 3 epochs on a Kaggle T4 from bundle `06ecc85`: AMI only 40 min, AMI + simulated 1.1 h. `experiments/results/2026-10-12-train-{amift,amimix}/`, checkpoints `models/model_{amift,amimix}-e{0,1,2}.pt`.

**Forgetting monitor (simulated `sir0_val`, not AMI).** Against the control fine-tune (same settings, simulated data only):
- **AMI only forgot part of the simulated task.** Present-target loss +1.5 / +3.0 / +3.2 against the control's ~−4.3, and the output only 10.8–13.0 dB quieter when the target is absent, against ~20 dB. It moved toward pass-through.
- **AMI + simulated did not.** Loss −4.2 / −4.4 / −4.2, matching the control; absent-target gap 15.9–17.0 dB, a mild move.
- Whisper probe (40 clips, noisy): AMI only 79.4 / 63.2 / 68.4, AMI + simulated 58.2 / 54.6 / 52.0, control 54.8 / 47.7 / 63.6.
- Figure: `experiments/results/2026-10-12-ami-ft-training-curves/forgetting_monitor.png`.

**Deviations from the entry above, fixed before any epoch is scored:**
1. **Selection uses the 150-window `ami_train_check` subset, not all 451.** Its anchors are already paid for, and it is a third of the calls. Rule unchanged (lowest LCF-WER per arm; tie → fewer invented words). `scripts/select_ami_ft.py`, render-only split `amitrchk`.
2. **Both arm winners are tested, not one** (Grant, 2026-10-12): each once on `ami_eval` and once on `eval_private` (n = 248). The arms were a designed comparison, so both are reported.
   - **The primary row is fixed now, not from the test:** the arm whose winner scores lower on `ami_train_check`. The other arm is secondary. No arm is promoted on `ami_eval` or `eval_private` results.
   - `eval_private` measures what each fine-tune costs the simulated task on unseen speakers, beside e21 and WeSep.
3. **Intervals:** paired against e21 and WeSep, resampling meetings (`ami_eval`) or target speakers (`eval_private`), 10,000 draws, seed 42. Rebuilt from the judge cache: `experiments/results/2026-10-12-ami-ft-final/paired_ci.py`.

Driver: `eval_ami_ft.sh` (`render` on CPU; `judge` runs selection, renders the winners and judges them).

**AI-use.** Claude-originated: the 150-window subset for selection; fixing the primary arm on the selection score; the forgetting-monitor reading.

---

## 2026-10-12 — Selection on `ami_train_check`: e21 beats every epoch, by muting

`gemini-3.7-flash`, aistudio, audio-in / text-out, prompt `d118b7d3bf30`, run 2026-10-12, one run, speech gate on. 150 training windows. `experiments/results/2026-10-12-select-ami-ft/`.

| LCF-WER, lower is better | e0 | e1 | e2 |
|---|---|---|---|
| AMI only | 96.08 | **92.27** | 95.04 |
| AMI + simulated | **92.61** | 94.45 | 96.12 |

- Reference rows: No processing 97.19, **e21 89.28**, Target alone (gated headset) 12.63.
- **Winners:** AMI only e1 (primary, the lower score) and AMI + simulated e0.
- **e21 scores best, and for a bad reason.** It is silent on 69/150 windows and misses 50.5 % of target words, yet leaks only 26 %. The fine-tunes halve the words missed (19–30 %) but leak 34–48 %.
- **On AMI, LCF-WER rewards silence.** Targets are short (~14 words), so every leaked word counts as an insertion against them, while a silent clip costs at most the target's own words. Never quote AMI LCF-WER without target words missed and silent outputs beside it.
- Later epochs mute less (silent 50 → 25) but garble more, so word error stays flat.

## 2026-10-13 — Final test: both fine-tunes are worse than e21 on new meetings

**Plain words.** Fine-tuning e21 on real AMI meetings did not help on meetings it never saw. Both recipes keep slightly more of the target, but let much more of the other speaker through, and end about 17 points worse than e21. They barely stopped muting (102–106 silent of 300, against 110). The muting fix seen on training windows was fit to those meetings, not learned. Mixing simulated data back in protected the simulated task, but changed nothing on AMI.

`gemini-3.7-flash`, aistudio, audio-in / text-out, prompt `d118b7d3bf30`, winners run 2026-10-13, one run each, speech gate on. Every timed-out call was re-asked: 0 failures. `experiments/results/2026-10-12-ami-ft-final/` (README, `paired_ci.py`).

**`ami_eval`**, n=300, unseen people and meetings from the same corpus. Lower is better.

| | No processing | e21 | AMI only e1 | AMI + sim e0 | WeSep |
|---|---|---|---|---|---|
| LCF-WER | 115.51 | 91.26 | 108.03 | 107.81 | **80.56** |
| target words missed | 3.46 | 45.25 | 33.92 | 34.91 | 27.71 |
| other speaker passed | 82.80 | 25.19 | 42.03 | 40.70 | **19.64** |
| invented / trial | 1.48 | 1.83 | 2.77 | 2.68 | 2.65 |
| silent (of 300) | 0 | 110 | 102 | 106 | 38 |

**`eval_private`**, n=248 (000372 removed for all), simulated, unseen speakers. Lower is better.

| | No processing | e21 | AMI only e1 | AMI + sim e0 | WeSep |
|---|---|---|---|---|---|
| LCF-WER | 52.05 | 41.81 | 47.17 | 45.26 | **34.08** |
| other speaker passed | 47.89 | 21.15 | 29.99 | 32.25 | **18.99** |

**Paired against e21,** LCF-WER points (meetings resampled on AMI, speakers on eval_private; 10,000 draws, seed 42):

| | ami_eval difference | ami_eval 95 % CI | eval_private difference | eval_private 95 % CI |
|---|---|---|---|---|
| AMI only e1 − e21 | +16.77 | [+10.89, +23.24] | +5.35 | [+2.29, +9.15] |
| AMI + sim e0 − e21 | +16.55 | [+9.93, +23.69] | +3.45 | [−0.59, +8.04] |
| AMI + sim e0 − AMI only e1 | −0.22 | [−4.98, +4.45] | −1.91 | [−4.51, +0.83] |

- **Both worse than e21 on `ami_eval`, beyond noise.** Both also trail WeSep by ~27 points, and WeSep never trained on AMI.
- **AMI only forgot the simulated task:** +5.35, beyond noise. **The 50/50 replay mostly prevented that:** +3.45, inside the noise.
- **The arms are indistinguishable** on both sets.
- **Reading:** the AMI fine-tune trades muting for leakage. Once it stops muting, it cannot separate two real far-field voices. Two signs point to separation ability as the limit, not missing real-audio exposure: the selection windows show the same trade, and WeSep is better on both measures.
- **Not pursued:** retraining from scratch with AMI. It would dilute the 1.91 h to ~4 % of the data, cost ~25 h of Kaggle, and it does not target the failure.
- **Claim wording:** "fine-tuning on AMI made extraction worse on unseen meetings from the same corpus". Never "does not transfer to real audio" in general. The headset reference is approximate.
- **Process:** the laptop shut down at 21:36 mid-render. The render resumed on 2026-10-13 (150 clips kept, 150 new; `runner.py` length check). `eval_ami_ft.sh` gained `render-test` / `judge-test`, and `judge` refuses to re-run selection.

**AI-use.** Claude-originated: the reading that LCF-WER rewards muting on AMI; the trade-off interpretation; the resume-safe split of the driver.
