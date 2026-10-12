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
