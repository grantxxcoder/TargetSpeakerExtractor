# Meeting notes

Newest first. Decisions taken here move to `docs/decisions/decisions.md`; anything left
open stays in `docs/decisions/decisions-pending.md`. Keep the item IDs so they can be
traced both ways.

---

## 2026-08-12 — Supervisor

**Present:**
**Phase:** 2, data preparation. Manifests built, renderer not started.

### Decided

| Item | Question | Supervisor's answer | Status |
|---|---|---|---|
| **A4** | Should the enrollment clip carry the room? | **No — clean voice samples, so the model knows what clean sounds like** | Done. `decisions.md` 2026-08-12. No code change: renderer unwritten, enrollment is already dry |
| **B3** | Fixed enrollment length? | **Yes, fixed 5 s, and it must stay a changeable parameter** | Done. `decisions.md` 2026-08-12. No code change: already `enrollment_length_s: 5.0` in config |
| **B4** | Silent-target trials in eval too? | **Yes — we are measuring intelligibility, so eval must include them** | **Decided, not implemented.** Needs a config change and an eval rebuild — see below |
| **A3** | Levels measured how? | **RMS** — reversed same day to **BS.1770 integrated loudness** | Done. `decisions.md` 2026-08-12. No code change: `pyloudnorm` 0.2.0 already in `tse_venv`, column names and ranges already correct |

### B4 — what is still outstanding

The answer settles *whether* absent trials appear in eval. Two things it does not
settle, both of which block implementation:

1. `eval_public` and `eval_private` are `target_absent_fraction: 0.0` in
   `experiments/configs/generator.yaml`. Changing them forces an eval rebuild.
   Fold into the B9/B10 rebuild rather than doing it twice.
2. **`metric-definitions.md` has no scoring rule for a trial with no reference
   text.** The main score cannot be computed on these. The standing proposal is a
   separate reported row measuring only how often the system invents speech that
   was not there, never folded into the headline number. Not yet confirmed.

→ **Fraction to use in eval:**

→ **How they are scored:**

### A3 — RMS was reversed to BS.1770 the same day

RMS was the initial answer. Reversed before anything was implemented, because it
was the *more* expensive option, not the cheaper one:

- **It forces a gating rule.** Measured at 16 kHz on identical speech, once
  continuous and once padded to 20 s: BS.1770 gives −23.35 vs −23.77 LUFS, plain
  RMS gives −26.01 vs −33.55 dBFS. **0.4 dB apart against 7.5 dB.** Under RMS
  `sir_db` means something different in every trial and level correlates with
  activity — a new entry for the §7 leak scoreboard. Worse once B9 varies
  `target_activity_ratio`.
- **It would have made `target_loudness_lufs` a misnomer**, forcing a rename
  across config, generator, six CSVs and the notebook, plus recalibration of the
  `[-33, -25]` range. BS.1770 keeps all of it valid.
- **`pyloudnorm` was already installed** (0.2.0 in `tse_venv`) — the "not
  currently installed" note in the pending doc was stale.

The one point for RMS, that gated loudness returns `-inf` on a silent stem, does
not apply: the 2026-08-11 anchor decision means a silent target is never measured.

Renderer constraints, both verified: a stem under 400 ms raises `ValueError`
(BS.1770 block size), so the generator needs a minimum target-speech guard; a
fully silent stem returns `-inf`, reachable only via a bug, so assert on it.

### Raised but not yet answered

Carried from `decisions-pending.md`. The first three are new this week.

- **B9** — silent-target trials are detectable without listening. Overlap and
  total speech both separate the classes perfectly (AUC 1.000, holds under a
  VAD-style check). Proposed: add `target_only_fraction`, and let
  `target_activity_ratio` vary. **NB**
- **B10** — the different-book enrollment guard split the speaker pool: 60.2 % of
  LibriSpeech speakers recorded one book, so they can never be a present target.
  Identity now predicts absence at AUC 0.795. Fix partly reverses B8. **NB**
- **B11** — 66.8 % of trials have reverb longer than the streaming window. My
  leaning: change the metric, never the data — score at 100/200/300/400/500 ms
  and report the decay.
- **A1** — reference signal. Still the biggest blocker on the renderer.
- Floor WER calibration target: is 60–80 % right?

→ **Notes:**

### Actions

| # | Action | Owner | By |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

### Follow-on already known

- If B9 and B10 are approved, one full manifest rebuild covers them plus B4.
- Still unimplemented: `noise_speech_rejection` (own docs call it critical —
  speech in the noise bed enters as an unlabelled third talker and gets scored as
  hallucination) and `length_mode`.
- Notebook: §2, §7 and the final health checks assume the current timing model
  and need revising after any rebuild; §3–§6 do not. §7.5 (leak plots) deferred
  to 2026-08-13 and should be written **before** the rebuild, so the leak
  scoreboard becomes a before/after rather than a claim.
