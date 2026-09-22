# Report todo — 2026-09-07

Submission **2026-11-05**. No experiment freeze; the binding constraint is the
date of the last judge pass, because everything downstream of the judge must be
written after it.

## Experiments

- **#8 M5 per-band gate** — build, train, evaluate, judge. D13, already
  specified. The close-systems test the metric needs. **Cut if not training by
  ~28 Sep.**
- **#9 Judge spread and confidence intervals** — free re-analysis of cached k=3
  repeats. Currently blocks any claim that the two models differ.
- **#10 AMI real-audio transfer check** — DNSMOS only; ceiling approximate from IHM.
- **#11 GPU latency**, and re-time the 4,976 checkpoint at 2250 chunks so the
  RTF rows are comparable.

## Writing

- **#2 Experiments setup** — text reference condition; statistical protocol;
  training setup; the split / n=103 sentence.
- **#3 Results** — all analysis prose; stale `% CLAIMS` comments; per-system
  descriptions; data-scaling section; caveats.
- **#4 Data construction** — overlap and the 250 ms definition;
  reference-signal decision; the flagged fixes.
- **#5 Literature review** — stub.
- **#12 Introduction** — stub. Settle the "metric is the contribution,
  extractor is the vehicle" framing here.
- **#13 Conclusion** — stub.
- **#18 THE CHEAP LISTENER MISRANKS. This is the metric contribution's proof.**
  Same audio, same 103 trials, three mask-hysteresis settings, two systems.
  `faster-whisper small.en` and `gemini-3.7-flash` **disagree on which setting
  is best, in BOTH systems** — Whisper picks `control`, the judge picks `mild`.

  | cost of `sharp` | judge | Whisper | error |
  |---|---|---|---|
  | ours | +1.72 | +18.48 | **10.7x overstated** |
  | WeSep | **-0.32** | +14.93 | **opposite sign** |

  On WeSep the proxy calls sharpening a 15-point disaster while the live model
  finds it marginally beneficial. Anyone tuning post-processing on a local ASR
  ships the wrong configuration. **A free stand-in cannot substitute for the
  live model** — asserted since M0, now demonstrated on two independent systems.

  **The same table argues AGAINST tuning to the judge (G1).** Its spread across
  all three arms is 2.44 points (ours) and 0.64 (WeSep) against a +-8 interval:
  the judge cannot tell the settings apart. It is ROBUST to the artefacts that
  wreck the ASR, and robustness means less exploitable structure, not more.
  Measure with it; do not train against it on this axis.

  Numbers: `TSE-listener-panel` worktree,
  `experiments/results/2026-09-21-holes-{,wesep-}{asr,gemini-3.7-flash}-*`.
  NOT BOOTSTRAPPED. The judge differences are smaller than its own filter noise
  (1 permanent block + 4 transient refusals across these arms), which is J5's
  unmeasured third noise source. Do not quote "mild wins" without it.
- **#17 THE LEAK/FABRICATION TRADE IS THE STRONGEST METRIC RESULT WE HAVE.**
  One knob (mask hysteresis), three settings, same estimates, same 103 trials,
  same judge, same day. Word error is FLAT — 26.92 / 26.28 / 26.60, a 0.64-point
  range against a +-8 interval — while leakage improves monotonically
  (ICR@2 23.30 -> 18.45, mean leak 12.87 -> 9.43) and fabrication worsens
  monotonically (FR@2 40.20 -> 44.66, invented/trial 1.80 -> 2.28).
  **The composition of the error changes completely underneath a headline that
  does not move:** insertions fall 8.73 -> 6.52 while substitutions rise
  10.56 -> 13.45, and they very nearly cancel. Removing the interferer and
  damaging the target's words in equal measure.
  **A system chosen on LCF-WER alone would call these three interchangeable.**
  That is the case for J4's metric system and for B13's "a headline aggregate
  must never appear alone", demonstrated rather than asserted — and on WeSep,
  so it does not depend on our own extractor being weak.
  Replicates the same trade already logged on our baseline family
  (decisions-m4.md, FR@2 34.31 -> 51.46). Two system families, one pattern.
  Numbers: `TSE-listener-panel` worktree,
  `experiments/results/2026-09-21-holes-wesep-gemini-3.7-flash-*`.
  NOT YET BOOTSTRAPPED — monotonicity over three settings is not a
  significance test. Run bootstrap_difference.py before quoting.
- **#16 O4 — WE ARE THE REAL-TSE BASELINE, and this reframes the negatives.**
  Our model IS `BSRNN_TFMAP_CAUSAL`, the challenge's own causal baseline. So the
  three negative M5 results (state teacher, mask structure, capacity) are not
  project-specific failures — they **independently reproduce the REAL-TSE
  organisers' published consensus** that gains came from data simulation, real-
  data adaptation, pseudo-labels and loss design, NOT from architecture. Write
  them as a reproduction, not as things that did not work.
  Carry with it: (a) the organisers' mid-challenge DNSMOS-OVRL gaming incident,
  which is first-hand evidence FOR this project's metric contribution — human-MOS
  correlation for OVRL on Track 1 was LCC +0.003, and they swapped to P.808 post
  hoc; (b) the four confounds in the WeSep comparison, which must travel with
  every mention of it — different data, offline (its normaliser is global, RTF
  2.854, cannot stream), 27.2 M against our 7.19 M, and out of domain by its own
  config. Full statement: `decisions-pending.md` O4, 2026-09-21.
- **#19 FIGURE: visualise the cue decomposition.** One mixture frame, worked
  through end to end, for the methodology chapter. Show: the enrolment's frames
  as a dictionary, the softmax weights picking which ones blend, the resulting
  template, and then the frame split into the part that matches it and the part
  that does not. Needs a same-gender trial (where the cue fails) beside a
  cross-gender one (where it works) — the contrast IS the argument for 1c.
  Data is already on disk; `experiments/results/2026-09-22-cue-directional-sir0`
  has the per-trial scores to pick good examples from.
- **#14 Abstract** — stub. Write last.

## Cleanup

- **#6 Symbol consistency sweep** — `N` means four things;
  `enrollment`/`enrolment` split 28–3.
- **#7 TODO markers** — 21 across the report.
- **#15 Front-matter admin** — **start first**, only item needing someone else:
  degree wording, declaration text, submission date. Plus the CARTSE expansion.

## Sequence

1. **Sep 7–21** — gate build and train; literature review and data chapter in
   parallel; kick off the front-matter admin.
2. **Sep 21–Oct 5** — evaluate the gate; judge spread; results prose.
3. **Oct 5–19** — introduction and conclusion; AMI leg if the gate went cleanly.
4. **Oct 19–Nov 2** — abstract, symbol sweep, TODO clear, read-through, caption
   and reference pass.
