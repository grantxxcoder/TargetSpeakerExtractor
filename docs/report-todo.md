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
