# Report todo — 2026-09-29

Submission **2026-11-05** (37 days). Supersedes the 2026-09-07 list (in git history).
`file:line` = `report/…`. **NB** = an examiner will catch it, or a claim is wrong.

## 1. NB — wrong or unsafe claims currently in the report

- [ ] **Judge "held out of training" is false.** `experiments/setup.tex:30`. Rule
  withdrawn 2026-09-15. Rewrite: the holdout moved from the model to the data
  (`sir0_privval`, `eval_private` never scored/selected in training); every claim
  says *optimised for Gemini*. Same fix in `metric-definitions.md` §4 (weak-point A7).
- [ ] **Headline numbers are on the split e21 was selected on.** All tables use
  `sir0_val` `both` n=103, and e21 was picked on it → optimistic. Say so in every
  caption; the only selection-free number is `eval_public` Case 3 (m4 2026-09-29).
- [ ] **`methodology/data.tex:54` misdescribes the eval set.** The 103 trials are the
  `both` slice of the 200-trial `sir0_val` (103/47/42/8), not a "final holdout with no
  target-absent trials". Name `sir0_val`, `eval_public`, `eval_private`, `sir0_privval` (dropped 09-24).
- [ ] **SIR rancges contradict.** `data.tex:46` U[−10, 10] dB vs appendix [−5, 15],
  `base` [0, 12], plus training-time gain re-draw (`dataset_loader.py`). State the real
  distribution per split. `eval_public` is easier than training (SIR +4.76 dB, 75 %
  target-louder, weak-point B1) → report it per SIR band.
- [ ] **Extension's gain credited to ECAPA.** `results/results.tex:46`. m2 2026-09-27:
  the extension is three inseparable changes (cue parts + context, content-WER selection
  and LR schedule, 21 vs 6 epochs). "Best streaming model", not "conditioning worked".
- [ ] **Judge noise figure is 3x understated.** `results.tex:241` comment says SEM ≈ 0.5;
  measured SEM 1.50, CI [0.98, 2.00] (A2). Systems judged once cannot be split below
  ±4.16 pts (A3).
- [ ] **No CIs or paired tests anywhere.** Table 5.1 has ±SD only. Add a separate 95 % CI
  column (house style) and paired bootstrap differences: e21 − baseline −16.36
  [−23.83, −9.18]. Case tables are one run each — say so.
- [ ] **Floor anchor bug (A1).** `load_once_index` is last-row-wins; plain run now gives
  floor 62.98 vs table 62.94. Fix, re-derive anchors, confirm Tables 5.1–5.3.
- [ ] **WeSep confounds must travel with every mention** (O4): different training data,
  offline (global normaliser, cannot stream), 27.2 M vs 7.19 M params, out of domain.
  `results.tex:46` calls it "best case of what BSRNN can achieve" — qualify.
- [ ] **Model-selection section describes the retired rule.** `setup.tex:43-49`
  ("10 dB down on absent branch"). Now: in-loop content WER (m2 09-23); e21 chosen on the
  judge, overruling in-loop e18 (m2 09-26). Gemini as selection criterion → record ID/prompt/date.
- [ ] **`w_m = 9.62` was calibrated on an abandoned distribution** (C6, ~65 % too high).
  `architecture.tex:261-267`. Re-derive on `sir0` or state it.
- [ ] **ICR is not an independent axis** (A6, r = 0.78 with WER). `metrics.tex:37-67`
  presents it as one. Restate as a split of WER's insertions; say FR *is* independent (r = 0.24).
- [ ] **Latency overclaim.** `results.tex:231` "comfortably within". Measured p99 margin is
  14–23 %, about the method's own error: claim "keeps up on an idle CPU". Footnote ‡ in
  Table 5.7 has no text.
- [ ] **Wording rules.** "two-speaker mixtures", never "conversation" (`litreview.tex:23`,
  `metrics.tex:16`). No REAL-TSE comparability — `results.tex:7` proposes adding CARTSE
  numbers: only with an explicit non-comparability line, or drop.
- [ ] **Stale `% CLAIMS` comments disagree with the tables** (error split 9.28→12.60 vs
  6.93/5.70; OVRL 2.237 vs 2.183; SAR/SIR 10.34 swapped). Write prose from the tables, delete comments.
- [ ] **TF-map equation omits the ×16 sharpening.** `architecture.tex` eq 3.5 writes
  softmax(S); baseline trained with `tfmap_scale: 16.0` (√F, `conditioning.py:85-97`).
  Without it the template is the clip's mean spectrum (measured 2026-08-25). State the
  departure from Zhang 2025 (they use un-normalised products) as borrowed-and-changed.

## 2. NB — missing content that must exist

- [ ] **The metric-contribution proof is not in the report.** Leak/fabrication trade and
  cheap-listener misranking (evidence below). Bootstrap first, then a results section.
- [ ] **Case 2 regression not discussed.** Every system makes target-only clips worse than
  doing nothing (1.65 → 3.56–5.40 LCF-WER). Report it.
- [ ] **e21 is all-or-nothing on Case 3.** Silences 18/42 (56/123 on `eval_public`);
  unsilenced clips pass ~the raw mixture's words. `results.tex:70` has one line with an
  unexplained n=123 — expand.
- [ ] **Extension architecture section** — methodology only covers the baseline;
  `setup.tex:14` empty.
- [ ] **WeSep described** — what it is, citation, why it cannot stream. `setup.tex:53`.
- [ ] **O4 framing** — our baseline *is* `BSRNN_TFMAP_CAUSAL`, the challenge baseline; write
  negative M5 arms (state teacher, mask structure, capacity) as a reproduction.
- [ ] **Causality probe result** — described in `setup.tex:41`, never reported. WeSep
  1.12e-2 vs ours 1.68e-8.
- [ ] **Latency component breakdown** — promised in `litreview.tex:21`, not delivered.
  Also state gap: quality scored whole-clip, latency chunked with state dropped (B2).
- [ ] **Text reference condition** — cut 2026-09-03 on latency (m4). One paragraph with the
  pass-through caveat.
- [ ] **Anchors + reporting protocol** — `metrics.tex:107-113` empty stubs.
- [ ] **Hyperparameter / config table** from the YAMLs (appendix); `setup.tex:8-18` empty.
- [ ] **Judge record** — appendix `Judge model: to be recorded`: `gemini-3.7-flash`,
  aistudio, audio-in/text-out, prompt sha `d118b7d3bf30`, run dates 09-02 → 09-29, and
  the JSON response schema (status field).
- [ ] **LLM-as-judge literature** (A8) — Zheng 2023 (arXiv:2306.05685), judge-bias papers;
  PESQetarian for metric gaming. Plus a gap statement → research questions.
- [ ] **Limitations / threats section** — two-speaker boundary, read speech, single seed
  (J5), selection on `sir0_val`, prompt sensitivity on n=4 (A5), AMI not run, the four
  points in `results.tex:234`.
- [ ] **Define the five systems once**, before results; use house labels ("No processing /
  Target alone (perfect)"), not Floor/Ceiling. Define "mean leak" and "invented/trial" in metrics.
- [ ] **Research questions + contributions** in the intro ("metric is the contribution,
  extractor is the vehicle").

## 3. Inline TODOs in the .tex

| Where | Todo |
|---|---|
| `frontmatter/title_page.tex:1,20` | degree wording, submission date (dept.) |
| `frontmatter/declaration.tex:1` | official SU declaration text |
| `frontmatter/abstract.tex` | empty — write last |
| `frontmatter/acknowledgements.tex:4` | empty |
| `frontmatter/nomenclature.tex:41` | CARTSE expansion; also add PSE, RTF, LLM, API, ECAPA-TDNN, LCF; symbols `w`, `w_m`, `τ`, `α`, `k` |
| `introduction/introduction.tex:4` | whole chapter |
| `litreview/litreview.tex:4,10,12` | chapter intro; paragraphs 2–3 of §TSE (speaker encoders, enrolment, WeSep) |
| `methodology/methodology.tex:4` | chapter intro |
| `methodology/data.tex:35` | enrolment: single 5 s clip (appendix says so) — confirm, delete TODO |
| `methodology/architecture.tex:24` | cite CARTSE for window-minus-hop |
| `methodology/architecture.tex:42` | band-plan experiment → cut to future work |
| `methodology/architecture.tex:74` | model size "not final" — capacity arm was confounded; finalise |
| `methodology/architecture.tex:103,108` | lookahead: crash fixed, ablation never run → run or cut |
| `methodology/architecture.tex:182` | objective: update to current loss |
| `methodology/architecture.tex:261` | weights paragraph (see `w_m` above) |
| `methodology/metrics.tex:77` | hallucination definition needs a citation (Koenecke 2024, Atwany 2025) |
| `experiments/experiments.tex:4` | chapter intro ("This section" → chapter) |
| `experiments/setup.tex:34` | name the VAD (Silero 6.2.1, same as data) |
| `experiments/setup.tex:53` | WeSep comparison |
| `results/results.tex:7` | capacity/data headroom; CARTSE (see wording rule) |
| `results/results.tex:234` | four caveats as prose |
| `appendices/appendices.tex:17` | judge row |
| `conclusion/conclusion.tex:4` | whole chapter |

## 4. Experiments — decide run or cut this week

- [x] **Bootstrap #17 and #18** — done 2026-09-30, `experiments/results/2026-09-30-bootstrap-holes-trade/`. Both narrowed; see evidence below.
- [ ] **Data scaling** — `data.tex:54` promises 1,989 / 4,976 / 9,955 results; add or delete.
- [ ] **Lookahead ablation** — run or remove from methodology.
- [ ] **Prompt sensitivity (A5)** — cheap; the primary contribution rests on it.
- [ ] **#8 per-band gate** — past its 28 Sep cut date → future work.
- [ ] **#10 AMI transfer** (DNSMOS only) — or state as not done.
- [ ] **#11 GPU latency** — the 4,976 re-time is obsolete (baseline is now 10000-e6).
- [ ] **Case 1a–1d** — defined in `metrics.tex:7-10`, never reported: report or drop.

## 5. Cleanup

- [ ] **#15 Front-matter admin — only item needing someone else. Start now.**
- [ ] #6 symbols: `N` means four things; `enrollment`/`enrolment` split.
- [ ] #19 figure: cue decomposition, same- vs cross-gender
  (`experiments/results/2026-09-22-cue-directional-sir0`).
- [ ] Typos in lit review (well establish, Human's, inate, seperate, adverserial pertubations); broken sentence `setup.tex:40` ("This is The second…").
- [ ] Check SU rules on list of figures/tables (commented out in `report.tex:129`).
- [ ] Final: caption pass, reference pass, every judge number carries ID/prompt/modality/date.

## 6. Sequence

1. **Sep 29–Oct 5** — section 1 fixes; run/cut decisions (section 4); bootstraps; admin emails.
2. **Oct 6–19** — section 2 content; results prose; lit review; limitations.
3. **Oct 20–26** — introduction, conclusion.
4. **Oct 27–Nov 2** — abstract, cleanup, full read-through. Nov 3–5 buffer.

---

## Evidence to write up (kept from 2026-09-07)

**#17 Leak trade — strongest metric result (narrowed by bootstrap 2026-09-30).** Mask
hysteresis, control / mild / sharp, same 103 `sir0_val` trials, same judge. Holm-corrected:
- **Total flat:** LCF-WER sharp − control WeSep −0.32 [−3.85, +2.97], ours +1.72 [−3.18, +6.78].
  Same-model arms, so the interval is ±3.5, not the ±8 between systems.
- **Leak falls — SIGNIFICANT in both:** mean leak WeSep −3.44 (p=0.012), ours −11.26 (p<0.001).
  ICR@2 only in ours (−8.74, p=0.034); WeSep's −4.85 is 5 trials, p=0.41.
- **What replaces it differs:** WeSep substitutions +2.88 (p=0.013); ours deletions +7.48 (p<0.001).
- **Fabrication rise NOT supported:** FR@2 p=0.76 / 0.16; invented/trial p=0.08 / 0.42. Drop "fabrication".
Claim: LCF-WER alone calls the arms interchangeable; the leak measure shows they are not.

**#18 Cheap listener misranks (confirmed for `sharp`, bootstrap 2026-09-30).** Cost of an
arm = arm − control LCF-WER. DiD = judge cost − `small.en` cost; negative = Whisper overstates.

| cost of `sharp` | judge | Whisper | DiD, 95 % CI, p(Holm) |
|---|---|---|---|
| ours | +1.72, n.s. | +18.48, p<0.001 | −16.76 [−31.25, −4.66], p=0.010 |
| WeSep | −0.32, n.s. | +14.93, p<0.001 | −15.25 [−27.14, −5.51], p=0.001 |

`mild`: WeSep DiD −12.43 (p=0.003); ours −6.87 (p=0.11, n.s.). `small.en` picks `control` in
96–100 % of resamples. **Retract** "judge picks `mild`" (judge arms indistinguishable, p=1.0),
"opposite sign" (judge cost is zero ±3.5) and the "10.7x" ratio (denominator is zero): say
"Whisper charges 15–18 points for a change the judge does not notice". Judge differences
are still below its re-measurement noise (A3, ±4.16).

**#16 O4.** Organisers' DNSMOS-OVRL gaming incident (Track 1 human-MOS LCC +0.003,
swapped to P.808 post hoc) is first-hand evidence for the metric contribution.
`decisions-pending.md` O4, 2026-09-21.
