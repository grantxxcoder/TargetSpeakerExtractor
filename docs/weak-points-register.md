# Weak-point register

**Read this first when reopening the audit.** Every row is a defect or gap found
against the project, with the evidence that established it and the paper or fix
that answers it. Status is the only field that changes.

Opened 2026-09-22. Audit method: code read + tests run against data already on
disk. Every paper below was verified by fetching its abstract page; none is from
memory. `[m2]`/`[m3]`/`[m4]` = the decision log holding the detail.

Status: **OPEN** · **FIXED** · **MEASURED** (quantified, not yet acted on) ·
**CLOSED** (superseded or answered elsewhere).

---

## A. Measuring instrument

| # | Weak point | Evidence | Answer | Status |
|---|---|---|---|---|
| A1 | **Floor anchor is not reproducible.** `load_once_index` is last-row-wins over an append-only CSV with no `repeat` filter, so `judge_spread.py`'s extra calls overwrite the canonical anchor. | Published 62.456345 == last-wins exactly; first-wins 63.242; a third value 63.271 published 9 h earlier. 0.81 pts = 23 % of the 3.49-pt claim. `judge.py:164-182` | One line: skip `repeat != 0` in `load_once_index`. Then re-publish every anchored number. | **OPEN** |
| A2 | **Judge noise understated 3x.** `project-state.md:224` claims SEM 0.50, extrapolated from 3 clips. | 31 clips x 5 identical calls: SEM **1.501**, 95 % CI **[0.98, 2.00]** — excludes 0.50. Pooled within-clip SD 15.2 pts. `2026-09-22-judge-retest` | Update `project-state.md`. Protocol: Blackwell arXiv:2410.03492; Tamba arXiv:2606.26185 (temp 0 != determinism); Haldar EMNLP 2025 arXiv:2510.27106 | **MEASURED** |
| A3 | **Two systems judged once each cannot be separated below +-4.16 pts** by judge noise alone, before trial sampling. | Follows from A2. The 2026-09-21 headline claim is 3.49. | Report as not distinguishable, or raise k and n. | **OPEN** |
| A4 | **k=1 everywhere.** Spec 5 says k>=3 is "not optional". All 1,957 estimate judgments are `repeat=0`. `judge_gate.yaml:64` declares `repeats: 3` and **no code reads that file**. | `judge.py:267`; grep `repeat` in eval scripts returns nothing | Wire `repeats` through; k=5 on ~100 trials. Cheapest at the shortlist stage, not across all epochs. | **OPEN** |
| A5 | **Prompt sensitivity closed on n=4 trials, verdict resting on one clip.** Spec 7 makes prompt dominance a *redesign trigger*. | `project-state.md:331` "answered, by accident"; cache holds 8+8 rows across 2 alt prompts | Run it properly. Cheap to close, expensive to leave open — it is the primary contribution. | **OPEN** |
| A6 | **ICR is not an independent axis.** Leaked interferer words are counted as WER insertions, so the scores share events. | My test, 2,197 judged clips: r(WER, leak) = **0.777**; r(insertions, leaked) = 0.630. **FR *is* independent: r = 0.24 / 0.05** | Keep FR (validated). Restate ICR as a decomposition of WER's insertions, not a second axis. | **OPEN** |
| A7 | **Metric spec asserts a withdrawn rule.** 4 still lists "the judge is held out from training" as one of four gaming-resistance properties. | `metric-definitions.md:374-376` vs `CLAUDE.md:19`, withdrawn 2026-09-15 | Rewrite 4 around the data holdout that replaced it. An examiner will find this. | **OPEN** |
| A8 | **Zero LLM-as-a-judge citations** under a thesis whose novelty is an LLM judge. | Repo-wide grep: no MT-Bench, no judge-bias literature | Zheng et al. NeurIPS 2023 arXiv:2306.05685 (position/verbosity/self-enhancement bias); Norman et al. arXiv:2606.19544 (minimum viable validation protocol) | **OPEN** |

## B. Data and protocol

| # | Weak point | Evidence | Answer | Status |
|---|---|---|---|---|
| B1 | **The final benchmark is easier than training.** Loudness shortcut removed from training, left in the benchmark. | `eval_private` SIR **+4.76 dB, 75.1 % target-louder**; `sir0_train` **+0.08 dB, 50.4 %**. Measured from manifests | Report per SIR band — `eval_by_case.py` already does it. Bias direction is **optimistic**. | **OPEN** |
| B2 | **Quality is whole-clip, latency is 80 ms chunked with state discarded.** No configuration has both numbers. | `runner.py:22-25`; `decisions-m3.md:656-661` | State the gap, or build the stateful path. | **OPEN** |
| B3 | Enrolment channel-match confound | **Checked, clean.** Same-chapter enrolment is +4.44 pts *worse*, CI [-0.56, +9.49] | — | **CLOSED** |
| B4 | Speaker disjointness / enrolment reuse / causality | **Checked, clean.** 0 speaker overlap (asserted mechanically); 0/200 enrolment reuse; first changed output sample 27.7 ms inside the declared 40 ms budget | — | **CLOSED** |

## C. Training and selection

| # | Weak point | Evidence | Answer | Status |
|---|---|---|---|---|
| C1 | **The selection rule did not track content.** Kept 1a e9 (worse than doing nothing) and rejected e15 (best in project). No reweighting reaches e15; it ranks **10th of 14**. | `reselect_epochs.py` over all 13 runs; e9 ASR 65.63 vs floor 65.22, e15 52.77 | WER in the loop: `content_probe.py`, `select_on: content_wer`. [m2] 2026-09-23 | **FIXED** |
| C2 | **`keep_top_k` deletes the winning epoch.** 1a e15 survived only because `_last.pt` is unconditional. | Retention kept e9, e2, e4 | `keep_stride` — a stride is the only cut that spans the run. | **FIXED** |
| C3 | **Proxy decoupling.** Third and sharpest instance: 12.9-pt ASR inversion between two epochs of one run. | 2026-09-04 (+12 % separation, WER worse); struct-e12 (-39 % selection score, judge flat); 1a e9-vs-e15 | Now one finding in [m2] rather than three arm post-mortems. | **FIXED** (documented) |
| C4 | **Picking the right speaker does not become words.** 1c e12 beats 1a on same-gender selection (+8/-0, p=0.008) yet content is indistinguishable (+1.85, CI [-2.44, +6.26]). 1c also leaks *more* (23.35 vs 19.41). | `2026-09-23-eval-cuecontext-e12-asr`; paired bootstrap B=20,000 | **The open research question.** Both are last epochs selected on `present_branch`; the 1c rerun under content selection is the test. | **OPEN** |
| C5 | **Same-gender is the failure mode.** Cross-gender 84.6 %, same-gender 67.1 %. 1a *widened* the gap; 1c is the first to clear the coin flip (51/76 vs a bar of 48). | Paired McNemar over 154 decisions | Onset-prompted conditioning arXiv:2505.05114; contrastive +/- enrolment (Xu et al., NeurIPS 2025) arXiv:2502.16611; speaker-consistency loss arXiv:2507.09510 | **OPEN** |
| C6 | **`w_m = 9.62` calibrated on an abandoned distribution**, documented as ~65 % too high, still shipping. | `decisions-m2.md:882`; no `wm-anchor-sir0` exists. 1c e12 reconstructs *worse than the raw mixture* while recovering 10.6 more points of content | Re-derive on `sir0`. `L_MR` is not measuring what it is being asked to. | **OPEN** |
| C7 | 16 epochs exceeds the 12 h Kaggle cap (12.16 h); the 16 was calibrated on a run 9 % faster than 1c | 1c measures 2,575 s/epoch | `EPOCHS = 14`; resume beyond, probe EMA now survives via `content_wer_history` | **FIXED** |
| C8 | Bundle shipped without the probe's modules; `jiwer`/`whisper-normalizer` absent from the Kaggle image and imported *lazily*; preflight blind to both; `--split` means different things in `make_estimates` vs `evaluate` | Found by building and by a live failure | All four fixed in `make_kaggle_bundle.py`, `make_kaggle_notebook.py`, `preflight_kaggle.py`, `select_by_wer.py` | **FIXED** |

---

## Papers, verified

Fetched and confirmed 2026-09-22/23. Preprints marked `[pre]`.

**Judge reliability** — Zheng et al., NeurIPS 2023, arXiv:2306.05685 · Haldar & Hockenmaier, EMNLP 2025, arXiv:2510.27106 · Blackwell et al., arXiv:2410.03492 · Tamba, arXiv:2606.26185 `[pre]` · Norman et al., arXiv:2606.19544 `[pre]`

**Speech-LLM evaluation** — **AudioJudge**, Manakul et al., arXiv:2507.12705 (*closest prior art to LCF; audio-in LLM as scorer, reports verbosity/position bias*) · AIR-Bench, ACL 2024, arXiv:2402.07729 (*position-swap de-biasing trick*) · Foo et al., arXiv:2604.24401 `[pre]` (*models keep 60-72 % of score with no audio — a ready-made gaming ablation*)

**The project's own premise, independently published** — **Fela & Mowlaee, EMNLP 2026, arXiv:2608.30348.** MetricGAN+ doubles LLM output divergence (0.318 vs 0.135) while *improving* PESQ. Cite and differentiate.

**Hallucination / fabrication** — Atwany et al., Findings of ACL 2025, arXiv:2502.12414 · Koenecke et al., FAccT 2024, arXiv:2402.08021 · Calm-Whisper, Interspeech 2025, arXiv:2505.12969

**Significance testing** — Liu & Peng, Interspeech 2020, arXiv:1912.09508 (*blockwise bootstrap, block by speaker — plain bootstrap understates on LibriSpeech-derived trials*) · Gillick & Cox, ICASSP 1989 (MAPSSWE) · Bisani & Ney, ICASSP 2004 · Dror et al., ACL 2018

**Metric gaming** — **PESQetarian**, de Oliveira et al., Interspeech 2024, arXiv:2406.03460 (*stronger cautionary tale than the DNSMOS anecdote*) · Close et al., EUSIPCO 2024, arXiv:2403.11732 · Huang & Toda, SLT 2026, arXiv:2606.31105

**Conditioning** — Xu et al., NeurIPS 2025, arXiv:2502.16611 · LExt, TASLP, arXiv:2505.05114 · Wu et al., arXiv:2507.09510 · Ma et al., arXiv:2609.20463 `[pre]` · EvoTSE, arXiv:2604.06810 `[pre]`

**Objectives** — DFA-PO, arXiv:2607.10191 `[pre]` (*closest to the thesis; WavLM-anchored DPO, documents reward hacking*) · Monir et al., Interspeech 2026, arXiv:2606.21635 (*consonant-weighted STFT loss*) · Sato et al., Frontiers in Signal Processing 2025 (*SSL-space loss*) · Berdoz et al., Interspeech 2026, arXiv:2606.21458

**Artefact control** — Li et al., Interspeech 2026, arXiv:2602.20967 (*training-free observation addition — note D11 already ran the global sweep; this supplies the adaptive rule*) · Huo et al., arXiv:2607.11157 `[pre]` (*polar projection; phase correction gives no recognition benefit — a useful negative*) · Qu et al., MLSP 2026, arXiv:2608.07781

**Streaming** — LaCo-SENet, Interspeech 2026, arXiv:2606.19688 (*latency as a tunable dial → a latency-vs-word-recovery curve*) · FastEnhancer, arXiv:2509.21867

**Three independent 2026 papers converge on one warning**: optimising a single audio-quality proxy reliably reward-hacks while the headline metric improves (arXiv:2607.10191, arXiv:2606.21458, arXiv:2608.30348). Matches the 2026-08-25 collapse-to-silence run.

---

## Reopening this

Highest value first, all cheap: **A1** (one line, invalidates anchored numbers until done) · **A2** (edit one figure) · **A7** (rewrite one section) · **A5** (the primary contribution rests on it). Then **C4/C5**, which are the research question rather than hygiene.
