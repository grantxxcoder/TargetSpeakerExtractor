# Supervisor feedback, 2026-10-30

Tick as done. Items marked (H) are also on `docs/report-handoff-2026-10-27.md`.

## Supervisor
- [ ] S1. Results are on the validation set, not the test set: motivate it, or switch.

## Overclaims and wording
- [x] W1. §5 intro states the masking trade-off as fact; §5.2 calls it a hypothesis. Hypothesis in both.
- [P] W2. §5.1 "established that LAMs are robust against noise": only white noise was tested (H). Pinned: affects the author's analysis. Same claim at litreview:32 and intro.
- [x] W3. §5.2 "each fine-tune improves its own score": the artefact weight's invented CI [−0.65, +0.25] crosses zero.
- [x] W4. §5.2 "hallucinated" → "invented". Also metrics.tex:40.
- [x] W5. §5.3 same-gender sentence describes different genders. Next sentence still credits ECAPA; caption checkpoint note not added.
- [x] W6. §5.1 "as pleasing to a human listener as WeSep": a predicted score.
- [x] W7. Rewrite the Zheng et al. "80 %" sentence (lit review).
- [x] W8. Rewrite the "filler words" and "grammatical issues" lines (ICR section).
- [x] W9. Rewrite the Experiments chapter intro paragraph.
- [x] W10. Rewrite "subjective to the actual perceptual system" (§5.1).

## Polish
Key: [x] done, [~] part done, [P] pinned, [ ] not started.

- [x] P1. Mixed voice. "our" (results.tex:194) and "I believe / I had / I would suggest" (conclusion.tex:12) now impersonal. AI declaration and nomenclature AI note left in first person on purpose.
- [~] P2. Mixed decimals. Done: abstract and conclusion to 2 dp (53.26, 39.59, 30.74, 46.88); 15.4 → 15.38 (results.tex:72). Left: one-decimal measured values need their source numbers (data shares 49.5/25.5/19.6/5.3; 5.8 %; −48.6/−29.2 dB; 71.8/52.6; 348.3 ms; 27.2 M; Appendix B AMI stats) and the w_m text −5.91/0.184 vs equation 5.9088/0.1842.
- [P] P3. p: the 0.3 exponent and the 0.297 silent fraction. Proposal: silent fraction → f_abs.
- [P] P4. τ used three ways. Proposal: τ = temperature only; loss floors → ε_pres, ε_abs.
- [P] P5. ℓ: the loss and the leaked word count. Proposal: loss ℓ → L; also fix line 126 ("(ŝ−s)²", "Gaussian").
- [x] P6. Interferer signal d → s_int (metrics.tex:60, nomenclature).
- [P] P7. Fig 5.1 caption τ pointer. Waits on P4. Proposal: add the 10⁻³ floor to the SDR/SIR/SAR equations + one sentence.
- [x] P8. "Section D.1" → "Appendix~\ref" (metrics.tex:7, setup.tex:45).
- [x] P9. Fig 5.2 caption: (\ref{ai:diagrams}).
- [~] P10. Nomenclature. Done: GLU, IHM removed; MLP kept (used in baseline figure); API, CPU, ECAPA-TDNN, ELBO, KL, LLM, RMS, RNN, TDNN added. Pinned: DNSMOS personalised-model sentence.
- [x] P11. Second "As a final test…" deleted (data.tex:40).
- [x] P12. Heading → "Trial cases" (data.tex:33).
- [x] P13. AI declaration: own page, ToC entry, ``…'' quotes.
- [x] P14. CARTSE URL added (PDF checked). SA-Mamba already had one.
- [x] P15. Pages added: [28] 1–5, [29] 3830–3834 (+ DOI). Crossref checked.
- [x] P16. VoxCeleb1 + VoxCeleb2 cited and added to bib (Crossref checked).
- [x] P17. [5] and [9] → @misc arXiv entries.
- [x] P18. All booktitles "Proc."; Interspeech entries carry the year.
- [P] P19. Conclusion RQ1–RQ3 answers. Draft below.
- [P] P20. Six limitations. Draft below.

## Follow-ups found while fixing
- [ ] Interspeech references print the year twice ("Proc. Interspeech 2017, 2017"). Proposal: drop the year from the booktitle.
- [ ] RIR and SLT in the nomenclature are unused.
- [ ] Appendix D prompt overflows the margin by 39 pt (H).
- [ ] Build is 52 pages; handoff gives a 50-page limit.

## Pinned drafts (Claude Code, 2026-10-30; not in the report)

**P19 Key findings**
- RQ1: SDR and SIR rank the five systems as Gemini does; DNSMOS does not (P.808 extension 3.03 vs WeSep 3.01, yet Gemini word error 39.59 vs 25.80). Per trial, among trials where SDR rose, Gemini still got worse on 32/82 baseline, 15/82 extension, 11/86 WeSep (Fig 5.1, both SIR groups summed).
- RQ2: leak 63.90 → 23.08 % while invented rose 1.27 → 2.20 per trial. Leak weight: leaked −3.26, invented +0.72, WER +0.47 [−3.20, +4.15]. Artefact weight: invented −0.20 [−0.65, +0.25] (not reliable), WER +9.34. Hypothesis: both errors come from how hard the mask suppresses shared frames.
- RQ3: test (248) 48.46 → 41.81, −6.65 [−10.66, −3.18]; same epoch 13: 58.47 vs 42.93; cue split alone = cue split + ECAPA = 45.23 (earlier protocol); compute 55.0 vs 55.4 ms per 80 ms chunk; 175 ms mean delay, RTF 0.69.

**P20 Limitations**
1. Checkpoints chosen on validation, so its numbers flatter the models; on test the baseline's gain over no processing is 3.59 [−9.33, +1.97], inside the noise.
2. Extension 28 epochs, baseline 14; it still leads at equal epochs, but the final comparison is not at equal training.
3. One seed (42); intervals cover trial sampling, not retraining.
4. Gemini is closed and can change; three runs on the same baseline outputs differed by up to 3.41 points.
5. AMI about 19 dB quieter than training (−48.6 vs −29.2 dB); the gate silenced 110/300 extension outputs; level and gate not separated from extraction.
6. Algorithmic latency 120 ms (80 ms chunk + 40 ms lookahead) > REAL-TSE's 100 ms.

## Done after the list (2026-10-30)
- [x] Nomenclature: RIR, SLT removed; unused symbols x_e, H, h[n] removed.
- [x] Appendix D prompt: \footnotesize, no overflow; prompt text unchanged.
