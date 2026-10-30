# Supervisor feedback, 2026-10-30

Tick as done. Items marked (H) are also on `docs/report-handoff-2026-10-27.md`.

## Supervisor
- [ ] S1. Results are on the validation set, not the test set: motivate it, or switch.

## Overclaims and wording
- [ ] W1. §5 intro states the masking trade-off as fact; §5.2 calls it a hypothesis. Hypothesis in both.
- [ ] W2. §5.1 "established that LAMs are robust against noise": only white noise was tested (H).
- [ ] W3. §5.2 "each fine-tune improves its own score": the artefact weight's invented CI [−0.65, +0.25] crosses zero.
- [ ] W4. §5.2 "hallucinated" → "invented".
- [ ] W5. §5.3 same-gender sentence describes different genders.
- [ ] W6. §5.1 "as pleasing to a human listener as WeSep": a predicted score.
- [ ] W7. Rewrite the Zheng et al. "80 %" sentence (lit review).
- [ ] W8. Rewrite the "filler words" and "grammatical issues" lines (ICR section).
- [ ] W9. Rewrite the Experiments chapter intro paragraph.
- [ ] W10. Rewrite "subjective to the actual perceptual system" (§5.1).

## Polish
- [x] P1. Mixed voice: "this work", "our" (§5.4), "I believe / I had limited time" (conclusion) (H).
- [~] P2. Mixed decimals: 53.3 vs 62.94 (abstract); 5.91 vs 5.9088 (weight calculation). Abstract, conclusion, 15.38 done; one-decimal measured values still need source lookups.
- [ ] P3. p: the 0.3 exponent and the 0.297 silent fraction.
- [ ] P4. τ used three ways.
- [ ] P5. ℓ: the loss and the leaked word count.
- [x] P6. d: the interferer signal and the run length.
- [ ] P7. Fig 5.1 caption points to §3.4.6 for τ; it is defined in §3.2.9. Waits on P4 (floor symbol).
- [x] P8. "Section D.1" → "Appendix D.1".
- [x] P9. Fig 5.2 caption: brackets around AI3.
- [~] P10. Nomenclature: IHM, GLU, MLP unused; DNSMOS "personalised model" never introduced (H). GLU/IHM removed, MLP kept (used in Fig 3.x), 9 acronyms added; DNSMOS sentence pinned.
- [x] P11. AMI section says "As a final test…" twice.
- [x] P12. "Target-absent trials" paragraph covers all four cases: rename.
- [x] P13. AI declaration sits under Conclusion, not in the ToC; wrong opening quote before "Guidelines…".
- [x] P14. References: CARTSE [1] and SA-Mamba [54] have no URL.
- [x] P15. References: [28] and [29] have no page numbers.
- [ ] P16. References: VoxCeleb not cited.
- [ ] P17. References: [5] says "Tech. Rep." twice.
- [ ] P18. References: "Proc." and "Proceedings" mixed.
- [ ] P19. Conclusion: answer RQ1, RQ2, RQ3 explicitly, one by one (H).
- [ ] P20. Limitations: validation set; extension trained 2× the epochs; one seed; Gemini drift; AMI input level and VAD gate; 120 ms algorithmic latency > 100 ms REAL-TSE limit.
