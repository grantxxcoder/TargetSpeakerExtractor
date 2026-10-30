# Supervisor feedback, 2026-10-30

Key: [x] done, [~] part done, [P] pinned, [-] dropped by the author, [ ] not started. (H) = also on `docs/report-handoff-2026-10-27.md`.

**Status: 27 of 31 done, 2 part done (S1, P2), 2 pinned (W2, P19).**

## Supervisor
- [~] S1. Results on the validation set, not the test set: motivate it.
  - Done: abstract (test numbers, "on the test set", "about a third"); Table 5.1 on test; Fig 5.2 on test + its two paragraphs; Case 1 sentence on test; conclusion RQ3 on test; test count 248 / 499 (data.tex, intro); Fig 5.5 caption says why validation is fine for delay; AMI closing paragraph says "On AMI".
  - Left: motivation paragraph at the start of §5; Fig 5.1 and Fig 5.3 on test (background jobs running); "validation" in the text of Table 5.2 (fine-tunes) and the voice-picking sentence.

## Overclaims and wording
- [x] W1. Masking trade-off stated as a hypothesis in the §5 intro.
- [P] W2. "LAMs are robust against noise": only white noise was tested (H). Pinned: affects the author's analysis. Same claim in litreview:32 and the intro.
- [x] W3. Fine-tune sentence gives intervals only.
- [x] W4. "hallucinated" → "invented" (results, metrics.tex:40).
- [x] W5. Same-gender sentence replaced with the author's wording.
- [x] W6. P.808 "predicted listener rating", said once.
- [x] W7. Zheng et al. sentence (checked against the paper's abstract).
- [x] W8. Content-words paragraph rewritten.
- [x] W9. Experiments intro rewritten in the author's style.
- [x] W10. DNSMOS sentence rewritten.

## Polish
- [x] P1. Impersonal voice in the body; AI declaration stays first person.
- [~] P2. Decimals. Done: abstract, conclusion, 15.38. Left (need source values): trial-case shares 49.5/25.5/19.6/5.3; 5.8 % silent crops; −48.6/−29.2 dB; 71.8 % (§3.3); WeSep 348.3 → 348.32 ms; WeSep parameters 27.2 M (timing log 27,242,122 vs test-render log 33.46 M: check which is right); Appendix B AMI stats; w_m text −5.91/0.184 vs equation 5.9088/0.1842.
- [x] P3. Silent share p → f_abs.
- [x] P4. Loss floors τ → ε_pres, ε_abs; τ is the gate temperature only.
- [x] P5. Gate-derivation loss ℓ → L; line 126 now "the loss of (3.12) on the gated output", temperature β (no Gaussian, no ν).
- [x] P6. Interferer signal d → s_int.
- [x] P7. Floor ε_pres added to SDR/SIR/SAR + "+30 dB cap" sentence; Fig 5.1 caption points to it.
- [x] P8. "Appendix D.1" (explicit Appendix~\ref).
- [x] P9. Fig 5.2 caption AI tag bracketed.
- [x] P10. Nomenclature cleaned (GLU, IHM, RIR, SLT, x_e, H, h[n] removed; 9 acronyms added). DNSMOS sentence dropped by the author.
- [x] P11. Duplicate "As a final test" removed.
- [x] P12. Heading "Trial cases".
- [x] P13. AI declaration on its own page, in the ToC, correct quotes.
- [x] P14. CARTSE URL added (SA-Mamba already had one).
- [x] P15. Pages for [28] and [29] (Crossref).
- [x] P16. VoxCeleb1 + VoxCeleb2 cited.
- [x] P17. arXiv entries as @misc.
- [x] P18. "Proc." everywhere; Interspeech year printed once.
- [P] P19. Conclusion answers RQ1–RQ3 one by one. Draft below.
- [x] P20. Author keeps the current limitations.

## Found while fixing
- [x] Interspeech year printed twice → booktitle "Proc. Interspeech".
- [x] Appendix D prompt overflow → \footnotesize, prompt text unchanged.
- [x] Gap at the bottom of page 18 → eq. 3.33 can break across pages.
- [x] "+30 dB" lost its plus sign → retain-explicit-plus.
- [x] Fine-tune "three of the four cases"; artefact weight alone worsens LCF-WER; "cannot be told apart from chance"; conclusion RQ2 hedged.
- [-] Unsupported claims (ECAPA credit in abstract/§5.3/conclusion; WeSep "cannot be used in a live system"; AMI "degrading the target speech"): author chose to leave them.
- [ ] N is both the STFT window length and the WER reference word count.
- [ ] Page count: 52 pages against a 50-page limit; S1 paragraph and P19 will add text.

## Background jobs (started 2026-10-30)
- Fig 5.1 on test: interferer stems rebuilt for 249 trials (0 mismatches, `experiments/results/2026-10-30-test-interferer-stems/`); signal + DNSMOS scoring into `experiments/results/2026-10-30-eval-private-*-signal-perceptual`.
- Fig 5.3 on test: Case 2–4 renders into `experiments/results/2026-10-30-est-private-*-{target_only,interferer_only,noise_only}`; then Gemini scoring (author runs it with the key).

## Pinned draft (Claude Code, 2026-10-30; not in the report)

**P19 Key findings** (RQ1 numbers wait for Fig 5.1 on test)
- RQ1: SDR and SIR rank the systems as Gemini does; DNSMOS does not (P.808 extension 3.03 vs WeSep 3.01, yet Gemini word error is higher on the extension).
- RQ2: on test, leak 47.89 → 21.15 % while invented rose 1.48 → 2.26 per trial. Fine-tunes (validation only): leak weight: leaked −3.26 [−6.59, −0.03], invented +0.72 [+0.27, +1.21], LCF-WER +0.47 [−3.20, +4.15]. Artefact weight: invented [−0.65, +0.25] (not reliable), LCF-WER +9.34. Hypothesis: both errors come from how hard the mask suppresses shared frames.
- RQ3: test (248) 48.46 → 41.81, −6.65 [−10.66, −3.18]; same epoch 13: 58.47 vs 42.93; compute 55.0 vs 55.4 ms per 80 ms chunk; 175 ms mean delay, RTF 0.69.
