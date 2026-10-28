# Report revision handoff, 2026-10-27

Supervisor feedback being applied: (1) frame the report around RQ1–RQ3, (2) more and richer figures, (3) Gemini Flash is not a live model, say so, (4) softer claims, (5) give the laptop specs for latency, (6) test-set results, (7) formatting and typos.

Companion files: `docs/report-review-2026-10-27.md` (full audit: claims, typos, consistency), `docs/ai-declaration-todo.md` (every Claude contribution still to declare and tag).

## Page budget (compiled 2026-10-27)
50 pages total (limit 50), main body ends p.34.1 (limit 35). `\setcounter{tocdepth}{2}` would free ~1 page if needed.

## Done
- Intro: RQ1–3 with section pointers; LCF-WER expanded as "Listener Content Fidelity".
- Lit review: bold (RQx) tags; AI tags in round brackets everywhere.
- §3.1 Data: done (author happy). Appendix A, enrolment figure removed.
- §3.3: typos, equation punctuation, symbol clashes (y, h̄, var, ω, ψ, β, ν, 𝒵), λ defined (β 0.01274, λ 0.04689), A_ij notation, KL left-hand side, lines 47 and 203 rewritten.
- §3.2: grammar at l.6, 26, 31, 36, 106, 116, 134, 166; broken STFT reference fixed.
- §3.4: RQ sentence in intro; SDR/SIR/SAR merged into one display.
- §4: hook added (author); setup:22 fixes; Training curves moved into §4.1; all cross-references now \autoref/\eqref; §4.2.2 epoch paragraph cut; Appendix F deleted.
- Figures: 3.1 pipeline (TikZ); 5.1 per-trial agreement (SDR rose → Gemini better/same/worse, split by SIR, 3-run mean); 5.2 leak-vs-invention map; 5.3 delay (laptop specs in caption); 5.4 three sets (validation/test/AMI, baseline now on test). Results floats kept under their headings with \FloatBarrier.
- §5 restructured: RQ1 / RQ2 / RQ3 / Unseen and real recordings, from the author's own sentences; answers left as `% TODO (author)` with numbers.
- Test set: matched baseline scored 2026-10-27 (logged in decisions-m4.md).

## Still to do, in suggested order
1. **§5 answers** (TODOs in `results/results.tex`): RQ3 first, then RQ1, RQ2, §5.4 test-set paragraph, §5 intro headline. Numbers below.
2. **§5 claim fixes**: "17.15" → 21.70 (mean leak gap); "decisive", "markedly", "shines", "noisy clips" (Case 3 is interferer-only); Case 2 bold on 3.56 (2.48 is lower); "clear reduction" (borderline); untested causes ("artefacts → hallucination", "diversity of training examples", "robustness of Gemini"); "significantly improved SIR"; "demonstrates"; "cannot be used in a live system" → cannot stream (clip-level normalisation); AMI "holding the same conclusions" and "due to the artefacts" (110/300 outputs muted, quiet input suspected); credit for ECAPA (ablation says cue split did the work); "live judge" wording; tables still say Floor/Ceiling (→ No processing / Target alone).
3. **§6 Conclusion**: Key findings as RQ1/RQ2/RQ3 answers; remove "do not correlate" and "robust to very low-level noise" and "I believe"; Limitations: add Flash-not-Live, two-speaker boundary, validation used for selection; future work: oracle gate −4.5 [−9.3, −0.2] on 97 clips.
4. **Abstract** last: one sentence per RQ; remove ECAPA credit and "holding the same conclusions".
5. **§4 Experiments**: augmentation was per Kaggle session not per epoch (persistent workers); L_abs weight 0 for 4 epochs then 3-epoch ramp (not "first 2 epochs"); fine-tuning runs paragraph (from e21, 6 epochs, lr 1.25e-4; leak weight ×2, artefact weight ×4 after AB-SDR `ochiai2024rethinking`; control); Flash-vs-Live paragraph `sec:flash` (author drafting, Claude continuation given); Listener paragraph fixes (wrong cite → google2026geminiflash, status-field claim contradicted 0/6, "connected to via"); VAD "prevents" → "reduces"; L_MR "fewer artefacts" (e21 val 0.281 vs mixture 0.270); training-curves caption ("seeb", gate row is a separate run); latency subsection ("buddet", define 80 ms chunk + 40 ms lookahead + compute); add extension parameter count 7,281,168; same-epoch comparison and selection-vs-report sentence in §4.1.3.
6. **§3 leftovers**: §3.2 merges (fold BSRNN and Band plan subsections, merge Residual LSTM + Band and sequence, shorten STFT prose); w_m = 9.62 came from an earlier training split (same rule on final data gives 1.03–1.75); grammar l.41, 140, 153, p clash l.157. §3.3: how the cue parts enter (5 channels, c_t repeated across bins, scales 6.94/0.51/1.28, +66,820 params) and the "no new information" argument; 1+γ fusion with Delcroix 2020 and FiLM (BibTeX given in session); 71.8/52.6 % are from an earlier baseline checkpoint; gate RQ2 opening sentence and outcome paragraph (label loss 15→4 train, flat 12.5–18.5 val; gated run 49.5 % vs 39.6 %; code at commit 9a6223b); ℓ is the composite loss of eq. 3.12 on the gated output (not squared error); l.180 run-on.
7. **§3.4 Metrics**: definition fixes (mean leak only over trials with ≥5 available words; invented = distinct content words; FR skips responses with no content words); +30 dB cap sentence; "Anchors"/"Reference Points" → plain labels, sentence case; ±SD definition in Uncertainty; `\mathbb{1}` prints a broken glyph (use `\mathbf{1}`); wording at l.3, 7, 11, 15, 20, 22, 43, 49.
8. **§2 / §1 leftovers**: check REAL-TSE 100 ms cap (algorithmic or end-to-end); "robust against noise" → added white noise; delete AIR-Bench role-swap sentence; "Gemini Flash 2.0" → "Gemini 2.0 Flash"; LAM naming; intro "fairly robust", "adapted by CARTSE".
9. **Formatting**: remaining typos in the review doc; leftover `% TODO`s; nomenclature (unused GLU/IHM/RIR/SLT; missing ECAPA, ELBO, KL, LLM, API, CPU, RMS, RNN); appendix prompt overflow (`\small`); AI declaration text (AI3 wording, new items) from the tracker.

## Numbers for the §5 answers
- **RQ1**: system ranking agrees for SDR, SIR, Whisper (63.04 / 48.92 / 41.01 / 29.13 / 1.98) and Gemini (62.94 / 53.26 / 39.59 / 25.80 / 1.19); DNSMOS SIG/OVRL and SAR disagree. Per trial (Fig 5.1, SDR rose): Gemini worse on target-quieter trials 16/45 baseline, 5/45 extension, 3/51 WeSep; on target-louder trials 16/37, 10/37, 8/35. Whisper vs Gemini per clip r = 0.75 (0.81 / 0.86 / 0.50). Source: `experiments/results/2026-10-27-per-trial-agreement/`.
- **RQ2**: leak 63.9 → 42.2 → 23.1 → 13.0 %; invented/trial 1.27 → 2.52 → 2.20 → 1.80 (no processing, baseline, extension, WeSep). Fine-tunes vs control: leak weight leak −3.26 [−6.59, −0.03], invented +0.72 [+0.27, +1.21], WER +0.47 [−3.20, +4.15]; artefact weight leak +14.66 [+9.09, +20.52], invented −0.20 [−0.65, +0.25], WER +9.34 [+3.65, +15.37]. Target-only clips: every extractor worse than no processing.
- **RQ3**: test (n = 248) baseline 48.46, extension 41.81, extension − baseline −6.65 [−10.66, −3.18]; same epoch 13: 58.47 vs 42.93; cue split only = cue split + ECAPA = 45.23 (earlier protocol); ECAPA same-gender voice picking −1.6 pp [−3.4, +0.2] on 1,079 crops; voice picking extension 89.7 / 72.4 % vs earlier baseline 71.8 / 52.6 %; compute 55.0 vs 55.4 ms per 80 ms chunk; 175 ms delay, RTF 0.69, i5-1135G7.
- **Test set** (n = 248): extension − no processing −10.24 [−18.06, −3.43]; baseline − no processing −3.59 [−9.33, +1.97] (within noise); WeSep − extension −7.73 [−12.02, −3.48]. Source: `experiments/results/2026-10-27-test-set-paired-ci/`.

## Working notes
- Claude only edits `report/` when told; otherwise suggests in the terminal. Close and reopen any .tex file Claude edited before typing in it (an editor save overwrote §3.3 fixes once).
- Figure scripts: `scripts/plot_{tradeoff_map,set_comparison,latency,per_trial_agreement}.py`; the last takes `--replot` and must run with `env -u GEMINI_API_KEY`.
- Compile check: `latexmk -pdf -outdir=<scratch> report.tex` from `report/`.
