# Blind examiner review — 2026-10-07

A Claude subagent with **no project context** read only the 52-page draft PDF (built 2026-10-07 15:24)
as an MSc examiner. Abstract and Conclusion excluded (unwritten). Model output, not a real examiner.
Page numbers are printed page numbers.

## Marks

| Dimension | Mark | Gist |
|---|---|---|
| Problem & contribution | 62 | Clear motivation; no explicit research questions; contribution 2 contradicted on p.31 |
| Literature review | 55 | 3.5 pp; missing SpeakerBeam / VoiceFilter / SpEx+ lineage and enhancement-for-ASR work; AIR-Bench link a non-sequitur |
| Methodology | 63 | Data + baseline careful; extension under-specified; inconsistencies p.7 vs Table C.1, p.11, p.15 |
| Experimental design | 48 | Baseline vs extension differ in epochs, LR schedule, selection; no ablation; one seed; selection on reported trials |
| Results & analysis | 55 | Rich tables + anchors; no trial-level CIs; undefined n=123; causal claims without ablation; §5.2 floor only |
| Writing & presentation | 60 | Readable, good figures; typos, internal notes in references, broken cross-refs, missing footnote |
| Rigour & honesty | 56 | Unusually candid about unfavourable results; undercut by a headline from undocumented data and overstatements |
| **Overall** | **57** | Pass, not yet "good" |

## Ten problems (reviewer's ranking) — with verification

| # | Reviewer's point | Verified? |
|---|---|---|
| 1 | Selection and test entangled: Fig 4.1 and Table 5.1 use the same 103 trials; stated rule (lowest Gemini WER) not followed — e27 39.2, baseline e15 51.8 | **Correct.** Baseline was chosen on loss, e21 tied e27 and won on invented words. Draft A (model selection) answers it. |
| 2 | Confounded comparison (epochs, LR schedule, selection, one seed); gain credited to ECAPA | **Correct.** Partial answer exists: the 1a / 1c arms in decisions-m2 09-27 are an ablation of the two changes. |
| 3 | No CIs over trials; "decisive", "significantly" unsupported | **Correct.** Paired bootstrap exists (e21 − baseline −16.36 [−23.83, −9.18]); not in the report. |
| 4 | "Five times as often as WeSep" from undefined n=123; 2.6× on documented 42; intro omits WeSep 13.8 pts better on Case 1 | **Correct.** n=123 is `eval_public`, never defined in the text. |
| 5 | Symmetric-SIR rationale contradicted by Table C.1 base SIR [0, 8] dB | **Caused by our error**: the appendix row is wrong (sir0 has no base SIR override). Fixing the row removes the objection. |
| 6 | Promised evidence missing (contribution 2; AMI; data scaling; causality probe; gender-wise selection) | **Correct.** All on the todo. |
| 7 | Extension inputs / embedding injection undescribed; target reference (dry vs early reflections) and chunk size not stated; w_g "derived" with no derivation; κ=16 testing unshown | **Correct.** |
| 8 | Inconsistencies: 31 vs 32 bands; "33,000 parameters" vs "no learned parameters"; causality claimed as own though baseline causal; "14.24" is the ICR gap not mean leak (15.28); Table 5.4 S+D+I ≠ LCF-WER; "LCF-WER" applied to Whisper; "196 ms well within" | **All correct.** S+D+I: floor 63.27 vs 62.94, baseline 55.59 vs 55.18, WeSep 26.39 vs 25.80 — error split is from a different run than the 3-run means. |
| 9 | FR counts misrecognitions as fabrication; leak/fabrication metrics not validated by humans; WeSep "upper bound" | **Correct.** Ceiling 0.22/trial is the misrecognition floor — say so. |
| 10 | Typos; internal notes printed in references [2], [9], [22], [34]–[37]; "Appendix C" ambiguity; Table 5.5 caption → wrong section; ‡ footnote missing; "wesep_16k" in Fig 3.3; self-contradictory AI-attribution sentence | **Correct** (bib `note` fields print annotations such as "Exact model identifier to be recorded…", "reference only, never run here"). "Appendix C" is Whisper's appendix — reword "Appendix C of the Whisper paper". |

## Strengths named
1. The leaked / invented split with formal definitions, showing a trade-off plain WER hides (Table 5.1).
2. Transparency: pinned instrument, exact prompt, full construction parameters, anchors in every table, judge variability, loss-vs-WER curves.
3. Honest reporting of unfavourable results (WeSep better; every extractor hurts target-only audio; DNSMOS diverges).

## Viva questions it would ask
1. Are Table 5.1's 103 trials the ones used to pick checkpoints? Scores on the untouched holdout?
2. Why epoch 21 (not 27) and epoch 6 (not 15)?
3. How much of the 15.6-point gain is the embedding, the cue split, longer training, and different selection?
4. What is the n=123 set, and why does the intro rely on it?
5. How do you separate misrecognition from fabrication? Human-checked?
6. With the target louder in most trials, how do you know the model uses the enrollment? Same-gender and AMI results?
