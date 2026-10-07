# Blind examiner review v2 — 2026-10-07 (evening)

Same brief as v1 (`2026-10-07-blind-examiner-review.md`), fresh no-context Claude subagent,
54-page build of 19:21. Model output, not a real examiner; ±2 points between runs is noise.

| Dimension | v1 | v2 | Note |
|---|---|---|---|
| Problem & contribution | 62 | 62 | contribution 2 still contradicted (p.32 "same conclusions") |
| Literature review | 55 | 55 | still no SpeakerBeam / VoiceFilter / SpEx; gap only "vs studies above" |
| Methodology | 63 | 66 | data/SIR fixed; cue-parts + ECAPA injection still undescribed; FR/ICR validity |
| Experimental design | 48 | 48 | selection on reported 103 trials; bundled changes; one seed |
| Results & analysis | 55 | 55 | AMI one sentence; §5.2 floor only; causal claims unsupported |
| Writing & presentation | 60 | 62 | 7 typos (verified), "wesep_16k", Fig 4.1 labels, bib notes, Claude sentence |
| Rigour & honesty | 56 | 58 | credited: pinned instrument, speaker bootstrap, 3 judge runs, AI attribution |
| **Overall** | **57** | **58** | "held-out headline + ablation would lift to ~65–70" |

## New in v2 (not in v1), verified where checkable
1. **Judge ≠ Live API**: motivation is Gemini Live; judge is `gemini-3.7-flash` (AI Studio) asked to transcribe. Justify or list as a limitation.
2. **Gain partly from suppression**: Gemini deletions 3.00 → 9.44 %; AMI 110/300 silenced, 45 % target words missed; LCF-WER falls mainly via fewer insertions. Flag it (CLAUDE.md: headline improving for a bad reason).
3. **FR denominator shrinks with silencing** (FR only over non-empty outputs).
4. **Judge SE 1.24 vs SD 0.34**: both true (1.24 from per-clip variation; 0.34 = SD of 3 totals, 2 df, unreliable) — the text must say which and why.
5. "roughly 15 hours" (setup.tex:11) — extension ~22 h over two sessions; baseline 10.5 h.
6. Fig 3.2 caption claims the causal time path as this work's; p.3 says REAL-TSE baseline already causal.
7. Undefined: "target words missed" (Table 5.3); AMI reference transcripts + stretch selection (App. D); chunk size/hardware for latency; 71.8 / 52.6 % method.
8. Typos (verified): data.tex:8 "organisersdescribe", baseline_architecture.tex:114 "per-bad", metrics.tex:7 "LibtriSpeech", metrics.tex:80 "bootsteap", litreview.tex:24 "pertubations", results.tex:40 "subsequentlyhas".
9. Suggestion: report the headline on an untouched split (eval_public `both`, 230 trials) — selection-free.

## Viva questions (v2)
Selection bias in 39.6 %; the 15.6-point decomposition and why e6 not e15; AMI drop extraction vs deleted insertions;
batch transcription as proxy for a live model; FR mishearing vs hallucination and its shrinking denominator;
the 123-clip set and SE 1.24 vs SD 0.34.
