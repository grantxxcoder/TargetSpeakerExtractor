# AI declaration to-do (report revision, from 2026-10-27)

Every Claude Code contribution to the report revision that still needs (a) a line in the declaration (`frontmatter/acknowledgements.tex`) and (b) a `(\ref{ai:...})` tag where it appears in the report. Tick both columns when done.

| # | Where in report | What Claude did | Declaration item | Declared | Tagged in text |
|---|---|---|---|---|---|
| 1 | Whole report | Audited claims, numbers, typos, structure (`docs/report-review-2026-10-27.md`); proposed the RQ-led restructure and the figure plan | AI2 writing assistant | [ ] | n/a |
| 2 | Intro, RQ2 | Suggested "Does reducing one type of error increase the other?" | AI2 | [ ] | [ ] |
| 3 | §3.4.2, nomenclature | Suggested renaming LCF-WER's expansion to "Listener Content Fidelity"; made the edit | AI2 or AI8 (metrics) | [ ] | [ ] |
| 4 | Lit review | Chose placement of the bold (RQ) tags; suggested closing sentences (only if adopted) | AI2 | [ ] | [ ] |
| 5 | Fig 3.1 pipeline (`figures/pipeline.tex`) | Wrote all TikZ from the author's spec; drafted caption | AI3 diagrams (reword: "wrote", not "assisted with layout") | [ ] | [x] caption tag present |
| 6 | Fig 5.1 trade-off map | Wrote `scripts/plot_tradeoff_map.py`; drafted caption | AI3 | [ ] | [ ] |
| 7 | Fig 5.3 three-set comparison | Wrote `scripts/plot_set_comparison.py`; drafted caption and the clause "…and Figure 5.3 compares…" in §5.2.2 | AI3 | [ ] | [ ] |
| 8 | Fig 5.5 delay | Wrote `scripts/plot_latency.py`; drafted caption | AI3 | [ ] | [ ] |
| 9 | §4.2.2 | Wrote the two sentences on choosing epoch 21 over 27 | AI2 | [ ] | [ ] |
| 10 | §5 test-set numbers and paired intervals | Wrote `experiments/results/2026-10-27-test-set-paired-ci/paired_ci_test.py` | AI6 code (hard) | [ ] | [ ] |
| 12 | §3.3.3 presence gate | Chose and applied the replacement symbols for the clashes (y, h̄, var, ω, ω₀, ψ, β, ν, 𝒵); wrote the sentence defining λ and giving β, λ; fixed typos and equation punctuation | AI7b / AI2 | [ ] | [ ] |
| 13 | §3.2 l.6, 26, 31, 36, 106, 116, 134, 166; §3.3 l.47, 203 | Rewrote these sentences for grammar at the author's instruction (wording is Claude's) | AI2 | [ ] | n/a |
| 14 | §3.4 intro, §3.4.6, §4.1.4 (iii), whole report | Wrote the metrics RQ sentence; merged the SDR/SIR/SAR displays; reworded WeSep reason (iii); moved Training curves; converted all manual cross-references to \autoref/\eqref | AI2 | [ ] | [ ] |
| 15 | Fig 5.1 per-trial agreement bar chart (RQ1), removed from the report 2026-10-28 (author found it confusing); script kept | Wrote `scripts/plot_per_trial_agreement.py` (per-trial SDR, Gemini and Whisper from caches; stacked bars of SDR-improved trials by Gemini outcome, split by SIR, three-run Gemini mean); drafted caption; per-trial data in `experiments/results/2026-10-27-per-trial-agreement/` | AI3 / AI6 | [ ] | [ ] |
| 16 | §5 structure | Rebuilt §5 into RQ1–RQ3 + unseen/real sections from the author's sentences (moved, split, two paragraphs and one figure deleted); wrote the TODO comments with numbers | AI2 | [ ] | n/a |
| 17 | Fig 5.1 conventional measures (RQ1) and Table 5.1 Gemini row | Wrote `scripts/plot_conventional_measures.py` (grouped bars, signal + DNSMOS, capped values left out); drafted both captions; inserted figure and table into results.tex at the author's instruction; recomputed WeSep signal + DNSMOS (`experiments/results/2026-10-28-eval-wesep-signal-perceptual`, matched the old row) | AI3 / AI6 | [ ] | [ ] |
| 18 | Fig 5.2 Gemini content and Fig 5.4 trial cases (RQ2), replacing Tables 5.2 and 5.3; CI row in Table 5.1 | Wrote `scripts/plot_gemini_content.py` and `scripts/plot_trial_cases.py`; drafted both captions; replaced the two tables in results.tex at the author's instruction; moved the LCF-WER 95 % intervals into Table 5.1 | AI3 | [ ] | [ ] |
| 19 | Table 5.2 fine-tune trade-off (RQ2), replacing Fig 5.3(b); Fig 5.3 cut to the systems panel; §4 epoch sentence given numbers; WeSep recoloured in all results figures | Built the 2 x 2 table (scores, paired changes, 95 % intervals) and drafted its caption; cut `plot_tradeoff_map.py` to one panel; replaced the panel reference in setup.tex with the e21/e27 numbers; proposed the within-model vs across-model reading of the trade-off (a forest-plot version was drawn and dropped at the author's request) | AI2 / AI3 | [ ] | [ ] |
| 20 | Fig 5.x voice picking (RQ3) | Wrote `scripts/plot_voice_picking.py` (rates from the directional-test per-crop files, crop bootstrap 95 % intervals and paired changes, `experiments/results/2026-10-28-voice-picking-ci/`); drafted caption; inserted it into results.tex at the author's instruction | AI3 / AI6 | [ ] | [ ] |
| 21 | §5 numbers behind every comparison | At the author's instruction, added bracketed values with metric names after each comparison in results.tex (from the result files), and corrected wording the numbers contradicted: WeSep not extension's output, higher not lower LCF-WER, SDR and SIR not all signal metrics, Case 3 is the other speaker not noise, two speakers not multiple, invented-word win scoped to AMI; trial-case figure Case 3 bracket | AI2 / AI3 | [ ] | [ ] |
| 11 | Results, references | Removed Table 5.1, the latency table, two AMI table rows; stripped bib notes; switched AI tags to round brackets (at the author's instruction) | AI2 | [ ] | n/a |

Rewrite every Claude-drafted caption and sentence in your own words before submission; the declaration then says Claude drafted, you rewrote.
