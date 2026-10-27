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
| 11 | Results, references | Removed Table 5.1, the latency table, two AMI table rows; stripped bib notes; switched AI tags to round brackets (at the author's instruction) | AI2 | [ ] | n/a |

Rewrite every Claude-drafted caption and sentence in your own words before submission; the declaration then says Claude drafted, you rewrote.
