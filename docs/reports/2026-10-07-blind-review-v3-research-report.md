# Blind review v3 — framed as an MSc RESEARCH REPORT (~35 pp) — 2026-10-07 21:14 build

Fresh no-context Claude subagent, PDF only. Model output, not a real examiner (±2–3 between runs).
Includes: metric-evidence table (Whisper column already removed by Grant), Whisper table in App. E,
signal+DNSMOS merged, speaker-clustered CIs, AMI table with meeting CIs.

| Dimension | v2 (thesis frame) | **v3 (report frame)** |
|---|---|---|
| Problem & contribution | 62 | **68** |
| Literature review | 55 | **60** |
| Methodology | 66 | **66** |
| Experimental design | 48 | **52** |
| Results & analysis | 55 | **58** |
| Writing & presentation | 62 | **63** |
| Rigour & honesty | 58 | **62** |
| **Overall** | **58** | **62 — good band (60–69)** |

Held below 70 by: (1) no held-out test (eval_private run in progress); (2) confounded comparison, no
ablation (matched baseline ready); (3) contribution 2 half-shown — Table 5.1 caption mentions Whisper but
has no Whisper column, and p.32 "same conclusions"; (4) FR validity unchecked — hand-audit a sample,
show ranking stable across k; (5) AMI gain partly from silencing (110/300, 45 % missed) — say so;
(6) single judge runs in Tables 5.1/5.4; (7) the Table 5.1 experiments undescribed in §4; artefact-weight
result (+9.34) undiscussed; (8) lit misses multi-talker speaker-attributed scoring (cpWER, SA-WER);
latency: CPU, chunk size, causality-probe result, 196 ms not in table, REAL-TSE 100 ms limit.
Also: §5.1 one sentence; Table 5.4 undiscussed; Fig 4.1 val loss rising undiscussed; typos incl. "seperation" p.33.

Strengths: reproducible data pipeline; serious uncertainty treatment (speaker bootstrap, judge resolution,
McNemar); working causal extractor at RTF 0.69 with per-case breakdown.

Viva: gain on eval_private?; FR misheard vs invented?; which change gives the gain?; is silencing 37 % of
AMI clips usable?; what does the leak/invent split add over cpWER / SA-WER?
