# Report todo — 2026-10-07 (full read-through)

Submission **2026-11-05** (29 days). Experiment freeze **2026-10-14**. Supersedes 2026-09-29 (git history).
Worked **section by section, in report order**. `file:line` = `report/…` as of 2026-10-07.
**NB** = wrong or unsafe claim; an examiner will catch it. Everything else is wording, structure or missing content.

## 0. PARKED until the section pass reaches Results: "Why the metric is needed"

The intro (done 2026-10-05) promises both of these.

- **Experiments → new subsection "Testing the metric"** (`exp:metric_tests`, after Evaluation setup). No numbers.
  Question: does the split, or the live model vs an offline transcriber, change a decision? Test 1: mask sharpening
  (hysteresis post-processing, no retraining; settings no change / mild 1.4-0.6-0.2 / sharp 1.5-0.5-0.0; baseline e6
  and WeSep; cite Canny 1986, Bregman 1990). Test 2: 5-epoch fine-tunes of e21 with leak weight / artefact weight
  (Ochiai 2024) / both vs control, registered before running. Listeners: Gemini + Whisper `small.en`, 103 `sir0_val`
  `both` clips, paired bootstrap 10,000 draws, seed 42, Holm.
- **Results → new FIRST subsection "Why the metric is needed"** (`sec:results_metric`, before Content fidelity).
  Same WER, different leak (#17); Whisper misprices it (#18); training changes swap errors (D10, AB-SDR); takeaway.
  Numbers: evidence block at the bottom + `experiments/results/2026-10-03-bootstrap-interf-ft/`.
- **Before writing:** add Canny 1986 + Bregman 1990 to the bib; control row is baseline e6's own output (judge 51.51
  on 09-21 vs Table 5.1's 55.18, same audio); one judge run per arm (Limitations); retractions below apply;
  #17/#18 still need a `decisions-m4.md` entry (audio + code on `m5-listener-panel` b7c54f2).

## 0b. NEW from the blind examiner review (2026-10-07, `docs/reports/2026-10-07-blind-examiner-review.md`)

Blind mark **57/100** (pass, not yet "good"). Items below are *not* already elsewhere in this file.
- [ ] **NB References print internal notes** — bib `note` fields in [2] google2026geminilive, [9] cosentino2020librimix, [22] wichern2019wham, [34] bird2009nltk, [35]–[36] ITU P.835/P.808 (and check all). Strip annotations from `references.bib`.
- [ ] **NB Table 5.4 S+D+I ≠ LCF-WER** (floor 63.27 vs 62.94, baseline 55.59 vs 55.18, WeSep 26.39 vs 25.80): error split is from a different run than the 3-run means. Use 3-run means or say which run.
- [ ] **NB `results.tex:39` "14.24 points"** is the ICR@2 gap; the mean-leak gap is 15.28.
- [ ] **NB `results.tex:89`** calls Whisper's score "LCF-WER" (defined live-model only) and says it reaches "the same conclusions" — contradicts contribution 2. Reword.
- [ ] **NB intro "five times as often as WeSep"** rests on `eval_public` n=123, never defined; on the documented 42 it is 18 vs 7 (2.6×). Name the set or use the 42.
- [ ] **Ablation**: report the 1a (cue parts) and 1c (+ embedding) arms as a partial ablation (decisions-m2 09-27: 1a −10.36 vs baseline) — answers "confounded comparison".
- [ ] Research questions: none stated; examiner expects them (intro).
- [ ] Lit review: add SpeakerBeam / VoiceFilter (bib has `zmolikova2019speakerbeam`, `wang2019voicefilter`), SpEx+; enhancement-for-ASR line.
- [ ] Method gaps: where the embedding enters the net + parameter count; target reference dry or with early reflections; streaming chunk size (80 ms); show the w_g derivation (1.24 / 2.30) or cite the log; κ = 16 measured numbers.
- [ ] Wording: "33,000 parameters" vs "no learned parameters" (TF map); causality claimed as own though the challenge baseline is causal — say what *this* work changed (per-frame norm, no centring); AI-attribution sentence `extension_architecture.tex:43` contradicts itself; "Appendix C of the Whisper paper"; Fig 3.3 title shows "wesep_16k"; Table 5.5 caption points to wrong section.
- [ ] Metric validity: say FR counts some misrecognitions (ceiling 0.22/trial is that floor); no human validation of leak/FR → Limitations.

---

## 1. Front matter

- [ ] `title_page.tex:1,20` degree wording, submission date (department).
- [ ] `declaration.tex:1` official SU declaration wording.
- [ ] **AI declaration section** — new front-matter page + SU form. List every Claude-drafted passage adopted:
  intro contributions + scope line; §2.4 judge paragraph + gap; STFT and TF-map condensations; ICR rewrite and
  symbol definitions; Output level paragraph; cue split (`extension_architecture.tex:43`); nomenclature rows.
- [ ] `acknowledgements.tex:5` empty.
- [ ] `abstract.tex` empty — write last.
- [ ] `nomenclature.tex`: add κ, ℓᵢ, aᵢ, k, 𝒯, 𝒯ₖ, Sₙ, 𝒩, p, 𝟙, 𝓘ᵢ; abbreviations PSE, LLM, API, ECAPA-TDNN.
  `N` means three things (STFT window, MR window variable `metrics`-side, WER reference length `metrics.tex:11`).
  Delete commented rows `:74`, `:76`.

## 2. Introduction — done 2026-10-05; leftovers

- [ ] `:3` "accoustically", "presented to" → "presented with".
- [ ] `:15` "estaablished"; `:16` "differentiated against" → "differentiated through".
- [ ] `:15` "made fully causal" needs the causality-probe result in Results (§7).
- [ ] `:8` promises §0 evidence; `:21` outline — add "and how the metric itself is tested" once §0 exists.

## 3. Literature review

- [ ] `:4` leftover `% TODO`.
- [ ] `:18` broken sentence "the assumption that a 200–300 ms budget is sufficient." (no verb); "identify quite quickly speaker identity".
- [ ] `:23` "compared of the unprocessed input"; reason (i) "extracting a wrong speaker can still yield a low signal score" — unclear (a wrong speaker *does* score low; the point is that an average hides it).
- [ ] `:25` "pertubations" → "perturbations".
- [ ] **NB** `:33` Koenecke did not find the language prior causes hallucination — report what they measured (~1 % of transcripts, 38 % harmful).
- [ ] `:33` cut the AIR-Bench → role-swap link (different problems).
- [ ] **NB** `:37` "report a single error instead of a rate of error" — Chondhekar reports a rate (semantic WER). Say "a single error rate, not split by source".
- [ ] Verify Foo 2026 "60–72 %" against the paper (only in the weak-points register).
- [ ] Optional: PESQetarian (de Oliveira 2024) for metric gaming (A8).

## 4. Methodology

### 4.0 Section intro
- [ ] `methodology.tex:4` leftover `% TODO`.

### 4.1 Data (`methodology/data.tex`)
- [ ] **NB `:62` Splits is wrong**: validation used all 200 `sir0_val` trials (not 103), and that set is where checkpoints were chosen — not an "evaluation holdout". Name `eval_public` / `eval_private`; drop the data-scaling promise or report it (§7). Draft B (2026-10-05) covers it.
- [ ] **Structure `:36–41`**: "Trial definitions in evaluation" + `\subsubsection{Text normalisation}` sit mid-Data, so Enrollment, Parameters, Target-absent, Splits and Limitations now render *under* "Text normalisation". Move both to the end of Data (or back to the start of Metrics).
- [ ] **NB `:54`** "would just learn to follow the speech that was loudest" — overstated (direct test was inconclusive). Use the measured fact: on 90 %-target-louder data a stranger's enrollment changed the output by 2.6 % (`decisions-m2.md` 2026-08-25). Use `\SIrange`.
- [ ] `:34` regimes: give the bands (base = SNR 8–20 dB, T60 ≤ 0.5 s, activity 45–78 %; hard = full ranges; SIR the same in both); gender → "half the two-speaker trials are same-gender".
- [ ] `:25` "a target speaker close to a recording device" — base does not narrow distance.
- [ ] `:3` "Below includes a description for…"; `:7` "three" → four, "(iii) needs to include" (no subject); `:16` "The attribution to the success came from…".
- [ ] `:60` heading "Target-absent trials" → "Trial cases"; use Case 1–4.
- [ ] **AMI described** (Carletta 2005; only real-audio check; REAL-T trial cutting cited as borrowed; headset target is approximate; point to App. D `tab:ami`). decisions-m0.md 2026-10-06.
- [ ] **NB appendix `appendices.tex:134`** base SIR row now `[0.0, 8.0]` dB — wrong: sir0 has no base SIR override (±10 dB in both). Delete row; note eval splits use [−5, 15] dB with no regimes.

### 4.2 Baseline architecture (`methodology/baseline_architecture.tex`)
- [ ] **NB `:27` band plan** lists 31 bands but says 32. Real plan: 3/6/16/64-bin bands + an 8-bin top band; edges ≈ 1.41 / 3.28 / 5.78 / 7.78 / 8 kHz (`bands.py`, verified).
- [ ] **NB `:32` caption** "lowest fifteen 100 Hz; single highest spans 2 kHz" — lowest are 93.75 Hz; the *widest* is 2 kHz; the highest is ~250 Hz.
- [x] **NB `:48`** "A batch is twelve … crops" → six (3 trials × both directions; `collate_pairs`). Same at `setup.tex:11`.
- [x] **NB `:109` TF map, two inversions**: the softmax spread weights *too evenly* (not "not enough"); the template does *not* indicate whether the target is speaking. Optional measured numbers: 619.6 of 628 frames used, template varies 4.7 %; with κ = 16 top 50 frames carry ~59 %.
- [ ] **NB CARTSE deviations table deleted**: the floor-on-‖s_proj‖² change (keeps L_pres scale-invariant; CARTSE floors on ‖s‖²) is now stated nowhere. Rule: borrowed-and-changed must be stated. Restore at least that + branch-wise means.
- [ ] **NB `:209` w_m**: "derived from the target present and absent losses" → from L_pres and L_MR (both present). **Keep 9.62** (both models trained with it) but say it was calibrated on the earlier data (−5.91 / 0.184); on the final data the same rule gives **7.45** (median −5.15 / 0.207, 300 `sir0_train` crops, `experiments/results/2026-10-07-wm-anchor-sir0/`), so the spectral term starts at 0.39 × |L_pres|, not 0.30. (Earlier "≈ 1.75" was wrong.)
- [ ] `:198` Combining branches: lost the *reason* for per-branch means (balance fixed by w regardless of how many silent crops a batch draws); currently conflated with crop-vs-label routing.
- [ ] `:185` Output level — Claude-drafted, reword in own words; w_g crops came from `sir0_val`; 2.30 is a rule of thumb.
- [ ] Wording: `:17` "centered" → "centred"; `:46` "the approach is to first it normalises"; `:79` "It is useful to not bound this mask…" (unclear); `:130` "invariable" → "invariant"; `:159` grammar; `:161` "quite" → "quiet"; `:166` "The model should penalise" → "The loss…", "in this works".
- [ ] Trim (after supervisor sign-off, §10): ~~Nyquist paragraph~~ (cut 10-07), residual-block textbook lines.

### 4.3 Extension (`methodology/extension_architecture.tex`)
- [x] **NB `:6`** energy step "as seen in WeSep implementation" → Zhang et al. 2025 §II-A (verified in the paper).
- [ ] `:45` "Speaker context embedding" — the *utterance-level* embedding is used (contextual was discarded); rename "Speaker embedding" here and at `setup.tex:20`.
- [ ] `:47` say where/how the 128-d embedding enters the network (`context_encoder.py`); drop "served locally on a device" (scope is server-class).
- [ ] `:3` "Below discusses"; `:20` "cannot tell distinguish".

### 4.4 Metrics (`methodology/metrics.tex`)
- [ ] **NB `:31` mean leak** is now averaged over 𝒯 — code averages over 𝒯₅ (aᵢ ≥ 5); over 𝒯 divides by zero when aᵢ = 0. Restore 𝒯₅ + its "where" clause.
- [ ] **NB `:53`** "negated indicator function" → indicator function.
- [ ] `:20` "leaked" is no longer defined — add "…that also appear in the live model's transcript"; first sentence gives the wrong reason for stopwords (not "easily hallucinated").
- [ ] `:18` add back that ICR is not independent of LCF-WER (A6): leaked words are already insertions/substitutions; ICR says where they came from.
- [ ] FR: say FR_count is the tables' "invented/trial"; target alone scores 0.22/trial, so only the excess over it belongs to a system.
- [ ] `:66` SDR/SIR/SAR lost its opening (name the three); add the +30 dB cap (τ = 10⁻³) — Table 5.6 caption points to it.
- [ ] `:78–81` Anchors: "upperbounds"; rename floor/ceiling → "No processing" / "Target alone (perfect)" (ICR text already uses these); delete comment `:80`.
- [ ] `:56` "In this works", "1-5" → "1--5".

## 5. Experiments (`experiments/setup.tex`)

- [ ] `experiments.tex:4` leftover `% TODO`.
- [ ] **NB `:15` Model selection**: baseline was *not* judged — picked on validation target-present loss with the silence bar (L_abs ≤ −10 dB), epoch 6 of 16. Extension: `small.en` on 40 clips → kept checkpoints → Gemini on 103 → e21. Add consequences: optimised for Gemini; selected on the reporting set (winner's curse −6.76 vs −1.51 [−4.07, +1.04]); different rules per model. Draft A (2026-10-05) covers it.
- [x] `:11` batch: "3 trials, each in both directions (6 crops)"; check "roughly 15 hours".
- [ ] `:20` (iv) "WeSep provides a potential upper bound" — overclaim (offline, different data); rename "speaker context embedding".
- [ ] `:24–28` **Training curves analysis** (task 7): why e21 ≠ lowest loss; baseline e7 57.0 % / e15 51.8 % on Gemini (one run each, 2026-10-04) — e15 by silencing 5 clips and deleting 13.5 % vs 3.0 %; re-plot with baseline Gemini diamonds.
- [ ] `:47` "it's result" → "its"; "for which … built for"; "transcribe the estimated audio … what it is able to hear". `:49` "reponse".
- [ ] `:53` speech-gate four-case list → one sentence; "support" → "supported".
- [ ] `:57` latency: state the framing cost — 40 ms by our convention (window + hop), = CARTSE's 24 + 8 ms; component breakdown promised at `litreview.tex:16`.
- [ ] Text reference condition: one paragraph (cut on latency 2026-09-03, m4) with the pass-through caveat.
- [ ] §0 "Testing the metric" subsection (parked).

## 6. Results (`results/results.tex`)

- [ ] **Section opening**: every two-speaker number is on `sir0_val`, where checkpoints were chosen → optimistic; the only selection-free results are `eval_public` Case 3 and AMI. House labels throughout ("No processing / Target alone (perfect)").
- [ ] **NB `:41`** credits the gain to ECAPA — the extension is three inseparable changes (cue + embedding, content-WER selection/LR schedule, 21 vs 6 epochs). "Best streaming model", not "conditioning worked".
- [ ] **NB `:66`** "18 noisy clips" are *other-speaker* clips; "attributed to the diversity of training examples" unsupported (same data); "extension better than WeSep" on Case 2 is within noise and all are worse than no processing; explain n = 123 `eval_public` (e21 − WeSep −3.64 [−5.67, −1.65] words/trial; silenced 56 vs 11; all-or-nothing: unsilenced clips 29.2 vs 28.0). Draft C (2026-10-05).
- [ ] **NB Table 5.1** (`:15`): add prompt + run dates (rules); "selected on these trials"; separate 95 % CI column + paired differences (e21 − baseline −16.36 [−23.83, −9.18]); floor anchor bug A1 (62.98 vs 62.94) — fix code first.
- [ ] **NB `:185–202` latency**: caption lost CPU spec (i5-1135G7, 4 threads, 2,250 × 80 ms chunks) and "WeSep single earlier run"; ‡ has no footnote text; "comfortably within" → "within" (p99 margin 14–23 %); worst case 196 ms is 4 ms under 200, not "well within".
- [ ] **NB `:205–215`** four caveats as prose; judge noise SEM is 1.50 [0.98, 2.00], not 0.5; systems judged once cannot be split below ±4.16.
- [ ] Delete stale `% CLAIMS` comments `:116–124`, `:146–156` (numbers disagree with the tables).
- [ ] `:39` "subsequentlyhas", "over the passing the audio"; `:144` "significantly" (no test) → "clearly"; `:130` caption points to `sec:results_errors` for the cap — point to Metrics.
- [ ] `:113` Error composition discusses only the floor — add extension/WeSep (deletions vs insertions story) from the table.
- [ ] **Causality probe result**: WeSep 1.12e-2 vs ours 1.68e-8 (needed for intro's "fully causal").
- [ ] **AMI results**: WeSep vs e21 table + near-mute finding (110/300 blocked, ~26 dB below input); label anchor "Target's headset"; no headroom share. decisions-m4.md 2026-10-06.
- [ ] `eval_public` per SIR band (B1) — or state why not.
- [ ] §0 "Why the metric is needed" as the first subsection (parked).

## 7. Experiments still to decide — before the 2026-10-14 freeze

- [ ] **Matched-protocol baseline** (registered decisions-m2.md 2026-10-07) — `bsrnn_baseline_matched.yaml`, 2 Kaggle sessions (14 → 28 epochs), then judge e13/e15/e18/e21/e27 + probe pick. Answers the blind review's "confounded comparison".
- [ ] **Prompt sensitivity (A5)** — cheap; the primary contribution rests on the judge prompt.
- [ ] **AMI open items**: meeting-clustered bootstrap e21 − WeSep; e21 on the 110 levelled inputs (quiet-input explanation).
- [ ] **Floor anchor bug (A1)** — fix `load_once_index`, re-derive anchors, confirm Tables 5.1–5.3.
- [x] **w_m on sir0** — measured 2026-10-07: rule gives 7.45 vs trained 9.62 (§4.2). Weak-point C6's "~65 % too high" should read ~29 %.
- [ ] Data scaling — report 2.135 / 2.584 / 2.900 dB (signal only) or cut.
- [ ] Drop: GPU latency (#11, obsolete); per-band gate (#8) → future work.

## 8. Conclusion

- [ ] `:4–5` Summary of contributions — answer the intro's contributions with the headline numbers + scope (two-speaker mixtures, optimised for Gemini, selected on `sir0_val`).
- [ ] `:6–7` **Limitations and future work** (one sentence now): two-speaker boundary; read speech, simulated shoebox rooms, English; selection on `sir0_val`; single seed; one judge run per arm; prompt sensitivity; AMI near-mute; quality scored whole-clip but latency chunked (B2). Future: band plan, per-band gate, hail-mary loss, w_m. Fold in `data.tex:64`.

## 9. Appendices

- [ ] `appendices.tex:17–19` delete TODO; `:32` judge row → AI Studio API, audio in / text out, prompt sha `d118b7d3bf30`, runs 2026-09-03 → 2026-10-06.
- [ ] `:134` base SIR row (see §4.1).
- [ ] Hyperparameter/config table from the YAMLs.
- [ ] "Enrolment" (App. D) vs "enrollment" (body) — pick one.

## 10. Final passes

- [ ] Restructure + trim to budget (body ≤ 35 pp, PDF ≤ 50) — outline to supervisor first.
- [ ] SU rule on list of figures/tables (`report.tex`, commented out).
- [ ] Caption pass; reference pass; every judge number carries model ID / prompt / modality / date.

## Done since 2026-09-29 (for the record)

Introduction (10-05); §2.4 judge literature + gap (10-06); STFT shortened; TF map condensed with κ = 16 and Zhang
credited for the energy step in §3.2; ICR rewritten with symbol definitions; lookahead removed (text, figure, caption);
"held out" judge claims removed; Silero VAD named; Koenecke cited for hallucination; 5 s clean enrollment; −20 dB floor;
w derivation step; numbered Cases 1–4; nomenclature ℜ/ℑ + CARTSE row; App. D AMI table; Chondhekar wording.

---

## Evidence to write up (kept from 2026-09-07)

**#17 Leak trade — strongest metric result (narrowed by bootstrap 2026-09-30).** Mask
hysteresis, control / mild / sharp, same 103 `sir0_val` trials, same judge. Holm-corrected:
- **Total flat:** LCF-WER sharp − control WeSep −0.32 [−3.85, +2.97], ours +1.72 [−3.18, +6.78].
  Same-model arms, so the interval is ±3.5, not the ±8 between systems.
- **Leak falls — SIGNIFICANT in both:** mean leak WeSep −3.44 (p=0.012), ours −11.26 (p<0.001).
  ICR@2 only in ours (−8.74, p=0.034); WeSep's −4.85 is 5 trials, p=0.41.
- **What replaces it differs:** WeSep substitutions +2.88 (p=0.013); ours deletions +7.48 (p<0.001).
- **Fabrication rise NOT supported:** FR@2 p=0.76 / 0.16; invented/trial p=0.08 / 0.42. Drop "fabrication".
Claim: LCF-WER alone calls the arms interchangeable; the leak measure shows they are not.

**#18 Cheap listener misranks (confirmed for `sharp`, bootstrap 2026-09-30).** Cost of an
arm = arm − control LCF-WER. DiD = judge cost − `small.en` cost; negative = Whisper overstates.

| cost of `sharp` | judge | Whisper | DiD, 95 % CI, p(Holm) |
|---|---|---|---|
| ours | +1.72, n.s. | +18.48, p<0.001 | −16.76 [−31.25, −4.66], p=0.010 |
| WeSep | −0.32, n.s. | +14.93, p<0.001 | −15.25 [−27.14, −5.51], p=0.001 |

`mild`: WeSep DiD −12.43 (p=0.003); ours −6.87 (p=0.11, n.s.). `small.en` picks `control` in
96–100 % of resamples. **Retract** "judge picks `mild`" (judge arms indistinguishable, p=1.0),
"opposite sign" (judge cost is zero ±3.5) and the "10.7x" ratio (denominator is zero): say
"Whisper charges 15–18 points for a change the judge does not notice". Judge differences
are still below its re-measurement noise (A3, ±4.16).

**D10 (2026-10-03).** Leak-weight fine-tune of e21, e5: LCF-WER +0.47 [−3.20, +4.15]; mean leak −3.26
[−6.59, −0.03]; invented/trial **+0.72 [+0.27, +1.21]** (the robust one). `small.en` would have called it a win.
AB-SDR 2x2: weighting error types moves errors between leaked and invented; no cell beats the plain loss.

**#16 O4.** Organisers' DNSMOS-OVRL gaming incident (Track 1 human-MOS LCC +0.003,
swapped to P.808 post hoc) is first-hand evidence for the metric contribution.
`decisions-pending.md` O4, 2026-09-21.
