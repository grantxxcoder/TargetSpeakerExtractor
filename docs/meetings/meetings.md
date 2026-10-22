# Meeting notes

Newest first. Decisions move to `docs/decisions/decisions-m0.md`; open items stay in `docs/decisions/decisions-pending.md`. Keep item IDs for tracing both ways.

---

## 2026-08-13 — Lecturer

**Present:**
**Phase:** 2, data preparation.

### Research posture

1. **This is research, not software engineering.** Deliver a defensible finding, not a well-built system. Engineer only what is needed to produce or trust the result.

2. **Quality over quantity.** One well-evidenced contribution beats several shallow ones: depth on the metric, a few clean, interpretable experiments.

3. **Innovate around the compute limit, do not brute-force it.** 11 weeks, constrained compute. Prefer a cheaper, *slightly worse* training method over a costly one; the shortcut is itself a contribution.

4. **Report the compute-limited gap explicitly.** For every efficiency shortcut: what was traded, its metric cost, and the expected result with full resources, with evidence where possible (e.g. a scaling point, an ablation or a short full-cost run).

**Implication for modelling:** compute is a design constraint alongside the ~200–300 ms latency budget. Levers to evaluate and report: shorter training crops, a subsampled training set, a smaller proxy encoder, fewer epochs with a stronger schedule, mixed precision, frozen-feature caching.

→ **Notes:**

---

## 2026-08-12 — Supervisor

**Present:**
**Phase:** 2, data preparation. Manifests built, renderer not started.

### Decided

| Item | Question | Supervisor's answer | Status |
|---|---|---|---|
| **A4** | Should the enrollment clip carry the room? | **No: clean voice samples, so the model knows what clean sounds like** | Done. `decisions-m0.md` 2026-08-12. No code change: renderer unwritten, enrollment already dry |
| **B3** | Fixed enrollment length? | **Yes, fixed 5 s, and it must stay a changeable parameter** | Done. `decisions-m0.md` 2026-08-12. No code change: already `enrollment_length_s: 5.0` in config |
| **B4** | Silent-target trials in eval too? | **Yes, we are measuring intelligibility, so eval must include them** | **Decided, not implemented.** Needs a config change and an eval rebuild (see below) |
| **A3** | Levels measured how? | **RMS**, reversed the same day to **BS.1770 integrated loudness** | Done. `decisions-m0.md` 2026-08-12. No code change: `pyloudnorm` 0.2.0 already in `tse_venv`, column names and ranges already correct |

### Difficulty, controllability and reporting

**Core concern: the data may be too hard to learn from, despite being realistic.** There is no dial between realism and trainability; everything below follows from that.

1. **Each parameter's distribution must be quickly changeable.** The generator hardcodes uniform sampling over `[lo, hi]`; the shape should come from config, not code. Room size was the example where uniform is probably wrong.

2. **Constraints, not only distributions.** Some conditions are *relational*, not ranges, e.g. **the target is always closer to the mic than the interferer**. The generator cannot express these yet.

3. **A realistic "average case" band per parameter.** Set the base condition to the easier part of each range so something is learnable; report the full range as a harder condition. For **all** parameters, not only the example.

   → **Clarify:** wall absorption "only accounts up to 0.5 of that uniform distribution": lower half of the range, coefficient ≤ 0.5, or middle 50 %? Absorption is not a config parameter; `pra.inverse_sabine` derives it from `t60_s`.

4. **Should the distribution *type* be configurable per parameter?** Idea only: name a distribution and its parameters in config (uniform / exponential / binomial / …). → **Decision needed.**

5. **Reporting must be stratified over every condition.** Final test-set metrics per condition (target present, target absent, interruptions), never one aggregate. Interpretability is strict.

   → **Clarify:** "interruptions": high-overlap trials or a new trial type? Nothing marks them today; the nearest column is `overlap_achieved`. Overlaps B9 (adds turn-taking and target-only trials).

6. **EDA per parameter, to check the realised distribution is the intended one.** The manifest notebook checks ranges (§3) and shortcuts (§7) but never plots realised shapes against the request.

**Own action:** comment every parameter in `experiments/configs/generator.yaml` in plain English: what it controls and a realistic average-case value.

→ **Notes:**
