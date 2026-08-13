# Meeting notes

Newest first. Decisions taken here move to `docs/decisions/decisions.md`; anything left
open stays in `docs/decisions/decisions-pending.md`. Keep the item IDs so they can be
traced both ways.

---

## 2026-08-13 — Lecturer

**Present:**
**Phase:** 2, data preparation.

### Research posture

1. **This is research, not software engineering.** Deliverable is a defensible
   finding, not a well-built system. Engineering effort only counts where it is
   needed to produce or trust the result.

2. **Quality over quantity.** One well-designed, well-evidenced contribution beats
   several shallow ones. Prefer depth on the metric and a small number of clean,
   interpretable experiments.

3. **Innovate around the compute limit — do not brute-force it.** 11 weeks and a
   constrained compute budget. A cheaper training method that is *slightly worse*
   is preferred over a costly one, and the shortcut is itself a contribution.

4. **Report the compute-limited gap explicitly.** Every efficiency shortcut must be
   written up as: what was traded, what it cost in metric terms, and what the model
   would be expected to reach with full resources. Evidence for the projection where
   possible (e.g. a scaling point, an ablation, or a short full-cost run).

**Implication for modelling:** treat compute as a first-class design constraint
alongside the ~200–300 ms latency budget. Candidate levers to evaluate and report —
shorter training crops, subsampled training set, smaller proxy encoder, fewer epochs
with a stronger schedule, mixed precision, frozen-feature caching.

→ **Notes:**

---

## 2026-08-12 — Supervisor

**Present:**
**Phase:** 2, data preparation. Manifests built, renderer not started.

### Decided

| Item | Question | Supervisor's answer | Status |
|---|---|---|---|
| **A4** | Should the enrollment clip carry the room? | **No — clean voice samples, so the model knows what clean sounds like** | Done. `decisions.md` 2026-08-12. No code change: renderer unwritten, enrollment is already dry |
| **B3** | Fixed enrollment length? | **Yes, fixed 5 s, and it must stay a changeable parameter** | Done. `decisions.md` 2026-08-12. No code change: already `enrollment_length_s: 5.0` in config |
| **B4** | Silent-target trials in eval too? | **Yes — we are measuring intelligibility, so eval must include them** | **Decided, not implemented.** Needs a config change and an eval rebuild — see below |
| **A3** | Levels measured how? | **RMS** — reversed same day to **BS.1770 integrated loudness** | Done. `decisions.md` 2026-08-12. No code change: `pyloudnorm` 0.2.0 already in `tse_venv`, column names and ranges already correct |

### Difficulty, controllability and reporting

**Core concern: the data may be too hard to learn from, despite being realistic.**
Realism and trainability are traded against each other with no dial to trade them
on. Everything below follows from that.

1. **Each parameter's distribution must be quickly changeable.** The generator
   hardcodes uniform sampling over `[lo, hi]` for every parameter. Needed: change
   the shape from config, not code. Room size given as the example where uniform
   is probably wrong.

2. **Constraints, not only distributions.** Some conditions are *relational* and
   cannot be written as a range — e.g. **the target is always closer to the mic
   than the interferer**. The generator cannot express this today.

3. **A realistic "average case" band per parameter.** Restrict the base condition
   to the easier part of each range so there is something learnable, and treat the
   full range as a harder reported condition. To apply to **all** parameters, not
   only the example given.

   → **Clarify:** the wall-absorption example, "only accounts up to 0.5 of that
   uniform distribution" — the lower half of the range, an absorption coefficient
   ≤ 0.5, or the middle 50 %? Absorption is not a config parameter at present; it
   is derived from `t60_s` by `pra.inverse_sabine`.

4. **Should the distribution *type* be configurable per parameter?** Idea only:
   name a distribution and its parameters in config (uniform / exponential /
   binomial / …) instead of always uniform. → **Decision needed.**

5. **Reporting must be stratified over every condition.** Final test-set metrics
   broken out for all conditions — target present, target absent, interruptions —
   never collapsed into one aggregate. Interpretability to be treated as strict.

   → **Clarify:** "interruptions" — high-overlap trials, or a new trial type?
   Nothing marks an interruption today; the nearest column is `overlap_achieved`.
   Overlaps with B9, which proposes adding turn-taking and target-only trials.

6. **EDA per parameter, to verify the realised distribution is the intended one.**
   The manifest notebook checks ranges (§3) and shortcuts (§7) but never plots
   each parameter's realised shape against what was asked for.

**Own action:** add a plain-English comment to every parameter in
`experiments/configs/generator.yaml` — what it controls, and a realistic
average-case value.

→ **Notes:**

