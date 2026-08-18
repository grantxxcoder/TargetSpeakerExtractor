# Decisions needed before audio can be generated

**Written 2026-08-10. Status 2026-08-13: one item left open.**

- **C2** — how hard the task should be (floor word error rate on the raw mixture).
  Needs the supervisor. Blocks nothing that can be done meanwhile.

**Every other decision is made** and recorded in `docs/decisions/decisions-m0.md`. Group A
is closed, so the renderer is unblocked. What remains is implementation, not choices:
B12's two PRs, then the manifest rebuild carrying B9, B10, B4 and the interruption
column.

Plain wording first, the jargon term in brackets so it can be matched to
`docs/data/data-construction-parameters.md`.

Move each answer into `docs/decisions/decisions-m0.md` once made.

---

## A. Blocks the renderer

### A1. *Decided 2026-08-13 — full reverberant, "what the mic heard". See `decisions-m0.md`.*

Pending supervisor sign-off. Dereverberation is an ablation only, if time allows.

### A2. *Decided 2026-08-11 — wrap around. See `decisions-m0.md`.*

Numbers are kept in `decisions-m0.md`. The gap in numbering is deliberate: A3-A6
keep their identifiers so earlier references stay valid.

### A3. *Decided 2026-08-12 — BS.1770 integrated loudness. See `decisions-m0.md`.*

### A4. *Decided 2026-08-12 — no room on the enrollment. See `decisions-m0.md`.*

### A5. *Decided 2026-08-13 — yes, pad by `t60_s`. See `decisions-m0.md`.*

### A6. *Decided 2026-08-13 — common-gain rescale at 0.95. See `decisions-m0.md`.*

**Group A is closed.** The renderer is unblocked.

---

## B. Needed before the real sets are generated

### B1. *Closed 2026-08-13 — subsumed by the difficulty dial. See `decisions-m0.md`.*

Not a standalone decision: `overlap_ratio` is one of the 14 parameters ranked in
`docs/data/difficulty-dial.md`, adjustable on request once B12 lands. Recorded there
as the narrowing to do **last**, because its 0.7 ceiling is deliberately matched to
REAL-TSE and changing it diverges from the anchor.

### B2. *Decided 2026-08-13 — measure from detected speech. See `decisions-m0.md`.*

Detector to be named in the PR that adds it.

### B3. *Decided 2026-08-12 — fixed 5 s, kept configurable. See `decisions-m0.md`.*

### B4. *Decided 2026-08-13 — yes, same fraction as train, scored on their own row. See `decisions-m0.md`.*

The fraction itself follows B9.

### B5. *Decided 2026-08-13 — Whisper `EnglishTextNormalizer`. See `decisions-m0.md`.*

### B6. *Decided 2026-08-13 — 500 generated, 200 the minimum scored. See `decisions-m0.md`.*

### B7. *Decided 2026-08-13 — off for the main run, kept as a switch. See `decisions-m0.md`.*

### B8. *Decided 2026-08-11 — different book. See `decisions-m0.md`.*

### B9. *Decided 2026-08-13 — 50 % both / 25 % absent / 25 % target-only, and a variable target activity ratio. See `decisions-m0.md`.*

Blocks the manifest rebuild until implemented. Sets B4's eval fraction at 0.25.

### B10. *Decided 2026-08-13 — three enrollment tiers, recorded per trial; eval pools redrawn. See `decisions-m0.md`.*

Not a reversal of B8: B8's cost note specified this fallback and its trigger. Folds
into the PR3 rebuild.

### B11. *Decided 2026-08-13 — report a latency decay curve, never cap T60. See `decisions-m0.md`.*

Largely defused by A1: with a full reverberant reference the model is no longer asked
to suppress a tail it has not heard.

### B12. *Architecture decided 2026-08-13 — two regimes, sampler layer, no relational constraints. See `decisions-m0.md`.*

**PR1 and PR2 landed 2026-08-14.** `src/data/sampling.py` holds the sampler;
`build_manifest.py` draws a regime per trial and records it. PR3 (B9/B10/B4 rebuild)
is next.

One band was deliberately not applied: `overlap_ratio` keeps its full `[0.2, 0.7]`
in `base`, because narrowing it diverges from the REAL-TSE anchor and needs
supervisor agreement (`difficulty-dial.md` §3). `target_activity_ratio` likewise
stays fixed until B9 decides what varying it means.

Band values live in `docs/data/difficulty-dial.md` §2; the how-to is
`docs/data/changing-the-data.md`.

Two sub-questions the original entry raised, both now answered in `decisions-m0.md`:
beta is dropped (no use case), and the wall-absorption ambiguity is resolved by
**not** capping absorption — it is derived from `t60_s` and volume, so a cap would
be a rejection rule, and rejection is what bends distributions. Raising the `t60_s`
floor achieves the same realism gain without rejection (`difficulty-dial.md` §1).

### B13. *Decided 2026-08-13 — per condition, no combinations, 100 trials per bucket. See `decisions-m0.md`.*

One part deferred: the **interruption** condition. Nothing marks an interruption
today, and defining one needs the turn-taking trials B9 introduces, so it is fixed
during that rebuild rather than before it.

---

## C. Ask the supervisor

1. **A1 — decided, needs sign-off only.** Reference is what the mic heard from the
   target (full reverberant): separate and denoise, do not dereverberate. Removing a
   0.6 s tail from a 300 ms causal window is not possible, and attempting it trades
   residue for artefacts, which are what degrade recognition most. Dereverberation
   kept as an ablation if time allows. `decisions-m0.md` 2026-08-13.
2. **How hard should the task be?** Measured as how badly an off-the-shelf
   transcriber does on the raw mixture (**floor word error rate**). Too easy and
   nothing distinguishes systems; too hard and nothing can be ranked. The current
   plan targets 60–80 %. **Still open — nothing else here can settle it.**
3. *B10 — decided 2026-08-13. Three enrollment tiers, recorded per trial. Not a
   reversal of B8: B8's own cost note specified this fallback and the trigger for it,
   and 60.2 % of speakers dropping out met that trigger. Worth mentioning, not
   asking. See `decisions-m0.md`.*
4. *B4 — answered 2026-08-12 and now fully decided. See `decisions-m0.md`.*
5. *B11 — decided 2026-08-13, and largely defused by A1. See `decisions-m0.md`.*
