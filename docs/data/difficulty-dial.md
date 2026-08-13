# Difficulty dial: which parameters to narrow, in order

**Written 2026-08-13.** Companion to `data-construction-parameters.md`, which says
what each parameter *is*. This says which ones to **narrow first** when the task is
too hard to learn from, and what realism each narrowing costs.

Two orderings, because they disagree — that disagreement is the useful part.
Percentages are measured on `data/manifests/train.csv` (19,569 rows).

---

## 1. Reverberation time in room terms

`t60_s` is the time echoes take to fade by 60 dB. Mid-frequency, occupied.

| T60 | Room |
|---|---|
| 0.0 s | Not physical. Free field / direct path only — no room exists with this |
| 0.1 s | Anechoic chamber; vocal booth; car interior. **This, not "soundproof", is the dead room** — soundproofing blocks sound entering, it does not shorten decay |
| 0.2 s | Broadcast/voiceover booth, small treated control room, edit suite |
| 0.3 s | Furnished living room, small private office, small carpeted meeting room |
| 0.4 s | Carpeted open-plan office, medium meeting room, ordinary classroom |
| 0.5 s | Larger conference room, classroom with a hard floor, treated restaurant |
| 0.6 s | Untreated classroom, lecture room, hard-floored open-plan office, hotel lobby. Upper limit in classroom acoustics standards |
| 0.7 s | Lecture theatre, small auditorium, courtroom, drama theatre |
| 0.8 s | Large lecture hall, hard-surfaced canteen, small church, atrium |
| 0.9 s | **Chapel** (your guess was right), medium church, gymnasium, large canteen |
| 1.0 s | Chamber-music hall, large chapel, station concourse |
| 1.5 s | Large church, untreated sports hall, mall atrium |
| 2.0 s | Symphony concert hall, large stone church, indoor swimming pool |
| 3.0 s+ | Cathedral, underground car park, tunnel. Cathedral naves reach 6–10 s |

**Current range `[0.15, 0.6]` = treated booth → untreated classroom.** Median 0.377.
Nothing above a lecture room is represented, so no auditorium, church or hall.

### T60 alone does not name a room — volume does too

Sabine: `T60 = 0.161·V / (S·α)`, so the same T60 in a bigger room needs more
absorptive walls. Your rooms are 77–395 m³ (median 191). Implied average wall
absorption in `train`:

| | absorption α | means |
|---|---|---|
| p25 | 0.293 | furnished office — plausible |
| p50 | 0.381 | well-furnished, some treatment |
| p75 | 0.545 | acoustically treated room |
| p90 | 0.730 | studio-grade treatment |
| p99 | 0.941 | anechoic-grade |

- **72.4 %** of trials are more absorptive than a furnished office (α > 0.30)
- **30.4 %** need deliberate acoustic treatment (α > 0.50)
- **11.9 %** need studio-grade treatment (α > 0.70)
- **2.1 %** are effectively 100–400 m³ anechoic chambers (α > 0.90)

Because α ∝ 1/T60, these are the **low**-T60 trials. So the unrealistic corner of
the room model is the *easy* end, not the boomy end — a 10×10×4 m room at
T60 0.15 s is not a small dead office, it is a hall lined with acoustic foam.

**Consequence: raising the T60 floor makes the data more realistic *and* harder at
the same time.** The two goals do not trade off here.

Feasibility is already handled — `sample_room` (`scripts/build_manifest.py:148-161`)
redraws when `pra.inverse_sabine` cannot solve, and 0 of 20,819 manifest rows are
infeasible. Verified 2026-08-13. It does mean large-room + short-T60 draws are
silently rejected, which is a rejection bias of the kind §7 of the manifest
notebook exists to catch; re-check it after any change here.

---

## 2. Parameters ordered by how much they hurt the model

Hardest first. "Narrow to" is the proposed learnable base case; the full range then
becomes a harder *reported* condition (B12 + B13), never deleted.

| # | Parameter | Now | Hard end | Damage | Realism of the hard end | Narrow to |
|---|---|---|---|---|---|---|
| 1 | `sir_db` | [-5, 15] | low | **Severe.** The config's own "main difficulty axis". At −5 dB the interferer is *louder* than the target, so nothing but the enrollment can identify which voice to keep | Plausible but uncommon — needs the interferer much closer to the mic | **[0, 12]** |
| 2 | `overlap_ratio` | [0.2, 0.7] | high | **Severe.** 70 % double-talk leaves little clean target speech to anchor on | **Unrealistic for conversation** (~0.10–0.15 in a meeting) but *matched to REAL-TSE at ~0.5 on purpose*. Narrowing = diverging from the anchor, a decision not a tweak | **[0.1, 0.45]** |
| 3 | `target_activity_ratio` | 0.75 fixed | n/a | **Structural, not gradual.** Being *fixed* forces high overlap and makes silent-target trials detectable at AUC 1.000 (B9) | A 14 s unbroken monologue is not conversation | **vary [0.25, 0.85]** |
| 4 | `snr_db` | [0, 20] | low | **High.** At 0 dB the noise bed is as loud as the target. Wordless noise cannot make the judge hear *wrong* words but it masks the right ones | 0 dB is beyond a busy cafe (config says 5–10 dB) | **[8, 20]** |
| 5 | `t60_s` | [0.15, 0.6] | **both** | **Moderate.** Under A1 no dereverberation is demanded, but smearing still degrades separation itself (Maciejewski et al., 2020) | Low end unrealistic (see §1), high end fine | **[0.25, 0.5]** — raises the floor |
| 6 | `source_distance_m` | [0.66, 2.0] | high | **Moderate.** Distance raises reverb-to-direct ratio and lowers direct level | Realistic — 2 m is an ordinary table mic | **[0.66, 1.4]** |
| 7 | `same_gender_fraction` | 0.5 | n/a | **Moderate.** Same-gender pairs are the hard case for speaker conditioning | 0.5 is exactly right for random pairing | **leave. Report stratified (B13)** |
| 8 | `enrollment_length_s` | 5.0 | short | **Moderate.** Less voice evidence. Already at the deliberate worst case (metric minimum) | Realistic | **leave; sweep as an experiment** |
| 9 | `enrollment_eq_prob` | 0.5 | high | **Low.** Channel mismatch on the conditioning path | Realistic — a stored profile is captured elsewhere | **leave** |
| 10 | `room_height_m` | [3.0, 4.0] | high | **Low.** Volume, hence absorption demand | **Config flags it as higher than typical** (2.4–2.7 m is ordinary) | **[2.5, 3.5]** — cheap realism gain |
| 11 | `mic_height_m` | [0.9, 1.8] | — | **Low.** | **Config flags it as higher than typical** (~0.75 m for a table mic) | **[0.7, 1.5]** |
| 12 | `mixture_length_s` | [15, 20] | — | **Negligible** for accuracy; matters for streaming-state tests | Fine | **leave** |
| 13 | `target_loudness_lufs` | [-33, -25] | — | **Negligible.** Absolute level, removed by any front-end normalisation | Fine | **leave** |
| 14 | `room_length_m` / `room_width_m` | [5, 10] | high | **Negligible** on its own; acts through T60 and absorption | Fine | **leave** |

---

## 3. Ordered by realism gained per unit of difficulty removed

The narrowings worth doing *first*, because they cost little realism or actively
improve it:

1. **`t60_s` floor 0.15 → 0.25.** Realism **improves** (drops the foam-lined-hall
   trials), difficulty rises slightly. Free win.
2. **`room_height_m` → [2.5, 3.5], `mic_height_m` → [0.7, 1.5].** Realism improves;
   the config already flags both as higher than typical. Nearly free.
3. **`snr_db` floor 0 → 8.** Removes a condition beyond a busy cafe. Small realism
   loss, decent difficulty relief.
4. **`sir_db` floor −5 → 0.** Biggest single difficulty relief. Real realism cost —
   the interferer genuinely can be the louder voice — so keep the full range as a
   reported condition.
5. **`source_distance_m` ceiling 2.0 → 1.4.** Moderate cost, moderate relief.
6. **`target_activity_ratio` varied (B9).** Realism improves substantially, but
   difficulty goes **up** in the short-utterance corner. Do it for correctness, not
   for relief.
7. **`overlap_ratio` ceiling 0.7 → 0.45.** *Do this last.* It is the only narrowing
   that breaks the deliberate match to REAL-TSE, so it needs a `decisions.md` entry
   and supervisor agreement, not a config edit.

**Never narrow:** `same_gender_fraction`, `enrollment_length_s`,
`target_absent_fraction`. Each is either already the deliberate worst case or the
thing an experimental variable is meant to measure.

---

## 4. Prerequisite

None of §2–§3 is a config edit today: `build_manifest.py` hardcodes
`rng.uniform(lo, hi)`, so config can move a range but not its shape, and there is
no way to express "narrow base case plus wide reported condition" at all. That is
`decisions-pending.md` **B12**, and it blocks using this document. Implement B12
first, then this becomes a config diff and a rebuild.

Both bands should be recorded per trial so any trial can be attributed to base case
or hard condition without recomputing.
