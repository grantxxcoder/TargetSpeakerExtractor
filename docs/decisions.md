# Decision Log

## 2026-08-04 — Repo workflow: branch protection + PR review
Chose to enforce PRs on main (no direct commits) even as a solo project,
so every code change gets a deliberate diff-review checkpoint before
becoming part of the "real" codebase — needed since all submitted code
must be individually understood and defensible.

## 2026-08-04 — Eval data: REAL-TSE dev/eval set unavailable
Challenge registration closed 31 May 2026, before we could register, so
we do not have access to the official dev/eval set. Decision: replicate
baseline training on Libri2Mix-100 + WHAM! (identical to the baselines'
own training data), and construct our own REAL-T-style eval set from
public training splits of AMI (then AliMeeting/AISHELL-4), following the
organizers' documented construction method (overlapping segments as
mixtures, ≥5s non-overlapping segments as enrollment). We use the
official REAL-TSE-Challenge scoring code for evaluation to keep numbers
methodologically comparable. Any comparison to published baseline
numbers will be caveated as "different eval audio, same eval method" —
not treated as a direct apples-to-apples result.
Also emailed organizers [date] to ask about academic access to the
official set; will update if granted.
