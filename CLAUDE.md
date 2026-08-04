# Project: Real-time Target Speaker Extraction (masters research project)

## Context
Replicating REAL-TSE Challenge online-track baselines and eval pipeline,
then exploring a novel low-latency, interpretable TSE architecture.
See docs/specification.md for the full brief.

Note: official REAL-TSE dev/eval data is not available to us (challenge
registration closed before we could register). We are instead:
1. Training/replicating baselines on Libri2Mix-100 + WHAM! (the same data
   the official baselines were trained on).
2. Building our own REAL-T-style eval set from the public training splits
   of AMI (and later AliMeeting/AISHELL-4), following the same
   overlapping-segment extraction method the challenge organizers describe.
3. Using the official REAL-TSE-Challenge eval pipeline code for scoring,
   so our numbers are methodologically comparable even though the eval
   audio itself differs.

## Non-negotiable rules
- NEVER commit directly to main. Always branch, then open a PR.
- Every experiment must have a YAML config in experiments/configs/ — no
  hardcoded hyperparameters in source files.
- Every experiment result gets logged in experiments/results/ with: the
  config used, the git commit hash, the metrics, and the date.
- Set and log a random seed for every run.
- When implementing something from a paper, cite it (author, year) in a
  code comment at the top of the file/function.
- After any nontrivial change, explain in plain language what the code
  does and why — I need to be able to defend every line in my thesis.
- Any place our evaluation setup differs from the official REAL-TSE
  protocol (different eval audio, etc.) must be noted in a comment and in
  docs/decisions.md — never presented as identical without caveat.
- Prefer small, single-purpose PRs over large ones.

## Stack
Python, PyTorch

## Current phase
Phase 1: literature review
