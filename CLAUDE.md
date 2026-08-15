# Project: Real-time Target Speaker Extraction (masters research project)

## Context
The objective is to build a streaming TSE model that maximises how
accurately a **live speech-to-speech model** (Gemini Live and similar)
recovers what the target speaker said from a mixture. We are optimising
for downstream live-model content fidelity, NOT for the perceptual or
signal quality of the separated audio. Quality may help, but it is not
the target. See docs/decisions/specification.md for the full brief (esp. note 10).

Working setup:
1. **Metric first.** The primary contribution is a defined, gaming-resistant
   metric for live-model content fidelity, plus the harness that computes
   it. See docs/data/metric-definitions.md.
2. **Train with differentiable proxies; judge with the live model.** A live
   API model cannot be backpropagated through, so it is a held-out judge
   only. Training uses differentiable proxies (frozen-ASR/SSL feature
   matching, optionally ASR cross-entropy, speaker and VAD terms).
   **The proxy must be a different model family from the judge** — training
   against your own evaluator makes the benchmark meaningless.
3. **Data is ours.** Training and primary eval are constructed mixtures
   (LibriSpeech-derived + real noise/reverb), because differentiable
   proxies need a clean target signal and exact ground-truth text — neither
   of which real conversational corpora provide. A smaller AMI-derived set
   is the secondary real-audio transfer check.
4. **Server-class compute is assumed.** On-device / small-model deployment
   is explicitly out of scope. Latency is a secondary objective with a
   ~200-300 ms streaming budget.
5. **The model outputs audio.** The live model accepts text too (spec note
   10), but a text path costs a whole ASR decode plus endpointing inside the
   latency budget, and it dissolves the research question — there are no
   audio artefacts to mishear if you hand the judge text. Text is therefore
   measured as a *reference condition* in the benchmark (extractor →
   off-the-shelf ASR → text → judge), never optimised for and never built.
   See docs/decisions/decisions.md and docs/data/metric-definitions.md §3.5.
6. **Two speakers, not multi-party.** A trial is the target, at most one
   other speaker, and noise. Two simultaneous non-target speakers never
   occur — a declared boundary of the task, decided 2026-08-14, not a gap to
   be filled. Do not propose adding a third talker; that decision was taken
   with the cost measured. Consequence to carry: "two overlapping voices"
   proves the target is present in our data, our own eval cannot detect a
   model exploiting that, and every claim must say *two-speaker mixtures*,
   never "conversation". See docs/decisions/decisions.md 2026-08-14.

We are NOT replicating the REAL-TSE Challenge baselines or its eval
pipeline (spec note 8). We borrow ideas, data-construction methods and
metric-design lessons from it, and cite it as the anchor benchmark for
real conversational TSE.

## Non-negotiable rules
- NEVER commit directly to main. Always branch, then open a PR.
- Every experiment must have a YAML config in experiments/configs/ — no
  hardcoded hyperparameters in source files.
- Every experiment result gets logged in experiments/results/ with: the
  config used, the git commit hash, the metrics, and the date.
- Set and log a random seed for every run.
- Any script or job taking over a minute gets a row in docs/run_times.md:
  date, command, scope, wall time. One line, no prose. Never estimate a
  runtime in conversation without checking that file first, and never
  report a projection as if it were measured.
- When implementing something from a paper, cite it (author, year) in a
  code comment at the top of the file/function.
- After any nontrivial change, explain in plain language what the code
  does and why — I need to be able to defend every line in my thesis.
- Never present our numbers as comparable to published REAL-TSE results.
  Different data, different metric, different protocol. Any borrowed
  method or metric must be cited as borrowed, and the difference noted in
  a code comment and in docs/decisions/decisions.md.
- Every live-model (judge) result must record the exact model ID, the
  exact prompt, the input modality (audio or text), and the run date.
  Closed models change silently, so comparisons across dates are invalid
  unless re-run.
- Never compare an audio-input judge result to a text-input one without
  stating that in the text condition the judge is close to a pass-through,
  so the number mostly reflects the front-end ASR, not the judge's
  listening.
- The judge model must never appear anywhere in the training loop, in any
  form, including as a proxy or a data filter.
- Prefer small, single-purpose PRs over large ones.
- When generating text/markdown files, I do not want long explanations. Short, concise answers are
  always preferred unless expressly asked otherwise.
- If needing a format for what to do always begin with the instruction first,
  highlighted, or made clear, followed by the short explanation as to why it
  is required only if necessary.

## Current phase
Phase 2: data preparation
