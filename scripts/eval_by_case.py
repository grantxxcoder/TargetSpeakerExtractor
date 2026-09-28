#!/usr/bin/env python3
"""Score one system on EVERY trial case separately, with the right metric per case.

    ../tse_venv/bin/python scripts/eval_by_case.py \
        --est experiments/results/2026-09-04-train-sir0-10000 \
        --split sir0_val --listener asr \
        --out experiments/results/<date>-case-suite-<tag>

WHY THIS EXISTS. Every content number this project has reported is on the
`both` condition. `both` is 51.5 % of `sir0_val` and 50.8 % of `sir0_privval`;
the other half of the data has never been scored on content at all. So we do
not know whether the model damages clean speech, whether it leaks a whole
speaker when the target never spoke, or whether it invents words out of noise
-- and those are three different failures with three different fixes.

THE CASES ARE NOT SCORED THE SAME WAY, AND THAT IS THE POINT.
The four conditions differ in whether the target spoke at all, so a single
metric cannot serve them. Word error against an empty reference is degenerate:
every emitted word is an insertion, so the score saturates at 100 % and stops
discriminating.

  | case              | target | interferer | the question                     |
  |-------------------|--------|------------|----------------------------------|
  | `target_only`     | yes    | no         | do we DAMAGE speech that needed  |
  |                   |        |            | no separation?                   |
  | `both`            | yes    | yes        | can we separate? (the only case  |
  |                   |        |            | measured before today)           |
  | `interferer_only` | no     | yes        | do we pass a whole stranger      |
  |                   |        |            | through as if it were the target?|
  | `noise_only`      | no     | no         | do we invent speech from noise?  |

PRESENT cases (`target_only`, `both`) are scored on LCF-WER and its error
breakdown, ICR@2 and leakage -- the established battery, against the floor
(raw mixture) and ceiling (clean target) measured ON THAT CASE. Anchors are
NOT transferable between cases: `target_only`'s floor is far easier than
`both`'s, because there is no interferer to confuse the listener.

ABSENT cases (`interferer_only`, `noise_only`) are scored on three things
instead, because there is no target script to align against:
  - SUPPRESSION, dB: output RMS against mixture RMS. More negative is better.
    This is the signal-domain measure of "did the system stay quiet".
  - FALSE-ALARM WORDS: how many words the listener emitted when the target
    said nothing. Zero is correct. This is the number that matters, because
    the judge's inserted words are indistinguishable from real content to a
    downstream consumer.
  - NON-RESPONSE RATE: the fraction of clips the listener declined to
    transcribe. On an absent case a HIGH rate is CORRECT behaviour, which is
    the opposite of its reading on a present case -- so it is reported with
    its direction stated, never pooled across cases.

WHY SUPPRESSION IS MEASURED AGAINST THE MIXTURE AND NOT IN ABSOLUTE dB.
Clips differ in level by construction (BS.1770 loudness targeting per
`decisions-m0.md` 2026-08-12), so an absolute output RMS is not comparable
across trials. The ratio to the input is.

THE `both` CASE IS ALSO SPLIT BY SIR. `decisions-m3.md` 2026-09-12 measured
the two ends of the SIR range as different problems -- at SIR < -5 the model
barely improves on doing nothing while leakage is 65 % of its wrong words, and
at SIR >= +5 leakage is 8 % and the remaining error is the model's own damage.
Reporting one blended `both` number hides that, so this script never does.

CORPUS WER, NOT PER-TRIAL WER -- AND THE 2026-09-12 SIR TABLE IS THE OTHER ONE.
`compute_lcf_wer` aggregates errors and reference words over the whole subset
(total errors / total reference words), so a long clip weighs more than a short
one. `decisions-m3.md` 2026-09-12's per-SIR-band table averaged PER-TRIAL WER
instead, which is why its numbers differ from this script's on the same trials
(e.g. SIR < -5: 89.5/88.2 there against 80.29/76.94 here). Both are correct;
they answer different questions and MUST NOT be quoted against each other.
Prefer headroom captured -- (floor - ours) / (floor - ceiling) -- when comparing
across bands, because the bands differ in how much room there is to win.

THE SPEECH GATE CAN OWN AN ABSENT-CASE RESULT ENTIRELY. Silero answers
speech-free clips locally without calling the listener, so a clean `noise_only`
score may be the gate's doing. Measured 2026-09-21: the gate blocked 8/8
`noise_only` clips for BOTH the mixture and our output, so that row says nothing
about the model. Always read `gate_blocked` in results.json before crediting a
system for silence.

NOT A NEW METRIC. Every content number comes from the same
`src/live_model_metric/evaluate.py` the rest of the project uses, at the same
settings; this script chooses which of them to report per case and adds the two
signal-domain readings the absent cases need. Anything here is comparable to a
previous `both` number and to nothing else.

CITATION / BORROWED-WITH-A-DIFFERENCE. The per-condition reporting structure
follows B13 (`decisions-m0.md` 2026-08-13, stratified reporting per condition,
no combinations). The absent-case framing -- suppression plus false-alarm
content rather than word error -- is ours; the REAL-TSE Challenge scores target
-absent segments inside one blended utterance-level TER, which this
deliberately does not do. Different data, different metric, different protocol:
nothing here is comparable to a published REAL-TSE number.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.live_model_metric.evaluate import (ASR, JUDGE, TRANSCRIPT_CACHE,  # noqa: E402
                                            load_trials, evaluate)
from src.run_log import timed  # noqa: E402

PRESENT_CASES = ("target_only", "both")
ABSENT_CASES = ("interferer_only", "noise_only")
ALL_CASES = PRESENT_CASES + ABSENT_CASES

# decisions-m3.md 2026-09-12. Edges are inclusive-low, exclusive-high.
SIR_BANDS = ((None, -5.0), (-5.0, 0.0), (0.0, 5.0), (5.0, None))


def band_label(lo, hi):
    if lo is None:
        return f"< {hi:+.0f}"
    if hi is None:
        return f">= {lo:+.0f}"
    return f"{lo:+.0f}..{hi:+.0f}"


def band_of(sir_db):
    if sir_db is None:
        return None
    for lo, hi in SIR_BANDS:
        if (lo is None or sir_db >= lo) and (hi is None or sir_db < hi):
            return band_label(lo, hi)
    return None


def rms(path):
    """Root-mean-square amplitude of a wav, or None if it is missing/empty."""
    if path is None or not Path(path).exists():
        return None
    audio, _ = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(audio))))


def suppression_db(estimate_path, mixture_path, floor=1e-12):
    """20*log10(RMS(estimate) / RMS(mixture)). MORE NEGATIVE IS BETTER here.

    On a target-absent trial the correct output is silence, so this says how
    far below its input the system pushed the clip. It is deliberately a ratio
    and not an absolute level -- see the module docstring.
    """
    a, b = rms(estimate_path), rms(mixture_path)
    if a is None or b is None or b <= floor:
        return None
    return 20.0 * np.log10(max(a, floor) / b)


def word_count(text):
    return len(str(text or "").split())


def score_absent_case(case, split, estimate_directory, listener, limit,
                      data_root, manifest_dir, judge_kwargs, cached_only,
                      verbose):
    """Suppression, false-alarm words and non-response, for a target-absent case.

    The listener still runs -- the whole question is what it TRANSCRIBES from a
    clip whose target said nothing -- but no alignment is possible, so the
    content battery is not called. Words emitted IS the error.
    """
    trials = load_trials(split, case, limit, data_root, manifest_dir,
                         estimate_directory, REPO_ROOT)
    if not trials:
        return None

    rows = {}
    for system, pick in (("floor", lambda t: t.mixture),
                         ("estimate", lambda t: t.estimate)):
        if system == "estimate" and estimate_directory is None:
            continue
        paths = [pick(t) for t in trials]

        # Reuse the project's own listening path so the transcripts land in the
        # same cache and the same speech gate applies. `evaluate` is called with
        # content metrics only; its word-error fields are meaningless here and
        # are dropped, but `responses` is exactly what we need.
        from src.live_model_metric.evaluate import _listen
        responses, gate_decisions, used_judge = _listen(
            paths, listener, split, manifest_dir, REPO_ROOT, TRANSCRIPT_CACHE,
            not cached_only, True, judge_kwargs, verbose)

        counts = [word_count(r) for r in responses]
        supp = [suppression_db(p, t.mixture) for p, t in zip(paths, trials)]
        supp = [s for s in supp if s is not None]

        rows[system] = {
            "n_trials": len(trials),
            "false_alarm_words_total": int(sum(counts)),
            "false_alarm_words_per_trial": float(np.mean(counts)) if counts else None,
            "clips_with_any_word": int(sum(1 for c in counts if c > 0)),
            "clips_with_any_word_pct": 100.0 * sum(1 for c in counts if c > 0) / len(counts),
            "non_response_rate_pct": 100.0 * sum(1 for c in counts if c == 0) / len(counts),
            "suppression_db_mean": float(np.mean(supp)) if supp else None,
            "suppression_db_median": float(np.median(supp)) if supp else None,
            "gate_blocked": sum(1 for d in gate_decisions if d.fired),
        }
        if used_judge is not None:
            rows[system]["judge_calls"] = used_judge.calls_made
            rows[system]["judge_cache_hits"] = used_judge.cache_hits
    return rows


def score_present_case(case, split, estimate_directory, listener, limit,
                       data_root, manifest_dir, judge_kwargs, cached_only,
                       verbose, systems):
    """The established content battery, with anchors measured on THIS case."""
    results = evaluate(split=split, condition=case,
                       estimate_directory=estimate_directory,
                       systems=systems, metrics=("content",), limit=limit,
                       data_root=data_root, manifest_dir=manifest_dir,
                       allow_new_transcripts=not cached_only, verbose=verbose,
                       repo_root=REPO_ROOT, listener=listener,
                       judge_kwargs=judge_kwargs)
    return {"n_trials": results.n_trials,
            "scores": dict(results.systems),
            "provenance": dict(results.provenance)}


def sir_breakdown(split, estimate_directory, listener, limit, data_root,
                  manifest_dir, judge_kwargs, cached_only, verbose):
    """`both`, split by SIR band. Never report `both` as one blended number."""
    trials = load_trials(split, "both", limit, data_root, manifest_dir,
                         estimate_directory, REPO_ROOT)
    by_band = {}
    for t in trials:
        label = band_of(t.signal_to_interference_db)
        if label is not None:
            by_band.setdefault(label, []).append(t.trial_id)

    from src.live_model_metric.evaluate import _listen
    from src.live_model_metric.lcf_wer import compute_lcf_wer

    out = {}
    for label, ids in by_band.items():
        subset = [t for t in trials if t.trial_id in set(ids)]
        row = {"n_trials": len(subset)}
        for system, pick in (("floor", lambda t: t.mixture),
                             ("estimate", lambda t: t.estimate),
                             ("ceiling", lambda t: t.clean)):
            if system == "estimate" and estimate_directory is None:
                continue
            paths = [pick(t) for t in subset]
            responses, _, _ = _listen(paths, listener, split, manifest_dir,
                                      REPO_ROOT, TRANSCRIPT_CACHE,
                                      not cached_only, True, judge_kwargs,
                                      verbose)
            word = compute_lcf_wer([t.target_text for t in subset], responses)
            row[system] = {"lcf_wer": word.word_error_rate,
                           "deletions": word.deletion_rate,
                           "insertions": word.insertion_rate,
                           "substitutions": word.substitution_rate}
        out[label] = row
    return out


def render(payload):
    """A flat text table. Present and absent cases are NEVER pooled."""
    lines = []
    add = lines.append
    add("")
    add(f"  CASE SUITE  split={payload['split']}  listener={payload['listener']}")
    add(f"  estimates: {payload['estimate_directory']}")
    add("")
    add("  PRESENT cases -- target spoke. Lower LCF-WER is better.")
    add("  " + "-" * 74)
    add(f"  {'case':<16}{'system':<10}{'n':>5}{'LCF-WER':>10}{'ICR@2':>9}{'leak%':>8}{'FR@2':>8}")
    for case in PRESENT_CASES:
        block = payload["cases"].get(case)
        if not block:
            continue
        for system, sc in (block.get("scores") or {}).items():
            add(f"  {case:<16}{system:<10}{block.get('n_trials', 0):>5}"
                f"{_fmt(sc.get('lcf_wer')):>10}{_fmt(sc.get('icr_at_2')):>9}"
                f"{_fmt(sc.get('mean_leak')):>8}{_fmt(sc.get('fr_at_2')):>8}")
    add("")
    add("  ABSENT cases -- target said NOTHING. Zero words is correct.")
    add("  Suppression: MORE NEGATIVE is better. Non-response: HIGHER is better here.")
    add("  " + "-" * 74)
    add(f"  {'case':<16}{'system':<10}{'n':>5}{'words/trial':>13}{'clips w/words':>15}{'supp dB':>10}")
    for case in ABSENT_CASES:
        block = payload["cases"].get(case)
        if not block:
            continue
        for system, sc in block.items():
            if not isinstance(sc, dict):
                continue
            add(f"  {case:<16}{system:<10}{sc.get('n_trials', 0):>5}"
                f"{_fmt(sc.get('false_alarm_words_per_trial')):>13}"
                f"{_fmt(sc.get('clips_with_any_word_pct')):>14}%"
                f"{_fmt(sc.get('suppression_db_mean')):>10}")
    if payload.get("both_by_sir"):
        add("")
        add("  `both`, BY SIR BAND. The two ends are different problems.")
        add("  " + "-" * 74)
        add(f"  {'SIR band':<12}{'n':>5}{'floor':>10}{'ours':>10}{'ceiling':>10}")
        for label in (band_label(*b) for b in SIR_BANDS):
            row = payload["both_by_sir"].get(label)
            if not row:
                continue
            add(f"  {label:<12}{row.get('n_trials', 0):>5}"
                f"{_fmt((row.get('floor') or {}).get('lcf_wer')):>10}"
                f"{_fmt((row.get('estimate') or {}).get('lcf_wer')):>10}"
                f"{_fmt((row.get('ceiling') or {}).get('lcf_wer')):>10}")
    add("")
    add("  Anchors are per-case and NOT transferable between cases.")
    add("  Nothing here is comparable to a published REAL-TSE number.")
    add("")
    return "\n".join(lines)


def _fmt(value):
    return "--" if value is None else f"{value:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", "--estimate-directory", dest="estimate_directory",
                    default=None, help="directory of rendered estimate.wav files")
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--cases", default=",".join(ALL_CASES))
    ap.add_argument("--listener", default=ASR, choices=[ASR, JUDGE])
    ap.add_argument("--limit", type=int, default=None,
                    help="per case, not in total")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--cached-only", action="store_true",
                    help="never transcribe anything new; fail soft instead")
    ap.add_argument("--no-sir-breakdown", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-rpm", type=int, default=10)
    ap.add_argument("--judge-max-new-calls", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42,
                    help="recorded for provenance; this script is deterministic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in ALL_CASES]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; known: {list(ALL_CASES)}")

    judge_kwargs = None
    if args.listener == JUDGE:
        # Judge.__init__'s name, as scripts/evaluate.py passes it. Was `rpm`,
        # which raised TypeError before the first call -- this path had never run.
        judge_kwargs = {"requests_per_minute": args.judge_rpm,
                        "max_new_calls": args.judge_max_new_calls}
        if args.judge_model:
            judge_kwargs["model_id"] = args.judge_model

    payload = {
        "date": date.today().isoformat(),
        "split": args.split,
        "listener": args.listener,
        "seed": args.seed,
        "estimate_directory": args.estimate_directory,
        "cases": {},
    }

    with timed(f"scripts/eval_by_case.py --split {args.split} "
               f"--cases {','.join(cases)} --listener {args.listener}",
               scope=lambda: f"{len(cases)} cases, {args.split}, "
                             f"listener {args.listener}"):
        for case in cases:
            print(f"\n######## {case} ########", flush=True)
            if case in PRESENT_CASES:
                payload["cases"][case] = score_present_case(
                    case, args.split, args.estimate_directory, args.listener,
                    args.limit, args.data_root, args.manifest_dir,
                    judge_kwargs, args.cached_only, True,
                    ("floor", "estimate", "ceiling") if args.estimate_directory
                    else ("floor", "ceiling"))
            else:
                payload["cases"][case] = score_absent_case(
                    case, args.split, args.estimate_directory, args.listener,
                    args.limit, args.data_root, args.manifest_dir,
                    judge_kwargs, args.cached_only, True)

        if "both" in cases and not args.no_sir_breakdown:
            print("\n######## both, by SIR band ########", flush=True)
            payload["both_by_sir"] = sir_breakdown(
                args.split, args.estimate_directory, args.listener, args.limit,
                args.data_root, args.manifest_dir, judge_kwargs,
                args.cached_only, False)

    text = render(payload)
    print(text)

    if args.out:
        out = REPO_ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(payload, indent=1, default=str))
        (out / "results.txt").write_text(text)
        (out / "meta.yaml").write_text(yaml.safe_dump({
            "date": payload["date"],
            "script": "scripts/eval_by_case.py",
            "seed": args.seed,
            "split": args.split,
            "cases": cases,
            "listener": args.listener,
            "estimate_directory": args.estimate_directory,
            "limit": args.limit,
        }, sort_keys=False))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
