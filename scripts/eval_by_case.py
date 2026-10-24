#!/usr/bin/env python3
"""Score one system on every trial case separately, with the right measure for each case.

    ../tse_venv/bin/python scripts/eval_by_case.py --est experiments/results/<estimates> --split sir0_val --listener asr --out experiments/results/<date>-case-suite-<tag>

The four conditions differ in whether the target spoke, so one measure cannot serve them all. Word error against an empty reference counts every word as an insertion and stops telling systems apart.
Where the target speaks (target_only, both) the usual content measures run against that case's own floor and ceiling, because the anchors are not transferable between cases.
Where the target is silent (interferer_only, noise_only) there is nothing to align against, so this measures how far below the mixture the output was pushed (more negative is better)
and how many words the listener reported (zero is correct). Suppression is a ratio to the mixture because clips differ in level by construction.
The speech gate answers speech-free clips without calling the listener, so a clean noise_only score can be the gate's doing. Always read gate_blocked before crediting a system for silence.
The per-condition reporting follows B13 (decisions-m0.md 2026-08-13). The target-absent measures are ours, and nothing here is comparable to a published REAL-TSE number.
"""

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
from src.live_model_metric.evaluate import ASR, JUDGE, evaluate, listen, load_trials, system_path
from src.run_log import timed

PRESENT_CASES = ("target_only", "both")
ABSENT_CASES = ("interferer_only", "noise_only")
ALL_CASES = PRESENT_CASES + ABSENT_CASES

# Keeps the suppression ratio finite when a clip is silent.
RMS_FLOOR = 1e-12
# Recorded for provenance only, because this script draws no random numbers.
SEED = 42


# this gives the root-mean-square amplitude of a wav file, or None if it is missing or empty
def rms(path):
    if path is None or not Path(path).exists():
        return None
    audio, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(audio))))


# this gives how far below the mixture the output was pushed, 20 log10 of the RMS ratio. More negative is better on a target-absent trial
def suppression_db(estimate_path, mixture_path):
    estimate_rms = rms(estimate_path)
    mixture_rms = rms(mixture_path)
    if estimate_rms is None or mixture_rms is None or mixture_rms <= RMS_FLOOR:
        return None
    return 20.0 * np.log10(max(estimate_rms, RMS_FLOOR) / mixture_rms)


# this scores a target-absent case: the words the listener reported when the target said nothing, and the suppression
def score_absent_case(case, split, estimate_directory, listener):
    """The listener still runs, because the question is what it reports from a clip whose target said nothing. A failed judge call scores as an empty answer, which here reads as a correct zero, so failures are recorded."""
    trials = load_trials(split, case, estimate_directory)
    if not trials:
        return None
    systems = ["floor"]
    if estimate_directory is not None:
        systems.append("estimate")

    rows = {}
    for system in systems:
        paths = []
        for trial in trials:
            paths.append(system_path(trial, system))
        responses, gate_decisions, used_judge = listen(paths, listener, split, 0, "small.en")

        word_counts = []
        for response in responses:
            word_counts.append(len(str(response or "").split()))
        suppressions = []
        for path, trial in zip(paths, trials):
            suppression = suppression_db(path, trial["mixture"])
            if suppression is not None:
                suppressions.append(suppression)
        clips_with_words = 0
        gate_blocked = 0
        for count in word_counts:
            if count > 0:
                clips_with_words += 1
        for decision in gate_decisions:
            if decision["fired"]:
                gate_blocked += 1

        rows[system] = {
            "n_trials": len(trials),
            "false_alarm_words_per_trial": float(np.mean(word_counts)),
            "clips_with_any_word_pct": 100.0 * clips_with_words / len(word_counts),
            "non_response_rate_pct": 100.0 * (len(word_counts) - clips_with_words) / len(word_counts),
            "suppression_db_mean": float(np.mean(suppressions)) if suppressions else None,
            "gate_blocked": gate_blocked,
        }
        if used_judge is not None:
            rows[system]["judge_calls"] = used_judge["calls_made"]
            rows[system]["judge_cache_hits"] = used_judge["cache_hits"]
            rows[system]["judge_failed"] = len(used_judge["failures"])
    return rows


# this scores a target-present case with the usual content measures, against that case's own floor and ceiling
def score_present_case(case, split, estimate_directory, listener):
    systems = ("floor", "ceiling")
    if estimate_directory:
        systems = ("floor", "estimate", "ceiling")
    results = evaluate(split, case, estimate_directory, systems, ("content",), listener, 0, "small.en")
    return {"n_trials": results["n_trials"], "scores": results["systems"], "provenance": results["provenance"]}


# Generated by Claude Code
def format_number(value):
    return "--" if value is None else f"{value:.2f}"


# Generated by Claude Code
def render(payload):
    lines = ["", f"  CASE SUITE  split={payload['split']}  listener={payload['listener']}", f"  estimates: {payload['estimate_directory']}", "", "  PRESENT cases -- target spoke. Lower LCF-WER is better.", "  " + "-" * 74, f"  {'case':<16}{'system':<10}{'n':>5}{'LCF-WER':>10}{'ICR@2':>9}{'leak%':>8}{'FR@2':>8}"]
    for case in PRESENT_CASES:
        block = payload["cases"].get(case)
        if not block:
            continue
        for system, scores in block["scores"].items():
            lines.append(f"  {case:<16}{system:<10}{block['n_trials']:>5}{format_number(scores.get('lcf_wer')):>10}{format_number(scores.get('icr_at_2')):>9}{format_number(scores.get('mean_leak')):>8}{format_number(scores.get('fr_at_2')):>8}")
    lines += ["", "  ABSENT cases -- target said NOTHING. Zero words is correct.", "  Suppression: MORE NEGATIVE is better. Non-response: HIGHER is better here.", "  " + "-" * 74, f"  {'case':<16}{'system':<10}{'n':>5}{'words/trial':>13}{'clips w/words':>15}{'supp dB':>10}"]
    failed = 0
    for case in ABSENT_CASES:
        block = payload["cases"].get(case)
        if not block:
            continue
        for system, scores in block.items():
            failed += scores.get("judge_failed", 0)
            lines.append(f"  {case:<16}{system:<10}{scores['n_trials']:>5}{format_number(scores['false_alarm_words_per_trial']):>13}{format_number(scores['clips_with_any_word_pct']):>14}%{format_number(scores['suppression_db_mean']):>10}")
    if failed:
        lines += ["", f"  WARNING: {failed} listener call(s) FAILED and were scored as empty.", "  On an absent case that reads as perfect silence. Do not report."]
    lines += ["", "  Anchors are per-case and NOT transferable between cases.", "  Nothing here is comparable to a published REAL-TSE number.", ""]
    return "\n".join(lines)


# this is where the case suite starts
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--est", "--estimate-directory", dest="estimate_directory", default=None, help="the folder of rendered estimate.wav files")
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--listener", default=ASR, choices=[ASR, JUDGE])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = {"date": date.today().isoformat(), "split": args.split, "listener": args.listener, "seed": SEED, "estimate_directory": args.estimate_directory, "cases": {}}
    with timed(f"scripts/eval_by_case.py --split {args.split} --listener {args.listener}", scope=lambda: f"{len(ALL_CASES)} cases, {args.split}, listener {args.listener}"):
        for case in ALL_CASES:
            print(f"\n######## {case} ########", flush=True)
            if case in PRESENT_CASES:
                payload["cases"][case] = score_present_case(case, args.split, args.estimate_directory, args.listener)
            else:
                payload["cases"][case] = score_absent_case(case, args.split, args.estimate_directory, args.listener)

    text = render(payload)
    print(text)
    if args.out:
        out = REPO_ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(payload, indent=1, default=str))
        (out / "results.txt").write_text(text)
        (out / "meta.yaml").write_text(yaml.safe_dump({"date": payload["date"], "script": "scripts/eval_by_case.py", "seed": SEED, "split": args.split, "cases": list(ALL_CASES), "listener": args.listener, "estimate_directory": args.estimate_directory}, sort_keys=False))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
