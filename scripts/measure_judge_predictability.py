#!/usr/bin/env python3
"""How well does a LOCAL listener predict the Gemini judge, PER TRIAL?

WHY THIS EXISTS. The 2026-09-15 proposal (decisions-pending.md, "the Gemini
tuner") is to train a local model that predicts what `gemini-3.7-flash` would
transcribe, and to use it as the differentiable training signal the live API
cannot provide. Before any of that is built, this script asks whether the
premise holds at all, using ONLY transcripts already on disk: no API calls, no
model inference, no GPU.

WHAT IT MEASURES. For every (system, trial) that has BOTH a cached judge
transcript and a cached faster-whisper `small.en` transcript OF THE SAME
estimate.wav, it scores each listener's LCF-WER against the trial's target
script (metric-definitions.md 3.1) and correlates them across trials.

HOW TO READ IT. The offline ASR is the strongest local listener the project
already owns, and it was never trained on judge behaviour. Its per-trial
agreement with the judge is therefore the STARTING POINT a trained surrogate
must beat: r^2 is how much of the judge's per-trial error a free local model
already explains, and MAE is how far off the level it is. A surrogate is worth
building to the extent it moves those two numbers -- and the room it has is
bounded above by the judge's own test-retest noise, which is measured
separately (only 3 clips carry repeats today; see the proposal).

CAVEAT, and it is the important one. `small.en` is not an untrained surrogate
-- it is a different, independently-trained ASR. Agreement here mixes "the
judge is predictable" with "two transcribers of English tend to agree". It
bounds the problem; it does not decompose it.

No seed: the computation is deterministic (cache lookups + jiwer alignment).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.live_model_metric.lcf_wer import count_errors, normalise_text  # noqa: E402
from src.live_model_metric import judge as J  # noqa: E402
from src.live_model_metric import evaluate as E  # noqa: E402

# Each entry is a system whose estimates were judged. The judge cache is keyed
# by the audio's own sha256, so naming the directory is enough -- a re-rendered
# checkpoint simply misses the cache rather than serving the wrong answer.
DEFAULT_SYSTEMS = {
    "baseline-5000-e7":  "experiments/results/2026-09-01-est-sir0-5000",
    "wesep":             "experiments/results/2026-09-03-est-wesep-tfmap-causal",
    "baseline-10000-e6": "experiments/results/2026-09-04-train-sir0-10000",
    "struct-e12":        "experiments/results/2026-09-15-est-struct-e12-sir0val",
}


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = average
        i = j + 1
    return out


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b))


def collect(split, systems, verbose=True):
    judge_cache = J.load_cache()
    asr_cache = E._load_cache(REPO_ROOT / "experiments/results/transcripts.csv")
    prompt_sha = J.prompt_sha()
    rendered = REPO_ROOT / "data/rendered" / split

    rows = []
    coverage = {}
    for name, directory in systems.items():
        estimate_root = REPO_ROOT / directory
        if not estimate_root.exists():
            coverage[name] = dict(paired=0, no_judge=0, no_asr=0, missing_dir=True)
            continue
        paired = no_judge = no_asr = 0
        for trial_directory in sorted(rendered.iterdir()):
            meta_file = trial_directory / "meta.json"
            if not meta_file.exists():
                continue
            wav = estimate_root / trial_directory.name / "estimate.wav"
            if not wav.exists():
                continue
            target_text = json.loads(meta_file.read_text()).get("target_text", "")
            # B4: a trial where the target never speaks has no reference, so
            # LCF-WER is 0/0 and the trial is excluded rather than scored 0.
            if not normalise_text(target_text).split():
                continue
            judged = judge_cache.get(
                J.cache_key(wav, J.DEFAULT_MODEL_ID, prompt_sha, 0, "aistudio"))
            heard = asr_cache.get(E._cache_key(wav))
            if judged is None:
                no_judge += 1
                continue
            if heard is None:
                no_asr += 1
                continue
            paired += 1
            jc = count_errors(target_text, judged[1])
            ac = count_errors(target_text, heard)
            rows.append(dict(
                system=name,
                trial=trial_directory.name,
                judge_wer=100 * jc.total_errors / jc.reference_word_count,
                asr_wer=100 * ac.total_errors / ac.reference_word_count,
                ref_words=jc.reference_word_count))
        coverage[name] = dict(paired=paired, no_judge=no_judge, no_asr=no_asr,
                              missing_dir=False)
        if verbose:
            print(f"  {name}: {paired} paired, {no_judge} no-judge, {no_asr} no-asr")
    return rows, coverage


def summarise(label, rows):
    judge = [r["judge_wer"] for r in rows]
    asr = [r["asr_wer"] for r in rows]
    r = pearson(judge, asr)
    return dict(
        group=label, n=len(rows),
        judge_mean=sum(judge) / len(judge),
        asr_mean=sum(asr) / len(asr),
        pearson=r, r_squared=r * r,
        spearman=spearman(judge, asr),
        mean_absolute_error=sum(abs(x - y) for x, y in zip(judge, asr)) / len(judge))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"pairing cached judge and ASR transcripts on {args.split}:")
    rows, coverage = collect(args.split, DEFAULT_SYSTEMS)
    if not rows:
        print("no paired trials -- nothing to correlate")
        return

    groups = [summarise("POOLED", rows)]
    for name in DEFAULT_SYSTEMS:
        subset = [r for r in rows if r["system"] == name]
        if subset:
            groups.append(summarise(name, subset))

    header = (f"{'group':<20}{'n':>5}{'judge mean':>12}{'asr mean':>10}"
              f"{'pearson r':>11}{'r^2':>7}{'spearman':>10}{'MAE':>8}")
    lines = [header]
    for g in groups:
        lines.append(f"{g['group']:<20}{g['n']:>5}{g['judge_mean']:>12.2f}"
                     f"{g['asr_mean']:>10.2f}{g['pearson']:>11.3f}"
                     f"{g['r_squared']:>7.3f}{g['spearman']:>10.3f}"
                     f"{g['mean_absolute_error']:>8.2f}")
    report = "\n".join(lines)
    print()
    print(report)

    if args.out:
        out = REPO_ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "per_trial.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / "report.txt").write_text(report + "\n")
        (out / "summary.json").write_text(json.dumps(
            dict(split=args.split, systems=DEFAULT_SYSTEMS, coverage=coverage,
                 groups=groups), indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
