"""Does the judge already KNOW the LibriSpeech text, without hearing anything?

    python3 scripts/probe_contamination.py --dry-run          # no API calls
    python3 scripts/probe_contamination.py --limit 20
    python3 scripts/probe_contamination.py --model gemini-X --out .../probe-X

WHY THIS IS GATE 1 OF THE PANEL. LibriSpeech is people reading public-domain
Project Gutenberg books aloud. Those books are on the open web, so a frontier
model has plausibly memorised them. That gives a judge TWO routes to the right
words: it listened and transcribed, or it recognised the passage and recalled
the rest. Every score in this project treats those identically -- both produce
text matching the reference, so LCF-WER counts both as correct and FR counts
neither as invented. The instrument is blind to the difference by construction.

The motivating number: the judge ceiling is 1.05 % LCF-WER where the offline ASR
reads 5.85 % on the SAME clean audio, and the judge's ceiling insertion rate is
0.03 % -- a transcriber that essentially never adds a word. Verified from cache
2026-09-21. Consistent with a much better listener; also consistent with recital.

HOW IT SEPARATES THEM: it takes the audio away. The model gets the first few
words as TEXT and is asked to continue. There is no legitimate route to the rest
of the passage, so anything it gets right, it already knew.

Two readings, both registered in the config before running:
  1. MATCHED vs SHUFFLED word error. The same continuation is scored against its
     own true remainder and against a different trial's. Generic plausible
     English scores about the same on both; a memorised passage scores far
     better on its own. The shuffled arm is free -- a re-score, not a new call.
  2. LONGEST EXACT RUN of consecutive words shared with the true remainder.
     Needs no statistics to read: generic English does not emit long exact runs.

WHAT A NEGATIVE RESULT BUYS, and it is worth the 103 calls either way: it turns
"the ceiling looks too good" from a worry into a closed question.

decisions-m4.md 2026-09-21. Config: experiments/configs/contamination_probe.yaml.
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_model_metric.evaluate import load_trials                  # noqa: E402
from src.live_model_metric.judge import Judge                           # noqa: E402
from src.live_model_metric.lcf_wer import count_errors, normalise_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/contamination_probe.yaml"

CONTINUATION_PROMPT = (
    "The following is the opening of a passage from a public-domain book.\n"
    "Continue it, reproducing the original wording exactly as it appears in "
    "the book. Output only the continuation, with no commentary.\n\n"
    "{prefix}"
)


def longest_exact_run(reference_words, hypothesis_words):
    """Longest run of consecutive words appearing in both, in order.

    Reported because it needs no baseline to interpret. Generic English shares
    short function-word runs; reproducing 8+ consecutive content words in order
    is recall, not coincidence.
    """
    if not reference_words or not hypothesis_words:
        return 0
    match = SequenceMatcher(a=reference_words, b=hypothesis_words,
                            autojunk=False).find_longest_match(
                                0, len(reference_words), 0, len(hypothesis_words))
    return match.size


def word_error_rate(reference_text, hypothesis_text):
    counts = count_errors(reference_text, hypothesis_text)
    if not counts.reference_word_count:
        return None
    return 100.0 * counts.total_errors / counts.reference_word_count


def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def load_cache(path):
    if not Path(path).exists():
        return {}
    with open(path, newline="") as handle:
        return {(r["model"], r["trial_id"], r["prefix_words"]): r
                for r in csv.DictReader(handle)}


def append_cache(path, row):
    path = Path(path)
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", default=None, help="override the config model ID")
    parser.add_argument("--backend", default=None, choices=["aistudio", "vertex"])
    parser.add_argument("--project", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prefix-words", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="build every prompt and print the plan, call nothing")
    parser.add_argument("--max-new-calls", type=int, default=150,
                        help="hard cap. Refuses rather than truncating; cached "
                             "work is kept and a re-run resumes.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    probe = config["probe"]
    model_id = args.model or config["model"]["model_id"]
    backend = args.backend or config["model"]["backend"]
    prefix_words = args.prefix_words or probe["prefix_words"]
    rpm = config["model"]["requests_per_minute"]
    cache_path = REPO_ROOT / config["output"]["cache"]

    random.seed(seed)

    trials = load_trials(probe["split"], condition=probe["condition"],
                         limit=args.limit)
    usable = []
    for trial in trials:
        words = normalise_text(trial.target_text).split()
        if len(words) < probe["min_total_words"]:
            continue
        usable.append((trial, " ".join(words[:prefix_words]),
                       " ".join(words[prefix_words:])))

    print(f"probe: {len(usable)} usable trials of {len(trials)} "
          f"(split={probe['split']}, prefix={prefix_words} words)")
    print(f"model: {model_id}  backend: {backend}  seed: {seed}")
    print("MODALITY: text only. No audio is sent.\n")

    if args.dry_run:
        trial, prefix, remainder = usable[0]
        print("--- example prompt ---")
        print(CONTINUATION_PROMPT.format(prefix=prefix))
        print(f"\n--- true remainder ({len(remainder.split())} words) ---")
        print(remainder[:300])
        print(f"\nwould make up to {len(usable)} calls. Nothing was sent.")
        return

    cache = load_cache(cache_path)
    judge = Judge(model_id=model_id, backend=backend, project=args.project)
    client = judge._ensure_client()
    interval = 60.0 / rpm if rpm else 0.0
    today = date.today().isoformat()

    new_calls = 0
    last_call_at = 0.0
    continuations = {}
    for trial, prefix, _remainder in usable:
        key = (model_id, trial.trial_id, str(prefix_words))
        if key in cache:
            continuations[trial.trial_id] = cache[key]["continuation"]
            continue
        if new_calls >= args.max_new_calls:
            raise SystemExit(
                f"hit --max-new-calls={args.max_new_calls} with "
                f"{len(usable) - len(continuations)} trials left. Cached work is "
                f"kept; re-run to resume.")
        wait = interval - (time.time() - last_call_at)
        if wait > 0:
            time.sleep(wait)
        interaction = client.interactions.create(
            model=model_id,
            input=[{"type": "text",
                    "text": CONTINUATION_PROMPT.format(prefix=prefix)}],
        )
        last_call_at = time.time()
        new_calls += 1
        text = (interaction.output_text or "").strip()
        continuations[trial.trial_id] = text
        append_cache(cache_path, {
            "model": model_id, "backend": backend, "trial_id": trial.trial_id,
            "prefix_words": prefix_words, "prefix": prefix,
            "continuation": text, "modality": "text", "run_date": today,
        })
        if new_calls % 10 == 0:
            print(f"  {new_calls} new calls...", flush=True)

    # -- scoring. Both arms are re-scores of text already bought -------------
    order = [t.trial_id for t, _, _ in usable]
    shuffled = order[:]
    random.shuffle(shuffled)
    # guarantee nobody is paired with themselves
    for i, tid in enumerate(shuffled):
        if tid == order[i]:
            swap = (i + 1) % len(shuffled)
            shuffled[i], shuffled[swap] = shuffled[swap], shuffled[i]
    remainder_by_id = {t.trial_id: rem for t, _, rem in usable}

    rows = []
    for index, (trial, _prefix, remainder) in enumerate(usable):
        hypothesis = continuations.get(trial.trial_id, "")
        other = remainder_by_id[shuffled[index]]
        rows.append({
            "trial_id": trial.trial_id,
            "matched_wer": word_error_rate(remainder, hypothesis),
            "shuffled_wer": word_error_rate(other, hypothesis),
            "matched_run": longest_exact_run(remainder.split(),
                                             normalise_text(hypothesis).split()),
            "shuffled_run": longest_exact_run(other.split(),
                                              normalise_text(hypothesis).split()),
            "continuation_words": len(normalise_text(hypothesis).split()),
        })

    def mean(field):
        values = [r[field] for r in rows if r[field] is not None]
        return sum(values) / len(values) if values else float("nan")

    matched_wer, shuffled_wer = mean("matched_wer"), mean("shuffled_wer")
    matched_run, shuffled_run = mean("matched_run"), mean("shuffled_run")
    gap = shuffled_wer - matched_wer
    verdict = config["verdict"]
    fired = (gap >= verdict["wer_gap_points"]
             or matched_run >= verdict["longest_run_words"])

    summary = {
        "date": today, "model_id": model_id, "backend": backend,
        "modality": "text", "seed": seed, "split": probe["split"],
        "prefix_words": prefix_words, "n_trials": len(rows),
        "new_calls": new_calls,
        "matched_wer": matched_wer, "shuffled_wer": shuffled_wer,
        "wer_gap_points": gap,
        "mean_longest_run_matched": matched_run,
        "mean_longest_run_shuffled": shuffled_run,
        "max_longest_run_matched": max((r["matched_run"] for r in rows), default=0),
        "verdict_registered": verdict,
        "contamination_detected": bool(fired),
    }

    print("\n=== contamination probe ===")
    print(f"  matched WER        {matched_wer:6.2f} %   (continuation vs ITS OWN remainder)")
    print(f"  shuffled WER       {shuffled_wer:6.2f} %   (vs another trial's remainder)")
    print(f"  gap                {gap:6.2f} points  (threshold {verdict['wer_gap_points']})")
    print(f"  longest exact run  {matched_run:6.2f} words matched, "
          f"{shuffled_run:.2f} shuffled (threshold {verdict['longest_run_words']})")
    print(f"  max exact run      {summary['max_longest_run_matched']} words")
    print(f"\n  VERDICT: {'CONTAMINATION DETECTED' if fired else 'no contamination detected'}")
    if fired:
        print("  The ceiling is not purely a measure of listening. Do not spend")
        print("  the panel budget until decisions-m4.md records what it now means.")
    else:
        print("  The ceiling stands as a listening result. Gate 1 passed.")

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / config["output"]["results_dir"] /
        f"{today}-contamination-probe-{model_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "per_trial": rows}, indent=2))
    with open(out_dir / "per_trial.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
