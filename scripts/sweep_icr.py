"""Report ICR at every k, per condition, for the floor and ceiling anchors.

    python scripts/sweep_icr.py --split eval_public

Satisfies metric-definitions.md 3.2's requirement that the threshold's
sensitivity be reported, and B13's that no aggregate appears alone.

The listener is swappable: `--listener cached-asr` reads transcripts already in
experiments/results/transcripts.csv. A live judge slots in as another listener
without changing anything below.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_model_metric.icr import K_VALUES, compute_icr  # noqa: E402

ASR_NAME = "faster-whisper==1.2.1:small.en:int8:cpu:greedy"


def cached_asr_listener(cache_path):
    with open(cache_path, newline="") as handle:
        cache = {row["key"]: row["text"] for row in csv.DictReader(handle)}

    def transcribe_one(audio_file_path):
        stat = Path(audio_file_path).stat()
        key = (f"small.en|{Path(audio_file_path).parent.name}"
               f"|{Path(audio_file_path).name}|{int(stat.st_mtime)}|{stat.st_size}")
        if key not in cache:
            raise SystemExit(
                f"no cached transcript for {audio_file_path}. Transcribe it first, "
                f"or the sweep would silently score a different trial set.")
        return cache[key]

    return transcribe_one


def strata_of(row):
    sir = float(row["sir_db"]) if row["sir_db"] else None
    overlap = float(row["overlap_achieved"]) if row["overlap_achieved"] else 0.0
    t60 = float(row["t60_s"]) if row["t60_s"] else None
    return {
        "louder": ("n/a" if sir is None else
                   "target" if sir > 1.0 else
                   "interferer" if sir < -1.0 else "balanced"),
        "overlap": ("none" if overlap <= 0.01 else
                    "low" if overlap < 0.3 else
                    "mid" if overlap < 0.6 else "high"),
        "t60": "n/a" if t60 is None else ("<=0.3s" if t60 <= 0.3 else ">0.3s"),
        "same_gender": ("n/a" if row["same_gender"] in ("", None) else
                        "same" if str(row["same_gender"]) in ("1", "1.0") else "diff"),
    }


def load(split, manifest_dir, data_root):
    manifest = Path(manifest_dir) / f"{split}.csv"
    audio_root = Path(data_root) / "rendered" / split
    trials = []
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(handle):
            trial_dir = audio_root / row["trial_id"]
            if not (trial_dir / "meta.json").exists():
                continue
            meta = json.loads((trial_dir / "meta.json").read_text())
            trials.append(dict(
                trial_id=row["trial_id"], condition=row["condition"],
                target_text=meta["target_text"],
                interferer_text=meta["interferer_text"],
                mixture=trial_dir / "mixture.wav",
                clean=trial_dir / "target.wav",
                strata=strata_of(row),
            ))
    return trials


def sweep_row(trials, transcribe_one, source):
    responses = [transcribe_one(trial[source]) for trial in trials]
    return compute_icr(responses,
                       [trial["target_text"] for trial in trials],
                       [trial["interferer_text"] for trial in trials])


def render(name, result):
    cells = "".join(
        f" {result.icr_at_k[k]:>5.1f} ({result.eligible_at_k[k]:>3}) |"
        if k in result.icr_at_k else f" {'—':>11} |"
        for k in K_VALUES
    )
    mean = ("—" if result.mean_leaked_percent is None
            else f"{result.mean_leaked_percent:.1f}")
    return (f"| {name:<22} | {result.trials_total:>4} |{cells}"
            f" {mean:>6} | {result.trials_ineligible:>4} |")


def header():
    ks = "".join(f" ICR@{k} (n)  |" for k in K_VALUES)
    rule = "|" + "-" * 24 + "|" + "-" * 6 + "|" + ("-" * 13 + "|") * len(K_VALUES) \
           + "-" * 8 + "|" + "-" * 6 + "|"
    return (f"| {'group':<22} | {'n':>4} |{ks} {'mean%':>6} | {'inel':>4} |\n{rule}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="eval_public")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--cache", default="experiments/results/transcripts.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    transcribe_one = cached_asr_listener(args.cache)
    trials = load(args.split, args.manifest_dir, args.data_root)
    print(f"loaded {len(trials)} trials from {args.split}")

    present = [t for t in trials if t["condition"] == "both"]
    absent = [t for t in trials if t["condition"] == "interferer_only"]

    blocks = []
    for label, group, source in (
        ("PRESENT (condition=both) — floor", present, "mixture"),
        ("PRESENT (condition=both) — ceiling", present, "clean"),
        ("ABSENT (interferer_only) — floor", absent, "mixture"),
        ("ABSENT (interferer_only) — ceiling", absent, "clean"),
    ):
        lines = [f"### {label}", "", header()]
        lines.append(render("all", sweep_row(group, transcribe_one, source)))
        for axis in ("louder", "overlap", "same_gender", "t60"):
            buckets = {}
            for trial in group:
                buckets.setdefault(trial["strata"][axis], []).append(trial)
            for value, subset in sorted(buckets.items()):
                lines.append(render(f"{axis}={value}",
                                    sweep_row(subset, transcribe_one, source)))
        blocks.append("\n".join(lines))
        print("\n" + blocks[-1])

    out_dir = Path(args.out or f"experiments/results/{date.today().isoformat()}-icr-sweep-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                           text=True).stdout.strip()
    (out_dir / "meta.yaml").write_text(yaml.safe_dump({
        "date": date.today().isoformat(),
        "script": "scripts/sweep_icr.py",
        "git_commit": commit + ("-dirty" if dirty else ""),
        "seed": args.seed,
        "split": args.split,
        "listener": ASR_NAME,
        "listener_role": "STAND-IN for the judge, not a live-model result",
        "k_values": list(K_VALUES),
        "n_trials_loaded": len(trials),
        "n_present": len(present),
        "n_absent": len(absent),
    }, sort_keys=False))
    (out_dir / "README.md").write_text(
        f"# ICR sweep — {args.split} — {date.today().isoformat()}\n\n"
        f"Listener: `{ASR_NAME}` — a STAND-IN for the judge, not a live-model "
        f"result.\n\n`inel` = trials ineligible (no exclusive interferer "
        f"content). `n` in each ICR cell is that k's own eligible count.\n\n"
        + "\n\n".join(blocks) + "\n")
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
