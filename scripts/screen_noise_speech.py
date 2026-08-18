#!/usr/bin/env python3
"""Detect speech hiding in the WHAM! noise beds, and measure what rejecting it costs.

    ../tse_venv/bin/python scripts/screen_noise_speech.py

Output: data/index/noise_speech_{tr,cv,tt}.csv  (clip, duration, speech_s,
        max_segment_s, n_segments, segments)
        data/index/noise_speech.meta.yaml

WHY
---
WHAM! was recorded in cafes, restaurants and city streets. Real places, with
real people in them. Any speech in the bed enters a trial as an unlabelled
third talker, and `docs/data/metric-definitions.md` scores the words it
contributes as the model hallucinating. It did not hallucinate; we put someone
else in the room and did not tell anyone.

`data-construction-parameters.md` calls `noise_speech_rejection` critical. It has
never been implemented, and it is the last unimplemented item that can silently
corrupt the headline metric.

THIS SCRIPT DOES NOT REJECT ANYTHING
------------------------------------
It measures, and it caches the measurement. The rejection RULE is deliberately
not applied here and not baked into the cache, because choosing it needs the
numbers this script produces -- and because a rule stored as a verdict cannot be
changed later without re-scanning 81.7 hours of audio. Cache the evidence, decide
the threshold in config, apply it at manifest-build time.

READ THE CAVEAT AT THE BOTTOM OF THE OUTPUT BEFORE TRUSTING ANY OF IT. A VAD
asked "is anyone talking in this cafe?" is working well outside the case it was
trained for, and the honest check is to listen. --sample-out writes clips for
exactly that.

Detector: Silero VAD, Silero Team (2021). Same settings as the utterance index,
so "speech" means the same thing on both sides of the mixture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import subprocess
import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import vad  # noqa: E402
from src.run_log import timed  # noqa: E402

COLUMNS = ["clip", "duration", "speech_s", "max_segment_s", "n_segments", "segments"]

# Candidate rejection rules. Applied to the CACHED measurements, so sweeping them
# is free -- no audio is re-read.
#
# max_segment_s is the criterion to watch. Total speech seconds treats twenty
# 0.1 s blips the same as one 2 s sentence, but only the sentence can put
# recognisable words into the judge's transcript. A word is ~0.3 s; a short
# phrase ~1 s.
MAX_SEG_RULES = [0.3, 0.5, 1.0, 2.0]
FRACTION_RULES = [0.02, 0.05, 0.10, 0.25]

# VAD thresholds for the sensitivity check. Higher = more confident before
# calling something speech. On babble the default 0.5 is the one most likely to
# fire on clatter and laughter, so it matters here in a way it did not on
# LibriSpeech.
SENSITIVITY_THRESHOLDS = [0.5, 0.7, 0.9]

_cfg = None
_sr = None
_root = None
_extra = None


def _init(cfg, sr, root, extra):
    global _cfg, _sr, _root, _extra
    _cfg, _sr, _root, _extra = cfg, sr, Path(root), extra
    import torch
    torch.set_num_threads(1)


def _work(item):
    """(split, clip, want_sensitivity) -> one cache row.

    The sensitivity flag rides in the work item rather than in a closure over
    the pool: a closure is not picklable, so imap would fail to dispatch it.
    Returns None for the row on a read failure -- reported, not hidden.
    """
    split, clip, want_sens = item
    try:
        import torch
        wav, sr = sf.read(_root / split / clip, dtype="float32")
        if sr != _sr:
            raise ValueError(f"{clip} is {sr} Hz, expected {_sr}")
        if wav.ndim > 1:                       # WHAM! ships stereo upstream
            wav = wav.mean(axis=1)
        t = torch.from_numpy(np.ascontiguousarray(wav))
        segs = vad.detect(t, vad.load_model(), _cfg, _sr)
        row = {
            "clip": clip,
            "duration": f"{len(wav) / sr:.4f}",
            "speech_s": f"{vad.total_speech(segs):.4f}",
            "max_segment_s": f"{max((b - a for a, b in segs), default=0.0):.4f}",
            "n_segments": len(segs),
            "segments": vad.format_segments(segs),
        }
        if not (_extra and want_sens):
            return split, row, None
        # Sensitivity subsample: same decode, several detector thresholds.
        alt = {}
        for thr in _extra:
            s = vad.detect(t, vad.load_model(), {**_cfg, "threshold": thr}, _sr)
            alt[thr] = (vad.total_speech(s) / (len(wav) / sr),
                        max((b - a for a, b in s), default=0.0))
        return split, row, alt
    except Exception as e:                      # noqa: BLE001
        print(f"\n  FAILED {split}/{clip}: {e}", file=sys.stderr)
        return split, None, None


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def load_cached(path):
    if not path.exists():
        return {}
    with path.open() as f:
        return {r["clip"]: r for r in csv.DictReader(f)}


def report(by_split, n_trials):
    """The table that decides the rule. All of it is post-processing on the
    cache, so re-running with different rules costs nothing."""
    print("\n" + "=" * 88)
    print("WHAT IS IN THE NOISE")
    print("=" * 88)
    print(f"\n  {'pool':<6} {'clips':>8} {'hours':>7} {'any speech':>11} "
          f"{'speech/dur':>11} {'max seg s':>10}")
    print("  " + "-" * 60)
    for split, rows in by_split.items():
        dur = np.array([float(r["duration"]) for r in rows])
        sp = np.array([float(r["speech_s"]) for r in rows])
        mx = np.array([float(r["max_segment_s"]) for r in rows])
        print(f"  {split:<6} {len(rows):>8,} {dur.sum()/3600:>7.1f} "
              f"{(sp > 0).mean():>10.1%} {(sp/dur).mean():>11.3f} {mx.mean():>10.2f}")

    print("\n" + "=" * 88)
    print("WHAT REJECTION WOULD COST  --  % of clips DROPPED, and what survives")
    print("=" * 88)
    print("\n  Rule A: drop a clip if its longest unbroken stretch of speech is >= X")
    print("  (a word is ~0.3 s, a short phrase ~1 s -- below that nothing")
    print("   recognisable can reach the judge)\n")
    print(f"  {'pool':<6} " + "".join(f"{'>=' + str(x) + 's':>16}" for x in MAX_SEG_RULES))
    print("  " + "-" * (6 + 16 * len(MAX_SEG_RULES)))
    for split, rows in by_split.items():
        mx = np.array([float(r["max_segment_s"]) for r in rows])
        dur = np.array([float(r["duration"]) for r in rows])
        cells = []
        for x in MAX_SEG_RULES:
            drop = mx >= x
            cells.append(f"{drop.mean():>7.1%} {len(rows)-drop.sum():>6,}kept")
        print(f"  {split:<6} " + "".join(f"{c:>16}" for c in cells))

    print("\n  Rule B: drop a clip if speech covers >= X of its duration\n")
    print(f"  {'pool':<6} " + "".join(f"{'>=' + str(int(x*100)) + '%':>16}"
                                      for x in FRACTION_RULES))
    print("  " + "-" * (6 + 16 * len(FRACTION_RULES)))
    for split, rows in by_split.items():
        sp = np.array([float(r["speech_s"]) for r in rows])
        dur = np.array([float(r["duration"]) for r in rows])
        for_row = []
        for x in FRACTION_RULES:
            drop = (sp / dur) >= x
            for_row.append(f"{drop.mean():>7.1%} {len(rows)-drop.sum():>6,}kept")
        print(f"  {split:<6} " + "".join(f"{c:>16}" for c in for_row))

    print("\n  Is the surviving pool big enough? Each trial draws one clip and a")
    print("  random offset into it, so clips are reused by design -- the question")
    print("  is variety, not supply.\n")
    print(f"  {'pool':<6} {'trials':>8} {'clips now':>10}  survivors needed to keep "
          f"clips-per-trial >= 0.25")
    print("  " + "-" * 78)
    for split, rows in by_split.items():
        n = n_trials.get(split, 0)
        print(f"  {split:<6} {n:>8,} {len(rows):>10,}  {max(1, n // 4):>8,}")


def sensitivity(alt, n):
    if not alt:
        return
    print("\n" + "=" * 88)
    print(f"DOES THE DETECTOR THRESHOLD CHANGE THE ANSWER?  ({n:,}-clip subsample)")
    print("=" * 88)
    print("\n  Higher threshold = more confident before calling something speech.")
    print("  If these rows disagree sharply, the detector is guessing and the")
    print("  rule must be set by listening, not by this table.\n")
    print(f"  {'threshold':<11} {'any speech':>11} {'speech/dur':>11} "
          + "".join(f"{'drop>=' + str(x) + 's':>14}" for x in MAX_SEG_RULES))
    print("  " + "-" * (33 + 14 * len(MAX_SEG_RULES)))
    for thr in SENSITIVITY_THRESHOLDS:
        fr = np.array([a[thr][0] for a in alt])
        mx = np.array([a[thr][1] for a in alt])
        cells = "".join(f"{(mx >= x).mean():>13.1%} " for x in MAX_SEG_RULES)
        print(f"  {thr:<11} {(fr > 0).mean():>10.1%} {fr.mean():>11.3f} {cells}")


def write_samples(by_split, root, out_dir, n):
    """Copy the most speech-heavy clips out so they can be listened to.

    The point of the whole script: a VAD asked about cafe babble is outside its
    training case, and no table settles that. Ranked most-speech-first, so the
    first few are the strongest evidence either way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in by_split.items():
        ranked = sorted(rows, key=lambda r: -float(r["max_segment_s"]))[:n]
        for i, r in enumerate(ranked):
            dst = out_dir / f"{split}_{i:02d}_maxseg{float(r['max_segment_s']):.1f}s_{r['clip']}"
            shutil.copy(root / split / r["clip"], dst)
    print(f"\n  Wrote {n} worst-offender clips per pool to {out_dir}")
    print("  Listen to them. If they are plates and laughter, the rule can be")
    print("  loose. If you can make out words, it must be strict.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--out-prefix", default="data/index/noise_speech")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N clips per pool, for a smoke test")
    ap.add_argument("--sensitivity-n", type=int, default=2000,
                    help="clips to also run at other detector thresholds; 0 to skip")
    ap.add_argument("--sample-out", default=None,
                    help="copy the worst offenders here so you can listen to them")
    ap.add_argument("--sample-n", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    cfg = vad.vad_config(config)
    sr = int(config["sample_rate"])
    root = Path(config["paths"]["wham_noise"])
    n_trials = {}
    for split, s in config["splits"].items():
        n_trials[s["noise_split"]] = n_trials.get(s["noise_split"], 0) + s["n_trials"]

    pools = sorted({s["noise_split"] for s in config["splits"].values()})
    print(f"Screening WHAM! for speech: pools {pools}")
    print(f"  silero-vad {vad.model_version()}  {cfg}")

    wanted, cached = [], {}
    for split in pools:
        path = Path(f"{args.out_prefix}_{split}.csv")
        cached[split] = {} if args.force else load_cached(path)
        clips = [r["clip"] for r in
                 csv.DictReader((Path(args.index_dir) / f"noise_{split}.csv").open())]
        if args.limit:
            clips = clips[:args.limit]
        wanted += [(split, c) for c in clips]

    todo = [(s, c) for s, c in wanted if c not in cached[s]]
    n_cached = len(wanted) - len(todo)
    print(f"  {len(wanted):,} clips, {n_cached:,} already cached, {len(todo):,} to do")

    # Sensitivity runs on a deterministic slice of the work, not a random draw,
    # so a resumed run does not silently change which clips it covers.
    extra = SENSITIVITY_THRESHOLDS if args.sensitivity_n else []
    sens_set = {todo[i] for i in range(0, len(todo),
                                       max(1, len(todo) // max(args.sensitivity_n, 1)))} \
        if extra else set()

    by_split = {s: [cached[s][c] for _, c in
                    [(x, y) for x, y in wanted if x == s and y in cached[s]]]
                for s in pools}
    alt, audio_s = [], [0.0]

    if todo:
        t0 = time.time()
        with timed("scripts/screen_noise_speech.py",
                   scope=lambda: f"{len(todo):,} clips / {audio_s[0]/3600:.0f} h"
                                 + (f" (resumed, {n_cached:,} cached)" if n_cached else ""),
                   rate=lambda: f"{audio_s[0]/max(time.time()-t0, 1e-9):.0f}x realtime, "
                                f"{args.workers} workers"):
            work = [(s, c, (s, c) in sens_set) for s, c in todo]
            with Pool(args.workers, initializer=_init,
                      initargs=(cfg, sr, str(root), extra)) as pool:
                for i, (split, row, a) in enumerate(
                        pool.imap(_work, work, chunksize=16), 1):
                    if row is not None:
                        by_split[split].append(row)
                        audio_s[0] += float(row["duration"])
                    if a:
                        alt.append(a)
                    if i % 200 == 0 or i == len(todo):
                        el = time.time() - t0
                        print(f"\r  {i}/{len(todo)}  [{el/60:5.1f} min elapsed, "
                              f"~{el/i*(len(todo)-i)/60:5.1f} min left]   ",
                              end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        for split in pools:
            path = Path(f"{args.out_prefix}_{split}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=COLUMNS)
                w.writeheader()
                w.writerows(by_split[split])
            print(f"Wrote {path}  ({len(by_split[split]):,} clips)")

    Path(f"{args.out_prefix}.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/screen_noise_speech.py",
        "git_commit": git_commit(),
        "config": args.config,
        "config_md5": hashlib.md5(Path(args.config).read_bytes()).hexdigest(),
        "detector": "silero-vad",
        "detector_version": vad.model_version(),
        "detector_settings": cfg,
        "pools": {s: len(by_split[s]) for s in pools},
        # Deliberately absent: any rejection verdict. The rule lives in config and
        # is applied at manifest-build time, so changing it never means re-scanning.
        "rejection_rule": "NOT APPLIED -- see decisions-m0.md, chosen from this report",
    }, sort_keys=False))

    report(by_split, n_trials)
    sensitivity(alt, len(alt))

    if args.sample_out:
        write_samples(by_split, root, Path(args.sample_out), args.sample_n)

    print("\n" + "=" * 88)
    print("BEFORE YOU TRUST ANY OF THIS")
    print("=" * 88)
    print("""
  1. The detector is outside its training case. Silero learns "is this person
     talking into a microphone". WHAM! is a cafe from across the room. It may
     fire on clattering plates and laughter, and it may miss a quiet
     conversation two tables away. Neither error is visible in the table above.

  2. Only Rule A tracks the thing that matters. Total speech seconds counts
     twenty 0.1 s blips the same as one 2 s sentence, but only the sentence can
     put recognisable words into the judge's transcript.

  3. Rejection is not free. Dropping the speechiest clips removes exactly the
     hardest noise, so the surviving bed is systematically easier than the one
     the task claims to model. If the drop rate is large, say so in the thesis
     rather than quietly shipping a gentler test set.

  4. Nothing has been rejected. This wrote measurements. Choose a rule, record
     it in decisions-m0.md with these numbers as the evidence, put it in
     generator.yaml, and apply it in build_manifest.py.

  Re-run with --sample-out /tmp/noise_check and listen.
""")


if __name__ == "__main__":
    main()
