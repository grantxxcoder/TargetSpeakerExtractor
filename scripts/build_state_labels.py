#!/usr/bin/env python3
"""Per-frame speaker-state labels for every rendered trial (decisions-pending.md D14).

    ../tse_venv/bin/python scripts/build_state_labels.py                    # every split
    ../tse_venv/bin/python scripts/build_state_labels.py --splits sir0_val  # one
    ../tse_venv/bin/python scripts/build_state_labels.py --limit 50         # smoke test

Output: data/index/state_<split>.csv        (trial_id, n_samples, target_spans,
                                             interferer_spans)
        data/index/state_<split>.meta.yaml

READS NO AUDIO. The manifest records which utterances went into each trial
(`target_utts`) and where each was placed (`target_onsets_s`); the VAD index
records where the speech is inside each utterance (B2). Shifting the second by
the first reconstructs the spans exactly. Why that matters beyond cost, and why
the reverberation tail is the one thing it cannot give: src/data/state_labels.py.

THE RECONSTRUCTION IS CHECKED, NOT TRUSTED
------------------------------------------
Four manifest columns were derived from these same spans by
scripts/build_manifest.py, so all four must come back out:

    target_speech_s / interferer_speech_s   sum of span lengths
    target_activity / interferer_activity   speech seconds / mixture_length_s
    target_footprint_s / interferer_...     first onset to last utterance end
    overlap_achieved                        shared seconds / mixture_length_s

That is a stronger test than any unit test on synthetic intervals could be: it
checks this script against the code that actually built the dataset. All four are
measured on the DRY spans, before the tail extension, because that is what
build_manifest.py measured.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import state_labels as sl  # noqa: E402
from src.data import vad  # noqa: E402
from src.run_log import timed  # noqa: E402

COLUMNS = ["trial_id", "n_samples", "target_spans", "interferer_spans"]

# The manifest stores lists pipe-separated, the same convention vad.py uses
# between segments.
LIST_SEP = "|"


def split_list(text):
    """"a|b" -> ["a", "b"];  "" -> []. A target-absent trial has no utterances
    at all, and an empty cell must not become [""]."""
    return text.split(LIST_SEP) if text else []


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:                           # noqa: BLE001
        return "UNKNOWN-not-a-git-checkout"


def read_vad_index(path):
    """{utt: [(start_s, end_s), ...]} and {utt: duration_s}.

    The one utterance in 137,876 with no detected speech parses to [] and is
    kept, not dropped: a trial could legitimately use it and must not crash.
    """
    segs, durs = {}, {}
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            segs[row["utt"]] = vad.parse_segments(row["segments"])
            durs[row["utt"]] = float(row["duration"])
    return segs, durs


def rendered_length_s(trial_dir, mixture_length_s, t60_s, sample_rate):
    """Length of the audio actually on disk, and whether meta.json supplied it.

    The manifest's `mixture_length_s` is the PRE-PAD length -- A5 pads the tail
    by `t60_s` so the reverberation is not truncated (decisions-m0.md
    2026-08-13), and `meta.json` records the result as `n_samples`. Labels are
    clipped to the rendered length, because that is the timeline the data loader
    crops from.

    Falls back to the computed length for a manifest row that was never
    rendered, and reports how many rows did so rather than hiding it.
    """
    meta_path = Path(trial_dir) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return int(meta["n_samples"]), True
    return int(round((mixture_length_s + t60_s) * sample_rate)), False


def check_row(row, t_dry, i_dry, durs, cfg, problems):
    """Compare the reconstruction against the manifest. Appends to `problems`.

    Measured on the DRY spans: build_manifest.py had no tail extension, so
    comparing extended spans against its columns would fail by construction.
    """
    tol_s, tol_r = cfg["tolerance_s"], cfg["tolerance_ratio"]
    length = float(row["mixture_length_s"])
    tid = row["trial_id"]

    for who, spans in (("target", t_dry), ("interferer", i_dry)):
        want_speech = float(row[f"{who}_speech_s"] or 0.0)
        got_speech = vad.total_speech(spans)
        if abs(got_speech - want_speech) > tol_s:
            problems.append(f"{tid} {who}_speech_s: manifest {want_speech:.4f}, "
                            f"rebuilt {got_speech:.4f}")

        want_act = float(row[f"{who}_activity"] or 0.0)
        got_act = got_speech / length if length else 0.0
        if abs(got_act - want_act) > tol_r:
            problems.append(f"{tid} {who}_activity: manifest {want_act:.4f}, "
                            f"rebuilt {got_act:.4f}")

        # Footprint is how much AUDIO the run carries, not the timeline it
        # spans: `pick_run` accumulates `footprint += d` over the utterances'
        # own durations, and `lay_out` then scatters the leftover time as gaps
        # BETWEEN them. So the gaps are deliberately excluded -- summing
        # durations is right and (last onset + duration) - (first onset) is not.
        # Durations come from the VAD index, which stores them beside the
        # segments and reads them from the same files build_manifest.py did.
        utts = split_list(row[f"{who}_utts"])
        want_fp = float(row[f"{who}_footprint_s"] or 0.0)
        got_fp = sum(durs.get(u, 0.0) for u in utts)
        if abs(got_fp - want_fp) > tol_s:
            problems.append(f"{tid} {who}_footprint_s: manifest {want_fp:.4f}, "
                            f"rebuilt {got_fp:.4f}")

    want_ov = float(row["overlap_achieved"] or 0.0)
    got_ov = vad.shared_seconds(t_dry, i_dry) / length if length else 0.0
    if abs(got_ov - want_ov) > tol_r:
        problems.append(f"{tid} overlap_achieved: manifest {want_ov:.4f}, "
                        f"rebuilt {got_ov:.4f}")


def build_split(manifest_path, segs, durs, cfg, limit=None):
    """One manifest -> (rows, stats, problems)."""
    sr = int(cfg["sample_rate"])
    mult = float(cfg["tail_t60_mult"])
    rows, problems = [], []
    stats = Counter()
    # Histograms at every reported hop, for the configured tail AND for the dry
    # arm, so the tail's effect on the class balance is measured not asserted.
    hist = {(hop, tail): np.zeros(sl.N_STATES, dtype=np.int64)
            for hop in cfg["report_hop_s"] for tail in (0.0, mult)}

    with Path(manifest_path).open() as f:
        manifest = list(csv.DictReader(f))
    if limit:
        manifest = manifest[:limit]

    for row in manifest:
        tid = row["trial_id"]
        length_pre = float(row["mixture_length_s"])
        t60 = float(row["t60_s"] or 0.0)

        t_utts, t_ons = split_list(row["target_utts"]), \
            [float(x) for x in split_list(row["target_onsets_s"])]
        i_utts, i_ons = split_list(row["interferer_utts"]), \
            [float(x) for x in split_list(row["interferer_onsets_s"])]

        # Dry spans first: these are what the manifest's columns were measured
        # from, so the checks must run before any tail is added.
        t_dry = vad.merge(vad.spans_of([segs.get(u, []) for u in t_utts], t_ons))
        i_dry = vad.merge(vad.spans_of([segs.get(u, []) for u in i_utts], i_ons))
        check_row(row, t_dry, i_dry, durs, cfg, problems)

        n_samples, from_meta = rendered_length_s(
            Path(cfg["paths"]["rendered"]) / row["split"] / tid,
            length_pre, t60, sr)
        stats["meta_present" if from_meta else "meta_missing"] += 1
        length_s = n_samples / sr

        t_spans = sl.build_spans(t_utts, t_ons, segs, t60, mult, length_s)
        i_spans = sl.build_spans(i_utts, i_ons, segs, t60, mult, length_s)

        rows.append({
            "trial_id": tid,
            "n_samples": n_samples,
            "target_spans": vad.format_segments(t_spans),
            "interferer_spans": vad.format_segments(i_spans),
        })

        for hop in cfg["report_hop_s"]:
            n_frames = int(n_samples / (hop * sr))
            for tail, (ts, is_) in ((0.0, (t_dry, i_dry)), (mult, (t_spans, i_spans))):
                if tail == 0.0:
                    ts = sl.clip_spans(ts, length_s)
                    is_ = sl.clip_spans(is_, length_s)
                states = sl.states_at_frames(ts, is_, n_frames, hop)
                hist[(hop, tail)] += sl.histogram(states)

        stats[row["condition"]] += 1
        stats["trials"] += 1

    return rows, stats, hist, problems


def write_split(out_dir, split, rows, stats, hist, cfg, args, mult):
    out = Path(out_dir) / f"state_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    def as_fractions(counts):
        total = int(counts.sum())
        return {name: round(float(c) / total, 4) if total else 0.0
                for name, c in zip(sl.STATE_NAMES, counts)}

    meta = {
        "generated": date.today().isoformat(),
        "generator": "scripts/build_state_labels.py",
        "git_commit": git_commit(),
        "config": args.config,
        "config_md5": hashlib.md5(Path(args.config).read_bytes()).hexdigest(),
        "seed": cfg["seed"],
        "split": split,
        "n_trials": stats["trials"],
        "sample_rate": cfg["sample_rate"],
        # Provenance of the labels themselves: they are a JOIN of these two, and
        # neither is in git, so this sidecar is the only travelling record.
        "manifest": str(Path(cfg["paths"]["manifests"]) / f"{split}.csv"),
        "vad_index": cfg["paths"]["vad_index"],
        "tail_t60_mult": mult,
        "meta_json_present": stats["meta_present"],
        "meta_json_missing": stats["meta_missing"],
        "state_encoding": {i: n for i, n in enumerate(sl.STATE_NAMES)},
        "class_fractions": {
            f"hop_{hop}s_tail_{tail}": as_fractions(counts)
            for (hop, tail), counts in sorted(hist.items())
        },
    }
    Path(str(out) .replace(".csv", ".meta.yaml")).write_text(
        yaml.safe_dump(meta, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/state_labels.yaml")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="split names; default every manifest found")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N trials per split, for a smoke test")
    ap.add_argument("--tail-t60-mult", type=float, default=None,
                    help="override the config's tail multiplier (0.0 = dry arm)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    gen = yaml.safe_load(Path("experiments/configs/generator.yaml").read_text())
    if int(cfg["sample_rate"]) != int(gen["sample_rate"]):
        raise ValueError(
            f"state_labels.yaml sample_rate {cfg['sample_rate']} != "
            f"generator.yaml {gen['sample_rate']}. The labels are indexed in "
            f"seconds but rasterised in frames, so a mismatch silently "
            f"mis-aligns every frame index derived from them.")
    if args.tail_t60_mult is not None:
        cfg["tail_t60_mult"] = args.tail_t60_mult
    mult = float(cfg["tail_t60_mult"])

    manifest_dir = Path(cfg["paths"]["manifests"])
    splits = args.splits or sorted(p.stem for p in manifest_dir.glob("*.csv"))

    print(f"state labels: {len(splits)} split(s), tail = {mult} x t60_s")
    segs, durs = read_vad_index(cfg["paths"]["vad_index"])
    print(f"  VAD index: {len(segs):,} utterances")

    total_trials = [0]
    with timed("scripts/build_state_labels.py",
               scope=lambda: f"{total_trials[0]:,} trials / {len(splits)} splits",
               rate=lambda: f"tail {mult}x t60, no audio read"):
        all_problems = []
        for split in splits:
            path = manifest_dir / f"{split}.csv"
            if not path.exists():
                print(f"  {split}: no manifest at {path}, skipped")
                continue
            rows, stats, hist, problems = build_split(path, segs, durs, cfg, args.limit)
            out = write_split(cfg["paths"]["out_dir"], split, rows, stats, hist,
                              cfg, args, mult)
            total_trials[0] += stats["trials"]
            all_problems += problems

            print(f"\n  {split}: {stats['trials']:,} trials -> {out}")
            if stats["meta_missing"]:
                print(f"    {stats['meta_missing']:,} rows had no meta.json; "
                      f"length computed as mixture_length_s + t60_s")
            for hop in cfg["report_hop_s"]:
                for tail in (0.0, mult):
                    counts = hist[(hop, tail)]
                    frac = counts / max(int(counts.sum()), 1)
                    label = f"hop {hop*1000:4.0f} ms  tail {tail:.1f}x"
                    print(f"    {label}  " + "  ".join(
                        f"{n} {f:.3f}" for n, f in zip(sl.STATE_NAMES, frac)))
            if problems:
                print(f"    {len(problems)} MANIFEST MISMATCHES")
                for p in problems[:10]:
                    print(f"      {p}")

        if all_problems:
            msg = (f"{len(all_problems)} manifest mismatches. The rebuilt spans "
                   f"disagree with the columns build_manifest.py derived from "
                   f"them, so the labels cannot be trusted.")
            if cfg.get("strict", True):
                raise SystemExit(f"FAILED: {msg}")
            print(f"WARNING: {msg}")

    print("\nThe class fractions above are the class weights for the detector's "
          "cross-entropy (inverse frequency) -- decisions-pending.md D14.")


if __name__ == "__main__":
    main()
