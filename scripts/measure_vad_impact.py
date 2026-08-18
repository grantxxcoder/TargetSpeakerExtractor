#!/usr/bin/env python3
"""Measure what B2 changes, and record it as an experiment result.

    ../tse_venv/bin/python scripts/measure_vad_impact.py \
        --out experiments/results/2026-08-15-vad-impact

This is the evidence behind decisions-m0.md 2026-08-15. It answers three questions
and writes the answers to disk with the config, commit hash, seed and date, so
the numbers can be re-derived rather than trusted.

  PART 1  How much of a LibriSpeech utterance file is actually speech, and how
          much does that depend on the detector settings? Swept, so the chosen
          250 ms is visibly a decision and not a default.

  PART 2  Taking real manifest rows with their recorded onsets UNCHANGED, how
          far apart are file-boundary overlap and speech overlap? What that
          means depends on which manifest you point it at, and the script
          detects which by looking for PR2's `target_footprint_s` column:

            pre-PR2 manifest   the gap IS the label error, because the stored
                               column is file-boundary overlap. This is the
                               measurement B2 was argued from.
            post-PR2 manifest  the stored column is already speech overlap, so
                               the gap is what B2 is worth -- how wrong these
                               labels would be had it not been done.

  PART 3  The three candidate definitions of `interrupted`, measured side by
          side. Option A was chosen 2026-08-15; this is what it was chosen over,
          and post-PR2 it is what the manifests carry.

Part 2's CHECK line prints first and must pass. It recomputes overlap the way
the manifest's own generation measured it and confirms that reproduces the
stored column. If it does not, this script is misreading build_manifest.py's
placement and every number below it is meaningless.

Pointing it at the pre-PR2 manifests still works and still reproduces the
2026-08-16 "before" figures, which is the point of detecting rather than
assuming: a recorded number nobody can re-derive is a claim, not a measurement.

Nothing is written outside --out. No manifest is rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import vad  # noqa: E402
from src.run_log import timed  # noqa: E402

# (label, threshold, min_silence_ms, min_speech_ms, speech_pad_ms)
GRID = [
    ("silero defaults",       0.5, 100, 250, 30),
    ("min_sil 100",           0.5, 100, 100, 30),
    ("min_sil 200",           0.5, 200, 100, 30),
    ("min_sil 250 (CHOSEN)",  0.5, 250, 100, 30),
    ("min_sil 300",           0.5, 300, 100, 30),
    ("min_sil 500",           0.5, 500, 100, 30),
    ("min_sil 250 thr 0.3",   0.3, 250, 100, 30),
    ("min_sil 250 thr 0.7",   0.7, 250, 100, 30),
]
CHOSEN = "min_sil 250 (CHOSEN)"

_sr = None
_ls_root = None
_subset = None
_grid = None


def _init(sr, ls_root, subset, grid):
    global _sr, _ls_root, _subset, _grid
    _sr, _ls_root, _subset, _grid = sr, Path(ls_root), subset, grid
    import torch
    torch.set_num_threads(1)


def _work(utt):
    """One decode, every setting in the grid -- so the sweep costs one read."""
    import torch
    speaker, chapter, _ = utt.split("-")
    wav, sr = sf.read(_ls_root / _subset[speaker] / speaker / chapter / f"{utt}.flac",
                      dtype="float32")
    t = torch.from_numpy(wav)
    model = vad.load_model()
    out = {}
    for label, thr, min_sil, min_sp, pad in _grid:
        out[label] = vad.detect(t, model, {
            "threshold": thr, "min_silence_duration_ms": min_sil,
            "min_speech_duration_ms": min_sp, "speech_pad_ms": pad}, _sr)
    return utt, len(wav) / sr, out


def run_pool(items, grid, sr, ls_root, subset, workers, label):
    t0 = time.time()
    out = []
    with Pool(workers, initializer=_init,
              initargs=(sr, str(ls_root), subset, grid)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, items, chunksize=4), 1):
            out.append(r)
            if i % 25 == 0 or i == len(items):
                el = time.time() - t0
                print(f"\r  {label}: {i}/{len(items)}  [{el:5.0f}s elapsed, "
                      f"~{el/i*(len(items)-i):4.0f}s left]   ",
                      end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


def read_speakers(root):
    out = {}
    for line in (root / "SPEAKERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        f = [p.strip() for p in line.split("|")]
        out[f[0]] = f[2]
    return out


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def pct(v, p):
    return float(np.percentile(v, p))


# --- part 1 ---------------------------------------------------------------

def part1(rng, index_dir, sr, ls_root, subset, n, workers):
    seen, utts = set(), []
    for path in sorted(index_dir.glob("utterances_*.csv")):
        for row in csv.DictReader(path.open()):
            if row["utt"] not in seen:
                seen.add(row["utt"])
                utts.append(row)
    sample = [utts[i] for i in rng.choice(len(utts), n, replace=False)]
    hours = sum(float(r["duration"]) for r in sample) / 3600

    print("=" * 86)
    print(f"PART 1  --  how much of an utterance file is speech")
    print(f"            {n} utterances sampled from {len(utts)} indexed ({hours:.2f} h)")
    print("=" * 86, flush=True)

    res = run_pool([r["utt"] for r in sample], GRID, sr, ls_root, subset,
                   workers, "part 1")

    print(f"\n{'setting':<24} {'speech/dur':>10} {'median':>8} {'p10':>7} {'p90':>7} "
          f"{'segs':>6} {'lead_s':>7} {'trail_s':>8}")
    print("-" * 86)
    table = {}
    for label, *_ in GRID:
        ratio, nseg, lead, trail = [], [], [], []
        for _, dur, out in res:
            segs = out[label]
            ratio.append(vad.total_speech(segs) / dur)
            nseg.append(len(segs))
            lead.append(segs[0][0] if segs else dur)
            trail.append(dur - segs[-1][1] if segs else 0.0)
        table[label] = {
            "speech_ratio_mean": round(float(np.mean(ratio)), 4),
            "speech_ratio_median": round(float(np.median(ratio)), 4),
            "speech_ratio_p10": round(pct(ratio, 10), 4),
            "speech_ratio_p90": round(pct(ratio, 90), 4),
            "segments_mean": round(float(np.mean(nseg)), 3),
            "leading_silence_s": round(float(np.mean(lead)), 4),
            "trailing_silence_s": round(float(np.mean(trail)), 4),
        }
        t = table[label]
        print(f"{label:<24} {t['speech_ratio_mean']:>10.3f} "
              f"{t['speech_ratio_median']:>8.3f} {t['speech_ratio_p10']:>7.3f} "
              f"{t['speech_ratio_p90']:>7.3f} {t['segments_mean']:>6.2f} "
              f"{t['leading_silence_s']:>7.3f} {t['trailing_silence_s']:>8.3f}")

    print("\n  speech/dur = fraction of the FILE that is detected speech")
    print("  segs       = speech segments per utterance (1.0 = no internal pause)")
    print("  lead/trail = mean silence before the first / after the last word")
    print("\n  The whole 100-500 ms range spans only "
          f"{table['min_sil 100']['speech_ratio_mean']:.3f}-"
          f"{table['min_sil 500']['speech_ratio_mean']:.3f}, so the headline is")
    print("  robust to this setting. That robustness is why 250 ms is defensible.")
    return table


# --- parts 2 and 3 --------------------------------------------------------

def part2(rng, manifest, sr, ls_root, subset, n, workers, tolerance):
    rows = [r for r in csv.DictReader(manifest.open()) if r["condition"] == "both"]
    sample = [rows[i] for i in rng.choice(len(rows), n, replace=False)]

    # PR2 added *_footprint_s at the same time as it changed what *_speech_s and
    # overlap_achieved mean, so the column's presence identifies the generation
    # of the manifest. Detected rather than assumed, so the pre-PR2 numbers in
    # decisions-m0.md 2026-08-16 stay re-derivable from the backed-up manifests.
    speech_based = "target_footprint_s" in rows[0]

    need = set()
    for r in sample:
        need |= set(r["target_utts"].split("|")) | set(r["interferer_utts"].split("|"))
    need = sorted(need)

    print("\n\n" + "=" * 86)
    headline = ("agreement between the manifest and the detector"
                if speech_based else "label error")
    print(f"PART 2  --  {headline} in {manifest}, onsets UNCHANGED")
    print(f"            {n} `both` trials, {len(need)} distinct utterances, "
          f"setting: {CHOSEN}")
    print("=" * 86, flush=True)

    grid = [g for g in GRID if g[0] == CHOSEN]
    res = run_pool(need, grid, sr, ls_root, subset, workers, "part 2")
    seg = {u: out[CHOSEN] for u, _, out in res}
    dur = {u: d for u, d, _ in res}

    rec, old, new = [], [], []
    t_old, t_new, i_old, i_new = [], [], [], []
    interr = {"old (file onsets)": [], "A: first onset per utterance": [],
              "A': first onset per trial": [], "B: every speech onset": []}

    for r in sample:
        L = float(r["mixture_length_s"])
        tu, iu = r["target_utts"].split("|"), r["interferer_utts"].split("|")
        to = [float(x) for x in r["target_onsets_s"].split("|")]
        io_ = [float(x) for x in r["interferer_onsets_s"].split("|")]

        ts_file = [(o, o + dur[u]) for u, o in zip(tu, to)]
        is_file = [(o, o + dur[u]) for u, o in zip(iu, io_)]
        ts_vad = vad.spans_of([seg[u] for u in tu], to)
        is_vad = vad.spans_of([seg[u] for u in iu], io_)

        rec.append(float(r["overlap_achieved"]))
        old.append(vad.shared_seconds(ts_file, is_file) / L)
        new.append(vad.shared_seconds(ts_vad, is_vad) / L)
        # The file-boundary activity has to be recomputed on a post-PR2 manifest:
        # its `target_activity` column is already the speech figure, so reading it
        # here would compare the VAD number against itself and print 0.0 % change
        # under a heading that says "file-bound". Pre-PR2 the column IS the
        # file-boundary figure, so it is read directly.
        if speech_based:
            t_old.append(float(r["target_footprint_s"]) / L)
            i_old.append(float(r["interferer_footprint_s"]) / L)
        else:
            t_old.append(float(r["target_activity"]))
            i_old.append(float(r["interferer_activity"]))
        t_new.append(vad.total_speech(ts_vad) / L)
        i_new.append(vad.total_speech(is_vad) / L)

        first = vad.onsets_of([seg[u] for u in iu], io_, first_only=True)
        every = vad.onsets_of([seg[u] for u in iu], io_, first_only=False)
        interr["old (file onsets)"].append(vad.is_interrupted(ts_file, io_))
        interr["A: first onset per utterance"].append(
            vad.is_interrupted(ts_vad, first))
        interr["A': first onset per trial"].append(
            vad.is_interrupted(ts_vad, first[:1]))
        interr["B: every speech onset"].append(vad.is_interrupted(ts_vad, every))

    rec, old, new = map(np.array, (rec, old, new))

    # Which convention built this manifest decides what the check asserts.
    # Pre-PR2 the stored column was file-boundary overlap, so `old` must
    # reproduce it. Post-PR2 it is speech overlap, so `new` must. Asserting the
    # wrong one is not a near miss -- the two differ by ~0.1 on every row.
    reference, ref_name = ((new, "VAD") if speech_based
                           else (old, "file-boundary"))
    d = float(np.abs(reference - rec).max())
    ok = d < 1e-3
    print(f"\n  CHECK   {ref_name} overlap recomputed from the recorded onsets vs the "
          f"stored\n          `overlap_achieved` column: max|diff| = {d:.5f}"
          f"   {'OK' if ok else '*** MISMATCH -- everything below is void ***'}")
    if speech_based:
        print("          Post-PR2 manifest: the column IS the speech overlap, so this")
        print("          is a regression check that the two have not drifted apart.")
    else:
        print("          Pre-PR2 manifest: the column is file-boundary overlap, so the")
        print("          `change` column below is live label error, not a what-if.")

    print(f"\n  {'quantity':<32} {'file-bound':>11} {'VAD':>9} {'change':>9}")
    print("  " + "-" * 64)
    summary = {}
    for name, a, b in [
        ("overlap_achieved (mean)",    float(old.mean()),   float(new.mean())),
        ("overlap_achieved (median)",  float(np.median(old)), float(np.median(new))),
        ("target_activity (mean)",     float(np.mean(t_old)), float(np.mean(t_new))),
        ("interferer_activity (mean)", float(np.mean(i_old)), float(np.mean(i_new))),
    ]:
        summary[name] = {"file_boundary": round(a, 4), "vad": round(b, 4),
                         "change_pct": round(100 * (b / a - 1), 2) if a else None}
        print(f"  {name:<32} {a:>11.3f} {b:>9.3f} {100*(b/a-1):>8.1f}%")

    zero = int((new < 1e-6).sum())
    missed = int(sum(abs(x - float(r["overlap_requested"])) > tolerance
                     for x, r in zip(reference, sample)))
    err = np.abs(old - new)
    print(f"\n  overlap collapses to ZERO:          {zero}/{len(new)} "
          f"({100*zero/len(new):.1f} %)")
    print("      both speak, never simultaneously -- the shared interval is one")
    print("      speaker's trailing silence and the other's leading silence")
    print(f"  outside overlap_tolerance {tolerance}:      {missed}/{len(new)} "
          f"({100*missed/len(new):.1f} %)")
    if speech_based:
        print("      trials whose ACHIEVED speech overlap missed what was REQUESTED.")
        print("      Should be ~0: best_onset optimises against this exact quantity,")
        print("      and a trial outside tolerance is rejected and redrawn.")
    else:
        print("      NOT a rejection rate: it measures how far placement must move")
        print("      once overlap is remeasured, and is an UPPER BOUND on rejections.")
    print(f"  per-trial |file - VAD| overlap:      mean {err.mean():.3f}  "
          f"p90 {pct(err, 90):.3f}  max {err.max():.3f}")
    if speech_based:
        print("      what B2 is worth on this placement: how wrong these labels")
        print("      WOULD be if overlap were still read off file boundaries")
    else:
        print("      varies per trial, so no correction factor can fix it -- this is")
        print("      what puts trials in the wrong B13 overlap bucket")

    print("\n\n" + "=" * 86)
    print("PART 3  --  candidate definitions of `interrupted`")
    print("=" * 86)
    print(f"\n  {'definition':<32} {'rate':>8}   note")
    print("  " + "-" * 74)
    notes = {
        "old (file onsets)": ("the pre-PR2 reading, kept for contrast"
                              if speech_based else "what the manifests carry today"),
        "A: first onset per utterance": ("CHOSEN 2026-08-15 -- what the manifests "
                                         "carry" if speech_based else
                                         "CHOSEN 2026-08-15 -- minimal correction"),
        "A': first onset per trial": "only the interferer's very first word",
        "B: every speech onset": "a breath pause counts as a new turn",
    }
    interrupted = {}
    for k, v in interr.items():
        interrupted[k] = round(float(np.mean(v)), 4)
        print(f"  {k:<32} {np.mean(v):>8.3f}   {notes[k]}")
    print("\n  The spread across definitions is larger than most effects this")
    print("  project will report, which is why it is pinned in decisions-m0.md.")

    return {
        "manifest_convention": "speech" if speech_based else "file_boundary",
        "check_reference": ref_name,
        "sanity_max_abs_diff": round(d, 6),
        "sanity_passed": ok,
        "summary": summary,
        "overlap_collapses_to_zero": zero,
        "outside_overlap_tolerance": missed,
        "overlap_tolerance": tolerance,
        "per_trial_abs_error": {"mean": round(float(err.mean()), 4),
                                "p90": round(pct(err, 90), 4),
                                "max": round(float(err.max()), 4)},
        "interrupted": interrupted,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--manifest", default="data/manifests/train.csv")
    ap.add_argument("--out", default="experiments/results/2026-08-15-vad-impact")
    ap.add_argument("--n-utterances", type=int, default=2000)
    ap.add_argument("--n-trials", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    sr = int(config["sample_rate"])
    ls_root = Path(config["paths"]["librispeech"])
    subset = read_speakers(ls_root)
    tolerance = float(config["defaults"]["overlap_tolerance"])

    manifest_meta = Path(str(args.manifest).replace(".csv", ".meta.yaml"))
    built_at = (yaml.safe_load(manifest_meta.read_text())
                if manifest_meta.exists() else {})

    buf = io.StringIO()
    with timed("scripts/measure_vad_impact.py",
               scope=f"{args.n_utterances:,} utts x {len(GRID)} settings "
                     f"+ {args.n_trials} trials",
               rate=f"{args.workers} workers"):
        with redirect_stdout(buf):
            rng = np.random.default_rng(args.seed)
            t1 = part1(rng, Path(args.index_dir), sr, ls_root, subset,
                       args.n_utterances, args.workers)
            t2 = part2(rng, Path(args.manifest), sr, ls_root, subset,
                       args.n_trials, args.workers, tolerance)
    report = buf.getvalue()
    print(report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text(report)
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "date": date.today().isoformat(),
        "script": "scripts/measure_vad_impact.py",
        "git_commit": git_commit(),
        "seed": args.seed,
        "config": args.config,
        "detector": "silero-vad",
        "detector_version": vad.model_version(),
        "manifest": args.manifest,
        "manifest_built_at_commit": built_at.get("git_commit"),
        "manifest_config_md5": built_at.get("config_md5"),
        "n_utterances_sampled": args.n_utterances,
        "n_trials_sampled": args.n_trials,
        "part1_settings_sweep": t1,
        "part2_label_error": t2,
    }, sort_keys=False))
    print(f"\nWrote {out}/report.txt and {out}/meta.yaml")
    if not t2["sanity_passed"]:
        sys.exit("SANITY CHECK FAILED -- results are void")


if __name__ == "__main__":
    main()
