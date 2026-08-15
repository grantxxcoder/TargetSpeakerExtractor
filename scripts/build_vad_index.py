#!/usr/bin/env python3
"""Run the voice-activity detector over every indexed utterance and cache where
the speech actually is (B2). Reads audio; writes one CSV.

    ../tse_venv/bin/python scripts/build_vad_index.py            # all splits
    ../tse_venv/bin/python scripts/build_vad_index.py --limit 200  # smoke test

Output: data/index/vad_segments.csv  (utt, duration, speech_s, segments)
        data/index/vad_segments.meta.yaml

WHY THE CACHE IS KEYED BY UTTERANCE, NOT BY SPLIT
-------------------------------------------------
`utterances_{split}.csv` is keyed by split NAME, which is why it went stale when
PR3 redrew the eval pools -- build_manifest.py:100 now carries a hand-written
guard for exactly that. VAD output for an utterance does not depend on which
split the utterance landed in, so one utterance-keyed store cannot go stale that
way. Redrawing splits just means some rows go unused; new speakers append.

The run is resumable. 137,876 utterances is ~30-60 min, and losing it to a
laptop sleep would be silly, so completed rows are flushed as they finish and a
re-run skips whatever is already present.

Nothing here is wired into build_manifest.py. That is PR2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

COLUMNS = ["utt", "duration", "speech_s", "segments"]

_cfg = None
_sr = None
_ls_root = None
_subset = None


def _init(cfg, sr, ls_root, subset):
    """Per-worker state. The model itself loads lazily on first use inside
    vad.load_model(), so it is never pickled across the process boundary."""
    global _cfg, _sr, _ls_root, _subset
    _cfg, _sr, _ls_root, _subset = cfg, sr, Path(ls_root), subset
    import torch
    torch.set_num_threads(1)  # 8 processes x N threads would oversubscribe


def _utt_path(utt):
    speaker, chapter, _ = utt.split("-")
    return _ls_root / _subset[speaker] / speaker / chapter / f"{utt}.flac"


def _work(utt):
    """One utterance -> its cache row. Returns None on a read failure so a single
    corrupt file cannot abort a 137k-file pass; failures are reported at the end."""
    try:
        wav, sr = sf.read(_utt_path(utt), dtype="float32")
        if sr != _sr:
            raise ValueError(f"{utt} is {sr} Hz, expected {_sr}")
        import torch
        segs = vad.detect(torch.from_numpy(wav), vad.load_model(), _cfg, _sr)
        return {
            "utt": utt,
            "duration": f"{len(wav) / sr:.4f}",
            "speech_s": f"{vad.total_speech(segs):.4f}",
            "segments": vad.format_segments(segs),
        }
    except Exception as e:                      # noqa: BLE001 - reported, not hidden
        print(f"\n  FAILED {utt}: {e}", file=sys.stderr)
        return None


def read_speakers(root):
    """{speaker_id: subset} from SPEAKERS.TXT, so an utterance id maps to a path."""
    out = {}
    for line in (root / "SPEAKERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        f = [p.strip() for p in line.split("|")]
        out[f[0]] = f[2]
    return out


def indexed_utterances(index_dir):
    """Every utterance named by any split's index, de-duplicated. Splits are
    speaker-disjoint, so this is a union with no overlap in practice."""
    seen = []
    known = set()
    for path in sorted(index_dir.glob("utterances_*.csv")):
        with path.open() as f:
            for row in csv.DictReader(f):
                if row["utt"] not in known:
                    known.add(row["utt"])
                    seen.append(row["utt"])
    return seen


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--index-dir", default="data/index")
    ap.add_argument("--out", default="data/index/vad_segments.csv")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N utterances, for a smoke test")
    ap.add_argument("--force", action="store_true",
                    help="ignore any existing cache and redo every utterance")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    cfg = vad.vad_config(config)
    sr = int(config["sample_rate"])
    ls_root = Path(config["paths"]["librispeech"])
    subset = read_speakers(ls_root)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = indexed_utterances(Path(args.index_dir))
    if args.limit:
        wanted = wanted[:args.limit]

    done = {}
    if out_path.exists() and not args.force:
        with out_path.open() as f:
            done = {r["utt"]: r for r in csv.DictReader(f)}
    todo = [u for u in wanted if u not in done]

    print(f"VAD index: {len(wanted)} utterances indexed, {len(done)} already cached, "
          f"{len(todo)} to do")
    print(f"  silero-vad {vad.model_version()}  {cfg}")

    rows = [done[u] for u in wanted if u in done]
    failed = []
    if todo:
        t0 = time.time()
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
            with Pool(args.workers, initializer=_init,
                      initargs=(cfg, sr, str(ls_root), subset)) as pool:
                for i, row in enumerate(pool.imap(_work, todo, chunksize=16), 1):
                    if row is None:
                        failed.append(i)
                    else:
                        rows.append(row)
                        w.writerow(row)
                    if i % 200 == 0 or i == len(todo):
                        f.flush()       # survive a kill without losing the pass
                        el = time.time() - t0
                        print(f"\r  {i}/{len(todo)}  [{el/60:5.1f} min elapsed, "
                              f"~{el/i*(len(todo)-i)/60:5.1f} min left]   ",
                              end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)

    dur = np.array([float(r["duration"]) for r in rows])
    speech = np.array([float(r["speech_s"]) for r in rows])
    nsegs = np.array([len(vad.parse_segments(r["segments"])) for r in rows])
    ratio = speech / dur
    lead = np.array([vad.parse_segments(r["segments"])[0][0]
                     if r["segments"] else float(r["duration"]) for r in rows])
    trail = np.array([float(r["duration"]) - vad.parse_segments(r["segments"])[-1][1]
                      if r["segments"] else 0.0 for r in rows])

    Path(str(out_path).replace(".csv", ".meta.yaml")).write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/build_vad_index.py",
        "git_commit": git_commit(),
        "config": args.config,
        "config_md5": hashlib.md5(Path(args.config).read_bytes()).hexdigest(),
        "detector": "silero-vad",
        "detector_version": vad.model_version(),
        "detector_settings": cfg,
        "sample_rate": sr,
        "n_utterances": len(rows),
        "n_failed": len(failed),
        "n_no_speech_detected": int((nsegs == 0).sum()),
        # Recorded here as well as in decisions.md: the manifests are not in git,
        # so this sidecar is the only travelling record of what produced them.
        "speech_ratio_mean": round(float(ratio.mean()), 4),
        "speech_ratio_median": round(float(np.median(ratio)), 4),
        "segments_per_utterance_mean": round(float(nsegs.mean()), 3),
        "leading_silence_s_mean": round(float(lead.mean()), 4),
        "trailing_silence_s_mean": round(float(trail.mean()), 4),
        "total_audio_hours": round(float(dur.sum() / 3600), 2),
        "total_speech_hours": round(float(speech.sum() / 3600), 2),
    }, sort_keys=False))

    print(f"\nWrote {out_path}  ({len(rows)} utterances, {len(failed)} failed)")
    print(f"  speech / duration      mean {ratio.mean():.4f}  "
          f"median {np.median(ratio):.4f}  p10 {np.percentile(ratio, 10):.4f}")
    print(f"  segments per utterance mean {nsegs.mean():.2f}")
    print(f"  leading silence        mean {lead.mean():.4f} s")
    print(f"  trailing silence       mean {trail.mean():.4f} s")
    print(f"  audio {dur.sum()/3600:.2f} h -> speech {speech.sum()/3600:.2f} h "
          f"({100*(1-speech.sum()/dur.sum()):.1f} % is silence)")
    if (nsegs == 0).any():
        print(f"  NOTE {int((nsegs==0).sum())} utterances had no speech detected; "
              f"they parse to [] and must not be chosen as enrollment")


if __name__ == "__main__":
    main()
