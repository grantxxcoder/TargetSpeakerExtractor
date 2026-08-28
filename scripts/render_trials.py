#!/usr/bin/env python3
"""Render a split's manifest to audio on disk.

    ../tse_venv/bin/python scripts/render_trials.py --split val
    ../tse_venv/bin/python scripts/render_trials.py --split train --limit 100

Layout, one directory per trial:

    data/rendered/<split>/<trial_id>/
        mixture.wav      what the model hears
        target.wav       A1's reference -- the target through its own room, alone
        enrollment.wav   dry, no room (A4)
        interferer.wav   the interferer through its own room, alone -- the
                         reference for the second training direction
        interferer_enrollment.wav   dry, the other speaker's conditioning clip
                         (a PHANTOM speaker when nobody interferes)
        meta.json        both transcripts, the gains applied, the EQ curves

16-bit WAV at 16 kHz, chosen 2026-08-16: decode-free, so a training step is a
pure disk read. That is the point of pre-rendering (decisions-m0.md 2026-08-15) and
it matters on a 4-vCPU box, where a CPU-bound loader starves the GPU.

Resumable. A trial whose four files all exist is skipped, so an interrupted run
is restarted by re-issuing the same command. `--force` re-renders regardless.

**Rendering is invalidated by any manifest rebuild.** The manifest fixes every
level, position and onset, so a rebuilt manifest describes different audio. The
run records the manifest's `config_md5` and refuses to extend a directory built
from a different one unless `--force` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import render  # noqa: E402
from src.run_log import timed  # noqa: E402

# Also gates the "already rendered?" check at line ~85, so adding a stem here
# is what makes existing trial directories count as INCOMPLETE and get redone.
# Omitting the new names would leave every old trial silently skipped.
STEMS = ("mixture", "target", "enrollment",
         "interferer", "interferer_enrollment")
_ctx = {}


def read_index(path):
    """utt -> (transcript, subset). The renderer needs the subset to find the flac."""
    with path.open() as f:
        return {r["utt"]: r for r in csv.DictReader(f)}


def flac_paths(ls_root, index, speaker_subset):
    """utt -> .flac path. Built once and shared, rather than derived per trial."""
    out = {}
    for utt, r in index.items():
        out[utt] = (ls_root / speaker_subset[r["speaker"]] / r["speaker"]
                    / r["chapter"] / f"{utt}.flac")
    return out


def read_speaker_subsets(ls_root):
    """Which LibriSpeech subset each speaker lives in, from SPEAKERS.TXT."""
    out = {}
    for line in (ls_root / "SPEAKERS.TXT").read_text().splitlines():
        if line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            out[parts[0]] = parts[2]
    return out


def _init(cfg, flac_of, texts, noise_root, noise_split, out_dir, force):
    _ctx.update(cfg=cfg, flac_of=flac_of, texts=texts, noise_root=noise_root,
                noise_split=noise_split, out_dir=out_dir, force=force)


def done(trial_dir):
    return (all((trial_dir / f"{s}.wav").exists() for s in STEMS)
            and (trial_dir / "meta.json").exists())


def _work(row):
    """Render one trial and write it. Returns (trial_id, status, note)."""
    trial_dir = _ctx["out_dir"] / row["trial_id"]
    if not _ctx["force"] and done(trial_dir):
        return row["trial_id"], "skipped", ""
    try:
        stems, meta = render.render_trial(
            row, _ctx["cfg"], _ctx["flac_of"], _ctx["texts"],
            _ctx["noise_root"], _ctx["noise_split"])
    except Exception as e:                                # noqa: BLE001
        return row["trial_id"], "failed", f"{type(e).__name__}: {e}"

    # Written to a temp dir and moved, so an interrupted write cannot leave a
    # half-finished trial that `done()` would later accept as complete.
    tmp = trial_dir.with_name(trial_dir.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name in STEMS:
        sf.write(tmp / f"{name}.wav", stems[name], _ctx["cfg"]["sample_rate"],
                 subtype="PCM_16")
    (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    tmp.rename(trial_dir)
    return row["trial_id"], "rendered", ""


def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:                                     # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--out-dir", default="data/rendered")
    ap.add_argument("--limit", type=int, default=None,
                    help="render only the first N trials -- use this to TIME the "
                         "job before committing to the full pass")
    ap.add_argument("--trials", default=None,
                    help="comma-separated trial ids, for inspecting one case "
                         "without rendering the split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="re-render trials that already exist")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    cfg = {"sample_rate": config["sample_rate"]}
    ls_root = Path(config["paths"]["librispeech"])
    noise_root = Path(config["paths"]["wham_noise"])
    noise_split = config["splits"][args.split]["noise_split"]

    manifest = Path(args.manifest_dir) / f"{args.split}.csv"
    manifest_meta = yaml.safe_load(
        (Path(args.manifest_dir) / f"{args.split}.meta.yaml").read_text())
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if args.trials:
        wanted = [t.strip() for t in args.trials.split(",") if t.strip()]
        by_id = {r["trial_id"]: r for r in rows}
        missing = [t for t in wanted if t not in by_id]
        if missing:
            raise SystemExit(f"not in {manifest}: {', '.join(missing)}")
        rows = [by_id[t] for t in wanted]
    if args.limit:
        rows = rows[:args.limit]

    index = read_index(Path("data/index") / f"utterances_{args.split}.csv")
    texts = {u: r["text"] for u, r in index.items()}
    flac_of = flac_paths(ls_root, index, read_speaker_subsets(ls_root))

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to mix audio from two different manifests in one directory. The
    # files would look complete and be silently inconsistent with each other.
    stamp = out_dir / "render.meta.yaml"
    if stamp.exists() and not args.force:
        prev = yaml.safe_load(stamp.read_text())
        if prev.get("manifest_config_md5") != manifest_meta.get("config_md5"):
            raise SystemExit(
                f"{out_dir} was rendered from a different manifest "
                f"(config_md5 {prev.get('manifest_config_md5')} vs "
                f"{manifest_meta.get('config_md5')}). A rebuild invalidates the "
                "audio -- delete the directory, or pass --force.")

    counts = {"rendered": 0, "skipped": 0, "failed": 0}
    failures = []
    with timed(f"scripts/render_trials.py --split {args.split}",
               scope=lambda: f"{counts['rendered']:,} trials rendered",
               rate=f"{args.workers} workers, 16 kHz PCM_16"):
        with Pool(args.workers, initializer=_init,
                  initargs=(cfg, flac_of, texts, noise_root, noise_split,
                            out_dir, args.force)) as pool:
            for i, (tid, status, note) in enumerate(
                    pool.imap_unordered(_work, rows, chunksize=4), 1):
                counts[status] += 1
                if status == "failed":
                    failures.append((tid, note))
                if i % 25 == 0 or i == len(rows):
                    print(f"\r  {i}/{len(rows)}  rendered {counts['rendered']}  "
                          f"skipped {counts['skipped']}  failed {counts['failed']}",
                          end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    rendered_s = sum(float(r["mixture_length_s"]) + float(r["t60_s"]) for r in rows)
    stamp.write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/render_trials.py",
        "split": args.split,
        "git_commit": git_commit(),
        "config": args.config,
        "sample_rate": cfg["sample_rate"],
        "format": "wav_pcm16",
        # The manifest this audio belongs to. Audio and manifest are one artefact;
        # a rebuild of either without the other is a silent inconsistency.
        "manifest": str(manifest),
        "manifest_config_md5": manifest_meta.get("config_md5"),
        "manifest_git_commit": manifest_meta.get("git_commit"),
        "n_trials": len(rows),
        "n_rendered": counts["rendered"],
        "n_skipped": counts["skipped"],
        "n_failed": counts["failed"],
        "partial": bool(args.limit or args.trials),
        "audio_hours": round(rendered_s / 3600, 3),
    }, sort_keys=False))

    if failures:
        (out_dir / "failed.txt").write_text(
            "\n".join(f"{t}\t{n}" for t, n in failures) + "\n")
        for t, n in failures[:5]:
            print(f"  FAILED {t}  {n}", file=sys.stderr)

    print(f"Wrote {out_dir}  ({counts['rendered']} rendered, "
          f"{counts['skipped']} skipped, {counts['failed']} failed)")
    print(f"  {rendered_s / 3600:.2f} h of audio across {len(rows)} trials")
    if args.limit:
        full = len(list(csv.DictReader(manifest.open())))
        print(f"  TIMING RUN: {len(rows)} of {full} trials. Multiply by "
              f"{full / len(rows):.1f} for the full split.")
    if counts["failed"]:
        sys.exit(f"{counts['failed']} trial(s) failed -- see {out_dir}/failed.txt")


if __name__ == "__main__":
    main()
