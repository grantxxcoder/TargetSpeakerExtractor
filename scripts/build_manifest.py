#!/usr/bin/env python3
"""Build the trial manifest for one split: one row per trial, every random
draw decided here. Reads file headers only, never audio samples.

    python scripts/build_manifest.py --split val

Transcripts are not copied into the manifest. They live in the utterance
index, keyed by utterance id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import yaml

COLUMNS = [
    "trial_id", "split",
    "target_speaker", "target_chapter", "target_utts", "target_onsets_s",
    "target_speech_s", "target_activity",
    "interferer_speaker", "interferer_chapter", "interferer_utts",
    "interferer_onsets_s", "interferer_speech_s", "interferer_activity",
    "enrollment_utt", "enrollment_offset_s", "enrollment_length_s", "enrollment_eq",
    "noise_clip", "noise_offset_s",
    "mixture_length_s", "sir_db", "snr_db", "target_loudness_lufs",
    "overlap_requested", "overlap_achieved",
    "t60_s", "room_l", "room_w", "room_h",
    "mic_x", "mic_y", "mic_z",
    "target_x", "target_y", "target_z",
    "interferer_x", "interferer_y", "interferer_z",
    "target_absent", "same_gender",
]


def rngs(trial_id, n):
    """Independent RNG streams for one trial, derived from its id alone.

    blake2b rather than hash(): Python salts string hashing per process, so
    hash() would give different audio on every run.
    """
    digest = hashlib.blake2b(trial_id.encode(), digest_size=8).digest()
    seeds = np.random.SeedSequence(int.from_bytes(digest, "big")).spawn(n)
    return [np.random.default_rng(s) for s in seeds]


def draw(rng, value):
    return float(rng.uniform(*value)) if isinstance(value, list) else float(value)


# --- corpus metadata ------------------------------------------------------

def read_speakers(root):
    """{speaker_id: (sex, subset)} from SPEAKERS.TXT."""
    out = {}
    for line in (root / "SPEAKERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        f = [p.strip() for p in line.split("|")]
        out[f[0]] = (f[1], f[2])
    return out


def read_books(root):
    """{chapter_id: book_id} from CHAPTERS.TXT."""
    out = {}
    for line in (root / "CHAPTERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        f = [p.strip() for p in line.split("|")]
        out[f[0]] = f[5]
    return out


def index_utterances(root, speakers, subset_of, cache):
    """One row per LibriSpeech utterance: id, speaker, chapter, duration, text."""
    if cache.exists():
        with cache.open() as f:
            return list(csv.DictReader(f))

    rows = []
    for i, sid in enumerate(speakers, 1):
        print(f"\rindexing speakers {i}/{len(speakers)}", end="", file=sys.stderr)
        for chapter in sorted((root / subset_of[sid] / sid).iterdir()):
            trans = {}
            for line in (chapter / f"{sid}-{chapter.name}.trans.txt").read_text().splitlines():
                utt, text = line.split(" ", 1)
                trans[utt] = text
            for flac in sorted(chapter.glob("*.flac")):
                rows.append({
                    "utt": flac.stem,
                    "speaker": sid,
                    "chapter": chapter.name,
                    "duration": f"{sf.info(flac).duration:.3f}",
                    "text": trans[flac.stem],
                })
    print(file=sys.stderr)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt", "speaker", "chapter", "duration", "text"])
        w.writeheader()
        w.writerows(rows)
    return rows


def index_noise(root, noise_split, cache):
    """One row per WHAM! noise clip: file name and duration."""
    if cache.exists():
        with cache.open() as f:
            return list(csv.DictReader(f))

    rows = [{"clip": p.name, "duration": f"{sf.info(p).duration:.3f}"}
            for p in sorted((root / noise_split).glob("*.flac"))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "duration"])
        w.writeheader()
        w.writerows(rows)
    return rows


# --- sampling ------------------------------------------------------------

def sample_room(rng, cfg):
    """Room dimensions, T60 and mic position. Redraws until the T60 is
    physically reachable in the room (small T60 in a big room is not)."""
    for _ in range(50):
        dims = [draw(rng, cfg["room_length_m"]),
                draw(rng, cfg["room_width_m"]),
                draw(rng, cfg["room_height_m"])]
        t60 = draw(rng, cfg["t60_s"])
        try:
            pra.inverse_sabine(t60, dims)
        except ValueError:
            continue
        mic = [rng.uniform(0.35 * dims[0], 0.65 * dims[0]),
               rng.uniform(0.35 * dims[1], 0.65 * dims[1]),
               draw(rng, cfg["mic_height_m"])]
        return dims, t60, mic
    raise RuntimeError("no feasible room after 50 draws")


def place_source(rng, cfg, dims, mic):
    """A point at a sampled distance and random angle from the mic, kept at
    least 0.3 m off every wall."""
    for _ in range(50):
        d = draw(rng, cfg["source_distance_m"])
        az = rng.uniform(0, 2 * np.pi)
        x, y = mic[0] + d * np.cos(az), mic[1] + d * np.sin(az)
        if 0.3 < x < dims[0] - 0.3 and 0.3 < y < dims[1] - 0.3:
            return [x, y, draw(rng, cfg["source_height_m"])]
    raise RuntimeError("no feasible source position after 50 draws")


def pick_run(rng, chapters, by_chapter, wanted_s, cap_s, max_n):
    """A run of consecutive utterances from one chapter, totalling between
    wanted_s and cap_s. Consecutive and single-chapter so the joined audio is
    continuous reading and the joined transcript is exact.
    """
    for _ in range(20):
        chapter = chapters[rng.integers(len(chapters))]
        utts = by_chapter[chapter]
        run, total = [], 0.0
        for u in utts[rng.integers(len(utts)):][:max_n]:
            d = float(u["duration"])
            if total + d > cap_s:
                break
            run.append(u)
            total += d
            if total >= wanted_s:
                return chapter, run, total
    return None


def lay_out(rng, run, span_s):
    """Onsets for a run of utterances inside span_s, with the leftover time
    scattered as gaps before, between and after them."""
    durations = [float(u["duration"]) for u in run]
    gaps = rng.dirichlet(np.ones(len(run) + 1)) * (span_s - sum(durations))
    onsets, t = [], gaps[0]
    for d, gap in zip(durations, gaps[1:]):
        onsets.append(t)
        t += d + gap
    return onsets


def spans(run, onsets):
    return [(o, o + float(u["duration"])) for u, o in zip(run, onsets)]


def shared_seconds(a, b):
    return sum(max(0.0, min(x2, y2) - max(x1, y1)) for x1, x2 in a for y1, y2 in b)


def best_onset(target_spans, block_s, length, wanted_s):
    """Slide the interferer block across the window and take the onset whose
    overlap with the target comes closest to wanted_s. Searching beats solving
    it in closed form because the target has gaps in it."""
    candidates = np.linspace(0.0, max(0.0, length - block_s), 400)
    overlaps = [shared_seconds(target_spans, [(c, c + block_s)]) for c in candidates]
    i = int(np.argmin([abs(o - wanted_s) for o in overlaps]))
    return float(candidates[i]), overlaps[i]


def build_trial(trial_id, split, cfg, speakers, sex, book, by_speaker,
                by_chapter, chapters_of, noise):
    pick, level, room, aug = rngs(trial_id, 4)

    length = draw(level, cfg["mixture_length_s"])
    absent = aug.random() < cfg["target_absent_fraction"]
    enroll_len = draw(level, cfg["enrollment_length_s"])
    same_gender = pick.random() < cfg["same_gender_fraction"]
    overlap_requested = 0.0 if absent else draw(level, cfg["overlap_ratio"])
    max_n = cfg["max_utterances_per_source"]

    for _ in range(20):
        target = str(pick.choice(speakers))
        pool = [s for s in speakers
                if s != target and (sex[s] == sex[target]) == same_gender]
        if not pool:
            continue
        interferer = str(pick.choice(pool))

        long_enough = [u for u in by_speaker[target]
                       if float(u["duration"]) >= enroll_len]
        if not long_enough:
            continue

        if absent:
            target_chapter, target_run, target_speech = None, [], 0.0
            target_spans = []
            target_onsets = []
            enrollment = long_enough[pick.integers(len(long_enough))]

            # The interferer's activity is drawn from the same distribution a
            # present trial would have produced. Pinning it at
            # target_activity_ratio (the previous behaviour) made absent trials
            # identifiable without listening: their interferer always talked
            # 0.75-0.85 of the window, where present-trial interferers span
            # roughly 0.2-0.9. A model could then emit silence whenever one
            # voice talks near-continuously and never consult the enrollment --
            # the exact shortcut target-absent trials exist to prevent.
            # decisions.md 2026-08-11.
            shadow_overlap = draw(level, cfg["overlap_ratio"])
            shadow_t_act = cfg["target_activity_ratio"] + level.uniform(
                0.0, cfg["activity_tolerance"])
            i_act = level.uniform(shadow_overlap,
                                  min(1.0, 1.0 - shadow_t_act + shadow_overlap))
        else:
            t_wanted = cfg["target_activity_ratio"] * length
            found = pick_run(pick, chapters_of[target], by_chapter, t_wanted,
                             t_wanted + cfg["activity_tolerance"] * length, max_n)
            if found is None:
                continue
            target_chapter, target_run, target_speech = found

            # The same guard the interferer gets: a different *book*, not merely a
            # different chapter. A LibriSpeech speaker usually reads consecutive
            # chapters of one book, so a different chapter of the same book still
            # shares narrative, characters, proper nouns and register -- enough
            # for the model to match enrollment to target on content rather than
            # on voice. decisions.md 2026-08-11 (was decisions-pending B8).
            other_book = [u for u in long_enough
                          if book[u["chapter"]] != book[target_chapter[1]]]
            if not other_book:
                continue
            enrollment = other_book[pick.integers(len(other_book))]

            target_onsets = lay_out(level, target_run, length)
            target_spans = spans(target_run, target_onsets)

            # The interferer must talk for at least the requested overlap, and
            # not so much that the two cannot avoid each other. Both bounds
            # follow from  max(0, t + i - 1) <= overlap <= min(t, i).
            t_act = target_speech / length
            lo, hi = overlap_requested, min(1.0, 1.0 - t_act + overlap_requested)
            if lo > hi:
                continue
            i_act = level.uniform(lo, hi)

        i_chapters = [c for c in chapters_of[interferer]
                      if target_chapter is None or book[c[1]] != book[target_chapter[1]]]
        if not i_chapters:
            continue
        found = pick_run(pick, i_chapters, by_chapter, i_act * length,
                         min((i_act + cfg["activity_tolerance"]) * length, length),
                         max_n)
        if found is None:
            continue
        interferer_chapter, interferer_run, interferer_speech = found

        # One contiguous block, so sliding it is the single free variable that
        # sets the overlap.
        if absent:
            i0, overlap_achieved = level.uniform(0, length - interferer_speech), 0.0
        else:
            i0, shared = best_onset(target_spans, interferer_speech, length,
                                    overlap_requested * length)
            overlap_achieved = shared / length
        if abs(overlap_achieved - overlap_requested) > cfg["overlap_tolerance"]:
            continue

        interferer_onsets, t = [], i0
        for u in interferer_run:
            interferer_onsets.append(t)
            t += float(u["duration"])
        break
    else:
        return None

    clip = noise[int(pick.integers(len(noise)))]
    noise_offset = pick.uniform(0, float(clip["duration"]))

    dims, t60, mic = sample_room(room, cfg)
    target_pos = place_source(room, cfg, dims, mic)
    interferer_pos = place_source(room, cfg, dims, mic)

    enroll_offset = pick.uniform(0, float(enrollment["duration"]) - enroll_len)

    assert target != interferer
    assert enrollment["speaker"] == target
    assert float(enrollment["duration"]) >= enroll_len >= 5.0
    assert target_onsets == sorted(target_onsets)
    assert all(0 <= o for o in target_onsets + interferer_onsets)
    assert interferer_onsets[-1] + float(interferer_run[-1]["duration"]) <= length + 1e-6
    if target_chapter is not None:
        assert book[enrollment["chapter"]] != book[target_chapter[1]]
        assert book[interferer_chapter[1]] != book[target_chapter[1]]
        assert target_onsets[-1] + float(target_run[-1]["duration"]) <= length + 1e-6

    return {
        "trial_id": trial_id,
        "split": split,
        "target_speaker": target,
        "target_chapter": "" if absent else target_chapter[1],
        "target_utts": "|".join(u["utt"] for u in target_run),
        "target_onsets_s": "|".join(f"{o:.4f}" for o in target_onsets),
        "target_speech_s": round(target_speech, 3),
        "target_activity": round(target_speech / length, 4),
        "interferer_speaker": interferer,
        "interferer_chapter": interferer_chapter[1],
        "interferer_utts": "|".join(u["utt"] for u in interferer_run),
        "interferer_onsets_s": "|".join(f"{o:.4f}" for o in interferer_onsets),
        "interferer_speech_s": round(interferer_speech, 3),
        "interferer_activity": round(interferer_speech / length, 4),
        "enrollment_utt": enrollment["utt"],
        "enrollment_offset_s": round(enroll_offset, 4),
        "enrollment_length_s": round(enroll_len, 3),
        "enrollment_eq": int(aug.random() < cfg["enrollment_eq_prob"]),
        "noise_clip": clip["clip"],
        "noise_offset_s": round(noise_offset, 4),
        "mixture_length_s": round(length, 3),
        "sir_db": "" if absent else round(draw(level, cfg["sir_db"]), 2),
        "snr_db": round(draw(level, cfg["snr_db"]), 2),
        "target_loudness_lufs": round(draw(level, cfg["target_loudness_lufs"]), 2),
        "overlap_requested": round(overlap_requested, 4),
        "overlap_achieved": round(overlap_achieved, 4),
        "t60_s": round(t60, 4),
        "room_l": round(dims[0], 3), "room_w": round(dims[1], 3), "room_h": round(dims[2], 3),
        "mic_x": round(mic[0], 3), "mic_y": round(mic[1], 3), "mic_z": round(mic[2], 3),
        "target_x": round(target_pos[0], 3), "target_y": round(target_pos[1], 3),
        "target_z": round(target_pos[2], 3),
        "interferer_x": round(interferer_pos[0], 3),
        "interferer_y": round(interferer_pos[1], 3),
        "interferer_z": round(interferer_pos[2], 3),
        "target_absent": int(absent),
        "same_gender": int(sex[interferer] == sex[target]),
    }


# --- entry point ---------------------------------------------------------

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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--out-dir", default="data/manifests")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    config_md5 = hashlib.md5(Path(args.config).read_bytes()).hexdigest()

    if args.split not in config["splits"]:
        sys.exit(f"unknown split '{args.split}'. Known: {sorted(config['splits'])}")
    cfg = {**config["defaults"], **config["splits"][args.split]}

    ls_root = Path(config["paths"]["librispeech"])
    splits = yaml.safe_load(Path(config["paths"]["splits"]).read_text())
    speakers = [str(s) for s in splits[args.split]]

    meta = read_speakers(ls_root)
    sex = {s: meta[s][0] for s in speakers}
    book = read_books(ls_root)
    subset_of = {s: meta[s][1] for s in speakers}

    index_dir = Path("data/index")
    utterances = index_utterances(ls_root, speakers, subset_of,
                                 index_dir / f"utterances_{args.split}.csv")
    noise = index_noise(Path(config["paths"]["wham_noise"]), cfg["noise_split"],
                        index_dir / f"noise_{cfg['noise_split']}.csv")

    by_speaker = defaultdict(list)
    by_chapter = defaultdict(list)
    for u in utterances:
        by_speaker[u["speaker"]].append(u)
        by_chapter[(u["speaker"], u["chapter"])].append(u)
    for key in by_chapter:
        by_chapter[key].sort(key=lambda u: u["utt"])
    chapters_of = defaultdict(list)
    for spk, chapter in sorted(by_chapter):
        chapters_of[spk].append((spk, chapter))

    rows, failed_ids = [], []
    for i in range(cfg["n_trials"]):
        trial_id = f"{args.split}-{config['seed']}-{i:06d}"
        row = build_trial(trial_id, args.split, cfg, speakers, sex, book,
                          by_speaker, by_chapter, chapters_of, noise)
        if row is None:
            failed_ids.append(trial_id)
        else:
            rows.append(row)
    failed = len(failed_ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.split}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # Unsatisfiable trials are dropped, and the ones that fail are the hard ones
    # -- tight overlap, low interferer activity, speakers with little material.
    # Their absence therefore biases the split. Record which ids were lost so the
    # bias can be characterised instead of guessed at.
    failed_file = out_dir / f"{args.split}.failed.txt"
    if failed_ids:
        failed_file.write_text("\n".join(failed_ids) + "\n")
    elif failed_file.exists():
        failed_file.unlink()

    (out_dir / f"{args.split}.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/build_manifest.py",
        "split": args.split,
        "seed": config["seed"],
        "config": args.config,
        "config_md5": config_md5,
        "git_commit": git_commit(),
        "n_trials": len(rows),
        "n_requested": cfg["n_trials"],
        "n_failed": failed,
        "failed_ids_file": failed_file.name if failed_ids else None,
    }, sort_keys=False))

    print(f"Wrote {out}  ({len(rows)} trials, {failed} unsatisfiable)")
    absent = sum(r["target_absent"] for r in rows)
    same = sum(r["same_gender"] for r in rows)
    present = [r for r in rows if not r["target_absent"]]
    print(f"  target absent  {absent / len(rows):.2f}")
    print(f"  same gender    {same / len(rows):.2f}")
    for name in ("overlap_requested", "overlap_achieved", "target_activity",
                 "interferer_activity"):
        v = [r[name] for r in present]
        if v:
            print(f"  {name:<20} mean {np.mean(v):.2f}  "
                  f"min {min(v):.2f}  max {max(v):.2f}")
    n_utts = [len(r["target_utts"].split("|")) for r in present]
    if n_utts:
        print(f"  utterances per target  mean {np.mean(n_utts):.1f}  max {max(n_utts)}")


if __name__ == "__main__":
    main()
