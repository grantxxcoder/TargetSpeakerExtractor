#!/usr/bin/env python3
"""Build the AMI real-audio transfer set (secondary eval) as trials on disk.

    ../tse_venv/bin/python scripts/build_ami_trials.py
    ../tse_venv/bin/python scripts/build_ami_trials.py --force   # rebuild

Source: AMI Meeting Corpus (Carletta et al., 2005), CC BY 4.0. Audio and
annotations under data/raw/ami/ (download: data/raw/ami/download.sh).

Construction borrowed from REAL-T (Li et al., Interspeech 2025): real
overlapping stretches of a meeting as mixtures, same-speaker solo stretches as
enrolment. Differences from REAL-T, recorded in decisions-m0.md:
  * exactly two active speakers per trial (decisions-m0.md 2026-08-14);
  * enrolment from a DIFFERENT meeting of the same group, matching our
    "different recording" rule (metric-definitions.md §2);
  * scored by the LCF judge, not REAL-T's protocol -- so never comparable to
    REAL-T numbers.

NOTHING IS MIXED. A trial is a cut of real audio at one start/end time:
    mixture.wav     table mic (Array1-01) -- what the model hears
    target.wav      target's headset, same times -- APPROXIMATE ceiling: it
                    carries bleed from the other speaker, never ground truth
    interferer.wav  other speaker's headset, same times
    enrollment.wav  5 s of the target alone, headset, sibling meeting
    meta.json       both speakers' transcript words, times
Same layout as scripts/render_trials.py, so make_estimates*.py and
evaluate.py read it unchanged. Split name `ami` in train.py SPLIT_MANIFESTS,
render-only: EVAL ONLY, never trained, filtered or selected on.

Level: left exactly as recorded -- no gain, no matching to our constructed
data. The point is audio the model has never seen, at the level a real
deployment would get it.
"""

import argparse
import csv
import glob
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from pathlib import Path

import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.estimates.runner import git_commit  # noqa: E402
from src.run_log import timed  # noqa: E402

SPLIT = "ami_eval"
SAMPLE_RATE = 16000


def load_speakers(annotations):
    """meeting -> {letter: (global speaker ID, headset channel)}.

    The letters A-D are NOT the same person across a group's meetings (4 of our
    15 groups, measured 2026-10-06), so anything matched across meetings must
    use the global ID.
    """
    speakers = {}
    root = ET.parse(Path(annotations) / "corpusResources/meetings.xml").getroot()
    for meeting in root.iter("meeting"):
        speakers[meeting.get("observation")] = {
            s.get("nxt_agent"): (s.get("global_name"), s.get("channel"))
            for s in meeting.iter("speaker")}
    return speakers


def load_transcript(annotations, meeting):
    """Per meeting: timed words, activity intervals, and 'unclear' marks.

    Activity = spoken words plus vocal sounds (laughs, coughs), so a laugh from a
    third person still breaks the two-speaker rule. Punctuation tokens carry no
    sound and are dropped.
    """
    words, activity, unclear = [], [], []
    for path in sorted(glob.glob(f"{annotations}/words/{meeting}.*.words.xml")):
        letter = path.split(".")[-3]
        for element in ET.parse(path).getroot():
            start, end = element.get("starttime"), element.get("endtime")
            if start is None or end is None:
                continue
            start, end = float(start), float(end)
            if element.tag in ("gap", "transformerror"):
                unclear.append(start)
            elif element.tag == "w" and element.get("punc") != "true":
                words.append((start, end, letter, element.text))
                if end > start:
                    activity.append((start, end, letter))
            elif element.tag == "vocalsound" and end > start:
                activity.append((start, end, letter))
    return words, sorted(activity), unclear


def merge_activity(activity, min_silence):
    """Join everyone's activity into stretches separated by >= min_silence of
    total silence. Each stretch is [start, end, {letters active in it}]."""
    stretches = []
    for start, end, letter in activity:
        if stretches and start - stretches[-1][1] < min_silence:
            stretches[-1][1] = max(stretches[-1][1], end)
            stretches[-1][2].add(letter)
        else:
            stretches.append([start, end, {letter}])
    return stretches


def find_windows(meeting, stretches, words, unclear, rules):
    """Non-overlapping 10-20 s windows with exactly two people talking.

    Cuts fall in the middle of an all-silent gap, so no word is chopped. From
    each cut the window is extended to the LONGEST valid end, then the search
    resumes after it -- greedy, so windows never share audio.
    """
    cuts = [(stretches[i][1] + stretches[i + 1][0]) / 2
            for i in range(len(stretches) - 1)]
    windows, last_end = [], -1.0
    for i in range(len(cuts)):
        if cuts[i] < last_end:
            continue
        best = None
        for j in range(i + 1, len(cuts)):
            start, end = cuts[i], cuts[j]
            if end - start > rules["max_s"]:
                break
            active = set().union(*(stretches[k][2] for k in range(i + 1, j + 1)))
            if len(active) > 2 or any(start <= t <= end for t in unclear):
                break
            if end - start < rules["min_s"] or len(active) != 2:
                continue
            counts = {letter: sum(1 for s, e, l, _ in words
                                  if l == letter and s >= start and e <= end)
                      for letter in active}
            if min(counts.values()) >= rules["min_words_each"]:
                best = {"meeting": meeting, "start_s": start, "end_s": end,
                        "letters": sorted(active)}
        if best:
            windows.append(best)
            last_end = best["end_s"]
    return windows


def solo_stretches(stretches, min_s):
    """Stretches where exactly one person talks for >= min_s: enrolment sources."""
    return [(next(iter(letters)), start, end) for start, end, letters in stretches
            if len(letters) == 1 and end - start >= min_s]


def text_between(words, letter, start, end):
    """One speaker's transcript words inside [start, end], markup stripped."""
    return " ".join(t for s, e, l, t in words if l == letter and s >= start and e <= end)


def cut(path, start, end):
    first = int(round(start * SAMPLE_RATE))
    audio, sr = sf.read(path, start=first, frames=int(round(end * SAMPLE_RATE)) - first,
                        dtype="float32")
    assert sr == SAMPLE_RATE, f"{path} is {sr} Hz"
    return audio


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="experiments/configs/ami_eval.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--force", action="store_true", help="delete and rebuild")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    rng = random.Random(cfg["seed"])
    annotations, audio_root = Path(cfg["annotations"]), Path(cfg["audio"])
    out_root = Path(args.data_root) / "rendered" / SPLIT
    manifest_path = Path(args.data_root) / "manifests" / f"{SPLIT}.csv"
    if out_root.exists() and any(out_root.iterdir()):
        if not args.force:
            raise SystemExit(f"{out_root} is not empty; pass --force to rebuild")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    speakers = load_speakers(annotations)
    groups = [g for site in cfg["groups"].values() for g in site]
    min_silence = cfg["window"]["edge_silence_s"]
    enroll_s = cfg["enrollment"]["length_s"]

    # 1. Windows and solo stretches, per meeting.
    transcripts, windows_by_group, solos = {}, defaultdict(list), defaultdict(list)
    for group in groups:
        for meeting in sorted(m for m in speakers if m.startswith(group)):
            words, activity, unclear = load_transcript(annotations, meeting)
            stretches = merge_activity(activity, min_silence)
            transcripts[meeting] = words
            for letter, start, end in solo_stretches(stretches, enroll_s):
                solos[speakers[meeting][letter][0]].append((meeting, letter, start, end))
            if meeting not in cfg["no_trials_from"]:
                windows_by_group[group] += find_windows(meeting, stretches, words,
                                                        unclear, cfg["window"])

    # 2. Target per window: only someone with a solo stretch in a SIBLING
    #    meeting can be a target; between two eligible people, a seeded coin.
    def sibling_solos(person, meeting):
        return [s for s in solos[person] if s[0] != meeting and s[0][:6] == meeting[:6]]

    usable = defaultdict(list)
    for group in groups:
        for window in windows_by_group[group]:
            meeting = window["meeting"]
            eligible = [l for l in window["letters"]
                        if sibling_solos(speakers[meeting][l][0], meeting)]
            if not eligible:
                continue
            target = rng.choice(eligible)
            interferer = next(l for l in window["letters"] if l != target)
            usable[group].append({**window, "target": target, "interferer": interferer})

    # 3. Sample: trials_per_group from each group; any shortfall topped up from
    #    the other groups' leftovers, seeded.
    chosen, leftovers = [], []
    for group in groups:
        pool = usable[group][:]
        rng.shuffle(pool)
        chosen += pool[:cfg["trials_per_group"]]
        leftovers += pool[cfg["trials_per_group"]:]
    rng.shuffle(leftovers)
    chosen += leftovers[:cfg["n_trials"] - len(chosen)]
    if len(chosen) != cfg["n_trials"]:
        raise SystemExit(f"only {len(chosen)} usable windows for {cfg['n_trials']} trials")
    chosen.sort(key=lambda w: (w["meeting"], w["start_s"]))

    # 4. Cut and write.
    rows = []
    with timed("scripts/build_ami_trials.py", scope=lambda: f"{len(rows)} trials, {SPLIT}",
               rate=lambda: "cpu"):
        for index, w in enumerate(chosen):
            trial_id = f"{SPLIT}-{cfg['seed']}-{index:06d}"
            meeting = w["meeting"]
            target_id, target_ch = speakers[meeting][w["target"]]
            interferer_id, interferer_ch = speakers[meeting][w["interferer"]]
            e_meeting, e_letter, e_start, _ = rng.choice(sibling_solos(target_id, meeting))
            e_channel = speakers[e_meeting][e_letter][1]

            def wav(m, channel):
                return audio_root / m / "audio" / f"{m}.{channel}.wav"

            stems = {
                "mixture": cut(wav(meeting, cfg["channels"]["mixture"]), w["start_s"], w["end_s"]),
                "target": cut(wav(meeting, f"Headset-{target_ch}"), w["start_s"], w["end_s"]),
                "interferer": cut(wav(meeting, f"Headset-{interferer_ch}"), w["start_s"], w["end_s"]),
                "enrollment": cut(wav(e_meeting, f"Headset-{e_channel}"), e_start, e_start + enroll_s),
            }
            trial_dir = out_root / trial_id
            trial_dir.mkdir()
            for name, audio in stems.items():
                sf.write(trial_dir / f"{name}.wav", audio, SAMPLE_RATE, subtype="PCM_16")

            meta = {
                "trial_id": trial_id, "sample_rate": SAMPLE_RATE,
                "n_samples": len(stems["mixture"]),
                "mixture_length_s": round(w["end_s"] - w["start_s"], 3),
                "meeting": meeting, "start_s": round(w["start_s"], 3), "end_s": round(w["end_s"], 3),
                "target_speaker": target_id, "interferer_speaker": interferer_id,
                "target_text": text_between(transcripts[meeting], w["target"], w["start_s"], w["end_s"]),
                "interferer_text": text_between(transcripts[meeting], w["interferer"], w["start_s"], w["end_s"]),
                "enrollment_meeting": e_meeting, "enrollment_start_s": round(e_start, 3),
                "enrollment_length_s": enroll_s,
                "ceiling_approximate": True,
            }
            (trial_dir / "meta.json").write_text(json.dumps(meta, indent=1))
            rows.append({k: meta[k] for k in ("trial_id", "meeting", "start_s", "end_s",
                                              "mixture_length_s", "target_speaker",
                                              "interferer_speaker", "enrollment_meeting",
                                              "enrollment_start_s", "enrollment_length_s")}
                        | {"split": SPLIT, "condition": "both"})

    fields = ["trial_id", "split", "condition"] + [k for k in rows[0]
                                                   if k not in ("trial_id", "split", "condition")]
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "generated": date.today().isoformat(), "generator": "scripts/build_ami_trials.py",
        "split": SPLIT, "seed": cfg["seed"], "config": str(config_path),
        "config_md5": hashlib.md5(config_path.read_bytes()).hexdigest(),
        "git_commit": git_commit(), "n_trials": len(rows),
        "source": "AMI Meeting Corpus, ami_public_manual_1.6.2 (Carletta et al., 2005)",
        "ceiling": "APPROXIMATE -- target headset, carries bleed",
    }
    manifest_path.with_suffix(".meta.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
    (out_root / "render.meta.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
    print(f"{len(rows)} trials -> {out_root}, manifest {manifest_path}")


if __name__ == "__main__":
    main()
