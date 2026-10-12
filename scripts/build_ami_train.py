#!/usr/bin/env python3
"""Build the AMI TRAINING split for M8 (decisions-m8.md 2026-10-12).

    ../tse_venv/bin/python scripts/build_ami_train.py --dry-run   # plan + bytes, no fetch
    ../tse_venv/bin/python scripts/build_ami_train.py
    ../tse_venv/bin/python scripts/build_ami_train.py --force     # rebuild

Separate from build_ami_trials.py on purpose: that script builds ami_eval and is
left untouched, so ami_eval cannot change. This one imports its window-finding
functions, so training clips are found by exactly the same rules.

Source: AMI Meeting Corpus (Carletta et al., 2005), CC BY 4.0. Construction
borrowed from REAL-T (Li et al., Interspeech 2025), as for ami_eval.

What differs from ami_eval, all because these trials are TRAINED on:
  * every usable window, both directions -- so both speakers need a sibling-
    meeting enrolment, plus a bank of `enrollment.variants` per speaker;
  * target.wav / interferer.wav are each speaker's headset ALIGNED, RESCALED
    and GATED (src/data/ami_targets.py) -- an approximate target, never truth;
  * audio comes from HTTP byte ranges (src/data/ami_fetch.py), not whole files;
  * manifest carries the columns the training loader reads, and remixable = 0:
    remix assumes mixture = target + interferer + noise, false across mics.

mixture.wav is the table mic exactly as recorded: real levels are the point.
"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_ami_trials import (find_windows, load_speakers,  # noqa: E402
                                      load_transcript, merge_activity,
                                      solo_stretches, text_between)
from src.data.ami_fetch import AmiFetcher  # noqa: E402
from src.data.ami_targets import activity_mask, estimate_lag, process_headset  # noqa: E402
from src.estimates.runner import git_commit  # noqa: E402
from src.run_log import timed  # noqa: E402

SAMPLE_RATE = 16000
SITE = {"ES": "Edinburgh", "IS": "IDIAP", "TS": "TNO"}


def check_disjoint(cfg, speakers, groups):
    """Refuse to build anything that shares a group, person or meeting with
    ami_eval. People are matched on AMI's global_name."""
    if cfg["split"] == "ami_eval":
        raise SystemExit("ami_eval is built by build_ami_trials.py, never here")
    eval_cfg = yaml.safe_load(Path(cfg["holdout"]["config"]).read_text())
    eval_groups = {g for site in eval_cfg["groups"].values() for g in site}
    shared = set(groups) & eval_groups
    if shared:
        raise SystemExit(f"groups shared with ami_eval: {sorted(shared)}")

    def people(group_set):
        return {name for m, letters in speakers.items() if m[:6] in group_set
                for name, _ in letters.values()}
    eval_people = people(eval_groups)
    with open(cfg["holdout"]["manifest"]) as fh:
        rows = list(csv.DictReader(fh))
    eval_people |= {r[k] for r in rows for k in ("target_speaker", "interferer_speaker")}
    eval_meetings = {r[k] for r in rows for k in ("meeting", "enrollment_meeting")}
    shared_people = people(set(groups)) & eval_people
    shared_meetings = {m for m in speakers if m[:6] in groups} & eval_meetings
    if shared_people or shared_meetings:
        raise SystemExit(f"shared with ami_eval: people {sorted(shared_people)}, "
                         f"meetings {sorted(shared_meetings)}")
    return len(eval_people)


def window_relative(activity, letter, start, end, reach=1.0):
    """One speaker's activity intervals, in seconds from `start`, including any
    within `reach` s outside the window (their gate padding can reach in)."""
    return [(a - start, b - start) for a, b, l in activity
            if l == letter and b > start - reach and a < end + reach]


def energy_db(x):
    return 10 * np.log10(float(np.dot(x, x)) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="experiments/configs/ami_train.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel range fetches; the mirror 403s at ~12+")
    ap.add_argument("--dry-run", action="store_true", help="plan and byte count only")
    ap.add_argument("--force", action="store_true", help="delete and rebuild")
    ap.add_argument("--resume", action="store_true",
                    help="keep complete trials already on disk (checked against a "
                         "recomputed meta.json), write the rest")
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    split = cfg["split"]
    rng = random.Random(cfg["seed"])
    annotations = Path(cfg["annotations"])
    speakers = load_speakers(annotations)
    groups = [g for site in cfg["groups"].values() for g in site]
    n_eval_people = check_disjoint(cfg, speakers, groups)

    out_root = Path(args.data_root) / "rendered" / split
    manifest_path = Path(args.data_root) / "manifests" / f"{split}.csv"
    if not args.dry_run and out_root.exists() and any(out_root.iterdir()) and not args.resume:
        if not args.force:
            raise SystemExit(f"{out_root} is not empty; pass --force to rebuild")
        shutil.rmtree(out_root)

    enroll_s, n_variants = cfg["enrollment"]["length_s"], cfg["enrollment"]["variants"]
    margin = int(round(cfg["target"]["fetch_margin_ms"] * SAMPLE_RATE / 1000))

    # 1. Windows, activity and solo stretches, per meeting -- ami_eval's rules.
    transcripts, activities, windows, solos = {}, {}, [], defaultdict(list)
    for group in groups:
        for meeting in sorted(m for m in speakers if m.startswith(group)):
            words, activity, unclear = load_transcript(annotations, meeting)
            stretches = merge_activity(activity, cfg["window"]["edge_silence_s"])
            transcripts[meeting], activities[meeting] = words, activity
            for letter, start, end in solo_stretches(stretches, enroll_s):
                solos[speakers[meeting][letter][0]].append((meeting, letter, start, end))
            if meeting not in cfg.get("no_trials_from", []):   # no table mic on the server
                windows += find_windows(meeting, stretches, words, unclear, cfg["window"])

    def sibling_solos(person, meeting):
        return [s for s in solos[person] if s[0] != meeting and s[0][:6] == meeting[:6]]

    # 2. Both speakers need a sibling-meeting enrolment: both directions train.
    #    Designated target (for the Gemini check) by a seeded coin.
    trials, dropped = [], 0
    for w in sorted(windows, key=lambda w: (w["meeting"], w["start_s"])):
        meeting = w["meeting"]
        if not all(sibling_solos(speakers[meeting][l][0], meeting) for l in w["letters"]):
            dropped += 1
            continue
        target = rng.choice(w["letters"])
        interferer = next(l for l in w["letters"] if l != target)
        bank = {}
        for role, letter in (("target", target), ("interferer", interferer)):
            pool = sibling_solos(speakers[meeting][letter][0], meeting)
            bank[role] = rng.sample(pool, min(n_variants, len(pool)))
        trials.append({**w, "target": target, "interferer": interferer, "bank": bank})

    # 3. Fetch plan: only the samples used.
    def span(start, end):
        first = int(round(start * SAMPLE_RATE))
        return first, int(round(end * SAMPLE_RATE)) - first

    requests = set()
    for t in trials:
        m, (first, n) = t["meeting"], span(t["start_s"], t["end_s"])
        requests.add((m, cfg["channels"]["mixture"], first, n))
        for letter in (t["target"], t["interferer"]):
            requests.add((m, f"Headset-{speakers[m][letter][1]}", first - margin, n + 2 * margin))
        for picks in t["bank"].values():
            for e_meeting, e_letter, e_start, _ in picks:
                requests.add((e_meeting, f"Headset-{speakers[e_meeting][e_letter][1]}",
                              *span(e_start, e_start + enroll_s)))
    n_bytes = 2 * sum(r[3] for r in requests)
    print(f"{len(windows)} windows, {dropped} dropped (a speaker has no sibling-meeting "
          f"enrolment), {len(trials)} trials, {sum(t['end_s'] - t['start_s'] for t in trials) / 3600:.2f} h")
    print(f"{len({p for t in trials for p in (speakers[t['meeting']][t['target']][0], speakers[t['meeting']][t['interferer']][0])})}"
          f" speakers, 0 shared with ami_eval's {n_eval_people}; "
          f"{len(requests)} pieces to fetch, {n_bytes / 1e9:.2f} GB")
    if args.dry_run:
        return

    fetcher = AmiFetcher(cfg["source_url"], cfg["segment_cache"])
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with timed("scripts/build_ami_train.py",
               scope=lambda: f"{len(rows)} trials, {split}, {n_bytes / 1e9:.2f} GB fetched by range",
               rate=lambda: f"cpu, {args.workers} fetch workers"):
        with ThreadPoolExecutor(args.workers) as pool:
            for done, _ in enumerate(pool.map(lambda r: fetcher.read(*r), sorted(requests)), 1):
                if done % 200 == 0:
                    print(f"  fetched {done}/{len(requests)}", flush=True)

        def roles(t):
            return (("target", t["target"], t["interferer"]),
                    ("interferer", t["interferer"], t["target"]))

        def audio_for(t, letter, other):
            m, (first, n) = t["meeting"], span(t["start_s"], t["end_s"])
            mixture = fetcher.read(m, cfg["channels"]["mixture"], first, n)
            headset = fetcher.read(m, f"Headset-{speakers[m][letter][1]}", first - margin, n + 2 * margin)
            return (mixture, headset,
                    window_relative(activities[m], letter, t["start_s"], t["end_s"]),
                    window_relative(activities[m], other, t["start_s"], t["end_s"]))

        # 4a. Delay of the table mic behind each headset, per window, then each
        #     person's value for the whole meeting from their CONFIDENT estimates.
        #     Measured on this build (decisions-m8.md 2026-10-12): estimates with
        #     a GCC-PHAT peak >= min_peak agree within a speaker-meeting to 0.5 ms
        #     (median range) and 3/480 are implausible; below 0.1, 55/134 are
        #     (negative, or > 8 ms). People mostly stay seated, so a weak
        #     window takes its speaker's meeting value instead of its own.
        min_peak = cfg["target"]["align"]["min_peak"]
        estimates, pooled = {}, defaultdict(lambda: ([], []))
        for index, t in enumerate(trials):
            for role, letter, other in roles(t):
                mixture, headset, own, oth = audio_for(t, letter, other)
                lag, peak = estimate_lag(mixture, headset, margin, own, oth, cfg["target"])
                estimates[index, role] = (lag, peak)
                key = (t["meeting"], speakers[t["meeting"]][letter][0])
                pooled[key][1].append(lag)
                if peak >= min_peak:
                    pooled[key][0].append(lag)
        reference = {key: ((int(round(np.median(confident))), "speaker_meeting_confident") if confident
                           else (int(round(np.median(every))), "speaker_meeting_all"))
                     for key, (confident, every) in pooled.items()}

        # 4b. Process and write.
        max_lag = int(round(cfg["target"]["align"]["max_lag_ms"] * SAMPLE_RATE / 1000))
        for index, t in enumerate(trials):
            trial_id = f"{split}-{cfg['seed']}-{index:06d}"
            m, (first, n) = t["meeting"], span(t["start_s"], t["end_s"])
            stems, info, ids = {}, {}, {}
            for role, letter, other in roles(t):
                ids[role] = speakers[m][letter][0]
                mixture, headset, own, oth = audio_for(t, letter, other)
                lag_est, peak = estimates[index, role]
                lag, source = ((lag_est, "own") if peak >= min_peak
                               else reference[m, ids[role]])
                stems[role], info[role] = process_headset(
                    mixture, headset, margin, own, oth, cfg["target"], lag)
                info[role].update({"lag_estimated_ms": round(1000 * lag_est / SAMPLE_RATE, 3),
                                   "align_peak": round(peak, 4), "lag_source": source,
                                   "at_bound": abs(lag_est) == max_lag})
            residual = mixture - stems["target"] - stems["interferer"]
            both = (activity_mask(window_relative(activities[m], t["target"], t["start_s"], t["end_s"], 0), 0.0, n)
                    & activity_mask(window_relative(activities[m], t["interferer"], t["start_s"], t["end_s"], 0), 0.0, n))

            trial_dir = out_root / trial_id
            # --resume: a trial with meta.json was finished (meta is written last),
            # so its recording-only files (table mic, enrolments, bank) are kept;
            # the PROCESSED stems and meta.json are always rewritten. Anything
            # without meta.json is a partial write and is redone whole.
            done_before = args.resume and (trial_dir / "meta.json").exists()
            if not done_before:
                shutil.rmtree(trial_dir, ignore_errors=True)
                trial_dir.mkdir()
            # The mixture is the recording untouched, so int16 full scale (-1.0)
            # round-trips exactly; AMI has a few clipped samples (7 in one IS1002c
            # window, 2026-10-12), recorded, not altered. The PROCESSED stems are
            # ours and must never reach full scale.
            full_scale = int((np.abs(mixture) >= 32767 / 32768).sum())
            assert np.abs(mixture).max() <= 1.0
            for name, audio in (("mixture", mixture), ("target", stems["target"]),
                                ("interferer", stems["interferer"])):
                assert name == "mixture" or np.abs(audio).max() < 1.0, f"{trial_id} {name} would clip"
                if not (done_before and name == "mixture"):
                    sf.write(trial_dir / f"{name}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
            bank_json = {"n_requested": n_variants, "variants": {}}
            for role, prefix in (("target", "enrollment"), ("interferer", "interferer_enrollment")):
                entries = []
                for k, (e_meeting, e_letter, e_start, _) in enumerate(t["bank"][role]):
                    channel = f"Headset-{speakers[e_meeting][e_letter][1]}"
                    audio = fetcher.read(e_meeting, channel, *span(e_start, e_start + enroll_s))
                    if not done_before:
                        sf.write(trial_dir / f"{prefix}_v{k:02d}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
                        if k == 0:   # as in sir0: enrollment.wav is variant 0
                            sf.write(trial_dir / f"{prefix}.wav", audio, SAMPLE_RATE, subtype="PCM_16")
                    entries.append({"meeting": e_meeting, "channel": channel,
                                    "start_s": round(e_start, 3), "length_s": enroll_s})
                bank_json["variants"][role] = {"speaker": ids[role], "n": len(entries),
                                               "entries": entries}
            if not done_before:
                (trial_dir / "enrollment_bank.json").write_text(json.dumps(bank_json, indent=1))

            e_t, e_i = bank_json["variants"]["target"]["entries"][0], bank_json["variants"]["interferer"]["entries"][0]
            meta = {
                "trial_id": trial_id, "sample_rate": SAMPLE_RATE, "n_samples": n,
                "mixture_length_s": round(t["end_s"] - t["start_s"], 3),
                "group": m[:6], "site": SITE[m[:2]],
                "meeting": m, "start_s": round(t["start_s"], 3), "end_s": round(t["end_s"], 3),
                "target_speaker": ids["target"], "interferer_speaker": ids["interferer"],
                "target_text": text_between(transcripts[m], t["target"], t["start_s"], t["end_s"]),
                "interferer_text": text_between(transcripts[m], t["interferer"], t["start_s"], t["end_s"]),
                "enrollment_meeting": e_t["meeting"], "enrollment_start_s": e_t["start_s"],
                "enrollment_length_s": enroll_s,
                "interferer_enrollment_meeting": e_i["meeting"],
                "interferer_enrollment_start_s": e_i["start_s"],
                "interferer_enrollment_length_s": enroll_s,
                "target_processing": info["target"], "interferer_processing": info["interferer"],
                "target_config": cfg["target"],
                "target_approximate": "headset, aligned + rescaled + gated; never ground truth",
                "mixture_full_scale_samples": full_scale,
            }
            (trial_dir / "meta.json").write_text(json.dumps(meta, indent=1))
            rows.append({
                "trial_id": trial_id, "split": split, "condition": "both",
                **{k: meta[k] for k in ("group", "site", "meeting", "start_s", "end_s",
                                        "mixture_length_s", "target_speaker",
                                        "interferer_speaker", "enrollment_meeting",
                                        "enrollment_start_s", "enrollment_length_s",
                                        "interferer_enrollment_meeting",
                                        "interferer_enrollment_start_s",
                                        "interferer_enrollment_length_s")},
                # Columns the training loader reads. SIR/SNR are MEASURED on the
                # processed stems, informational only: remixable = 0.
                "sir_db": round(energy_db(stems["target"]) - energy_db(stems["interferer"]), 2),
                "snr_db": round(energy_db(stems["target"]) - energy_db(residual), 2),
                "target_absent": 0,
                "overlap_achieved": round(float(both.mean()), 4),
                "regime": "ami",
                "same_gender": int(ids["target"][0] == ids["interferer"][0]),
                "interferer_enrollment_phantom": 0,
                "remixable": 0,
                "mixture_full_scale_samples": full_scale,
                **{f"{role}_{k}": info[role][k] for role in ("target", "interferer")
                   for k in ("lag_ms", "lag_estimated_ms", "align_peak", "lag_source",
                             "at_bound", "gain_db", "gated_frac")},
            })

    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = Path(cfg["segment_cache"]) / "index.csv"
    provenance = {
        "generated": date.today().isoformat(), "generator": "scripts/build_ami_train.py",
        "split": split, "seed": cfg["seed"], "config": str(config_path),
        "config_md5": hashlib.md5(config_path.read_bytes()).hexdigest(),
        "git_commit": git_commit(), "n_trials": len(rows),
        "n_windows_found": len(windows), "n_dropped_no_sibling_enrolment": dropped,
        "hours": round(sum(t["end_s"] - t["start_s"] for t in trials) / 3600, 3),
        "source": "AMI Meeting Corpus, ami_public_manual_1.6.2 (Carletta et al., 2005)",
        "audio": f"HTTP byte ranges from {cfg['source_url']}, {n_bytes} bytes planned",
        "segment_index": str(index),
        "segment_index_md5": hashlib.md5(index.read_bytes()).hexdigest(),
        "target": "APPROXIMATE -- headset aligned + rescaled + gated (src/data/ami_targets.py)",
        "disjoint_from": "ami_eval: no shared group, speaker or meeting (checked at build)",
    }
    manifest_path.with_suffix(".meta.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
    (out_root / "render.meta.yaml").write_text(yaml.safe_dump(provenance, sort_keys=False))
    print(f"{len(rows)} trials -> {out_root}, manifest {manifest_path}")


if __name__ == "__main__":
    main()
