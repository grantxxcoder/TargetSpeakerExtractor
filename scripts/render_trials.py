"""Render a split's manifest to audio on disk.

    ../tse_venv/bin/python scripts/render_trials.py --split sir0_train

Each trial gets a folder data/rendered/<split>/<trial_id>/ holding mixture.wav (what the model hears), target.wav (A1's reference, the target through its own room alone),
enrollment.wav (dry, A4), interferer.wav and interferer_enrollment.wav (the second training direction, with a phantom speaker when nobody interferes), and meta.json (both transcripts, the gains and the EQ curves).
The audio is 16-bit WAV at 16 kHz, so a training step is a pure disk read with no decoding (decisions-m0.md 2026-08-15).
A trial whose files all exist is skipped, which is how sir0_train was rendered in two passes. A rebuilt manifest describes different audio, so a folder rendered from another manifest is refused.
"""

import argparse
import csv
import json
import shutil
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_manifest import read_speakers
from src.data import render
from src.run_log import timed

CONFIG_PATH = "experiments/configs/generator.yaml"
MANIFEST_DIR = Path("data/manifests")
RENDERED_DIR = Path("data/rendered")
INDEX_DIR = Path("data/index")
WORKERS = 8

# A trial only counts as rendered when every one of these exists.
STEMS = ("mixture", "target", "enrollment", "interferer", "interferer_enrollment")

# What every worker process needs, set once when the worker starts.
worker_state = {}


# this reads the split's utterance index into {utterance id: row}
def read_index(path):
    index = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            index[row["utt"]] = row
    return index


# this gives the flac path of every utterance in the index, built once rather than per trial
def flac_paths(librispeech_root, index, speaker_metadata):
    paths = {}
    for utterance, row in index.items():
        subset = speaker_metadata[row["speaker"]][1]
        paths[utterance] = librispeech_root / subset / row["speaker"] / row["chapter"] / f"{utterance}.flac"
    return paths


# this stores what the workers share, once per worker process
def start_worker(sample_rate, audio_paths, texts, noise_root, noise_split, out_dir):
    worker_state["sample_rate"] = sample_rate
    worker_state["audio_paths"] = audio_paths
    worker_state["texts"] = texts
    worker_state["noise_root"] = noise_root
    worker_state["noise_split"] = noise_split
    worker_state["out_dir"] = out_dir


# this checks whether a trial's folder already holds every stem and its meta
def is_rendered(trial_dir):
    for stem in STEMS:
        if not (trial_dir / f"{stem}.wav").exists():
            return False
    return (trial_dir / "meta.json").exists()


# this renders one trial and writes it, returning (trial id, status, note)
def render_one(row):
    trial_dir = worker_state["out_dir"] / row["trial_id"]
    if is_rendered(trial_dir):
        return row["trial_id"], "skipped", ""
    try:
        stems, meta = render.render_trial(row, worker_state["sample_rate"], worker_state["audio_paths"], worker_state["texts"], worker_state["noise_root"], worker_state["noise_split"])
    except Exception as error:
        return row["trial_id"], "failed", f"{type(error).__name__}: {error}"

    # Written to a temporary folder and then moved, so an interrupted write never leaves a half trial that looks complete.
    partial_dir = trial_dir.with_name(trial_dir.name + ".partial")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)
    for stem in STEMS:
        sf.write(partial_dir / f"{stem}.wav", stems[stem], worker_state["sample_rate"], subtype="PCM_16")
    (partial_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    partial_dir.rename(trial_dir)
    return row["trial_id"], "rendered", ""


# this is where rendering starts
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    sample_rate = config["sample_rate"]
    librispeech_root = Path(config["paths"]["librispeech"])
    noise_root = Path(config["paths"]["wham_noise"])
    noise_split = config["splits"][args.split]["noise_split"]

    manifest = MANIFEST_DIR / f"{args.split}.csv"
    manifest_meta = yaml.safe_load((MANIFEST_DIR / f"{args.split}.meta.yaml").read_text())
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))

    index = read_index(INDEX_DIR / f"utterances_{args.split}.csv")
    texts = {}
    for utterance, row in index.items():
        texts[utterance] = row["text"]
    audio_paths = flac_paths(librispeech_root, index, read_speakers(librispeech_root))

    out_dir = RENDERED_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Audio from two different manifests in one folder would look complete and be silently inconsistent.
    stamp = out_dir / "render.meta.yaml"
    if stamp.exists():
        previous = yaml.safe_load(stamp.read_text())
        if previous.get("manifest_config_md5") != manifest_meta.get("config_md5"):
            raise SystemExit(f"{out_dir} was rendered from a different manifest (config_md5 {previous.get('manifest_config_md5')} against {manifest_meta.get('config_md5')}). Delete the folder to render this manifest.")

    counts = {"rendered": 0, "skipped": 0, "failed": 0}
    failures = []
    with timed(f"scripts/render_trials.py --split {args.split}", scope=lambda: f"{counts['rendered']:,} trials rendered", rate=f"{WORKERS} workers, 16 kHz PCM_16"):
        with Pool(WORKERS, initializer=start_worker, initargs=(sample_rate, audio_paths, texts, noise_root, noise_split, out_dir)) as pool:
            for trial_id, status, note in pool.imap_unordered(render_one, rows, chunksize=4):
                counts[status] += 1
                if status == "failed":
                    failures.append((trial_id, note))

    rendered_lengths = []
    for row in rows:
        rendered_lengths.append(float(row["mixture_length_s"]) + float(row["t60_s"]))
    rendered_seconds = sum(rendered_lengths)
    stamp.write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/render_trials.py",
        "split": args.split,
        "config": CONFIG_PATH,
        "sample_rate": sample_rate,
        "format": "wav_pcm16",
        # Audio and manifest are one artefact, so the manifest it belongs to is recorded.
        "manifest": str(manifest),
        "manifest_config_md5": manifest_meta.get("config_md5"),
        "n_trials": len(rows),
        "n_rendered": counts["rendered"],
        "n_skipped": counts["skipped"],
        "n_failed": counts["failed"],
        "audio_hours": round(rendered_seconds / 3600, 3),
    }, sort_keys=False))

    # Generated by Claude Code
    print(f"Wrote {out_dir}  ({counts['rendered']} rendered, {counts['skipped']} skipped, {counts['failed']} failed)")
    print(f"  {rendered_seconds / 3600:.2f} h of audio across {len(rows)} trials")
    if failures:
        lines = []
        for trial_id, note in failures:
            lines.append(f"{trial_id}\t{note}")
        (out_dir / "failed.txt").write_text("\n".join(lines) + "\n")
        sys.exit(f"{len(failures)} trial(s) failed, see {out_dir}/failed.txt")


if __name__ == "__main__":
    main()
