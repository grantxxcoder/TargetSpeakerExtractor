"""Run the voice-activity detector over every indexed utterance and cache where the speech is (B2, decisions-m0.md 2026-08-15).

    ../tse_venv/bin/python scripts/build_vad_index.py

It writes data/index/vad_segments.csv (utt, duration, speech_s, segments) and data/index/vad_segments.meta.yaml.
The cache is keyed by utterance rather than by split, because where the speech sits in a recording does not depend on which split the recording landed in.
"""

import argparse
import csv
import hashlib
import sys
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_manifest import read_speakers
from src.data import vad
from src.run_log import timed

CONFIG_PATH = "experiments/configs/generator.yaml"
INDEX_DIR = Path("data/index")
OUT_PATH = INDEX_DIR / "vad_segments.csv"
META_PATH = INDEX_DIR / "vad_segments.meta.yaml"
WORKERS = 8
COLUMNS = ["utt", "duration", "speech_s", "segments"]

# What every worker process needs, set once when the worker starts.
worker_state = {}


# this stores what the workers share. The model loads inside vad.load_model on first use, so it is never sent between processes
def start_worker(settings, sample_rate, librispeech_root, speaker_metadata):
    worker_state["settings"] = settings
    worker_state["sample_rate"] = sample_rate
    worker_state["librispeech_root"] = librispeech_root
    worker_state["speaker_metadata"] = speaker_metadata
    # Eight processes with several threads each would oversubscribe the CPU.
    torch.set_num_threads(1)


# this detects the speech in one utterance and returns its cache row
def detect_one(utterance):
    speaker, chapter, number = utterance.split("-")
    path = worker_state["librispeech_root"] / worker_state["speaker_metadata"][speaker][1] / speaker / chapter / f"{utterance}.flac"
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != worker_state["sample_rate"]:
        raise ValueError(f"{utterance} is {sample_rate} Hz, but {worker_state['sample_rate']} Hz was expected.")
    segments = vad.detect(torch.from_numpy(waveform), vad.load_model(), worker_state["settings"], sample_rate)
    return {"utt": utterance, "duration": f"{len(waveform) / sample_rate:.4f}", "speech_s": f"{vad.total_speech(segments):.4f}", "segments": vad.format_segments(segments)}


# this gives every utterance named by any split's utterance index, without repeats
def indexed_utterances(index_dir):
    utterances = []
    seen = set()
    for path in sorted(index_dir.glob("utterances_*.csv")):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["utt"] not in seen:
                    seen.add(row["utt"])
                    utterances.append(row["utt"])
    return utterances


# this is where the VAD pass starts
def main():
    # It takes no options, but parsing still answers --help instead of starting the job.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    settings = vad.vad_config(config)
    sample_rate = int(config["sample_rate"])
    librispeech_root = Path(config["paths"]["librispeech"])
    utterances = indexed_utterances(INDEX_DIR)

    rows = []
    audio_seconds = [0.0]
    start_time = time.time()
    with timed("scripts/build_vad_index.py", scope=lambda: f"{len(utterances):,} utts / {audio_seconds[0] / 3600:.0f} h", rate=lambda: f"{audio_seconds[0] / max(time.time() - start_time, 1e-9):.0f}x realtime, {WORKERS} workers"):
        with Pool(WORKERS, initializer=start_worker, initargs=(settings, sample_rate, librispeech_root, read_speakers(librispeech_root))) as pool:
            for row in pool.imap(detect_one, utterances, chunksize=16):
                rows.append(row)
                audio_seconds[0] += float(row["duration"])
    with OUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # The summary numbers quoted in decisions-m0.md 2026-08-15 come from here.
    durations = []
    speech = []
    segment_counts = []
    leading_silence = []
    trailing_silence = []
    for row in rows:
        segments = vad.parse_segments(row["segments"])
        durations.append(float(row["duration"]))
        speech.append(float(row["speech_s"]))
        segment_counts.append(len(segments))
        if segments:
            leading_silence.append(segments[0][0])
            trailing_silence.append(float(row["duration"]) - segments[-1][1])
        else:
            leading_silence.append(float(row["duration"]))
            trailing_silence.append(0.0)
    durations = np.array(durations)
    speech = np.array(speech)
    segment_counts = np.array(segment_counts)
    speech_ratio = speech / durations

    META_PATH.write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/build_vad_index.py",
        "config": CONFIG_PATH,
        "config_md5": hashlib.md5(Path(CONFIG_PATH).read_bytes()).hexdigest(),
        "detector": "silero-vad",
        "detector_version": vad.model_version(),
        "detector_settings": settings,
        "sample_rate": sample_rate,
        "n_utterances": len(rows),
        "n_no_speech_detected": int((segment_counts == 0).sum()),
        "speech_ratio_mean": round(float(speech_ratio.mean()), 4),
        "speech_ratio_median": round(float(np.median(speech_ratio)), 4),
        "segments_per_utterance_mean": round(float(segment_counts.mean()), 3),
        "leading_silence_s_mean": round(float(np.mean(leading_silence)), 4),
        "trailing_silence_s_mean": round(float(np.mean(trailing_silence)), 4),
        "total_audio_hours": round(float(durations.sum() / 3600), 2),
        "total_speech_hours": round(float(speech.sum() / 3600), 2),
    }, sort_keys=False))

    # Generated by Claude Code
    print(f"Wrote {OUT_PATH}  ({len(rows)} utterances)")
    print(f"  speech / duration  mean {speech_ratio.mean():.4f}  median {np.median(speech_ratio):.4f}")
    print(f"  leading silence    mean {np.mean(leading_silence):.4f} s   trailing silence mean {np.mean(trailing_silence):.4f} s")


if __name__ == "__main__":
    main()
