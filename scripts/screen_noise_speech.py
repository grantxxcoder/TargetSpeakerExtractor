"""Measure the speech hiding in every WHAM! noise bed (B2, decisions-m0.md 2026-08-16).

    ../tse_venv/bin/python scripts/screen_noise_speech.py

It writes data/index/noise_speech_{tr,cv,tt}.csv (clip, duration, speech_s, max_segment_s, n_segments, segments) and data/index/noise_speech.meta.yaml.
WHAM! was recorded in real cafes and streets, so a bed can hold background talkers whose words would score as invented speech. This only measures.
The cutoff, max_speech_run_s in generator.yaml, is applied when the manifest is built, so changing it never needs a rescan.
The detector is Silero VAD (Silero Team, 2021) with the same settings as the utterance index, so "speech" means the same thing on both sides of the mixture.
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
from src.data import vad
from src.run_log import timed

CONFIG_PATH = "experiments/configs/generator.yaml"
INDEX_DIR = Path("data/index")
WORKERS = 8
COLUMNS = ["clip", "duration", "speech_s", "max_segment_s", "n_segments", "segments"]

# What every worker process needs, set once when the worker starts.
worker_state = {}


# this stores what the workers share, once per worker process
def start_worker(settings, sample_rate, noise_root):
    worker_state["settings"] = settings
    worker_state["sample_rate"] = sample_rate
    worker_state["noise_root"] = noise_root
    torch.set_num_threads(1)


# this measures the speech in one noise clip and returns (pool, cache row)
def screen_one(item):
    pool_name, clip = item
    waveform, sample_rate = sf.read(worker_state["noise_root"] / pool_name / clip, dtype="float32")
    if sample_rate != worker_state["sample_rate"]:
        raise ValueError(f"{clip} is {sample_rate} Hz, but {worker_state['sample_rate']} Hz was expected.")
    # WHAM! ships stereo upstream, so the channels are averaged.
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    segments = vad.detect(torch.from_numpy(np.ascontiguousarray(waveform)), vad.load_model(), worker_state["settings"], worker_state["sample_rate"])
    longest_segment = 0.0
    for start, end in segments:
        longest_segment = max(longest_segment, end - start)
    row = {"clip": clip, "duration": f"{len(waveform) / sample_rate:.4f}", "speech_s": f"{vad.total_speech(segments):.4f}", "max_segment_s": f"{longest_segment:.4f}", "n_segments": len(segments), "segments": vad.format_segments(segments)}
    return pool_name, row


# this is where the screening starts
def main():
    # It takes no options, but parsing still answers --help instead of starting the job.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    settings = vad.vad_config(config)
    sample_rate = int(config["sample_rate"])
    noise_root = Path(config["paths"]["wham_noise"])

    pool_names = set()
    for split_settings in config["splits"].values():
        pool_names.add(split_settings["noise_split"])
    pool_names = sorted(pool_names)

    work = []
    for pool_name in pool_names:
        with (INDEX_DIR / f"noise_{pool_name}.csv").open() as handle:
            for row in csv.DictReader(handle):
                work.append((pool_name, row["clip"]))

    rows_by_pool = {}
    for pool_name in pool_names:
        rows_by_pool[pool_name] = []
    audio_seconds = [0.0]
    start_time = time.time()
    with timed("scripts/screen_noise_speech.py", scope=lambda: f"{len(work):,} clips / {audio_seconds[0] / 3600:.0f} h", rate=lambda: f"{audio_seconds[0] / max(time.time() - start_time, 1e-9):.0f}x realtime, {WORKERS} workers"):
        with Pool(WORKERS, initializer=start_worker, initargs=(settings, sample_rate, noise_root)) as pool:
            for pool_name, row in pool.imap(screen_one, work, chunksize=16):
                rows_by_pool[pool_name].append(row)
                audio_seconds[0] += float(row["duration"])

    for pool_name in pool_names:
        path = INDEX_DIR / f"noise_speech_{pool_name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows_by_pool[pool_name])
        print(f"Wrote {path}  ({len(rows_by_pool[pool_name]):,} clips)")

    pool_sizes = {}
    for pool_name in pool_names:
        pool_sizes[pool_name] = len(rows_by_pool[pool_name])
    (INDEX_DIR / "noise_speech.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/screen_noise_speech.py",
        "config": CONFIG_PATH,
        "config_md5": hashlib.md5(Path(CONFIG_PATH).read_bytes()).hexdigest(),
        "detector": "silero-vad",
        "detector_version": vad.model_version(),
        "detector_settings": settings,
        "pools": pool_sizes,
        # The rejection rule is applied by build_manifest.py, not stored here.
        "rejection_rule": "NOT APPLIED -- see decisions-m0.md, chosen from this report",
    }, sort_keys=False))


if __name__ == "__main__":
    main()
