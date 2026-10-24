"""Write estimate.wav for every trial in a manifest, for any extractor.

This is the first half of the evaluation. scripts/evaluate.py reads the estimate folders and never knows which model wrote them, so our model and the borrowed WeSep model are compared on equal terms.
It only imports the standard library, numpy, soundfile and yaml, because WeSep runs in its own virtualenv without torch 2.13. Do not import torch or anything from src.models here.

Every extractor is held to the same rules: the whole clip in one forward pass, no chunking, and float32 output left unnormalised, so a level error stays visible and no system gets a free level correction.
"""

import csv
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

# A length difference this big is a bug, not STFT framing. WeSep returns 16 samples (1 ms) short, which the metrics tolerate, but a twentieth of a second means something is misaligned.
LENGTH_WARN_S = 0.05
PROGRESS_EVERY = 25


# this reads the trials of a manifest in order, as dictionaries of trial_id, directory and condition. condition None keeps every trial
def read_trials(manifest_csv, audio_root, limit, condition):
    manifest_csv = Path(manifest_csv)
    audio_root = Path(audio_root)
    if not manifest_csv.exists():
        raise SystemExit(f"No manifest at {manifest_csv}.")
    trials = []
    with open(manifest_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if condition is not None and row.get("condition", "") != condition:
                continue
            trials.append({"trial_id": str(row["trial_id"]), "directory": audio_root / str(row["trial_id"]), "condition": str(row.get("condition", ""))})
    if not trials:
        raise SystemExit(f"No trials with condition={condition!r} in {manifest_csv}.")
    if limit is None:
        return trials
    return trials[:limit]


# this reads a whole wav file as one mono float32 channel. Estimates use the whole clip, not the 4 s training crop
def read_mono(path):
    if not path.exists():
        raise SystemExit(f"Missing {path}.")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


# this runs extract over every trial, writes out_root/<trial_id>/estimate.wav, then one meta.yaml for the whole pass, and returns that meta
def write_estimates(extract, trials, out_root, sample_rate, provenance):
    """extract takes (mixture, enrolment, sample rate) as 1-D arrays and returns the estimate as a 1-D array."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    deltas = []
    for trial in trials:
        mixture = read_mono(trial["directory"] / "mixture.wav")
        enrollment = read_mono(trial["directory"] / "enrollment.wav")
        estimate = extract(mixture, enrollment, sample_rate)
        if estimate is None:
            raise SystemExit(f"{trial['trial_id']}: the extractor returned no audio. WeSep does this when its own VAD finds the enrolment silent, and that gate belongs to the metric, not to a system under test.")
        estimate = np.asarray(estimate, dtype=np.float32).reshape(-1)
        deltas.append(len(estimate) - len(mixture))
        estimate_path = out_root / trial["trial_id"] / "estimate.wav"
        estimate_path.parent.mkdir(parents=True, exist_ok=True)
        # FLOAT keeps the file float32 and unnormalised. PCM_16 would clip anything the model pushed past full scale.
        sf.write(str(estimate_path), estimate, sample_rate, subtype="FLOAT")
        written += 1
        if written % PROGRESS_EVERY == 0 or written == len(trials):
            print(f"  {written}/{len(trials)}", flush=True)

    worst = max(deltas, key=abs) if deltas else 0
    if abs(worst) > LENGTH_WARN_S * sample_rate:
        print(f"  WARNING: estimate length differs from the mixture by up to {worst} samples ({worst / sample_rate * 1000:+.0f} ms). The metrics truncate to the shortest input, so this does not crash, but check for a real misalignment before trusting the row.")

    meta = {"date": date.today().isoformat()}
    for key, value in provenance.items():
        meta[key] = value
    meta["n_trials"] = written
    meta["sample_rate"] = sample_rate
    meta["audio"] = "estimate.wav, float32, unnormalised, whole clip"
    meta["estimate_length_delta_samples"] = {"min": min(deltas, default=0), "max": max(deltas, default=0)}
    with open(out_root / "meta.yaml", "w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)
    print(f"\n  wrote {written} estimates -> {out_root}/")
    return meta
