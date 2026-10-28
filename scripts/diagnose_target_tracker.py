"""Tests the target-tracker idea without any training: tests D1 and D2 of docs/extra/2026-10-28-target-tracker-plan.md.

    ../tse_venv/bin/python scripts/diagnose_target_tracker.py

Both tests edit e21's stored validation outputs (sir0_val, the 103 two-speaker trials) band by band, only in frames where both speakers talk, and never make a band louder.
D1, oracle energy: each band is turned down to the clean target's own band energy. This is the most any band-level turn-down could gain, so it is the ceiling for D2.
D2, oracle trust: a Kalman filter per band (Kalman, 1960) tracks the target's log band energy from earlier frames. The true speech labels say how far to trust e21's output: fully when the target talks alone, not at all when it is silent, and by r_overlap in overlap.
The random-walk variance of each band is fitted on clean targets of the training split, the way the presence gate's transition matrix was fitted from labels.
Speech labels come from Silero VAD with generator.yaml's settings (src/data/vad.py), run on the clean target and on the other speaker's track.
The settings were registered before running, in decisions-pending.md 2026-10-28.
Claude Code wrote this script from the author's idea (2026-10-28).
"""

import csv
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from statistics import median

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.vad import detect, load_model, model_version, vad_config
from src.models.bands import band_plan
from src.models.stft import build_stft, stft_forward, stft_inverse
from src.run_log import timed

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments/results"
RENDERED = REPO_ROOT / "data/rendered"
E21_ESTIMATES = RESULTS / "2026-09-25-est-cuecontext-wer-e21"
PRIOR_DIRECTORY = RESULTS / "2026-10-28-tracker-prior"
SUMMARY_DIRECTORY = RESULTS / "2026-10-28-tracker-diagnostics"

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128
SEED = 42
PRIOR_TRIALS = 200
# Added to every band energy before the log, so a silent band has a finite log energy.
ENERGY_FLOOR = 1e-8
# The observation variance when the target talks alone, small enough that the filter follows e21's output there.
ALONE_VARIANCE = 1e-4
# The three registered overlap settings, as multiples of the median random-walk variance.
OVERLAP_MULTIPLES = [1, 10, 100]


# this reads a wav file as one float32 channel
def read_mono(path):
    waveform, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"{path} is at {sample_rate} Hz, not {SAMPLE_RATE}.")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform


# this gives the complex spectrogram of one waveform, shaped (frequency bins, frames), with the model's own STFT
def spectrogram_of(stft, waveform):
    with torch.no_grad():
        return stft_forward(stft, torch.from_numpy(waveform)[None])[0]


# this sums the energy of the bins in each band, giving an array shaped (bands, frames)
def band_energy(spectrogram, band_widths):
    power = (spectrogram.abs() ** 2).numpy()
    energies = []
    start = 0
    for width in band_widths:
        energies.append(power[start:start + width].sum(axis=0))
        start += width
    return np.stack(energies)


# this marks the frames whose centre falls inside a detected speech segment
def frame_activity(waveform, num_frames, stft, vad_model, vad_settings):
    """Frame t covers samples t * hop - padding to t * hop - padding + n_fft, so its centre is at t * hop - padding + n_fft / 2."""
    speech = np.zeros(len(waveform), dtype=bool)
    for start_seconds, end_seconds in detect(torch.from_numpy(waveform), vad_model, vad_settings, SAMPLE_RATE):
        speech[int(start_seconds * SAMPLE_RATE):int(end_seconds * SAMPLE_RATE)] = True
    centres = np.arange(num_frames) * stft.hop_length - stft.padding + stft.n_fft // 2
    centres = np.clip(centres, 0, len(waveform) - 1)
    return speech[centres]


# this fits the random-walk variance of each band: how much the clean target's log band energy moves from one frame to the next while the target keeps talking
def fit_random_walk_variance(stft, band_widths, vad_model, vad_settings):
    trial_ids = []
    with open(REPO_ROOT / "data/manifests/sir0_train.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] in ("both", "target_only") and (RENDERED / "sir0_train" / row["trial_id"] / "target.wav").exists():
                trial_ids.append(row["trial_id"])
    trial_ids.sort()
    chosen = np.random.default_rng(SEED).choice(len(trial_ids), size=PRIOR_TRIALS, replace=False)
    differences = []
    for index in chosen:
        target = read_mono(RENDERED / "sir0_train" / trial_ids[index] / "target.wav")
        log_energy = np.log(band_energy(spectrogram_of(stft, target), band_widths) + ENERGY_FLOOR)
        active = frame_activity(target, log_energy.shape[1], stft, vad_model, vad_settings)
        # Only steps where the target talks in both frames count, so a pause starting or ending is not a step.
        both_active = active[1:] & active[:-1]
        differences.append((log_energy[:, 1:] - log_energy[:, :-1])[:, both_active])
    differences = np.concatenate(differences, axis=1)
    variance = differences.var(axis=1, ddof=1)
    trials_used = []
    for index in chosen:
        trials_used.append(trial_ids[index])
    return variance, differences.shape[1], trials_used


# this gives D1's gains: in overlap frames, each band turned down to the clean target's energy, never up
def oracle_energy_gains(estimate_energy, target_energy, overlap):
    gains = np.ones_like(estimate_energy)
    ratio = np.sqrt((target_energy + ENERGY_FLOOR) / (estimate_energy + ENERGY_FLOOR))
    gains[:, overlap] = np.minimum(1.0, ratio[:, overlap])
    return gains


# this gives D2's gains: a Kalman filter per band tracks the target's log band energy, and in overlap frames a band louder than the tracked target is turned down to it
def kalman_gains(estimate_log_energy, target_active, overlap, random_walk_variance, overlap_variance):
    """Predict: mean stays, variance grows by the random-walk variance. Correct: move the mean towards e21's output by the Kalman gain (Kalman, 1960).

    When the target is silent there is nothing to learn about it, so only the prediction step runs. Returns the gains, shaped (bands, frames), and the Kalman gains of the overlap frames.
    """
    num_bands, num_frames = estimate_log_energy.shape
    gains = np.ones((num_bands, num_frames))
    overlap_kalman_gains = []
    mean = None
    variance = None
    for frame in range(num_frames):
        if not target_active[frame]:
            if variance is not None:
                variance = variance + random_walk_variance
            continue
        observation = estimate_log_energy[:, frame]
        observation_variance = ALONE_VARIANCE
        if overlap[frame]:
            observation_variance = overlap_variance
        if mean is None:
            # The first frame the target talks in starts the track at e21's own output.
            mean = observation.copy()
            variance = np.full(num_bands, observation_variance)
            continue
        predicted_variance = variance + random_walk_variance
        kalman_gain = predicted_variance / (predicted_variance + observation_variance)
        mean = mean + kalman_gain * (observation - mean)
        variance = (1.0 - kalman_gain) * predicted_variance
        if overlap[frame]:
            # The tracked energy over e21's energy, as an amplitude ratio, capped at 1 so nothing is made louder.
            gains[:, frame] = np.minimum(1.0, np.exp((mean - observation) / 2.0))
            overlap_kalman_gains.append(kalman_gain)
    return gains, overlap_kalman_gains


# this multiplies every bin of each band by that band's gain
def apply_band_gains(spectrogram, gains, band_widths):
    per_bin = np.repeat(gains, band_widths, axis=0)
    return spectrogram * torch.from_numpy(per_bin.astype(np.float32))


# this gives the scale-invariant SDR of an estimate against the clean target (Le Roux et al., 2019), a quick check before the judge
def si_sdr(estimate, target):
    scale = np.dot(estimate, target) / (np.dot(target, target) + 1e-12)
    projection = scale * target
    error = estimate - projection
    return 10 * np.log10((np.dot(projection, projection) + 1e-12) / (np.dot(error, error) + 1e-12))


# this reads the two-speaker validation trials in manifest order
def validation_trials():
    trial_ids = []
    with open(REPO_ROOT / "data/manifests/sir0_val.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "both":
                trial_ids.append(row["trial_id"])
    return trial_ids


def main():
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    stft = build_stft(N_FFT, HOP, SAMPLE_RATE)
    band_widths = band_plan(SAMPLE_RATE, N_FFT)
    vad_model = load_model()
    with open(REPO_ROOT / "experiments/configs/generator.yaml") as handle:
        vad_settings = vad_config(yaml.safe_load(handle))

    print("fitting the random-walk variance on training targets", flush=True)
    random_walk_variance, steps_used, prior_trials = fit_random_walk_variance(stft, band_widths, vad_model, vad_settings)
    median_variance = median(random_walk_variance.tolist())
    PRIOR_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prior_record = {"date": date.today().isoformat(), "git_commit": git_commit, "seed": SEED, "split": "sir0_train", "n_trials": PRIOR_TRIALS, "frame_steps_used": int(steps_used), "energy_floor": ENERGY_FLOOR, "silero_vad": model_version(), "random_walk_variance_per_band": random_walk_variance.round(5).tolist(), "median_variance": round(median_variance, 5), "trials": prior_trials}
    with open(PRIOR_DIRECTORY / "prior.json", "w") as handle:
        json.dump(prior_record, handle, indent=2)

    variants = {"d1-oracle-energy": None}
    for multiple in OVERLAP_MULTIPLES:
        variants[f"d2-kalman-r{multiple}q"] = multiple * median_variance
    output_directories = {}
    for name in variants:
        output_directories[name] = RESULTS / f"2026-10-28-est-e21-tracker-{name}"
        output_directories[name].mkdir(parents=True, exist_ok=True)

    sums = {"e21": []}
    for name in variants:
        sums[name] = {"si_sdr": [], "gain_db": [], "kalman_gain": []}
    overlap_shares = []
    trial_ids = validation_trials()
    for count, trial_id in enumerate(trial_ids, 1):
        trial_directory = RENDERED / "sir0_val" / trial_id
        target = read_mono(trial_directory / "target.wav")
        interferer = read_mono(trial_directory / "interferer.wav")
        estimate = read_mono(E21_ESTIMATES / trial_id / "estimate.wav")
        estimate_spectrogram = spectrogram_of(stft, estimate)
        estimate_energy = band_energy(estimate_spectrogram, band_widths)
        target_energy = band_energy(spectrogram_of(stft, target), band_widths)
        num_frames = estimate_energy.shape[1]
        target_active = frame_activity(target, num_frames, stft, vad_model, vad_settings)
        interferer_active = frame_activity(interferer, num_frames, stft, vad_model, vad_settings)
        overlap = target_active & interferer_active
        overlap_shares.append(overlap.mean())
        sums["e21"].append(si_sdr(estimate, target))

        for name, overlap_variance in variants.items():
            if overlap_variance is None:
                gains = oracle_energy_gains(estimate_energy, target_energy, overlap)
            else:
                gains, overlap_kalman_gains = kalman_gains(np.log(estimate_energy + ENERGY_FLOOR), target_active, overlap, random_walk_variance, overlap_variance)
                if overlap_kalman_gains:
                    sums[name]["kalman_gain"].append(float(np.mean(overlap_kalman_gains)))
            with torch.no_grad():
                edited = stft_inverse(stft, apply_band_gains(estimate_spectrogram, gains, band_widths)[None], len(estimate))[0].numpy()
            (output_directories[name] / trial_id).mkdir(exist_ok=True)
            sf.write(str(output_directories[name] / trial_id / "estimate.wav"), edited, SAMPLE_RATE, subtype="FLOAT")
            sums[name]["si_sdr"].append(si_sdr(edited, target))
            if overlap.any():
                sums[name]["gain_db"].append(float(np.mean(20 * np.log10(gains[:, overlap]))))
        if count % 25 == 0:
            print(f"  {count}/{len(trial_ids)} trials", flush=True)

    for name, overlap_variance in variants.items():
        meta = {"date": date.today().isoformat(), "git_commit": git_commit, "script": "scripts/diagnose_target_tracker.py", "system": f"e21 + target tracker, {name} (forward-only, oracle labels)", "base_estimates": str(E21_ESTIMATES.relative_to(REPO_ROOT)), "seed": SEED, "split": "sir0", "manifest": "sir0_val", "condition": "both", "n_trials": len(trial_ids), "random_walk_prior": str((PRIOR_DIRECTORY / "prior.json").relative_to(REPO_ROOT)), "alone_variance": ALONE_VARIANCE, "overlap_variance": overlap_variance, "sample_rate": SAMPLE_RATE, "audio": "estimate.wav, float32, unnormalised, whole clip"}
        with open(output_directories[name] / "meta.yaml", "w") as handle:
            yaml.safe_dump(meta, handle, sort_keys=False)

    summary = {"date": date.today().isoformat(), "git_commit": git_commit, "n_trials": len(trial_ids), "overlap_share_of_frames_mean": round(float(np.mean(overlap_shares)), 4), "median_random_walk_variance": round(median_variance, 5), "e21_si_sdr_mean_db": round(float(np.mean(sums["e21"])), 3), "variants": {}}
    for name, overlap_variance in variants.items():
        record = {"overlap_variance": overlap_variance, "si_sdr_mean_db": round(float(np.mean(sums[name]["si_sdr"])), 3), "si_sdr_change_db": round(float(np.mean(sums[name]["si_sdr"]) - np.mean(sums["e21"])), 3), "mean_gain_in_overlap_db": round(float(np.mean(sums[name]["gain_db"])), 3)}
        if sums[name]["kalman_gain"]:
            record["mean_kalman_gain_in_overlap"] = round(float(np.mean(sums[name]["kalman_gain"])), 4)
        summary["variants"][name] = record
    SUMMARY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_DIRECTORY / "signal_check.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    # Generated by Claude Code
    print(f"median random-walk variance {median_variance:.4f} over {steps_used} frame steps; overlap is {100 * summary['overlap_share_of_frames_mean']:.1f} % of frames on average")
    print(f"e21 SI-SDR {summary['e21_si_sdr_mean_db']:.2f} dB")
    for name, record in summary["variants"].items():
        print(f"  {name:18s} SI-SDR change {record['si_sdr_change_db']:+.2f} dB   mean gain in overlap {record['mean_gain_in_overlap_db']:+.2f} dB   Kalman gain {record.get('mean_kalman_gain_in_overlap', '-')}")


if __name__ == "__main__":
    with timed("scripts/diagnose_target_tracker.py", lambda: "sir0_val both, 103 trials, 4 variants + 200-trial prior"):
        main()
