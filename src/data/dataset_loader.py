"""Turns manifest rows into waveform tensors for training. This is the read side of the data pipeline.

It returns waveforms rather than spectrograms, because the STFT lives in the model and one waveform feeds all four loss resolutions (decisions-m1.md 2026-08-18).
Each trial gives two examples: the same mixture asked for the target and for the other speaker. That pairing is the only thing in the data that forces the model to read the enrolment (decisions-m2.md 2026-08-26).
In training, every epoch draws a new 4 s crop, one of the speaker's enrolment recordings, and a new SIR and SNR lent by another trial (decisions-m2.md 2026-08-30). Validation keeps all three fixed, so its numbers stay comparable across epochs.
Every random draw is seeded by (seed, epoch, trial), not by a shared generator, so it is the same in any worker process.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import default_collate

# This must equal CLIP_CEILING in src/data/render.py, because the remix repeats the renderer's last step. It is copied rather than imported so the loader workers do not load scipy and pyloudnorm.
CLIP_CEILING = 0.95


# this flattens each trial's pair of examples into one batch, so batch_size counts trials and a step sees twice as many examples
def collate_pairs(batch):
    examples = []
    for pair in batch:
        examples.extend(pair)
    return default_collate(examples)


# this reads one mono wav file, or a piece of it, as a float32 tensor
def read_wav(path, sample_rate, start, frames):
    with sf.SoundFile(str(path)) as sound_file:
        if sound_file.samplerate != sample_rate:
            raise ValueError(f"{path} is at {sound_file.samplerate} Hz, but {sample_rate} Hz was expected.")
        if sound_file.channels != 1:
            raise ValueError(f"{path} has {sound_file.channels} channels, but it should be mono.")
        if start > 0:
            sound_file.seek(start)
        audio = sound_file.read(frames, dtype="float32", always_2d=False)
    return torch.from_numpy(np.ascontiguousarray(audio))


# this picks where the crop starts
def crop_start(seed, epoch, trial_index, clip_frames, chunk_frames):
    latest_start = clip_frames - chunk_frames
    if latest_start < 0:
        raise ValueError(f"A clip of {clip_frames} samples is shorter than the {chunk_frames} sample crop.")
    generator = np.random.default_rng((seed, epoch, trial_index))
    return int(generator.integers(0, latest_start + 1))


# this picks which of the speaker's enrolment recordings to use this epoch
def enrollment_path(trial_directory, name, seed, epoch, trial_index, stream, enrollment_variants):
    """The two speakers draw from separate streams, so their choices are independent.

    A speaker with only a few long utterances can have a short bank, so a missing variant falls back to variant 0.
    """
    generator = np.random.default_rng((seed, epoch, trial_index, stream))
    variant = int(generator.integers(0, enrollment_variants))
    path = trial_directory / f"{name}_v{variant:02d}.wav"
    if not path.exists():
        return trial_directory / f"{name}_v00.wav"
    return path


# this collects, for each difficulty regime, the SIR and SNR values another trial can lend to the remix
def gain_pools(manifest):
    """SIR values come from trials with both speakers, and SNR values from trials where the target speaks, because only there was each level actually applied."""
    pools = {}
    for regime, group in manifest.groupby("regime"):
        sir_values = group.loc[group["condition"] == "both", "sir_db"].to_numpy(float)
        snr_values = group.loc[group["target_absent"] == 0, "snr_db"].to_numpy(float)
        pools[str(regime)] = (sir_values, snr_values)
    return pools


# this draws another trial's SIR and SNR from the same regime, so every remixed level is one the generator really produced
def draw_gains(pools, row, seed, epoch, trial_index):
    """Returns None for a level that does not apply. A target-absent trial keeps its rendered levels, and only a trial with both speakers gets a new SIR."""
    if int(row["target_absent"]):
        return None, None
    # Stream 2 keeps this draw apart from the crop and the two enrolment draws.
    generator = np.random.default_rng((seed, epoch, trial_index, 2))
    sir_values, snr_values = pools[str(row["regime"])]
    new_sir = None
    if str(row["condition"]) == "both" and len(sir_values):
        new_sir = float(sir_values[generator.integers(len(sir_values))])
    new_snr = None
    if len(snr_values):
        new_snr = float(snr_values[generator.integers(len(snr_values))])
    return new_sir, new_snr


# this rebuilds the mixture at the new levels
def remix(mixture, target, interferer, row, new_sir, new_snr):
    """The noise comes back by subtraction, because the mixture was rendered as target plus interferer plus noise.

    Only the mixture should change, but if the new sum would clip, all three take the same gain, as the renderer does.
    """
    if new_sir is None and new_snr is None:
        return mixture, target, interferer
    noise = mixture - target - interferer
    if new_sir is not None:
        # SIR is the target level minus the interferer level, so a lower SIR means a louder interferer.
        interferer = interferer * (10.0 ** ((float(row["sir_db"]) - new_sir) / 20.0))
    if new_snr is not None:
        noise = noise * (10.0 ** ((float(row["snr_db"]) - new_snr) / 20.0))
    mixture = target + interferer + noise
    peak = max(float(mixture.abs().max()), float(target.abs().max()), float(interferer.abs().max()))
    if peak > CLIP_CEILING:
        common_gain = CLIP_CEILING / peak
        mixture = mixture * common_gain
        target = target * common_gain
        interferer = interferer * common_gain
    return mixture, target, interferer


class TrialDataset(torch.utils.data.Dataset):
    """DataLoader needs an object it can index and take the length of, which is why this one stays a class.

    Each item is a list of two examples, one per speaker, so both directions of a trial always land in the same batch.
    """

    def __init__(self, manifest_csv, audio_root, chunk_s, sample_rate, seed, training, enrollment_variants):
        self.manifest = pd.read_csv(manifest_csv)
        self.audio_root = Path(audio_root)
        self.chunk_frames = int(chunk_s * sample_rate)
        self.sample_rate = sample_rate
        self.seed = seed
        self.training = training
        self.enrollment_variants = enrollment_variants
        self.epoch = 0
        if training:
            self.gain_pools = gain_pools(self.manifest)

    def __len__(self):
        return len(self.manifest)

    # this is called at the start of every training epoch, so the crop, enrolment and remix draws change
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, trial_index):
        row = self.manifest.iloc[trial_index]
        trial_directory = self.audio_root / row["trial_id"]
        mixture_path = trial_directory / "mixture.wav"
        # Validation always takes the epoch 0 crop, so its crops never move.
        crop_epoch = self.epoch if self.training else 0
        start = crop_start(self.seed, crop_epoch, trial_index, sf.info(str(mixture_path)).frames, self.chunk_frames)
        mixture = read_wav(mixture_path, self.sample_rate, start, self.chunk_frames)
        target = read_wav(trial_directory / "target.wav", self.sample_rate, start, self.chunk_frames)
        interferer = read_wav(trial_directory / "interferer.wav", self.sample_rate, start, self.chunk_frames)
        if self.training:
            new_sir, new_snr = draw_gains(self.gain_pools, row, self.seed, self.epoch, trial_index)
            mixture, target, interferer = remix(mixture, target, interferer, row, new_sir, new_snr)
        # Both directions hear the same mixture. Only the enrolment and the reference change.
        target_example = self.example(trial_directory, trial_index, "enrollment", 0, mixture, target)
        interferer_example = self.example(trial_directory, trial_index, "interferer_enrollment", 1, mixture, interferer)
        return [target_example, interferer_example]

    # this builds one example: the mixture, the speaker asked for, their enrolment, and whether they are silent in this crop
    def example(self, trial_directory, trial_index, enrollment_name, stream, mixture, reference):
        if self.training:
            path = enrollment_path(trial_directory, enrollment_name, self.seed, self.epoch, trial_index, stream, self.enrollment_variants)
        else:
            path = trial_directory / f"{enrollment_name}.wav"
        enrollment = read_wav(path, self.sample_rate, 0, -1)
        # The crop's own audio decides whether the speaker is silent, not the trial's label, because a 4 s crop of a speaking trial can hold only silence.
        crop_absent = bool(reference.abs().max() == 0)
        return {"mixture": mixture, "target": reference, "enrollment": enrollment, "crop_absent": crop_absent}
