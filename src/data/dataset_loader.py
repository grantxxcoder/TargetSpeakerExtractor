"""Manifest row -> waveform tensors. The read side of the data pipeline.

Waveforms, never spectrograms: the STFT lives in the model so one waveform feeds
all four loss resolutions, and a complex64 spectrogram is ~8x the PCM_16 on disk
(~185 GB vs 27 GB for train).

Decisions, all decisions-m1.md 2026-08-18:
  chunk_s = 4.0, matching CARTSE Track 1 (the only online causal TSE candidate).
  Crop offsets uniform -- VAD-aware cropping measured and rejected, leakage 5.8 %
  and confined to `hard` below target_activity 0.4.
  `crop_absent` comes from the CROPPED target stem, not the manifest's clip-level
  label: a 4 s crop of a `both` trial may hold no target speech, and target.wav is
  exactly zero in silence, so the crop's own audio is exact ground truth.

Offsets derive from (seed, epoch, idx), not an ambient RNG -- reproducible, and it
sidesteps each DataLoader worker holding its own RNG copy. Call set_epoch() each
epoch or you re-crop the same window and use a sixth of the audio.

CPU tensors: creating CUDA tensors in a forked worker raises "Cannot
re-initialize CUDA in forked subprocess".
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data._utils.collate import default_collate


def collate_pairs(batch):
    """Flatten the per-trial lists, then collate normally.

    Keeps train.py untouched: it still receives a flat batch of examples. Note
    that `batch_size` therefore counts TRIALS, and a step sees twice as many
    examples when both_directions is on -- which is what the GPU memory probe in
    the Kaggle notebook measures, so it adapts on its own.
    """
    return default_collate([ex for group in batch for ex in group])


class TrialDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, data_root, split, chunk_s, sample_rate, seed,
                 random_crop=True, both_directions=False):
        self.data_root    = Path(data_root)
        self.split        = split                   
        self.chunk_s      = chunk_s        
        self.sample_rate  = sample_rate
        self.seed         = seed
        self.random_crop  = random_crop
        # BOTH DIRECTIONS: one trial -> two examples, the same mixture asked for
        # the target and for the other speaker. decisions-m1.md 2026-08-26.
        # This is the ONLY thing in the data that makes reading the enrollment
        # compulsory -- with one direction an enrollment-ignoring model fits
        # every example, and measured, it did. `interferer_only` trials are the
        # sharpest case: same audio, silence one way and a voice the other.
        self.both_directions = both_directions
        self.chunk_frames = int(chunk_s * sample_rate)
        self.epoch        = 0                       

        self.manifest_df = pd.read_csv(manifest_csv)

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        """A LIST of examples, one per direction. Flattened by collate_pairs.

        A list rather than one dict so both directions of a trial are guaranteed
        to land in the SAME batch: the contrast has to be inside each gradient
        step, and shuffling flat indices would separate them.
        """
        out = [self._example(idx, "target")]
        if self.both_directions:
            out.append(self._example(idx, "interferer"))
        return out

    def _example(self, idx, which):
        """One training example. `which` selects the direction:

            "target"      target.wav                + enrollment.wav
            "interferer"  interferer.wav            + interferer_enrollment.wav

        Both read the SAME mixture crop -- same offset, same audio. That is the
        point: the input is identical and only the enrollment differs.
        """
        row = self.manifest_df.iloc[idx]
        trial_directory = self.data_root / "rendered" / self.split / row["trial_id"]
        mixture_directory = trial_directory / "mixture.wav"
        number_of_frames = sf.info(str(mixture_directory)).frames

        # keyed on idx, not on the direction, so the two directions share a crop
        start_offset = self._crop_offset_start(idx, number_of_frames)
        stem, enrol = (("target.wav", "enrollment.wav") if which == "target"
                       else ("interferer.wav", "interferer_enrollment.wav"))

        mixture_audio = self._read_in_wav(mixture_directory, start=start_offset, frames=self.chunk_frames)
        target_audio = self._read_in_wav(trial_directory / stem, start=start_offset, frames=self.chunk_frames)
        enrollment_audio = self._read_in_wav(trial_directory / enrol)

        # Same rule as before, now applied to whichever stem is the target for
        # this direction: the crop's own audio is the ground truth, never the
        # manifest's clip-level label. A phantom interferer is silent for the
        # whole clip, so this is exactly zero and the example is absent.
        crop_absent = bool(target_audio.abs().max() == 0)
        return {
            "mixture": mixture_audio,
            "target": target_audio,
            "enrollment": enrollment_audio,
            "crop_absent": crop_absent,
            "trial_id": str(row["trial_id"]),
            "direction": which,
            "meta": {
                "condition":        str(row["condition"]),
                "clip_absent":      bool(row["target_absent"]),
                "sir_db":           float(row["sir_db"]),
                "snr_db":           float(row["snr_db"]),
                "overlap_achieved": float(row["overlap_achieved"]),
                "regime":           str(row["regime"]),
                "same_gender":      float(row["same_gender"]),
                "phantom":          bool(row.get("interferer_enrollment_phantom", 0))
                                    and which == "interferer",
            },
        }


    def _read_in_wav(self, path, start=0, frames=-1):
        # Read in a wav file and return a torch tensor of shape (frames,)
        with sf.SoundFile(str(path)) as f:
            assert f.samplerate == self.sample_rate, f"Sample rate mismatch: {f.samplerate} != {self.sample_rate}"
            assert f.channels == 1, f"Channel mismatch: {f.channels} != 1"

            if start > 0:
                f.seek(start)
            
            x = f.read(frames, dtype='float32', always_2d=False)

        return torch.from_numpy(np.ascontiguousarray(x))

    def _crop_offset_start(self, idx, n_frames):
        # This is used to determine the starting point of the crop for the audio. It is important to note that this is only used for the mixture and target audio, not the enrollment audio. The enrollment audio is always read in full.

        max_start = n_frames - self.chunk_frames
        assert max_start >= 0, f"clip {n_frames} shorter than chunk {self.chunk_frames}"

        epoch = self.epoch if self.random_crop else 0
        rng = np.random.default_rng((self.seed, epoch, idx))
        return int(rng.integers(0, max_start + 1))

    def set_epoch(self, epoch):
        # call this at the top of each training epoch to ensure that the random cropping is consistent across all samples in the dataset. This is important for reproducibility and to ensure that the model sees the same data in each epoch.
        self.epoch = epoch
