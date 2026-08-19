"""Manifest row -> waveform tensors. The read side of the data pipeline.

Returns waveforms, never spectrograms: the STFT lives in the model so one
waveform can feed all four resolutions the multi-resolution loss needs, and
because a complex64 spectrogram is ~8x the size of the PCM_16 wav on disk
(~185 GB for the train split against 27 GB). See decisions-m1.md 2026-08-18.

The decisions this file implements, all in docs/decisions/decisions-m1.md:

  2026-08-18  chunk_s = 4.0, matching CARTSE Track 1 -- the only online, causal
              TSE system among the candidates.
  2026-08-18  crop offsets are uniform. VAD-aware cropping was measured and
              rejected: leakage is 5.8 %, and confined to the `hard` regime
              below target_activity 0.4.
  2026-08-18  `crop_absent` is computed from the cropped target stem, NOT from
              the manifest's clip-level `target_absent`. The manifest describes
              the whole clip; a 4 s crop of a `both` trial may contain no target
              speech at all. target.wav is exactly zero when the target is
              silent, so the crop's own audio is exact ground truth.

Crop offsets derive from (seed, epoch, idx) rather than an ambient RNG: that is
reproducible, and it sidesteps the num_workers problem where each worker process
holds its own RNG copy. Call set_epoch() each epoch or every epoch re-crops the
same window and you use a sixth of the audio.

Returns CPU tensors. Moving to device is the training loop's job -- creating CUDA
tensors in a forked DataLoader worker raises "Cannot re-initialize CUDA in forked
subprocess".
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch


class TrialDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_csv, data_root, split, chunk_s, sample_rate, seed, random_crop=True):
        self.data_root    = Path(data_root)
        self.split        = split                   
        self.chunk_s      = chunk_s        
        self.sample_rate  = sample_rate
        self.seed         = seed
        self.random_crop  = random_crop
        self.chunk_frames = int(chunk_s * sample_rate)
        self.epoch        = 0                       

        self.manifest_df = pd.read_csv(manifest_csv)

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        # there are a couple of things that I need to get:
        # 1. the mixture audio
        # 2. the target speaker audio
        # 3. the enrollment audio

        # Then there are a few things that are not necessary but are useful:
        # 1. the meta data
        # 2. the trial_id
        # 3. the crop absent

        row = self.manifest_df.iloc[idx]
        trial_directory = self.data_root / "rendered" / self.split / row["trial_id"]
        mixture_directory = trial_directory / "mixture.wav"
        number_of_frames = sf.info(str(mixture_directory)).frames

        start_offset = self._crop_offset_start(idx, number_of_frames)
        mixture_audio = self._read_in_wav(mixture_directory, start=start_offset, frames=self.chunk_frames)
        target_audio = self._read_in_wav(trial_directory / "target.wav", start=start_offset, frames=self.chunk_frames)
        enrollment_audio = self._read_in_wav(trial_directory / "enrollment.wav")

        crop_absent = bool(target_audio.abs().max() == 0)
        return {
            "mixture": mixture_audio,
            "target": target_audio,
            "enrollment": enrollment_audio,
            "crop_absent": crop_absent,
            "trial_id": str(row["trial_id"]),
            "meta": {
                "condition":        str(row["condition"]),
                "clip_absent":      bool(row["target_absent"]),
                "sir_db":           float(row["sir_db"]),
                "snr_db":           float(row["snr_db"]),
                "overlap_achieved": float(row["overlap_achieved"]),
                "regime":           str(row["regime"]),
                "same_gender":      float(row["same_gender"]),
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
