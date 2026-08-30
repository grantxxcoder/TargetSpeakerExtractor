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

`enrollment_variants` > 1 additionally rotates the CUE per epoch, reading one of
K enrollment recordings rendered by scripts/render_enrollment_bank.py. The crop
rotation alone does not do this: it resamples the same scene, so the identity
cue stayed a fixed waveform across all 24 epochs of the 2026-08-29 run and was
memorisable. decisions-m1.md 2026-08-30.

`remix_gains` rebuilds the mixture per epoch at a different SIR and SNR, so the
same trial is a different DIFFICULTY each time it is seen. Together the two
rotate both halves of the example -- the cue and the mixture -- which is the
pair a memorised input/output mapping needs to stay stable.

CPU tensors: creating CUDA tensors in a forked worker raises "Cannot
re-initialize CUDA in forked subprocess".
"""

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data._utils.collate import default_collate


# A6's clip ceiling. MUST equal src/data/render.CLIP_CEILING -- the remix re-runs
# the renderer's final step, and a mismatch would silently rescale every crop it
# touches. tests/test_remix_gains.py pins the two together rather than importing
# render here, which would pull scipy and pyloudnorm into every DataLoader worker.
CLIP_CEILING = 0.95


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
                 random_crop=True, both_directions=False, enrollment_variants=1,
                 remix_gains=False):
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
        # ENROLLMENT BANK: 1 = the single rendered enrollment.wav, i.e. the
        # behaviour every run up to 2026-08-29 had. >1 rotates through
        # enrollment_v00..v{K-1}.wav, which are distinct UTTERANCES by the same
        # speaker, each with its own EQ curve and all levelled identically.
        # v00 reproduces enrollment.wav exactly, so K=1 and a bank of 1 are the
        # same data and the arm is a clean ablation.
        self.enrollment_variants = int(enrollment_variants)
        # random_crop=False marks a FIXED evaluation set. The crop is pinned, and
        # the cue is pinned with it -- to `enrollment.wav` itself, not to
        # whichever variant epoch 0 happens to draw. That is what keeps a val
        # number comparable with every run from before the bank existed, and it
        # means an eval split needs no bank rendered at all.
        if not random_crop:
            self.enrollment_variants = 1
        self.chunk_frames = int(chunk_s * sample_rate)
        self.epoch        = 0                       

        # REMIX: rebuild the mixture each epoch at another trial's SIR and SNR.
        # False = the mixture exactly as rendered, i.e. every run up to
        # 2026-08-29. Pinned off on a fixed evaluation set for the same reason
        # the crop and the cue are. decisions-m1.md 2026-08-30.
        self.remix_gains = bool(remix_gains) and random_crop

        self.manifest_df = pd.read_csv(manifest_csv)
        if self.enrollment_variants > 1:
            self._check_bank()
        if self.remix_gains:
            self._build_gain_pools()

    def __len__(self):
        return len(self.manifest_df)

    def __getitem__(self, idx):
        """A LIST of examples, one per direction. Flattened by collate_pairs.

        A list rather than one dict so both directions of a trial are guaranteed
        to land in the SAME batch: the contrast has to be inside each gradient
        step, and shuffling flat indices would separate them.

        The mixture crop is read ONCE here rather than once inside each
        direction. Both directions share it by construction -- same offset, same
        audio -- so the second read was always a duplicate, and the remix has to
        happen before the split or the two directions would hear different
        mixtures and the contrast the pairing exists for would be broken.
        """
        row = self.manifest_df.iloc[idx]
        trial_directory = self.data_root / "rendered" / self.split / row["trial_id"]
        mixture_path = trial_directory / "mixture.wav"
        number_of_frames = sf.info(str(mixture_path)).frames

        # keyed on idx, not on the direction, so the two directions share a crop
        start_offset = self._crop_offset_start(idx, number_of_frames)

        mixture_audio = self._read_in_wav(mixture_path, start=start_offset,
                                          frames=self.chunk_frames)
        target_audio = self._read_in_wav(trial_directory / "target.wav",
                                         start=start_offset, frames=self.chunk_frames)
        # Wanted by the second direction, and by the remix for its noise
        # recovery. Nothing else needs it, and older single-direction splits may
        # not have the stem on disk at all.
        interferer_audio = (
            self._read_in_wav(trial_directory / "interferer.wav",
                              start=start_offset, frames=self.chunk_frames)
            if self.both_directions or self.remix_gains else None)

        # The REALISED levels, which are the manifest's own unless the remix
        # changed them. Reported in meta so a stratified diagnostic describes the
        # audio the model actually heard rather than the audio on disk.
        sir_used, snr_used = float(row["sir_db"]), float(row["snr_db"])
        if self.remix_gains:
            (mixture_audio, target_audio, interferer_audio,
             sir_new, snr_new) = self._remix(
                idx, row, mixture_audio, target_audio, interferer_audio)
            sir_used = sir_used if sir_new is None else sir_new
            snr_used = snr_used if snr_new is None else snr_new

        out = [self._example(idx, row, trial_directory, "target",
                             mixture_audio, target_audio, sir_used, snr_used)]
        if self.both_directions:
            out.append(self._example(idx, row, trial_directory, "interferer",
                                     mixture_audio, interferer_audio,
                                     sir_used, snr_used))
        return out

    def _remix(self, idx, row, mixture, target, interferer):
        """Rebuild the mixture at another trial's SIR and SNR.

        The three ingredients were summed at render time and two of them are on
        disk, so the third comes back by subtraction:

            noise = mixture - target - interferer

        exactly, on EVERY trial -- including the 3.5 % where A6's clip guard
        fired, because A6 scales the mixture and both stems by the same factor
        (render.render_trial step 4), so the equality survives it.

        Only the mixture is meant to change. `target` is the training reference
        and is returned untouched unless the new sum would clip, in which case it
        takes the same common gain the mixture does -- A6's rule, applied here to
        the crop rather than the clip, because the crop is all we have. That
        keeps the output/reference level relationship `L_gain` measures intact.
        """
        sir_new, snr_new = self._draw_gains(idx, row)
        if sir_new is None and snr_new is None:
            return mixture, target, interferer, None, None

        noise = mixture - target - interferer
        if sir_new is not None:
            # SIR is target level MINUS interferer level, so a lower new SIR
            # means a louder interferer.
            interferer = interferer * (10.0 ** ((float(row["sir_db"]) - sir_new) / 20.0))
        if snr_new is not None:
            noise = noise * (10.0 ** ((float(row["snr_db"]) - snr_new) / 20.0))
        mixture = target + interferer + noise

        peak = max(float(mixture.abs().max()), float(target.abs().max()),
                   float(interferer.abs().max()))
        if peak > CLIP_CEILING:
            common_gain = CLIP_CEILING / peak
            mixture, target = mixture * common_gain, target * common_gain
            interferer = interferer * common_gain
        return mixture, target, interferer, sir_new, snr_new

    def _draw_gains(self, idx, row):
        """Another trial's (SIR, SNR) from the same difficulty regime.

        RESAMPLED FROM THE MANIFEST, not from the ranges in generator.yaml.
        Three reasons, and they are the argument for the whole design: every
        value is one the generator actually produced, so no epoch can train on
        an out-of-distribution mixture; the regime mix and any within-regime
        correlation between SIR and SNR survive for free; and it needs no second
        copy of the sampling config to drift from the one the manifest was built
        with. The difficulty dial keeps meaning exactly what it meant.

        Returns (None, None) where the numbers do not compose. SIR is only
        meaningful with both speakers actually in the audio, and SNR only where
        the target is the loudness anchor -- on a target-absent trial the anchor
        is the interferer or the noise itself (render.render_trial, 2026-08-11),
        so those 24.7 % pass through as rendered.
        """
        if int(row["target_absent"]):
            return None, None
        # A distinct substream from the crop (seed, epoch, idx) and the
        # enrollment (seed, epoch, idx, 0|1), so adding the remix cannot shift
        # either and make runs before and after incomparable for a second reason.
        rng = np.random.default_rng((self.seed, self.epoch, idx, 2))
        sir_pool, snr_pool = self._gain_pools[str(row["regime"])]
        # target_only has no interferer to rebalance -- the stem is silence, so
        # scaling it is a no-op, and drawing a SIR would just be noise in the log.
        sir_new = (float(sir_pool[rng.integers(len(sir_pool))])
                   if str(row["condition"]) == "both" and len(sir_pool) else None)
        snr_new = (float(snr_pool[rng.integers(len(snr_pool))])
                   if len(snr_pool) else None)
        return sir_new, snr_new

    def _build_gain_pools(self):
        """regime -> (SIR values, SNR values) eligible to be donated.

        Restricted to rows where each number was actually applied: SIR from
        `both` trials, SNR from any target-present trial. A donor drawn from a
        target-absent row would carry a figure the renderer never used as a
        target-relative level.
        """
        df = self.manifest_df
        self._gain_pools = {}
        for regime, group in df.groupby("regime"):
            self._gain_pools[str(regime)] = (
                group.loc[group["condition"] == "both", "sir_db"].to_numpy(float),
                group.loc[group["target_absent"] == 0, "snr_db"].to_numpy(float))

    def _example(self, idx, row, trial_directory, which, mixture_audio,
                 target_audio, sir_used, snr_used):
        """One training example. `which` selects the direction:

            "target"      target.wav                + enrollment.wav
            "interferer"  interferer.wav            + interferer_enrollment.wav

        Both directions receive the SAME mixture object -- same offset, same
        audio, and after any remix the same rebuilt mixture. That is the point:
        the input is identical and only the enrollment differs.
        """
        enrol = ("enrollment.wav" if which == "target"
                 else "interferer_enrollment.wav")
        enrollment_audio = self._read_in_wav(
            self._enrollment_path(trial_directory, idx, enrol, which))

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
                # REALISED, not as rendered: these differ from the manifest
                # whenever remix_gains re-drew them for this epoch.
                "sir_db":           float(sir_used),
                "snr_db":           float(snr_used),
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

    def _enrollment_path(self, trial_directory, idx, name, which):
        """Which of the speaker's enrollment recordings to condition on this epoch.

        Keyed on the DIRECTION as well as (seed, epoch, idx), unlike the crop
        offset, which is deliberately shared so both directions see identical
        mixture audio. The two enrollments are different speakers entirely, so
        tying their variant choice together would buy nothing and would halve
        the number of distinct (target cue, interferer cue) pairs the model sees.

        Never reached on a fixed set: `random_crop=False` forces
        `enrollment_variants` to 1 in the constructor, so validation reads
        `enrollment.wav` and its curve stays readable across epochs.
        """
        if self.enrollment_variants <= 1:
            return trial_directory / name
        stream = 0 if which == "target" else 1
        rng = np.random.default_rng((self.seed, self.epoch, idx, stream))
        k = int(rng.integers(0, self.enrollment_variants))
        path = trial_directory / f"{name[:-4]}_v{k:02d}.wav"
        if not path.exists():
            # A speaker with few long utterances legitimately gets a short bank
            # (render_enrollment_bank.py reports how many), so fall back DOWN to
            # variant 0 rather than failing the run -- but only after the
            # constructor has confirmed a bank exists at all. A missing bank is
            # a configuration error and must not be papered over here.
            return trial_directory / f"{name[:-4]}_v00.wav"
        return path

    def _check_bank(self):
        """Fail at construction, not at epoch 1, if the bank is not on disk.

        `enrollment_variants` changes what data a run trains on, so a config
        claiming a bank that is not there would produce a history.csv that is
        unreadable next to its own config -- the same reason `amp` is
        config-driven.
        """
        first = self.manifest_df.iloc[0]["trial_id"]
        probe = (self.data_root / "rendered" / self.split / first
                 / "enrollment_v00.wav")
        if not probe.exists():
            raise FileNotFoundError(
                f"enrollment_variants={self.enrollment_variants} but {probe} is "
                "missing. Render the bank first:\n"
                f"  python scripts/render_enrollment_bank.py --split {self.split} "
                f"--variants {self.enrollment_variants}")

    def set_epoch(self, epoch):
        # call this at the top of each training epoch to ensure that the random cropping is consistent across all samples in the dataset. This is important for reproducibility and to ensure that the model sees the same data in each epoch.
        self.epoch = epoch
