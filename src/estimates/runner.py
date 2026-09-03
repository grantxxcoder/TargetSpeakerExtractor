"""Write estimate.wav for every trial in a manifest, for ANY extractor.

Stage 1 of a two-stage evaluation. Stage 2 is scripts/evaluate.py, which reads
estimate directories and never knows which model made them -- so making a new
system comparable is entirely a matter of writing its estimates in this layout.

Shared by:
  scripts/make_estimates.py        -- our BSRNN checkpoint
  scripts/make_estimates_wesep.py  -- the REAL-TSE WeSep pretrained baseline

VENV-NEUTRAL ON PURPOSE. This module imports only the standard library, numpy,
soundfile and yaml, because the two systems CANNOT share an interpreter: ours
needs torch 2.13 / numpy 2.5.2 (requirements.txt pins them, and every rendered
trial and VAD figure depends on those pins), while WeSep needs torch 2.7.1 and
downgrades numpy to 1.26.4. So the front-end scripts run under two different
virtualenvs and meet here, on the file format. Do not import torch, pandas or
anything from src.models in this file -- doing so silently breaks the WeSep
front-end, whose venv has none of them.

What every extractor is held to, so that two systems differ only by their model:

  * WHOLE CLIP, ONE FORWARD PASS, no chunking. Our model is causal, so
    appending later audio cannot change earlier output (measured 2026-08-24,
    1.68e-08), and stitching chunks reinjects the overlap-add tail at every
    seam.
  * float32, UNNORMALISED output. Normalising would hide the gain error that
    L_gain exists to catch, and it would hand a level correction to whichever
    system happened to apply it -- a difference between systems that is not a
    difference in extraction.
  * One meta.yaml for the whole run, not one per trial: provenance is a
    property of the pass, not of each file.
"""

from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import soundfile as sf
import yaml

# A length difference this big is a bug, not framing. WeSep returns 16 samples
# (1 ms) short of its input because of STFT framing, which the metrics tolerate
# -- they truncate to the shortest input -- but a tenth of a second would mean
# something is actually misaligned, so it gets said out loud.
LENGTH_WARN_S = 0.05

# The extractor contract: mixture and enrollment as 1-D float arrays at
# `sample_rate`, estimate back as a 1-D float array. Whole clips, not crops.
Extractor = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Trial:
    trial_id: str
    directory: Path
    condition: str


def git_commit() -> str:
    """HEAD, suffixed -dirty when the tree is not clean.

    Duplicated from scripts/train.py rather than imported, matching how the
    other scripts do it -- and here duplication is load-bearing, not just
    convention: train.py imports torch, which this module must not. `-dirty`
    matters because a result logged against a dirty tree is not reproducible
    from that hash.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:                            # noqa: BLE001
        return "UNKNOWN-not-a-git-checkout"


def read_trials(manifest_csv, audio_root, limit=None, condition=None) -> list[Trial]:
    """Trials in manifest order, optionally filtered by condition.

    Read with the stdlib csv module rather than pandas, both to stay
    venv-neutral and because nothing here needs a dataframe. Order and count
    match src.data.dataset_loader.TrialDataset, which does a bare read_csv and
    filters nothing, so switching to this changed neither which trials are
    rendered nor in what order.
    """
    manifest_csv, audio_root = Path(manifest_csv), Path(audio_root)
    if not manifest_csv.exists():
        raise SystemExit(f"no manifest at {manifest_csv}")
    with open(manifest_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{manifest_csv} has no rows")
    if "trial_id" not in rows[0]:
        raise SystemExit(f"{manifest_csv} has no trial_id column; "
                         f"columns are {sorted(rows[0])}")

    trials = [Trial(trial_id=str(r["trial_id"]),
                    directory=audio_root / str(r["trial_id"]),
                    condition=str(r.get("condition", "")))
              for r in rows
              if condition is None or str(r.get("condition", "")) == condition]
    if condition is not None and not trials:
        seen = sorted({str(r.get("condition", "")) for r in rows})
        raise SystemExit(f"no trials with condition={condition!r} in "
                         f"{manifest_csv}; conditions present: {seen}")
    return trials if limit is None else trials[:limit]


def write_estimates(extract: Extractor,
                    trials: Sequence[Trial],
                    out_root,
                    sample_rate: int,
                    provenance: dict,
                    progress_every: int = 25) -> dict:
    """Run `extract` over every trial and write <out_root>/<trial_id>/estimate.wav.

    Returns the provenance dict actually written, so a caller can log it.
    Deliberately NOT wrapped in src.run_log.timed here -- the front-end script
    owns that, because the run_times.md row should name the script the user
    typed, not this helper.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Check the provenance serialises BEFORE spending the compute. meta.yaml is
    # written last, so a value yaml.safe_dump cannot represent -- torch.__version__
    # is a str SUBCLASS and is refused -- would otherwise throw away a whole pass
    # over the split at the very end, with the estimates on disk but no record of
    # what made them, which is the same as not having run it.
    try:
        yaml.safe_dump(provenance)
    except yaml.YAMLError as exc:
        raise SystemExit(f"provenance will not serialise, refusing to run: {exc}")

    written, deltas = 0, []

    for trial in trials:
        mixture = _read_mono(trial.directory / "mixture.wav")
        enrollment = _read_mono(trial.directory / "enrollment.wav")

        estimate = extract(mixture, enrollment, sample_rate)
        if estimate is None:
            raise SystemExit(
                f"{trial.trial_id}: the extractor returned no audio. WeSep does "
                f"this when its own VAD is enabled and finds the enrollment "
                f"silent -- that gate belongs to the metric (speech_gate.py), "
                f"not inside a system under test.")
        estimate = np.asarray(estimate, dtype=np.float32).reshape(-1)

        deltas.append(len(estimate) - len(mixture))
        trial_dir = out_root / trial.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        # subtype FLOAT, so the written file is float32 and unnormalised; a
        # PCM_16 write would clip anything the model pushed past full scale
        # instead of leaving the overshoot visible.
        sf.write(str(trial_dir / "estimate.wav"), estimate, sample_rate,
                 subtype="FLOAT")

        written += 1
        if written % progress_every == 0 or written == len(trials):
            print(f"  {written}/{len(trials)}", flush=True)

    worst = max(deltas, key=abs) if deltas else 0
    if abs(worst) > LENGTH_WARN_S * sample_rate:
        print(f"  WARNING: estimate length differs from the mixture by up to "
              f"{worst} samples ({worst / sample_rate * 1000:+.0f} ms). The "
              f"metrics truncate to the shortest input, so this does not crash "
              f"-- but check for a real misalignment before trusting the row.")

    meta = {
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        **provenance,
        "n_trials": written,
        "sample_rate": sample_rate,
        "audio": "estimate.wav, float32, unnormalised, whole clip",
        "estimate_length_delta_samples": {"min": min(deltas, default=0),
                                          "max": max(deltas, default=0)},
    }
    with open(out_root / "meta.yaml", "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    print(f"\n  wrote {written} estimates -> {out_root}/")
    return meta


def _read_mono(path: Path) -> np.ndarray:
    """Whole file as 1-D float32. Not the 4 s training crop: the dataset loader
    crops for training, so estimates read the files directly."""
    if not path.exists():
        raise SystemExit(f"missing {path}")
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    return audio.mean(axis=1) if audio.ndim > 1 else audio
