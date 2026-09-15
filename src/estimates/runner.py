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


# Written before the first trial and deleted after the last, so its PRESENCE
# means "a pass over this directory started and did not finish". meta.yaml keeps
# its old meaning untouched -- written last, so it still marks completion. Two
# files, two questions: "did this finish?" and "what was it going to be?".
RESUME_LEDGER = "run.provenance.yaml"

# Keys meta.yaml adds on top of the caller's provenance. Needed to read an old
# COMPLETED directory back as a provenance dict, since write_estimates flattens
# provenance into meta rather than nesting it.
_META_ONLY = frozenset({"date", "git_commit", "n_trials", "n_written",
                        "n_reused", "sample_rate", "audio",
                        "estimate_length_delta_samples"})


def _reusable(estimate_path: Path, mixture_path: Path, sample_rate: int) -> bool:
    """Is this estimate.wav from a finished write, and therefore safe to keep?

    Ctrl-C during a 3-hour render lands mid-write often enough to matter, and a
    half-written wav opens fine while being silently short. So an existing file
    is reused only if its length matches its mixture to the same tolerance the
    run already warns at -- the check costs a header read, not a decode.

    NOT a checksum. It catches the interrupted write, which is the failure this
    exists for; it cannot catch a file that a different model wrote at the right
    length. That is what the provenance ledger is for.
    """
    if not estimate_path.exists():
        return False
    try:
        frames = sf.info(str(estimate_path)).frames
        expected = sf.info(str(mixture_path)).frames
    except RuntimeError:            # unreadable header = interrupted write
        return False
    return frames > 0 and abs(frames - expected) <= LENGTH_WARN_S * sample_rate


def _prior_provenance(out_root: Path, ledger_path: Path):
    """What the estimates already in this directory were made by, or None.

    Reads the ledger of an interrupted run first, then meta.yaml of a completed
    one, so a resume is checked against either.
    """
    if ledger_path.exists():
        return yaml.safe_load(ledger_path.read_text())
    meta_path = out_root / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        return {"provenance": {k: v for k, v in meta.items()
                               if k not in _META_ONLY},
                "n_trials_requested": meta.get("n_trials")}
    return None


def _ledger_differences(prior: dict, current: dict):
    """Every field on which a resume disagrees with what is already on disk.

    WHY REFUSE RATHER THAN WARN. Reusing one checkpoint's audio under another
    checkpoint's meta.yaml produces a directory that is a blend of two systems
    and says it is one. Nothing downstream can detect that -- evaluate.py reads
    wav files and believes the meta -- so it has to be impossible, not merely
    discouraged.

    n_trials_requested IS RECORDED BUT NOT COMPARED. A pass over more trials
    than the one before it is the ordinary case -- `--limit 2` to check the
    wiring, then the whole split -- and the two estimates already rendered are
    the same model's on the same audio, so refusing them would block exactly
    what resuming is for. What decides whether a file is reusable is which
    system and which audio produced it, and `split`, `manifest`, `condition`
    and `checkpoint` in the provenance already say all four.
    """
    differences = []
    prior_provenance = prior.get("provenance") or {}
    for key in sorted(set(prior_provenance) | set(current["provenance"])):
        was, now = prior_provenance.get(key, "<absent>"), current["provenance"].get(key, "<absent>")
        if was != now:
            differences.append((key, was, now))
    return differences


def write_estimates(extract: Extractor,
                    trials: Sequence[Trial],
                    out_root,
                    sample_rate: int,
                    provenance: dict,
                    progress_every: int = 25,
                    force: bool = False) -> dict:
    """Run `extract` over every trial and write <out_root>/<trial_id>/estimate.wav.

    Returns the provenance dict actually written, so a caller can log it.
    Deliberately NOT wrapped in src.run_log.timed here -- the front-end script
    owns that, because the run_times.md row should name the script the user
    typed, not this helper.

    RESUMES BY DEFAULT. A trial whose estimate.wav is already on disk and
    complete is kept, not re-rendered. A full pass over sir0_privval is 3.4 h
    measured (docs/run_times.md 2026-09-14), which is long enough that it gets
    interrupted -- and before this, an interrupted pass had to start from zero,
    so the cost of stopping a render was the whole render. Inference is
    deterministic under no_grad, so a reused file is the file this pass would
    have written.

    THE RESUME IS GUARDED, and the guard is the point. Reuse is only safe if the
    audio already there came from the SAME system, so the run records its
    provenance in RESUME_LEDGER before the first trial and refuses to resume
    over a directory whose recorded provenance differs. Without that, pointing a
    second checkpoint at a directory would silently blend two systems into one
    set of estimates and label it as one -- and nothing downstream could tell,
    because evaluate.py reads wav files and believes meta.yaml.

    force=True re-renders everything and skips the guard, for the case where the
    files on disk are known bad rather than merely different.
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

    ledger_path = out_root / RESUME_LEDGER
    ledger = {"provenance": provenance, "n_trials_requested": len(trials)}

    if not force:
        prior = _prior_provenance(out_root, ledger_path)
        differences = _ledger_differences(prior, ledger) if prior else []
        if differences:
            lines = "\n".join(f"    {key}: on disk {was!r}, this run {now!r}"
                              for key, was, now in differences)
            raise SystemExit(
                f"refusing to resume {out_root}: the estimates already there "
                f"were made by a different run.\n{lines}\n"
                f"  Use a different --out, or --force to re-render the "
                f"directory from scratch.")
    with open(ledger_path, "w") as fh:
        yaml.safe_dump(ledger, fh, sort_keys=False)

    written, reused, deltas = 0, 0, []

    for trial in trials:
        mixture_path = trial.directory / "mixture.wav"
        estimate_path = out_root / trial.trial_id / "estimate.wav"
        if not force and _reusable(estimate_path, mixture_path, sample_rate):
            # Header reads only -- the whole saving is in not decoding the
            # mixture and not running the model.
            deltas.append(sf.info(str(estimate_path)).frames
                          - sf.info(str(mixture_path)).frames)
            reused += 1
            if reused % progress_every == 0:
                print(f"  reused {reused} existing", flush=True)
            continue

        mixture = _read_mono(mixture_path)
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
        estimate_path.parent.mkdir(parents=True, exist_ok=True)
        # subtype FLOAT, so the written file is float32 and unnormalised; a
        # PCM_16 write would clip anything the model pushed past full scale
        # instead of leaving the overshoot visible.
        sf.write(str(estimate_path), estimate, sample_rate, subtype="FLOAT")

        written += 1
        if (written + reused) % progress_every == 0 or written + reused == len(trials):
            print(f"  {written + reused}/{len(trials)}", flush=True)

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
        # STILL "how many trials are in this directory", so every meta.yaml
        # written before resuming existed still reads the same way. The split
        # into rendered-now and kept is recorded alongside, not inside it.
        "n_trials": written + reused,
        "n_written": written,
        "n_reused": reused,
        "sample_rate": sample_rate,
        "audio": "estimate.wav, float32, unnormalised, whole clip",
        "estimate_length_delta_samples": {"min": min(deltas, default=0),
                                          "max": max(deltas, default=0)},
    }
    with open(out_root / "meta.yaml", "w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)
    # Last, and only on success: meta.yaml now marks the directory complete, so
    # the "a pass is unfinished" ledger would be a lie if it outlived it.
    ledger_path.unlink(missing_ok=True)
    kept = f", reused {reused}" if reused else ""
    print(f"\n  wrote {written} estimates{kept} -> {out_root}/")
    return meta


def _read_mono(path: Path) -> np.ndarray:
    """Whole file as 1-D float32. Not the 4 s training crop: the dataset loader
    crops for training, so estimates read the files directly."""
    if not path.exists():
        raise SystemExit(f"missing {path}")
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    return audio.mean(axis=1) if audio.ndim > 1 else audio
