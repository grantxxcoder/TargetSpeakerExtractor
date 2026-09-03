"""Tests for the shared estimate runner and the two front-ends that use it.

The point of src/estimates/runner.py is that our model and the borrowed WeSep
checkpoint are written to disk under identical conventions, so that
scripts/evaluate.py compares two MODELS rather than two pipelines. These tests
pin the conventions that make that true: manifest order, the audio format, and
the fact that the WeSep front-end's copy of the split table still agrees with
the one in scripts/train.py.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from src.estimates.runner import Trial, read_trials, write_estimates  # noqa: E402

SAMPLE_RATE = 16000
COLUMNS = ["trial_id", "condition"]


def _manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _trial_audio(audio_root, trial_id, n=8000, seed=0):
    d = audio_root / trial_id
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in ("mixture.wav", "enrollment.wav"):
        sf.write(str(d / name), rng.standard_normal(n).astype(np.float32) * 0.1,
                 SAMPLE_RATE, subtype="FLOAT")
    return d


@pytest.fixture
def split(tmp_path):
    """Six trials, three conditions, in a deliberately non-alphabetical order."""
    rows = [{"trial_id": "t-003", "condition": "both"},
            {"trial_id": "t-001", "condition": "both"},
            {"trial_id": "t-005", "condition": "noise_only"},
            {"trial_id": "t-002", "condition": "both"},
            {"trial_id": "t-004", "condition": "target_absent"},
            {"trial_id": "t-006", "condition": "both"}]
    audio_root = tmp_path / "rendered"
    for i, row in enumerate(rows):
        _trial_audio(audio_root, row["trial_id"], seed=i)
    return _manifest(tmp_path, rows), audio_root, rows


# --- read_trials ---------------------------------------------------------

def test_read_trials_preserves_manifest_order_and_count(split):
    manifest, audio_root, rows = split
    trials = read_trials(manifest, audio_root)
    # Manifest order, NOT sorted: TrialDataset does a bare read_csv and indexes
    # with .iloc, so reordering here would silently change which trials --limit
    # selects and break comparability with every earlier run.
    assert [t.trial_id for t in trials] == [r["trial_id"] for r in rows]


def test_read_trials_filters_nothing_by_default(split):
    manifest, audio_root, rows = split
    assert len(read_trials(manifest, audio_root)) == len(rows)


def test_read_trials_condition_filter(split):
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both")
    assert [t.trial_id for t in trials] == ["t-003", "t-001", "t-002", "t-006"]


def test_read_trials_limit_applies_after_the_filter(split):
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both", limit=2)
    assert [t.trial_id for t in trials] == ["t-003", "t-001"]


def test_read_trials_rejects_an_absent_condition_and_says_what_exists(split):
    manifest, audio_root, _ = split
    with pytest.raises(SystemExit, match="conditions present"):
        read_trials(manifest, audio_root, condition="nonesuch")


def test_read_trials_rejects_a_missing_manifest(tmp_path):
    with pytest.raises(SystemExit, match="no manifest"):
        read_trials(tmp_path / "absent.csv", tmp_path)


def test_read_trials_points_at_the_trial_directory(split):
    manifest, audio_root, _ = split
    trial = read_trials(manifest, audio_root)[0]
    assert trial.directory == audio_root / trial.trial_id


# --- write_estimates -----------------------------------------------------

def _passthrough(mixture, enrollment, sample_rate):     # noqa: ARG001
    return mixture


def test_write_estimates_layout_and_count(split, tmp_path):
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both")
    out = tmp_path / "out"
    meta = write_estimates(_passthrough, trials, out, SAMPLE_RATE, {"system": "test"})

    assert meta["n_trials"] == 4
    for trial in trials:
        assert (out / trial.trial_id / "estimate.wav").exists()
    # One meta.yaml for the pass, not one per trial.
    assert (out / "meta.yaml").exists()
    assert not (out / trials[0].trial_id / "meta.yaml").exists()


def test_write_estimates_is_float32_and_unnormalised(split, tmp_path):
    """A loud estimate must survive as written. PCM_16 would clip it and
    normalising would rescale it -- either one hides the gain error L_gain
    exists to catch, and hands one system a level correction the other
    does not get."""
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both", limit=1)
    out = tmp_path / "out"

    def loud(mixture, enrollment, sample_rate):         # noqa: ARG001
        return mixture * 0.0 + 2.5                       # deliberately past full scale

    write_estimates(loud, trials, out, SAMPLE_RATE, {})
    written, sr = sf.read(str(out / trials[0].trial_id / "estimate.wav"),
                          dtype="float32")
    assert sr == SAMPLE_RATE
    assert written.dtype == np.float32
    assert np.allclose(written, 2.5), "estimate was clipped or normalised"


def test_write_estimates_records_provenance_and_commit(split, tmp_path):
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both", limit=1)
    out = tmp_path / "out"
    write_estimates(_passthrough, trials, out, SAMPLE_RATE,
                    {"system": "test", "checkpoint": {"path": "x"}})
    meta = yaml.safe_load(open(out / "meta.yaml"))
    for key in ("date", "git_commit", "system", "checkpoint", "n_trials",
                "sample_rate", "audio", "estimate_length_delta_samples"):
        assert key in meta, f"provenance lost {key}"


def test_write_estimates_records_a_length_difference(split, tmp_path):
    """WeSep returns 16 samples short of its input because of STFT framing. The
    metrics truncate to the shortest input so it is harmless, but it is recorded
    rather than absorbed silently."""
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both", limit=1)
    out = tmp_path / "out"

    def short(mixture, enrollment, sample_rate):        # noqa: ARG001
        return mixture[:-16]

    meta = write_estimates(short, trials, out, SAMPLE_RATE, {})
    assert meta["estimate_length_delta_samples"] == {"min": -16, "max": -16}


def test_write_estimates_refuses_when_the_extractor_returns_nothing(split, tmp_path):
    """WeSep returns None when its own VAD is on and the enrollment is silent.
    That gate belongs to the metric, not inside a system under test, so this
    must fail loudly rather than skip a trial and shrink n."""
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both", limit=1)
    with pytest.raises(SystemExit, match="returned no audio"):
        write_estimates(lambda m, e, sr: None, trials, tmp_path / "out",
                        SAMPLE_RATE, {})


def test_write_estimates_rejects_unserialisable_provenance_before_running(split, tmp_path):
    """meta.yaml is written last, so a provenance value yaml.safe_dump cannot
    represent would otherwise be discovered only after a full pass over the
    split -- estimates on disk, no record of what made them. torch.__version__
    is exactly this: a str subclass, which safe_dump refuses."""
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both")
    out = tmp_path / "out"

    class TorchVersion(str):
        pass

    calls = []

    def counted(mixture, enrollment, sample_rate):      # noqa: ARG001
        calls.append(1)
        return mixture

    with pytest.raises(SystemExit, match="will not serialise"):
        write_estimates(counted, trials, out, SAMPLE_RATE,
                        {"torch": TorchVersion("2.7.1+cpu")})
    assert calls == [], "refused only after doing the work"


def test_write_estimates_flattens_a_channel_dimension(split, tmp_path):
    """Both front-ends hand back (1, N) from a batched forward pass; the file on
    disk has to be mono 1-D either way."""
    manifest, audio_root, _ = split
    trials = read_trials(manifest, audio_root, condition="both", limit=1)
    out = tmp_path / "out"
    write_estimates(lambda m, e, sr: m.reshape(1, -1), trials, out, SAMPLE_RATE, {})
    written, _ = sf.read(str(out / trials[0].trial_id / "estimate.wav"))
    assert written.ndim == 1


def test_write_estimates_reports_a_missing_trial_directory(tmp_path):
    out = tmp_path / "out"
    trials = [Trial(trial_id="absent", directory=tmp_path / "nope", condition="both")]
    with pytest.raises(SystemExit, match="missing"):
        write_estimates(_passthrough, trials, out, SAMPLE_RATE, {})


# --- the duplicated split table --------------------------------------------

def test_wesep_split_table_matches_train():
    """scripts/make_estimates_wesep.py carries its own copy of the split table
    because it runs in a venv with no torch 2.13 and no src.models, so it cannot
    import scripts/train.py. This is the guard on that copy: if train.py gains a
    split or repoints one, the WeSep front-end would otherwise keep rendering
    the previous audio directory against the new manifest and nothing would say
    so."""
    from make_estimates_wesep import VAL_SPLITS
    from train import SPLIT_MANIFESTS

    assert VAL_SPLITS == {split: pairs[1] for split, pairs in SPLIT_MANIFESTS.items()}
