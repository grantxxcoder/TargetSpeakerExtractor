"""The probe's loop guard: a listener loop must not count, leakage still must.

2026-09-24: `small.en` looped on sir0_val-42-000098 (166 words for a 55-word
target) and one such clip moved the 40-clip probe ~12 points.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.live_model_metric.content_probe import (  # noqa: E402
    ContentProbe, clip_errors, load_probe_trials, words_spoken)
from src.live_model_metric.lcf_wer import count_errors  # noqa: E402

MANIFEST = ROOT / "data/manifests/sir0_val.csv"
AUDIO = ROOT / "data/rendered/sir0_val"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="needs rendered sir0_val")


def one_trial():
    return load_probe_trials(MANIFEST, AUDIO, limit=1, condition="both")


def texts(trial):
    meta = json.loads((Path(trial[1]).parent / "meta.json").read_text())
    return meta["target_text"], meta["interferer_text"]


def test_uncapped_is_the_reported_rule():
    counts = count_errors("a b c", "x y z w w w w")
    assert clip_errors(counts) == counts.substitutions + counts.deletions + counts.insertions


def test_a_loop_is_capped_at_the_words_spoken():
    counts = count_errors("a b c", "a b c " + "c " * 50)
    assert clip_errors(counts, spoken=5) == 5


def test_a_wrong_but_not_looping_transcript_is_never_capped():
    # Entirely the OTHER speaker: full leakage, still charged in full.
    target, interferer = "a b c", "d e f g"
    counts = count_errors(target, interferer)
    raw = clip_errors(counts)
    assert clip_errors(counts, spoken=len(target.split()) + len(interferer.split())) == raw


def test_words_spoken_counts_both_speakers():
    # Counted AFTER the scorer's normaliser, which expands contractions
    # ("don't" -> "do not"), so it matches count_errors' reference_word_count.
    from src.live_model_metric.lcf_wer import normalise_text
    trial = one_trial()[0]
    target, interferer = texts(trial)
    assert words_spoken(Path(trial[1]).parent) == (len(normalise_text(target).split())
                                                   + len(normalise_text(interferer).split()))


def test_the_guard_is_off_unless_asked():
    trial = one_trial()
    loop = texts(trial[0])[0] + " again" * 400
    off = ContentProbe(trial, transcriber=lambda audio: loop)
    on = ContentProbe(trial, transcriber=lambda audio: loop, cap_errors_at_spoken=True)
    extract = lambda mixture, enrolment, sr: mixture
    wer_off, wer_on = off.score(extract), on.score(extract)
    assert wer_off > 100.0                   # the loop counts in full, as before
    assert wer_on < wer_off
    assert on.capped[-1][0] == 1 and off.capped[-1] == (0, 0)
    assert "loop guard" in on.verdict() or len(on.history) < 2


def test_train_py_reads_the_switch():
    import train
    base = {"content_probe": {"enabled": True, "split": "sir0_val", "n_trials": 1,
                              "data_root": str(ROOT / "data"),
                              "manifest_dir": str(ROOT / "data/manifests")}}
    assert train.build_content_probe(base, verbose=False).cap_errors_at_spoken is False
    base["content_probe"]["cap_errors_at_spoken"] = True
    assert train.build_content_probe(base, verbose=False).cap_errors_at_spoken is True
