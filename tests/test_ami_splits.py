"""ami_train never touches ami_eval, and is in the shape training expects.
decisions-m8.md 2026-10-12. Reads the BUILT data; skips where it is not on disk.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRAIN = ROOT / "data/manifests/ami_train.csv"
EVAL = ROOT / "data/manifests/ami_eval.csv"
pytestmark = pytest.mark.skipif(not (TRAIN.exists() and EVAL.exists()),
                                reason="ami_train / ami_eval not built here")


@pytest.fixture(scope="module")
def train():
    return pd.read_csv(TRAIN)


@pytest.fixture(scope="module")
def evaluation():
    return pd.read_csv(EVAL)


def test_no_person_is_in_both(train, evaluation):
    people = lambda df: set(df.target_speaker) | set(df.interferer_speaker)  # noqa: E731
    assert not people(train) & people(evaluation)


def test_no_meeting_is_in_both_including_enrolment_meetings(train, evaluation):
    train_meetings = (set(train.meeting) | set(train.enrollment_meeting)
                      | set(train.interferer_enrollment_meeting))
    eval_meetings = set(evaluation.meeting) | set(evaluation.enrollment_meeting)
    assert not train_meetings & eval_meetings


def test_no_scenario_group_is_in_both(train):
    eval_groups = {g for site in yaml.safe_load(
        (ROOT / "experiments/configs/ami_eval.yaml").read_text())["groups"].values() for g in site}
    assert not set(train.meeting.str[:6]) & eval_groups


def test_the_loader_columns_are_present_and_remix_is_off(train):
    for col in ("sir_db", "snr_db", "target_absent", "overlap_achieved", "regime",
                "same_gender", "interferer_enrollment_phantom", "remixable"):
        assert col in train and train[col].notna().all(), col
    assert (train.remixable == 0).all() and (train.regime == "ami").all()
    assert (train.mixture_length_s >= 4.008).all()


def test_every_trial_has_its_files_and_a_bank(train):
    for tid in train.trial_id:
        d = ROOT / "data/rendered/ami_train" / tid
        for f in ("mixture.wav", "target.wav", "interferer.wav", "enrollment.wav",
                  "interferer_enrollment.wav", "enrollment_v00.wav",
                  "interferer_enrollment_v00.wav", "meta.json", "enrollment_bank.json"):
            assert (d / f).exists(), d / f
        bank = json.loads((d / "enrollment_bank.json").read_text())["variants"]
        assert bank["target"]["n"] >= 1 and bank["interferer"]["n"] >= 1
