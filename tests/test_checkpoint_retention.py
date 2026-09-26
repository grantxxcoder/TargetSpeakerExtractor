"""Which checkpoints survive a run (scripts/train.py:checkpoints_to_drop).

The failure this guards against is unrecoverable and it has already happened
once. Item 1a's epoch 15 is the best content result the project has produced
(ASR LCF-WER 52.77 against a 65.22 floor) and it ranks TENTH OF FOURTEEN
eligible epochs on the signal score that decides retention. It survived only
because `_last.pt` is written unconditionally. A longer run would have deleted
it before anyone scored it. decisions-m2.md 2026-09-23.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import checkpoints_to_drop  # noqa: E402

# Item 1a's real ranking, (score, epoch) sorted by score, from
# experiments/results/2026-09-23-train-sir0-cueparts/history.csv.
CUEPARTS = [(4.319, 9), (4.599, 2), (4.948, 4), (4.992, 7), (5.393, 3),
            (5.474, 10), (5.539, 5), (5.601, 11), (6.323, 14), (6.443, 15),
            (6.772, 12), (6.981, 13), (7.033, 8), (7.630, 6)]


def test_default_is_unchanged_top_k():
    """stride off == the pre-2026-09-23 policy, so old configs are untouched."""
    kept = [(1.0, 9), (2.0, 2), (3.0, 4), (4.0, 7), (5.0, 15)]
    assert checkpoints_to_drop(kept, keep_top_k=3) == [7, 15]


def test_keep_top_k_zero_drops_everything():
    kept = [(1.0, 9), (2.0, 2)]
    assert checkpoints_to_drop(kept, keep_top_k=0) == [2, 9]


def test_stride_keeps_the_epoch_top_k_throws_away():
    """The regression. Epoch 15 is rank 10 of 14 and must still survive."""
    dropped = checkpoints_to_drop(CUEPARTS, keep_top_k=3, keep_stride=3,
                                  last_epoch=15)
    assert 15 not in dropped, "the best content epoch was deleted"
    assert 9 not in dropped, "the top-scoring epoch was deleted"
    for epoch in (3, 6, 12):
        assert epoch not in dropped, f"stride epoch {epoch} was deleted"


def test_stride_is_a_union_with_top_k_not_a_replacement():
    dropped = set(checkpoints_to_drop(CUEPARTS, keep_top_k=3, keep_stride=3,
                                      last_epoch=15))
    survivors = {e for _, e in CUEPARTS} - dropped
    assert {9, 2, 4} <= survivors, "top-3 by score must survive"
    assert {3, 6, 12, 15} <= survivors, "stride epochs must survive"


def test_stride_bounds_the_disk_cost():
    """Spread, not everything: a 16-epoch run keeps well under half its epochs."""
    dropped = checkpoints_to_drop(CUEPARTS, keep_top_k=3, keep_stride=3,
                                  last_epoch=15)
    survivors = len(CUEPARTS) - len(dropped)
    assert survivors <= 8, f"{survivors} checkpoints is too many to keep"
    assert len(dropped) > 0, "stride must still delete something"
