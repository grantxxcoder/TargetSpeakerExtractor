"""LCF — Live-model Content Fidelity. docs/data/metric-definitions.md.

Built one metric at a time, on purpose. Present so far:

    lcf_wer.py   LCF-WER, the primary score (3.1)

Still to come, each as its own module: ICR (3.2), NRR (3.3).

NOTHING HERE MAY BE IMPORTED BY THE TRAINING LOOP. CLAUDE.md: the judge model
never appears in training, in any form, including as a proxy or a data filter.
"""

from .lcf_wer import (NORMALISER_ID, Edits, LcfWer, edit_counts,
                      headroom_captured, lcf_wer, normalise)

__all__ = ["NORMALISER_ID", "Edits", "LcfWer", "edit_counts",
           "headroom_captured", "lcf_wer", "normalise"]
