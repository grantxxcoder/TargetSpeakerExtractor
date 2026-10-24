"""LCF-WER, the primary score of docs/data/metric-definitions.md 3.1: the word error rate of what a listener reports hearing.

Scoring works on text. The listener, the offline ASR or the live judge, turns each clip into text first, and the same scoring applies to both.
"""

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

NORMALISER = EnglishTextNormalizer()


# this normalises text the way Whisper's English normaliser does, so spelling and punctuation differences are not counted as errors
def normalise_text(text):
    if text is None:
        return ""
    return NORMALISER(str(text))


# this counts the substitutions, deletions and insertions between one reference and one hypothesis
def count_errors(reference_text, hypothesis_text):
    reference = normalise_text(reference_text)
    hypothesis = normalise_text(hypothesis_text)
    reference_word_count = len(reference.split())

    # No reference words means the rate is 0/0, so the caller decides what to do with the trial.
    if reference_word_count == 0:
        return {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_word_count": 0, "total_errors": 0}

    measures = jiwer.process_words([reference], [hypothesis])
    return {"substitutions": measures.substitutions, "deletions": measures.deletions, "insertions": measures.insertions, "reference_word_count": reference_word_count, "total_errors": measures.substitutions + measures.deletions + measures.insertions}


# this gives the corpus word error rate over many trials, with its substitution, deletion and insertion parts, all in percent
def compute_lcf_wer(reference_texts, hypothesis_texts):
    """A corpus rate: the total errors over the total reference words, not a mean of each trial's rate. Every value is None when no trial has reference words."""
    reference_texts = list(reference_texts)
    hypothesis_texts = list(hypothesis_texts)
    if len(reference_texts) != len(hypothesis_texts):
        raise ValueError(f"{len(reference_texts)} references but {len(hypothesis_texts)} hypotheses")

    reference_word_count = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    for reference_text, hypothesis_text in zip(reference_texts, hypothesis_texts):
        counts = count_errors(reference_text, hypothesis_text)
        reference_word_count += counts["reference_word_count"]
        total_substitutions += counts["substitutions"]
        total_deletions += counts["deletions"]
        total_insertions += counts["insertions"]

    result = {"word_error_rate": None,
              "substitution_rate": None,
              "deletion_rate": None,
              "insertion_rate": None}
    if reference_word_count == 0:
        return result

    # Multiply before dividing, so a fully wrong set lands on exactly 100.0.
    total_errors = total_substitutions + total_deletions + total_insertions
    result["word_error_rate"] = total_errors * 100.0 / reference_word_count
    result["substitution_rate"] = total_substitutions * 100.0 / reference_word_count
    result["deletion_rate"] = total_deletions * 100.0 / reference_word_count
    result["insertion_rate"] = total_insertions * 100.0 / reference_word_count
    return result
