"""LCF-WER, the primary score of docs/data/metric-definitions.md 3.1.

Scoring works on TEXT. Audio only enters through `transcribe_one`, a callable
of (audio_file_path) -> text. Swap that callable to change the listener from
the offline ASR to the live judge; nothing else changes.
"""

from dataclasses import dataclass

NORMALISER_NAME = "whisper-normalizer==0.1.15:EnglishTextNormalizer"

_normaliser = None


def normalise_text(text):
    global _normaliser
    if text is None:
        return ""
    if isinstance(text, float) and text != text:
        return ""
    if _normaliser is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _normaliser = EnglishTextNormalizer()
    return _normaliser(str(text))


@dataclass
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_word_count: int = 0

    @property
    def total_errors(self):
        return self.substitutions + self.deletions + self.insertions


def count_errors(reference_text, hypothesis_text):
    import jiwer

    reference = normalise_text(reference_text)
    hypothesis = normalise_text(hypothesis_text)
    reference_word_count = len(reference.split())

    # No reference words means the rate is 0/0. The caller decides what to do.
    if reference_word_count == 0:
        return ErrorCounts()

    measures = jiwer.process_words([reference], [hypothesis])
    return ErrorCounts(
        substitutions=measures.substitutions,
        deletions=measures.deletions,
        insertions=measures.insertions,
        reference_word_count=reference_word_count,
    )


@dataclass
class LcfWerResult:
    word_error_rate: float = None
    substitution_rate: float = None
    deletion_rate: float = None
    insertion_rate: float = None
    trials_scored: int = 0
    trials_without_reference: int = 0
    reference_word_count: int = 0
    total_errors: int = 0

    def __str__(self):
        if self.word_error_rate is None:
            return "LCF-WER: undefined, no scorable trials"
        return (
            f"LCF-WER {self.word_error_rate:.2f} % "
            f"(substitutions {self.substitution_rate:.2f}, "
            f"deletions {self.deletion_rate:.2f}, "
            f"insertions {self.insertion_rate:.2f}) "
            f"over {self.trials_scored} trials, "
            f"{self.reference_word_count} reference words"
        )


def compute_lcf_wer(reference_texts, hypothesis_texts):
    reference_texts = list(reference_texts)
    hypothesis_texts = list(hypothesis_texts)
    if len(reference_texts) != len(hypothesis_texts):
        raise ValueError(
            f"{len(reference_texts)} references but {len(hypothesis_texts)} hypotheses"
        )

    result = LcfWerResult()
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0

    for reference_text, hypothesis_text in zip(reference_texts, hypothesis_texts):
        counts = count_errors(reference_text, hypothesis_text)
        if counts.reference_word_count == 0:
            result.trials_without_reference += 1
            continue
        result.trials_scored += 1
        result.reference_word_count += counts.reference_word_count
        result.total_errors += counts.total_errors
        total_substitutions += counts.substitutions
        total_deletions += counts.deletions
        total_insertions += counts.insertions

    if result.reference_word_count == 0:
        return result

    # Multiply before dividing so a fully-wrong set lands on exactly 100.0.
    word_count = result.reference_word_count
    result.word_error_rate = result.total_errors * 100.0 / word_count
    result.substitution_rate = total_substitutions * 100.0 / word_count
    result.deletion_rate = total_deletions * 100.0 / word_count
    result.insertion_rate = total_insertions * 100.0 / word_count
    return result


def transcribe_audio_files(audio_file_paths, transcribe_one):
    return [transcribe_one(audio_file_path) for audio_file_path in audio_file_paths]


def score_audio_files(audio_file_paths, reference_texts, transcribe_one):
    hypothesis_texts = transcribe_audio_files(audio_file_paths, transcribe_one)
    return compute_lcf_wer(reference_texts, hypothesis_texts)
