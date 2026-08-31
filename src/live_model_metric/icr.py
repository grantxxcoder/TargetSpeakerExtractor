"""ICR, the interferer content rate of docs/data/metric-definitions.md 3.2.

Answers one question: did the other speaker's words come through?

Scoring works on TEXT, like lcf_wer. Audio only enters through `transcribe_one`,
a callable of (audio_file_path) -> text, so the same swap takes it from the
offline ASR to the live judge.

Target-absent trials are scored by calling `compute_icr` on them as a SEPARATE
group. `t` is empty there, so every interferer content word counts as exclusive
and any leak is unambiguous. See decisions-m4.md 2026-08-31.
"""

from dataclasses import dataclass, field
from pathlib import Path

from .lcf_wer import normalise_text

STOPWORDS_FILE = Path(__file__).parent / "stopwords_nltk_english.txt"

K_VALUES = (1, 2, 3, 5)
HEADLINE_K = 2
MINIMUM_AVAILABLE_FOR_FRACTION = 5

_stopwords = None


def load_stopwords():
    global _stopwords
    if _stopwords is None:
        lines = STOPWORDS_FILE.read_text().splitlines()
        entries = [line for line in lines if line and not line.startswith("#")]
        # Normalise, then SPLIT. 45 of the 198 entries are apostrophe forms and
        # the normaliser expands them into two words ("don't" -> "do not").
        # Stored whole, such an entry is a set member that no single token can
        # ever equal, so it would silently filter nothing.
        _stopwords = set()
        for entry in entries:
            _stopwords.update(normalise_text(entry).split())
    return _stopwords


def content_words(text):
    stopwords = load_stopwords()
    return {word for word in normalise_text(text).split() if word not in stopwords}


def interferer_exclusive_content(target_text, interferer_text):
    return content_words(interferer_text) - content_words(target_text)


@dataclass
class TrialLeakage:
    leaked_count: int = 0
    available_count: int = 0
    leaked_words: set = field(default_factory=set)

    @property
    def leaked_fraction(self):
        if self.available_count == 0:
            return None
        return self.leaked_count / self.available_count


def measure_leakage(response_text, target_text, interferer_text):
    available_words = interferer_exclusive_content(target_text, interferer_text)
    leaked_words = content_words(response_text) & available_words
    return TrialLeakage(len(leaked_words), len(available_words), leaked_words)


@dataclass
class IcrResult:
    icr_at_k: dict = field(default_factory=dict)
    eligible_at_k: dict = field(default_factory=dict)
    mean_leaked_percent: float = None
    trials_for_fraction: int = 0
    trials_ineligible: int = 0
    trials_total: int = 0

    @property
    def headline(self):
        return self.icr_at_k.get(HEADLINE_K)

    def __str__(self):
        if not self.icr_at_k:
            return "ICR: undefined, no eligible trials"
        at_k = "  ".join(
            f"@{k} {self.icr_at_k[k]:.1f}% (n={self.eligible_at_k[k]})"
            for k in K_VALUES if k in self.icr_at_k
        )
        mean = ("—" if self.mean_leaked_percent is None
                else f"{self.mean_leaked_percent:.1f}%")
        return (f"ICR {at_k}\n"
                f"  mean leaked {mean} over {self.trials_for_fraction} trials "
                f"with >={MINIMUM_AVAILABLE_FOR_FRACTION} words available\n"
                f"  {self.trials_ineligible} of {self.trials_total} trials "
                f"ineligible (no exclusive interferer content)")


def compute_icr(response_texts, target_texts, interferer_texts):
    response_texts = list(response_texts)
    target_texts = list(target_texts)
    interferer_texts = list(interferer_texts)
    if not len(response_texts) == len(target_texts) == len(interferer_texts):
        raise ValueError(
            f"{len(response_texts)} responses, {len(target_texts)} targets, "
            f"{len(interferer_texts)} interferers -- must be parallel"
        )

    leakages = [
        measure_leakage(response_text, target_text, interferer_text)
        for response_text, target_text, interferer_text
        in zip(response_texts, target_texts, interferer_texts)
    ]

    result = IcrResult(trials_total=len(leakages))
    result.trials_ineligible = sum(1 for l in leakages if l.available_count == 0)

    # Eligibility is per k: a trial with 1 exclusive word available cannot
    # score @2, so counting it in that denominator would drag the rate down
    # for a reason about the trial rather than the system.
    for k in K_VALUES:
        eligible = [l for l in leakages if l.available_count >= k]
        result.eligible_at_k[k] = len(eligible)
        if eligible:
            fired = sum(1 for l in eligible if l.leaked_count >= k)
            result.icr_at_k[k] = fired * 100.0 / len(eligible)

    # A fraction over a tiny denominator is not a fraction: 1 of 1 is 100%.
    for_fraction = [l for l in leakages
                    if l.available_count >= MINIMUM_AVAILABLE_FOR_FRACTION]
    result.trials_for_fraction = len(for_fraction)
    if for_fraction:
        result.mean_leaked_percent = (
            sum(l.leaked_fraction for l in for_fraction) * 100.0 / len(for_fraction)
        )
    return result


def score_audio_files(audio_file_paths, target_texts, interferer_texts, transcribe_one):
    response_texts = [transcribe_one(path) for path in audio_file_paths]
    return compute_icr(response_texts, target_texts, interferer_texts)
