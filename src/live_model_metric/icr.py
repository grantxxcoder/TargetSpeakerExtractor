"""ICR, the interferer content rate of docs/data/metric-definitions.md 3.2: did the other speaker's words come through?

A trial counts as leaked when at least two content words that only the interferer said appear in the listener's response. ICR@2 is the share of eligible trials that leaked.
Scoring works on text, like lcf_wer.
"""

from pathlib import Path

from .lcf_wer import normalise_text

STOPWORDS_FILE = Path(__file__).parent / "stopwords_nltk_english.txt"

# A trial leaks when at least this many of the interferer's own content words come through. This is the k of the reported ICR@2.
LEAK_THRESHOLD = 2

# A fraction over a tiny denominator is not a fraction (1 of 1 is 100 %), so the mean leaked share only uses trials with at least this many words available.
MINIMUM_AVAILABLE_FOR_FRACTION = 5


# this reads the stopword list. It normalises each entry and then splits it, because the normaliser expands forms like "don't" into two words
def load_stopwords():
    stopwords = set()
    for line in STOPWORDS_FILE.read_text().splitlines():
        if line and not line.startswith("#"):
            stopwords.update(normalise_text(line).split())
    return stopwords


STOPWORDS = load_stopwords()


# this gives the set of content words in a text: its normalised words minus the stopwords
def content_words(text):
    words = set()
    for word in normalise_text(text).split():
        if word not in STOPWORDS:
            words.add(word)
    return words


# this gives the content words only the interferer said, the only words that can prove a leak
def interferer_exclusive_content(target_text, interferer_text):
    return content_words(interferer_text) - content_words(target_text)


# this measures one trial: how many of the interferer's exclusive words came through, out of how many were available
def measure_leakage(response_text, target_text, interferer_text):
    """leaked_fraction is None when the interferer said nothing the target did not also say."""
    available_words = interferer_exclusive_content(target_text, interferer_text)
    leaked_words = content_words(response_text) & available_words
    leaked_fraction = None
    if available_words:
        leaked_fraction = len(leaked_words) / len(available_words)
    return {"leaked_count": len(leaked_words), "available_count": len(available_words), "leaked_fraction": leaked_fraction}


# this gives ICR@2 and the mean share of interferer words that leaked, over parallel lists of responses, target scripts and interferer scripts
def compute_icr(response_texts, target_texts, interferer_texts):
    """Only trials with at least two exclusive interferer words can reach the threshold, so only they count towards ICR@2. Both values are None when no trial qualifies."""
    response_texts = list(response_texts)
    target_texts = list(target_texts)
    interferer_texts = list(interferer_texts)
    if not len(response_texts) == len(target_texts) == len(interferer_texts):
        raise ValueError(f"{len(response_texts)} responses, {len(target_texts)} targets, {len(interferer_texts)} interferers -- must be parallel")

    eligible_trials = 0
    leaked_trials = 0
    leaked_fractions = []
    for response_text, target_text, interferer_text in zip(response_texts, target_texts, interferer_texts):
        leakage = measure_leakage(response_text, target_text, interferer_text)
        if leakage["available_count"] >= LEAK_THRESHOLD:
            eligible_trials += 1
            if leakage["leaked_count"] >= LEAK_THRESHOLD:
                leaked_trials += 1
        if leakage["available_count"] >= MINIMUM_AVAILABLE_FOR_FRACTION:
            leaked_fractions.append(leakage["leaked_fraction"])

    result = {"icr_at_2": None, "mean_leaked_percent": None}
    if eligible_trials:
        result["icr_at_2"] = leaked_trials * 100.0 / eligible_trials
    # sum() adds floats with compensated summation, which is more accurate than a running total.
    if leaked_fractions:
        result["mean_leaked_percent"] = sum(leaked_fractions) * 100.0 / len(leaked_fractions)
    return result
