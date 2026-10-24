"""FR, the fabrication rate of docs/data/metric-definitions.md 3.3b: did the listener report words that nobody said?

Word error rate measures content recovered and ICR measures content leaked from the other speaker. FR measures content invented: in the response but in neither speaker's script (decisions-m4.md 2026-09-03).
Compare systems on invented words per trial, not on the percentage, because the percentage divides by the response's own length and so punishes a shorter response.
FR's ceiling is not zero, because a misheard word also counts, so always read it against the clean-target row.
Scoring works on text, like lcf_wer and icr.
"""

from .icr import content_words

# A trial counts as fabricating when at least this many invented content words appear. This is the k of the reported FR@2.
FABRICATION_THRESHOLD = 2

# The same guard as ICR's: the mean invented share only uses responses with at least this many content words.
MINIMUM_RESPONSE_WORDS_FOR_FRACTION = 5


# this gives the response's content words that neither speaker said. An empty script on a target-only or target-absent trial needs no special case
def invented_content(response_text, target_text, interferer_text):
    return content_words(response_text) - content_words(target_text) - content_words(interferer_text)


# this gives FR@2, the invented words per trial, the mean invented share, and how many trials had no response at all
def compute_fr(response_texts, target_texts, interferer_texts):
    """An empty response is left out rather than scored as 0 % invented, because a muting system would otherwise lower its fabrication by saying nothing. Empty responses are counted separately."""
    response_texts = list(response_texts)
    target_texts = list(target_texts)
    interferer_texts = list(interferer_texts)
    if not len(response_texts) == len(target_texts) == len(interferer_texts):
        raise ValueError(f"{len(response_texts)} responses, {len(target_texts)} targets, {len(interferer_texts)} interferers -- must be parallel")

    scored_trials = 0
    fabricating_trials = 0
    invented_total = 0
    invented_fractions = []
    for response_text, target_text, interferer_text in zip(response_texts, target_texts, interferer_texts):
        response_count = len(content_words(response_text))
        if response_count == 0:
            continue
        invented_count = len(invented_content(response_text, target_text, interferer_text))
        scored_trials += 1
        invented_total += invented_count
        if invented_count >= FABRICATION_THRESHOLD:
            fabricating_trials += 1
        if response_count >= MINIMUM_RESPONSE_WORDS_FOR_FRACTION:
            invented_fractions.append(invented_count / response_count)

    trials_total = len(response_texts)
    result = {"fr_at_2": None, "invented_per_trial": None, "mean_invented_percent": None, "trials_empty": trials_total - scored_trials, "trials_total": trials_total}
    if scored_trials:
        result["fr_at_2"] = fabricating_trials * 100.0 / scored_trials
        result["invented_per_trial"] = invented_total / scored_trials
    # sum() adds floats with compensated summation, which is more accurate than a running total.
    if invented_fractions:
        result["mean_invented_percent"] = sum(invented_fractions) * 100.0 / len(invented_fractions)
    return result
