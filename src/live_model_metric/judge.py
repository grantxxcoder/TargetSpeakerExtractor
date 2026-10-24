"""The live judge: Gemini hears a clip under a fixed prompt and reports what was said, in place of the offline ASR.

The judge does not separate anything. It only listens, so its transcript is scored with the same word error rate code as the ASR's (metric-definitions.md 3.1).
The prompt is part of the instrument, so its hash is in every cache key: editing the prompt makes every answer new rather than reusing stale ones.
"Nothing heard" comes back in a separate status field, because the text normaliser turns any in-band marker into an empty string.
Every answer is appended to the cache as soon as it arrives, because a run spans days and gets interrupted.
make_judge returns a plain dictionary holding the settings and the running counters (calls_made, cache_hits, filter_blocks and so on).
"""

import base64
import csv
import hashlib
import json
import os
import random
import time
from datetime import date
from pathlib import Path

from google import genai
from google.genai import types

REPO_ROOT = Path(__file__).resolve().parents[2]

# The instrument: one model, one prompt, served through AI Studio. Every reported judge result used exactly these.
MODEL_ID = "gemini-3.7-flash"
BACKEND = "aistudio"
PROMPT_FILE = REPO_ROOT / "src/live_model_metric/judge_prompt.txt"
DEFAULT_CACHE = REPO_ROOT / "experiments/results/judge_responses.csv"

CACHE_FIELDS = ["key", "model", "backend", "prompt_sha", "trial_id", "file", "repeat", "status", "text", "run_date"]

# The judge answers in JSON, with the status kept outside the transcript so it never reaches the normaliser.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["speech", "no_speech"]},
        "transcript": {"type": "string"},
    },
    "required": ["status", "transcript"],
}

# A 17 s clip at 16 kHz is about 557 kB, far inside the 20 MB inline limit, so audio is sent inline and the Files API is not needed.
AUDIO_MIME = "audio/wav"

# Retries for throttling, and separate retries for the safety filter, which is not deterministic (a refused clip passed on a re-run, 2026-09-02).
MAX_RETRIES = 5
FILTER_RETRIES = 2
BACKOFF_INITIAL_S = 2.0
TIMEOUT_S = 60.0

# The rendered clips no model can change, so their answer is only ever bought once. estimate.wav is model output and can be judged again.
RUN_ONCE_CLIPS = frozenset({"target", "mixture", "interferer"})


# These two stop the whole run instead of failing one clip. Python can only tell errors apart in an except by their class, so they stay classes.
class QuotaExhausted(RuntimeError):
    """The daily quota is spent. Everything answered is cached, so a re-run tomorrow resumes."""


class MissingCredentials(RuntimeError):
    """There is no API key, so no call can succeed. Scored as failures, every clip would read as an empty answer, which on a target-absent trial looks like perfect silence."""


# this reads the judge's fixed prompt
def prompt_text():
    return PROMPT_FILE.read_text()


# this gives a short hash of the prompt, which goes into every cache key
def prompt_sha():
    return hashlib.sha256(prompt_text().encode()).hexdigest()[:12]


# this hashes a file's bytes. Estimates are keyed on their content, so a retrained checkpoint writing to the same path is never served the old answer
def audio_fingerprint(audio_path):
    """Not memoised on the file's timestamp: two checkpoints written to one path within the same second would otherwise share a digest."""
    digest = hashlib.sha256()
    with open(Path(audio_path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


# this says whether a clip is one of the rendered clips no model can change
def runs_once(audio_path):
    return Path(audio_path).stem.lower() in RUN_ONCE_CLIPS


# this gives the cache key for a clip: once per trial for the rendered clips, and per content and repeat for an estimate
def cache_key(audio_path, sha, repeat):
    path = Path(audio_path)
    if runs_once(path):
        return f"{MODEL_ID}@{BACKEND}|{sha}|{path.parent.name}|{path.stem.lower()}|once"
    return f"{MODEL_ID}@{BACKEND}|{sha}|{path.parent.name}|{path.name}|sha{audio_fingerprint(path)}|r{repeat}"


# this reads the answer cache as key to (status, text). A missing file is an empty cache
def load_cache(cache_path):
    if not Path(cache_path).exists():
        return {}
    cache = {}
    with open(cache_path, newline="") as handle:
        for row in csv.DictReader(handle):
            cache[row["key"]] = (row["status"], row["text"])
    return cache


# this indexes the rendered clips' answers by their columns rather than their key, so an answer is found whatever key format wrote it
def load_once_index(cache_path):
    if not Path(cache_path).exists():
        return {}
    index = {}
    with open(cache_path, newline="") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["file"]).stem.lower()
            if stem in RUN_ONCE_CLIPS:
                index[(row["model"], row["prompt_sha"], row.get("backend", BACKEND), row["trial_id"], stem)] = (row["status"], row["text"])
    return index


# this appends one answer to the cache file
def append_cache(cache_path, row):
    cache_path = Path(cache_path)
    exists = cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# this says whether an error is Gemini's safety filter refusing the input. That is a measurement event, not a failure (metric-definitions.md 3.3)
def is_content_blocked(error):
    text = str(error).lower()
    return "content_blocked" in text or "input blocked" in text or "blocked by gemini's filters" in text or "blocked by the safety filter" in text


# this reads an error message and says whether it is a throttle worth retrying, and whether it is the daily cap
def is_quota_error(error):
    """The SDK has no typed quota error, so this reads the message. A quota with a limit of 0, depleted prepaid credit, or disabled billing is a permanent refusal that no wait fixes, so it is never retried."""
    text = str(error).lower()
    permanent = "limit: 0" in text or "limit 0" in text or 'quota_limit_value: "0"' in text or "limit_value: 0" in text or "credits are depleted" in text or "prepayment credits" in text or "billing" in text and "disabled" in text
    if permanent:
        return False, False
    daily = "per day" in text or "perday" in text or "daily" in text or "requests per day" in text
    throttled = "429" in text or "resource_exhausted" in text or "quota" in text or "rate limit" in text
    return throttled, daily


# this makes a judge. repeat separates repeated runs of the same estimate, and cache_path is where its answers are kept
def make_judge(repeat, cache_path, requests_per_minute, verbose):
    min_interval_s = 0.0
    if requests_per_minute:
        min_interval_s = 60.0 / requests_per_minute
    return {
        "repeat": repeat,
        "cache_path": Path(cache_path),
        "min_interval_s": min_interval_s,
        "verbose": verbose,
        "prompt": prompt_text(),
        "sha": prompt_sha(),
        "cache": load_cache(cache_path),
        "once": load_once_index(cache_path),
        "client": None,
        "last_call_at": 0.0,
        "calls_made": 0,           # this process only. The cache is the real ledger.
        "cache_hits": 0,
        "filter_blocks": 0,        # refused on every retry, recorded as a non-response
        "filter_transients": 0,    # refused once, then passed on a retry
        "failures": [],            # (clip, error) for clips that failed and were scored empty, filled in by evaluate.listen
    }


# this is what the scoring calls: a clip in, the judge's transcript out. A no-speech answer is an empty transcript, scored as all deletions
def judge_transcribe(judge, audio_path):
    status, text = judge_clip(judge, audio_path)
    return text


# this gives a rendered clip's place in the once-only index
def once_slot(judge, audio_path):
    path = Path(audio_path)
    return (MODEL_ID, judge["sha"], BACKEND, path.parent.name, path.stem.lower())


# this gives the cached (status, text) for a clip, or None if it has never been paid for
def judge_cached(judge, audio_path):
    if runs_once(audio_path):
        return judge["once"].get(once_slot(judge, audio_path))
    return judge["cache"].get(cache_key(audio_path, judge["sha"], judge["repeat"]))


# this gives (status, text) for a clip: from the cache when it is there, otherwise from one paid call, which is cached straight away
def judge_clip(judge, audio_path):
    path = Path(audio_path)
    hit = judge_cached(judge, path)
    if hit is not None:
        judge["cache_hits"] += 1
        return hit

    status, text = call_with_retries(judge, path)
    key = cache_key(path, judge["sha"], judge["repeat"])
    append_cache(judge["cache_path"], {"key": key, "model": MODEL_ID, "backend": BACKEND, "prompt_sha": judge["sha"], "trial_id": path.parent.name, "file": path.name, "repeat": judge["repeat"], "status": status, "text": text, "run_date": date.today().isoformat()})
    judge["cache"][key] = (status, text)
    if runs_once(path):
        judge["once"][once_slot(judge, path)] = (status, text)
    judge["calls_made"] += 1
    return status, text


# this builds the Gemini client on first use, with an API key from aistudio.google.com
def ensure_client(judge):
    """The SDK's own retries are switched off, so only call_with_retries retries, and a permanent refusal surfaces at once instead of looking like a hang. The timeout is in milliseconds."""
    if judge["client"] is not None:
        return judge["client"]
    if not os.environ.get("GEMINI_API_KEY"):
        raise MissingCredentials("GEMINI_API_KEY is not set. Export it before running the judge.")
    http_options = types.HttpOptions(timeout=int(TIMEOUT_S * 1000), retry_options=types.HttpRetryOptions(attempts=1))
    judge["client"] = genai.Client(http_options=http_options)
    return judge["client"]


# this waits long enough to keep under the requests-per-minute limit
def throttle(judge):
    wait = judge["min_interval_s"] - (time.time() - judge["last_call_at"])
    if wait > 0:
        time.sleep(wait)


# this calls the judge, retrying throttles with backoff and safety-filter refusals separately, and stops the run on the daily cap
def call_with_retries(judge, path):
    """A refusal on every filter retry is recorded as blocked_by_filter with an empty transcript. Filter retries do not use up the throttle retries."""
    delay = BACKOFF_INITIAL_S
    filter_attempts = 0
    for attempt in range(1, MAX_RETRIES + FILTER_RETRIES + 1):
        throttle(judge)
        try:
            judge["last_call_at"] = time.time()
            result = call_once(judge, path)
            if filter_attempts:
                judge["filter_transients"] += 1
                if judge["verbose"]:
                    print("    ...passed on retry -- the earlier refusal was transient, NOT recorded", flush=True)
            return result
        except QuotaExhausted:
            raise
        except Exception as error:
            if is_content_blocked(error):
                filter_attempts += 1
                if filter_attempts <= FILTER_RETRIES:
                    if judge["verbose"]:
                        print(f"    safety filter refused {Path(path).parent.name}/{Path(path).name} -- retrying ({filter_attempts}/{FILTER_RETRIES}); the filter is not deterministic", flush=True)
                    time.sleep(BACKOFF_INITIAL_S)
                    continue
                judge["filter_blocks"] += 1
                if judge["verbose"]:
                    print(f"    safety filter refused {Path(path).parent.name}/{Path(path).name} {filter_attempts}x -- recorded as a non-response", flush=True)
                return "blocked_by_filter", ""
            throttled, daily = is_quota_error(error)
            if daily:
                raise QuotaExhausted(f"daily quota spent after {judge['calls_made']} calls this process. Cached work is safe -- rerun tomorrow and it resumes. Original: {error}") from error
            if not throttled or (attempt - filter_attempts) >= MAX_RETRIES:
                raise
            sleep_s = delay + random.uniform(0, delay * 0.25)
            if judge["verbose"]:
                reason = " ".join(str(error).split())[:300]
                print(f"    throttled (attempt {attempt}/{MAX_RETRIES}), sleeping {sleep_s:.1f}s\n      reason: {reason}", flush=True)
            time.sleep(sleep_s)
            delay *= 2
    raise RuntimeError("unreachable")


# this makes one call: the prompt and the clip as base64 audio in, the parsed (status, text) out
def call_once(judge, path):
    client = ensure_client(judge)
    audio_b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    interaction = client.interactions.create(model=MODEL_ID, input=[{"type": "text", "text": judge["prompt"]}, {"type": "audio", "data": audio_b64, "mime_type": AUDIO_MIME}], response_format={"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA})
    return parse_response(interaction.output_text)


# this parses the judge's JSON answer. An answer that does not parse is recorded with its own status rather than crashing the run
def parse_response(output_text):
    raw = (output_text or "").strip()
    if not raw:
        return "empty_response", ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "unparsed", raw
    status = payload.get("status") or "unknown"
    transcript = (payload.get("transcript") or "").strip()
    if status == "no_speech":
        return "no_speech", ""
    return status, transcript
