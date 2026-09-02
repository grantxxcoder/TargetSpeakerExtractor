"""The live judge: audio in, text out. Swaps in for the offline ASR.

WHAT THIS IS. `metric-definitions.md` 3.1 scores LCF-WER by presenting audio to
a live model under a fixed prompt and reading back what it heard. The judge does
NOT separate anything -- the extractor does that. The judge only listens.

WHY IT NEEDS NO NEW METRIC CODE. lcf_wer.py, icr.py and nrr.py all take a
`transcribe_one(path) -> text` callable. A judge is exactly that signature, so
`make_judge()` returns a drop-in replacement for `evaluate.transcribe`'s ASR.

    from src.live_model_metric.judge import make_judge
    judge = make_judge(repeat=0)
    text = judge("data/rendered/sir0_val/t0001/mixture.wav")

DESIGN CONSTRAINTS, all recorded in experiments/configs/judge_gate.yaml:

  * The prompt is part of the instrument. Its sha256 is in every cache key, so
    editing the prompt does not silently reuse stale answers -- it re-runs.
  * The "nothing heard" signal is a STRUCTURED field, never in-band text.
    Measured 2026-09-02: whisper-normalizer maps '####' and '[no speech]' both
    to '', so an in-band sentinel is destroyed by the normaliser and becomes
    indistinguishable from a non-response. That would blunt NRR, whose whole
    job is telling a declining judge from a silent one.
  * Raw text is appended to the cache after EVERY call, not per batch. A
    free-tier run spans days and will be interrupted; and J3 (the ICR
    threshold) is still open, so its sensitivity sweep must be a re-score of
    stored text rather than a re-run.
  * The binding free-tier limit is per-DAY, not per-minute. QuotaExhausted is
    raised as its own type so a caller can stop cleanly and resume tomorrow
    rather than treating it as a failure.

decisions-pending.md J2 (which judge), J1 (audio-in / text-out, closed
2026-08-31), J3 (the prompt must not instruct the judge to pick a speaker).
"""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import random
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_ID = "gemini-3.7-flash"
DEFAULT_PROMPT_FILE = REPO_ROOT / "src/live_model_metric/judge_prompt.txt"
DEFAULT_CACHE = REPO_ROOT / "experiments/results/judge_responses.csv"

CACHE_FIELDS = ["key", "model", "backend", "prompt_sha", "trial_id", "file",
                "repeat", "status", "text", "run_date"]

# Structured output. The status field lives OUTSIDE the transcript so it never
# reaches the normaliser -- see the module docstring.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["speech", "no_speech"]},
        "transcript": {"type": "string"},
    },
    "required": ["status", "transcript"],
}

# 16 kHz mono, ~17.4 s => ~557 kB, far inside the 20 MB inline request cap, so
# audio goes inline as base64 and the Files API is not needed. One less moving
# part in the measuring instrument, and nothing to clean up server-side.
AUDIO_MIME = "audio/wav"


class QuotaExhausted(RuntimeError):
    """The daily free-tier cap is spent. Not a failure -- resume tomorrow."""


class NewCallLimitReached(RuntimeError):
    """max_new_calls was hit. A guard against an accidental large spend, not an
    error: everything already answered is on disk and a re-run resumes."""


def prompt_text(prompt_file=None):
    return Path(prompt_file or DEFAULT_PROMPT_FILE).read_text()


def prompt_sha(prompt_file=None):
    """Short hash, recorded per call. A changed prompt invalidates the cache."""
    return hashlib.sha256(prompt_text(prompt_file).encode()).hexdigest()[:12]


def audio_fingerprint(audio_path):
    """sha256 of the file's BYTES, truncated. Used only for estimate.wav.

    NOT MEMOISED, deliberately. An earlier version cached the digest against
    (path, mtime, size) and was wrong: mtime has one-second resolution, so two
    different checkpoints written to the same path within the same second at
    the same size returned the FIRST file's digest -- which would serve model
    A's judge answer for model B's audio and silently corrupt the results
    table. Hashing ~557 kB costs about a millisecond against a network call,
    so there is nothing to optimise here.
    """
    digest = hashlib.sha256()
    with open(Path(audio_path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


# THE UNCHANGEABLES. These clips are not produced by any model -- they are the
# rendered trial itself -- so their judge answer can only ever be bought once.
# Anything else (estimate.wav) is model output and may legitimately be judged
# again, because a retrained checkpoint writes different audio to the same path.
RUN_ONCE_CLIPS = frozenset({"target", "mixture", "interferer"})


def runs_once(audio_path):
    """True for target/mixture/interferer, i.e. audio no model can change."""
    return Path(audio_path).stem.lower() in RUN_ONCE_CLIPS


def once_key(audio_path, model_id, sha, backend="aistudio"):
    """Identity for a run-once clip. Deliberately has NO repeat and NO content
    hash: one trial's mixture is one clip, so one answer is all it can ever
    need. Model, prompt and backend stay in the key because they are the
    measuring instrument -- change one and it is a different measurement."""
    path = Path(audio_path)
    return (f"{model_id}@{backend}|{sha}|{path.parent.name}|{path.stem.lower()}"
            f"|once")


def cache_key(audio_path, model_id, sha, repeat, backend="aistudio",
              force_repeat=False):
    """Identity for a re-runnable clip (estimate.wav).

    Keeps the content hash, and that part is load-bearing rather than
    decorative: a retrained checkpoint overwrites estimate.wav at the SAME
    path, so without the hash model B would be served model A's answer.

    force_repeat=True opts a run-once clip INTO repeat keying. Used only by the
    deliberate spread study (scripts/judge_spread.py), which has to judge one
    unchanging clip several times to measure how much the judge's answer varies
    between calls. That number is M4's gate criterion -- run-to-run spread must
    be smaller than the floor-to-ceiling gap -- and it bounds the smallest
    system difference the metric can honestly claim.
    """
    if runs_once(audio_path) and not force_repeat:
        return once_key(audio_path, model_id, sha, backend)
    path = Path(audio_path)
    return (f"{model_id}@{backend}|{sha}|{path.parent.name}|{path.name}"
            f"|sha{audio_fingerprint(path)}|r{repeat}")


def load_cache(cache_path=None):
    """key -> (status, text). Missing file is an empty cache, not an error."""
    cache_path = Path(cache_path or DEFAULT_CACHE)
    if not cache_path.exists():
        return {}
    with open(cache_path, newline="") as handle:
        return {row["key"]: (row["status"], row["text"])
                for row in csv.DictReader(handle)}


def load_once_index(cache_path=None):
    """(model, prompt_sha, backend, trial_id, clip stem) -> (status, text).

    Built from the CSV COLUMNS rather than the key string, so a run-once clip
    already answered is found no matter what key format wrote it. That is what
    lets the key format change without ever re-buying an answer.
    """
    cache_path = Path(cache_path or DEFAULT_CACHE)
    if not cache_path.exists():
        return {}
    index = {}
    with open(cache_path, newline="") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["file"]).stem.lower()
            if stem not in RUN_ONCE_CLIPS:
                continue
            index[(row["model"], row["prompt_sha"], row.get("backend", "aistudio"),
                   row["trial_id"], stem)] = (row["status"], row["text"])
    return index


def _append_cache(cache_path, row):
    cache_path = Path(cache_path)
    exists = cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _is_content_blocked(exc):
    """Did a safety filter refuse this input?

    Observed 2026-09-02 on an ESTIMATE clip, not on a mixture or a clean target:
    400 `content_blocked`, "Input blocked: This request was blocked by Gemini's
    filters." The prompt is identical across every call and had already
    succeeded 40+ times, so what the filter reacted to is the extractor's own
    output audio.

    THIS IS A MEASUREMENT EVENT, NOT AN ERROR. metric-definitions.md 3.3 names
    it: "a live model can refuse, hit a safety filter, or judge the audio
    unusable", and separating that from "misheard" is NRR's second stated
    purpose.

    THE FILTER IS NON-DETERMINISTIC, measured 2026-09-02. The first version of
    this code retried it zero times and cached the refusal, on the assumption
    that a filter verdict is a fixed property of the input. It is not: on the
    re-run every one of the same 20 estimates passed, including the clip that
    had been refused. Caching a transient refusal would permanently score a
    trial as a non-response because of a coin flip, silently penalising the
    system that produced it. So a block is now RETRIED (see
    `filter_retries`) and only recorded if it persists.
    """
    text = str(exc).lower()
    return ("content_blocked" in text
            or "input blocked" in text
            or "blocked by gemini's filters" in text
            or "blocked by the safety filter" in text)


def _is_quota_error(exc):
    """google-genai does not expose a typed daily-quota error, so this reads the
    message. Deliberately conservative: anything mentioning a per-day quota is
    treated as the daily cap, everything else 429-ish is a per-minute throttle
    worth retrying.

    THE ZERO-LIMIT CASE, added 2026-09-02 after it bit on the first smoke call.
    When a model is not available on your tier, Google answers 429
    RESOURCE_EXHAUSTED with a quota whose LIMIT IS 0 -- not a throttle, a
    permanent refusal. Retrying it can never succeed, and the old heuristic saw
    the word "quota" and burned all five attempts on it. A zero limit is
    therefore reported as fatal (throttled=False) so it raises at once with the
    real message attached.
    """
    text = str(exc).lower()
    # THREE 429 SHAPES THAT ARE NOT THROTTLES. Each is a permanent refusal that
    # no amount of waiting fixes, so retrying only sleeps through the backoff:
    #   * a quota whose LIMIT IS 0 -- the model is not on this tier
    #   * depleted PREPAY credits  -- observed 2026-09-02 on the first real
    #     call; the project was on a prepay plan with a zero balance, and the
    #     old heuristic saw "429" and retried it five times
    #   * billing disabled / not enabled for the project
    permanent = ("limit: 0" in text or "limit 0" in text
                 or 'quota_limit_value: "0"' in text
                 or "limit_value: 0" in text
                 or "credits are depleted" in text
                 or "prepayment credits" in text
                 or "billing" in text and "disabled" in text)
    if permanent:
        return False, False
    daily = ("per day" in text or "perday" in text or "daily" in text
             or "requests per day" in text)
    throttled = ("429" in text or "resource_exhausted" in text
                 or "quota" in text or "rate limit" in text)
    return throttled, daily


class Judge:
    """One judge, one prompt, one model. Callable as `transcribe_one`."""

    def __init__(self, model_id=DEFAULT_MODEL_ID, prompt_file=None, repeat=0,
                 cache_path=None, requests_per_minute=10, max_retries=5,
                 backoff_initial_s=2.0, allow_new=True, verbose=True,
                 backend="aistudio", project=None, location="us-central1",
                 timeout_s=60.0, sdk_attempts=1, max_new_calls=None,
                 repeat_run_once=False, filter_retries=2):
        # WHICH SURFACE SERVES THE MODEL IS PART OF THE INSTRUMENT. The same
        # model_id on AI Studio and on Vertex should be the same weights, but
        # defaults (safety settings, exact served version) are not guaranteed
        # identical, so every SCORED call in one anchor set must come from ONE
        # backend. It is recorded in the cache key for exactly that reason.
        if backend not in ("aistudio", "vertex"):
            raise ValueError(f"backend must be 'aistudio' or 'vertex', got {backend!r}")
        self.backend = backend
        self.project = project
        self.location = location
        self.model_id = model_id
        self.prompt_file = prompt_file
        self.prompt = prompt_text(prompt_file)
        self.sha = prompt_sha(prompt_file)
        self.repeat = repeat
        self.cache_path = Path(cache_path or DEFAULT_CACHE)
        self.min_interval_s = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.max_retries = max_retries
        self.backoff_initial_s = backoff_initial_s
        self.allow_new = allow_new
        self.verbose = verbose
        # ONE RETRY LAYER, AND IT IS THIS ONE. google-genai retries internally
        # (default: 408/429/5xx with exponential backoff), which sits BELOW this
        # class -- so a permanent 429 like a depleted prepay balance took minutes
        # to surface and the run merely looked hung. Observed 2026-09-02.
        # sdk_attempts=1 means no SDK retries, so _is_quota_error sees the first
        # failure and can fail fast on the shapes that no wait will fix.
        self.timeout_s = timeout_s
        self.sdk_attempts = sdk_attempts

        self.max_new_calls = max_new_calls
        # Opt-in ONLY for the spread study. With this on, target/mixture/
        # interferer are keyed by repeat like an estimate, so the same clip can
        # deliberately be judged several times to measure run-to-run variance.
        # Off everywhere else, which is the run-once guarantee.
        self.repeat_run_once = repeat_run_once
        # A safety-filter refusal is NON-DETERMINISTIC (measured 2026-09-02:
        # a refused estimate passed on re-run). Retry before believing it,
        # or a coin flip is permanently cached as a non-response.
        self.filter_retries = filter_retries
        self.filter_blocks = 0        # persistent refusals, this process
        self.filter_transients = 0    # refused once, passed on retry

        self._cache = load_cache(self.cache_path)
        self._once = load_once_index(self.cache_path)
        self._client = None
        self._last_call_at = 0.0
        self.calls_made = 0          # THIS process only. The cache is the real ledger.
        self.cache_hits = 0

    # -- the transcribe_one seam -------------------------------------------
    def __call__(self, audio_path):
        status, text = self.judge(audio_path)
        # LCF-WER scores TEXT. A no_speech answer is an empty hypothesis, which
        # jiwer scores as all-deletions -- metric-definitions.md 3.1's stated
        # and deliberate treatment of a non-response.
        return text

    def _once_slot(self, audio_path):
        path = Path(audio_path)
        return (self.model_id, self.sha, self.backend, path.parent.name,
                path.stem.lower())

    def cached(self, audio_path):
        """(status, text) if this clip has already been paid for, else None.

        target / mixture / interferer are RUN-ONCE: one answer per trial per
        instrument, repeat index ignored, because no model can change that
        audio. estimate.wav is keyed normally so it can be judged again.
        """
        if runs_once(audio_path) and not self.repeat_run_once:
            return self._once.get(self._once_slot(audio_path))
        key = cache_key(audio_path, self.model_id, self.sha, self.repeat,
                        self.backend, force_repeat=self.repeat_run_once)
        return self._cache.get(key)

    def preflight(self, audio_paths):
        """(already_paid, would_spend) for a planned run. Makes NO calls.

        Report this before any run. The point is that the number of NEW calls is
        knowable in advance, so a run's cost is never a surprise and a stale
        prompt hash shows up as "everything is new" rather than as a bill.
        """
        already, new = 0, 0
        for path in audio_paths:
            if path is None or not Path(path).exists():
                continue
            if self.cached(path) is not None:
                already += 1
            else:
                new += 1
        return already, new

    def judge(self, audio_path):
        """(status, text). Serves the cache first; never re-calls a cached key."""
        path = Path(audio_path)
        key = cache_key(path, self.model_id, self.sha, self.repeat, self.backend,
                        force_repeat=self.repeat_run_once)

        hit = self.cached(path)
        if hit is not None:
            self.cache_hits += 1
            return hit

        if not self.allow_new:
            raise RuntimeError(
                f"{path} is not in the judge cache and allow_new=False. "
                f"Re-run with allow_new=True to spend a call on it.")
        # HARD SPEND CAP. Refuses rather than truncates, so an accidentally huge
        # run cannot quietly bill through: nothing already cached is lost, and
        # re-running resumes from where this stopped.
        if self.max_new_calls is not None and self.calls_made >= self.max_new_calls:
            raise NewCallLimitReached(
                f"max_new_calls={self.max_new_calls} reached; refusing to spend "
                f"another call. {self.cache_hits} clips were served from cache. "
                f"Raise max_new_calls to continue -- cached work is kept.")

        status, text = self._call_with_retries(path)
        row = {"key": key, "model": self.model_id, "backend": self.backend,
               "prompt_sha": self.sha,
               "trial_id": path.parent.name, "file": path.name,
               "repeat": self.repeat, "status": status, "text": text,
               "run_date": date.today().isoformat()}
        _append_cache(self.cache_path, row)      # per call, before returning
        self._cache[key] = (status, text)
        if runs_once(path) and not self.repeat_run_once:
            self._once[self._once_slot(path)] = (status, text)
        self.calls_made += 1
        return status, text

    # -- transport ---------------------------------------------------------
    def _ensure_client(self):
        """Two surfaces, one SDK.

        aistudio -- an API key from aistudio.google.com. Free tier available;
                    content may be used to improve Google's products.
        vertex   -- a GCP project on Application Default Credentials
                    (`gcloud auth application-default login`). Paid-tier data
                    terms, much higher rate limits, and the surface the $300
                    Google Cloud credit is NOT explicitly carved out of --
                    the published exclusion names "Gemini API in AI Studio".
                    Whether a given credit actually covers it is only
                    confirmable from the billing report after one real call.
        """
        if self._client is not None:
            return self._client

        # Credentials are checked BEFORE the SDK import, so a missing key
        # reports "set GEMINI_API_KEY" rather than "No module named google".
        project = None
        if self.backend == "vertex":
            project = self.project or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError(
                    "backend='vertex' needs a GCP project: pass project=, or set "
                    "GOOGLE_CLOUD_PROJECT. Auth is Application Default "
                    "Credentials -- run `gcloud auth application-default login`.")
        elif not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in .env (gitignored) and "
                "export it, or run with GEMINI_API_KEY=... ")

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "the Gemini SDK is not installed. Run: pip install -U google-genai"
            ) from exc

        kwargs = {}
        if self.backend == "vertex":
            kwargs = {"vertexai": True, "project": project, "location": self.location}

        # Built defensively: HttpRetryOptions is not in every google-genai
        # version, and a missing knob must not stop the run. Timeout is in
        # MILLISECONDS.
        try:
            from google.genai import types
            http_options = types.HttpOptions(
                timeout=int(self.timeout_s * 1000),
                retry_options=types.HttpRetryOptions(attempts=self.sdk_attempts),
            )
            self._client = genai.Client(http_options=http_options, **kwargs)
        except Exception:                                  # noqa: BLE001
            try:
                from google.genai import types
                self._client = genai.Client(
                    http_options=types.HttpOptions(timeout=int(self.timeout_s * 1000)),
                    **kwargs)
                if self.verbose:
                    print("    note: SDK retry_options unavailable; SDK may retry "
                          "beneath this class", flush=True)
            except Exception:                              # noqa: BLE001
                self._client = genai.Client(**kwargs)
                if self.verbose:
                    print("    note: could not set SDK timeout/retries; a hung "
                          "call will not self-bound", flush=True)
        return self._client

    def _throttle(self):
        wait = self.min_interval_s - (time.time() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)

    def _call_with_retries(self, path):
        delay = self.backoff_initial_s
        filter_attempts = 0
        # A filter retry is not a throttle retry: it must not consume the quota
        # backoff budget, so the loop is generous enough to hold both.
        budget = self.max_retries + self.filter_retries
        for attempt in range(1, budget + 1):
            self._throttle()
            try:
                self._last_call_at = time.time()
                result = self._call_once(path)
                if filter_attempts:
                    self.filter_transients += 1
                    if self.verbose:
                        print(f"    ...passed on retry -- the earlier refusal "
                              f"was transient, NOT recorded", flush=True)
                return result
            except QuotaExhausted:
                raise
            except Exception as exc:                       # noqa: BLE001
                if _is_content_blocked(exc):
                    filter_attempts += 1
                    if filter_attempts <= self.filter_retries:
                        if self.verbose:
                            print(f"    safety filter refused "
                                  f"{Path(path).parent.name}/{Path(path).name} "
                                  f"-- retrying ({filter_attempts}/"
                                  f"{self.filter_retries}); the filter is not "
                                  f"deterministic", flush=True)
                        time.sleep(self.backoff_initial_s)
                        continue
                    # Refused every time. Now it is a finding about this clip
                    # rather than a coin flip, so record it.
                    self.filter_blocks += 1
                    if self.verbose:
                        print(f"    safety filter refused "
                              f"{Path(path).parent.name}/{Path(path).name} "
                              f"{filter_attempts}x -- recorded as a "
                              f"non-response", flush=True)
                    return "blocked_by_filter", ""
                throttled, daily = _is_quota_error(exc)
                if daily:
                    raise QuotaExhausted(
                        f"daily quota spent after {self.calls_made} calls this "
                        f"process. Cached work is safe -- rerun tomorrow and it "
                        f"resumes. Original: {exc}") from exc
                if not throttled or (attempt - filter_attempts) >= self.max_retries:
                    raise
                sleep_s = delay + random.uniform(0, delay * 0.25)
                if self.verbose:
                    # ALWAYS show the reason. Printing only "throttled" hid a
                    # zero-limit refusal behind five pointless retries on the
                    # first real call, 2026-09-02.
                    reason = " ".join(str(exc).split())[:300]
                    print(f"    throttled (attempt {attempt}/{self.max_retries}), "
                          f"sleeping {sleep_s:.1f}s\n      reason: {reason}",
                          flush=True)
                time.sleep(sleep_s)
                delay *= 2
        raise RuntimeError("unreachable")

    def _call_once(self, path):
        client = self._ensure_client()
        audio_b64 = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
        interaction = client.interactions.create(
            model=self.model_id,
            input=[
                {"type": "text", "text": self.prompt},
                {"type": "audio", "data": audio_b64, "mime_type": AUDIO_MIME},
            ],
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": RESPONSE_SCHEMA},
        )
        return self._parse(interaction.output_text)

    @staticmethod
    def _parse(output_text):
        """Structured output should always parse. If it does not, that is itself
        a finding about the judge (J2: a judge that wanders off-prompt lands in
        NRR and is indistinguishable from a degenerate extractor), so record it
        as a distinct status rather than crashing the run."""
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


def make_judge(**kwargs):
    """Convenience: a `transcribe_one`-compatible callable."""
    return Judge(**kwargs)
