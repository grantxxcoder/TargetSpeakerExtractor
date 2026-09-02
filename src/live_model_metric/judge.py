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


def prompt_text(prompt_file=None):
    return Path(prompt_file or DEFAULT_PROMPT_FILE).read_text()


def prompt_sha(prompt_file=None):
    """Short hash, recorded per call. A changed prompt invalidates the cache."""
    return hashlib.sha256(prompt_text(prompt_file).encode()).hexdigest()[:12]


def cache_key(audio_path, model_id, sha, repeat, backend="aistudio"):
    """Mirrors evaluate._cache_key, plus the four things that make a judge
    answer reproducible: which model, which serving backend, which prompt,
    which repeat. Backend is in the key so an AI Studio answer is never served
    to a Vertex run, or the reverse -- they are different instruments until
    measured to be the same."""
    path = Path(audio_path)
    stat = path.stat()
    return (f"{model_id}@{backend}|{sha}|{path.parent.name}|{path.name}"
            f"|{int(stat.st_mtime)}|{stat.st_size}|r{repeat}")


def load_cache(cache_path=None):
    """key -> (status, text). Missing file is an empty cache, not an error."""
    cache_path = Path(cache_path or DEFAULT_CACHE)
    if not cache_path.exists():
        return {}
    with open(cache_path, newline="") as handle:
        return {row["key"]: (row["status"], row["text"])
                for row in csv.DictReader(handle)}


def _append_cache(cache_path, row):
    cache_path = Path(cache_path)
    exists = cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _is_quota_error(exc):
    """google-genai does not expose a typed daily-quota error, so this reads the
    message. Deliberately conservative: anything mentioning a per-day quota is
    treated as the daily cap, everything else 429-ish is a per-minute throttle
    worth retrying."""
    text = str(exc).lower()
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
                 backend="aistudio", project=None, location="us-central1"):
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

        self._cache = load_cache(self.cache_path)
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

    def judge(self, audio_path):
        """(status, text). Serves the cache first; never re-calls a cached key."""
        path = Path(audio_path)
        key = cache_key(path, self.model_id, self.sha, self.repeat, self.backend)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        if not self.allow_new:
            raise RuntimeError(
                f"{path} is not in the judge cache and allow_new=False. "
                f"Re-run with allow_new=True to spend a call on it.")

        status, text = self._call_with_retries(path)
        row = {"key": key, "model": self.model_id, "backend": self.backend,
               "prompt_sha": self.sha,
               "trial_id": path.parent.name, "file": path.name,
               "repeat": self.repeat, "status": status, "text": text,
               "run_date": date.today().isoformat()}
        _append_cache(self.cache_path, row)      # per call, before returning
        self._cache[key] = (status, text)
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

        if self.backend == "vertex":
            self._client = genai.Client(vertexai=True, project=project,
                                        location=self.location)
        else:
            self._client = genai.Client()
        return self._client

    def _throttle(self):
        wait = self.min_interval_s - (time.time() - self._last_call_at)
        if wait > 0:
            time.sleep(wait)

    def _call_with_retries(self, path):
        delay = self.backoff_initial_s
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                self._last_call_at = time.time()
                return self._call_once(path)
            except QuotaExhausted:
                raise
            except Exception as exc:                       # noqa: BLE001
                throttled, daily = _is_quota_error(exc)
                if daily:
                    raise QuotaExhausted(
                        f"daily quota spent after {self.calls_made} calls this "
                        f"process. Cached work is safe -- rerun tomorrow and it "
                        f"resumes. Original: {exc}") from exc
                if not throttled or attempt == self.max_retries:
                    raise
                sleep_s = delay + random.uniform(0, delay * 0.25)
                if self.verbose:
                    print(f"    throttled (attempt {attempt}/{self.max_retries}), "
                          f"sleeping {sleep_s:.1f}s", flush=True)
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
