"""Fetch only the seconds of an AMI recording we use, by HTTP byte range.

AMI serves each channel as one whole-meeting WAV (16 kHz mono PCM_16). The
ami_train windows are ~2 h inside ~38 h of meetings, so whole files cost ~22 GB
for ~0.8 GB of use (decisions-m8.md 2026-10-12). The mirror answers Range
requests (HTTP 206, checked 2026-10-12), and PCM_16 is fixed-width: sample n of
a file whose data chunk starts at byte D is bytes D + 2n and D + 2n + 1.

Every piece is cached as its own WAV and logged in <cache>/index.csv with its
URL, byte range and sha256, so a build can be checked against the index and
rerun without the network. Samples outside the file are returned as zeros.

Found 2026-10-12 while building ami_train (decisions-m8.md):
  * ES2010d is stored STEREO with both channels bit-identical. Read as frame
    pairs, checked L == R on every piece, left kept. Unequal channels raise.
  * The mirror answers 403 under ~12+ parallel requests and 200 a moment
    later: retried with backoff. 404 is a missing file (IS1007d.Array1-01,
    like IS1003b for ami_eval) and fails at once.
"""

import csv
import hashlib
import http.client
import struct
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
INDEX_FIELDS = ["url", "meeting", "channel", "first_sample", "n_samples",
                "byte_start", "byte_end", "file_channels", "sha256", "fetched"]


def _get(url, byte_start=None, byte_end=None, tries=6):
    """Bytes [byte_start, byte_end] inclusive (whole file if None). Retries
    transient failures (403 under load, 5xx, timeouts); a 404 raises at once."""
    request = urllib.request.Request(url)
    if byte_start is not None:
        request.add_header("Range", f"bytes={byte_start}-{byte_end}")
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                if byte_start is not None:
                    # 200 would mean the server ignored Range and sent the whole
                    # file: the bytes would then be at the wrong offsets.
                    if response.status != 206:
                        raise IOError(f"{url}: expected 206, got {response.status}")
                    if len(body) != byte_end - byte_start + 1:
                        raise IOError(f"{url}: asked {byte_end - byte_start + 1} B, got {len(body)}")
                return body
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(url) from error
            if attempt == tries - 1:
                raise
            time.sleep(2 * 2 ** attempt)
        except (OSError, IOError, http.client.HTTPException):   # incl. IncompleteRead
            if attempt == tries - 1:
                raise
            time.sleep(2 * 2 ** attempt)


def parse_wav_header(head):
    """(data_offset, data_bytes, channels) of a 16 kHz PCM_16 RIFF header, mono
    or stereo. Walks the chunks rather than assuming 44 bytes; refuses any
    other format."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt = 12, None
    while pos + 8 <= len(head):
        chunk, size = head[pos:pos + 4], struct.unpack("<I", head[pos + 4:pos + 8])[0]
        if chunk == b"fmt ":
            fmt = struct.unpack("<HHIIHH", head[pos + 8:pos + 24])
        elif chunk == b"data":
            if fmt is None:
                raise ValueError("data chunk before fmt chunk")
            audio_format, channels, rate, _, _, bits = fmt
            if (audio_format, rate, bits) != (1, SAMPLE_RATE, 16) or channels not in (1, 2):
                raise ValueError(f"unexpected format {fmt}")
            return pos + 8, size, channels
        pos += 8 + size + (size & 1)
    raise ValueError("no data chunk in the first bytes")


class AmiFetcher:
    """Range-reads AMI channels into a local cache. Thread-safe."""

    def __init__(self, base_url, cache_dir):
        self.base_url = base_url.rstrip("/")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.index = self.cache / "index.csv"
        self._lock = threading.Lock()
        self._headers = {}
        self.bytes_fetched = 0

    def url(self, meeting, channel):
        return f"{self.base_url}/{meeting}/audio/{meeting}.{channel}.wav"

    def _header(self, url):
        with self._lock:
            if url in self._headers:
                return self._headers[url]
        found = parse_wav_header(_get(url, 0, 4095))
        with self._lock:
            self._headers[url] = found
        return found

    def path(self, meeting, channel, first, n):
        return self.cache / meeting / f"{meeting}.{channel}.{first}-{first + n}.wav"

    def read(self, meeting, channel, first, n):
        """float32 samples [first, first + n) of one channel, as sf.read would
        give from the whole file (int16 / 32768). Zeros outside the file."""
        path = self.path(meeting, channel, first, n)
        if path.exists():
            audio, rate = sf.read(path, dtype="float32")
            assert rate == SAMPLE_RATE and len(audio) == n, path
            return audio
        url = self.url(meeting, channel)
        data_offset, data_bytes, channels = self._header(url)
        frame = 2 * channels                       # bytes per sample frame
        total = data_bytes // frame
        lo, hi = max(first, 0), min(first + n, total)
        out = np.zeros(n, dtype=np.int16)
        digest = ""
        if hi > lo:
            byte_start = data_offset + frame * lo
            byte_end = data_offset + frame * hi - 1
            body = _get(url, byte_start, byte_end)
            samples = np.frombuffer(body, dtype="<i2").reshape(-1, channels)
            if channels == 2 and not np.array_equal(samples[:, 0], samples[:, 1]):
                raise ValueError(f"{url}: stereo with DIFFERENT channels at "
                                 f"samples {lo}-{hi}; refusing to pick one")
            out[lo - first:hi - first] = samples[:, 0]
            digest = hashlib.sha256(body).hexdigest()
        else:
            byte_start = byte_end = -1
        # Atomic: write a temp file, log it, THEN rename into place. A killed run
        # leaves at worst an orphan temp file or a logged piece that is fetched
        # again -- never a cached piece without its index row, or a truncated one.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        sf.write(tmp, out, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        with self._lock:
            self.bytes_fetched += max(0, byte_end - byte_start + 1)
            new = not self.index.exists()
            with open(self.index, "a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow({"url": url, "meeting": meeting, "channel": channel,
                                 "first_sample": first, "n_samples": n,
                                 "byte_start": byte_start, "byte_end": byte_end,
                                 "file_channels": channels,
                                 "sha256": digest, "fetched": date.today().isoformat()})
        tmp.replace(path)
        return out.astype(np.float32) / 32768.0
