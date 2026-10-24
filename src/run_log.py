"""Adds a wall-clock row to docs/run_times.md when a job that took over a minute finishes, so a runtime is looked up rather than guessed (CLAUDE.md).

    with timed("scripts/build_vad_index.py", lambda: f"{count:,} utterances"):
        ...

The scope is a function rather than a string, because the interesting scope is usually only known once the work is done. Logging never raises, so it cannot fail a two-hour render.
"""

import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

# Rows go directly after this line, newest first, so the table header stays put.
MARKER = "<!-- rows appended below by src/run_log.py -->"
RUN_TIMES = Path(__file__).resolve().parents[1] / "docs/run_times.md"
MIN_SECONDS = 60


# this writes a duration the way the table shows it: 58 s, 26 min or 2.5 h
def human(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# this inserts one row, and returns whether it was written
def record(command, scope, wall_seconds, rate):
    if wall_seconds < MIN_SECONDS:
        return False
    try:
        row = f"| {date.today().isoformat()} | `{command}` | {scope} | {human(wall_seconds)} | {rate} |"
        text = RUN_TIMES.read_text()
        if MARKER not in text:
            print(f"  run_log: no marker in {RUN_TIMES}, appending at end", file=sys.stderr)
            RUN_TIMES.write_text(text.rstrip() + "\n" + row + "\n")
        else:
            RUN_TIMES.write_text(text.replace(MARKER, MARKER + "\n" + row, 1))
        print(f"  run time logged: {human(wall_seconds)} in {RUN_TIMES.name}", file=sys.stderr)
        return True
    except Exception as error:  # A logging failure must never end the job.
        print(f"  run_log: could not write {RUN_TIMES}: {error}", file=sys.stderr)
        return False


# this times a job and logs it. A job that raises is still logged, marked failed, because knowing a render died 40 minutes in is worth not rediscovering
@contextmanager
def timed(command, scope=lambda: "", rate=lambda: ""):
    start = time.time()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.time() - start
        try:
            scope_text = scope()
            rate_text = rate()
        except Exception:
            scope_text, rate_text = "?", ""
        if failed:
            scope_text = scope_text + " **(failed)**"
        record(command, scope_text, elapsed, rate_text)
