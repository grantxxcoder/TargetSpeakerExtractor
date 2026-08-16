"""Append a wall-clock row to docs/run_times.md when a job finishes.

Implements the CLAUDE.md rule: anything over a minute gets a row, so a runtime
is looked up rather than guessed. Jobs under the threshold write nothing, which
is what keeps the file short enough to be worth reading.

    from src.run_log import timed

    with timed("build_vad_index.py", lambda: f"{n:,} utts / {hours:.0f} h"):
        ...work...

`scope` is a callable, not a string, because the interesting scope is usually
only known once the work is done ("21,324 utterances", not "all of them").
It is evaluated at exit, inside the same try block, so a bad f-string cannot
take down a job that already succeeded.

Never raises. A logging failure must not fail a 2-hour render.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

# Rows are inserted directly after this line, newest first, so the file needs no
# markdown parsing and the table header stays put.
MARKER = "<!-- rows appended below by src/run_log.py -->"
RUN_TIMES = Path(__file__).resolve().parents[1] / "docs/run_times.md"
MIN_SECONDS = 60


def human(seconds):
    """`58 s`, `26 min`, `2.5 h`. Two significant figures is plenty -- this is
    for planning a coffee break, not for benchmarking."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def record(command, scope, wall_s, rate="", path=None):
    """Insert one row. Returns True if written, False if skipped or failed.

    `path` resolves at CALL time, not import time. As a default argument it
    bound the module-level RUN_TIMES once, so tests that patched RUN_TIMES
    still wrote to the real docs/run_times.md -- which is exactly what happened
    the first time these tests ran.
    """
    path = Path(path) if path is not None else RUN_TIMES
    if os.environ.get("RUN_LOG") == "0":
        return False
    if wall_s < MIN_SECONDS:
        return False
    try:
        row = (f"| {date.today().isoformat()} | `{command}` | {scope} | "
               f"{human(wall_s)} | {rate} |")
        text = path.read_text()
        if MARKER not in text:
            print(f"  run_log: no marker in {path}, appending at end",
                  file=sys.stderr)
            path.write_text(text.rstrip() + "\n" + row + "\n")
        else:
            path.write_text(text.replace(MARKER, MARKER + "\n" + row, 1))
        print(f"  run time logged: {human(wall_s)} -> {path.name}",
              file=sys.stderr)
        return True
    except Exception as e:                      # noqa: BLE001 - never fatal
        print(f"  run_log: could not write {path}: {e}", file=sys.stderr)
        return False


@contextmanager
def timed(command, scope=lambda: "", rate=lambda: ""):
    """Time a job and log it. `scope` and `rate` are callables evaluated at exit.

    A job that raises is still logged, marked `(failed)`: knowing that a render
    died 40 minutes in is exactly the kind of thing worth not rediscovering.
    """
    t0 = time.time()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.time() - t0
        try:
            s = scope() if callable(scope) else str(scope)
            r = rate() if callable(rate) else str(rate)
        except Exception:                       # noqa: BLE001
            s, r = "?", ""
        record(command, s + (" **(failed)**" if failed else ""), elapsed, r)
