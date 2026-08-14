#!/usr/bin/env python3
"""PR2 acceptance test: prove the sampler wiring changed no value.

    python scripts/check_manifest_parity.py --a OLD_DIR --b NEW_DIR

Compares two manifest directories byte for byte, ignoring columns named with
--ignore. Manifests are not tracked in git, so the reference has to be a copy of
the previous output taken before the change; there is nothing to diff against
otherwise.

Byte comparison, not a pandas frame comparison: rounding is applied when the row
is written, so a float that reads back equal can still have been written
differently. The written bytes are what the renderer will consume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPLITS = ["smoke_train", "smoke_val", "val", "eval_public", "eval_private", "train"]


def strip_columns(path, ignore):
    """File bytes with the ignored columns removed, line endings preserved."""
    raw = path.read_bytes()
    lines = raw.split(b"\n")
    header = lines[0].rstrip(b"\r").split(b",")
    keep = [i for i, name in enumerate(header) if name.decode() not in ignore]
    out = []
    for line in lines:
        if not line.strip():
            out.append(line)
            continue
        crlf = line.endswith(b"\r")
        fields = line.rstrip(b"\r").split(b",")
        # A row with a different column count is itself the finding; keep it
        # whole so the diff shows it rather than silently reindexing.
        if len(fields) != len(header):
            out.append(line)
            continue
        out.append(b",".join(fields[i] for i in keep) + (b"\r" if crlf else b""))
    return b"\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="reference manifest dir")
    ap.add_argument("--b", required=True, help="new manifest dir")
    ap.add_argument("--ignore", default="regime",
                    help="comma-separated columns to drop before comparing")
    ap.add_argument("--splits", default=",".join(SPLITS))
    args = ap.parse_args()

    ignore = {c for c in args.ignore.split(",") if c}
    bad = 0
    for split in args.splits.split(","):
        a, b = Path(args.a) / f"{split}.csv", Path(args.b) / f"{split}.csv"
        if not a.exists() or not b.exists():
            print(f"{split:<14} SKIP (missing {'a' if not a.exists() else 'b'})")
            continue
        if strip_columns(a, ignore) == strip_columns(b, ignore):
            print(f"{split:<14} identical")
        else:
            print(f"{split:<14} DIFFERS")
            bad += 1

    print(f"\nignored columns: {sorted(ignore) or 'none'}")
    print("PASS" if not bad else f"FAIL — {bad} split(s) differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
