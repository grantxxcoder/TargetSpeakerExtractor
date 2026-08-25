#!/usr/bin/env python3
"""Write a deterministic, condition-stratified SUBSET of an existing manifest.

WHY THIS EXISTS
---------------
The 50-trial `smoke` split has only 20 speakers. Training on it produced an
enrolment-blind mute: swapping a crop's enrolment moved the output by -17.15 dB
and cost only 0.62 dB of L_pres, so the model was not doing target extraction at
all. It could not tell target-present from target-absent, and under a single
shared output gain the mute is the genuinely optimal answer to the objective.
See docs/decisions/decisions-m1.md 2026-08-25.

Learning to use the enrolment needs speaker diversity, which trial count alone
does not buy. `train` already has 19,938 trials over 1,172 speakers and is
already rendered, so the cheapest correct mid-size split is a subset of it: no
new trials, no new audio, and the sample is drawn from the real training
distribution rather than a 20-speaker corner of it.

This script does NOT generate trials. scripts/build_manifest.py does that.
This only selects rows from a manifest that already exists.

WHAT IT GUARANTEES
------------------
  deterministic   rows are chosen by hash order of (seed, tag, trial_id), the
                  same idiom as deterministic_sample() in make_splits.py. Same
                  seed and same source manifest always give the same subset.
  stratified      the condition mix is preserved proportionally. This matters:
                  loss.w was calibrated against a measured absent rate, so a
                  subset that drifted on `condition` would silently change what
                  the absent half of the objective is worth.
  provenanced     writes <name>.meta.yaml recording the source manifest, its
                  own md5, the seed, the git commit and the realised mix, so a
                  training run logged against this subset is reproducible.

USAGE
-----
    python scripts/make_subset_manifest.py --subset mid_train
    python scripts/make_subset_manifest.py --subset mid_train --subset mid_val
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def git_commit() -> str:
    """Current commit hash, or a clear marker. Same helper as make_splits.py --
    duplicated the way the other scripts do it. `-dirty` matters: a manifest
    logged against a dirty tree is not reproducible from that hash."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def hash_order(ids, seed: int, tag: str) -> list:
    """Reproducible shuffle by sha256 of (seed, tag, id).

    Not random.sample: this must not depend on Python's RNG state, on the order
    rows happen to sit in the source file, or on how many ids are asked for. The
    tag keeps mid_train and mid_val from drawing the same hash order.
    """
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{tag}:{i}".encode()).hexdigest())


def stratified_subset(df: pd.DataFrame, n: int, seed: int, tag: str,
                      stratify_by: str | None) -> pd.DataFrame:
    """n rows of df, hash-ordered, with the `stratify_by` mix preserved.

    Largest-remainder apportionment, not round(): rounding each stratum
    independently overshoots or undershoots n, and silently returning 1,997 rows
    when 2,000 were asked for is the kind of thing that never gets noticed.
    """
    if stratify_by is None:
        keep = hash_order(df["trial_id"].tolist(), seed, tag)[:n]
        return df[df["trial_id"].isin(set(keep))].copy()

    groups = {k: g for k, g in df.groupby(stratify_by, sort=True)}
    exact = {k: len(g) * n / len(df) for k, g in groups.items()}
    quota = {k: int(v) for k, v in exact.items()}

    # hand out the leftover seats to the largest fractional parts, ties broken by
    # stratum name so the result does not depend on dict ordering
    short = n - sum(quota.values())
    for k in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:short]:
        quota[k] += 1

    keep = []
    for k, g in groups.items():
        take = min(quota[k], len(g))
        if take < quota[k]:
            print(f"  WARNING: stratum {stratify_by}={k} has only {len(g)} rows, "
                  f"wanted {quota[k]}", file=sys.stderr)
        keep += hash_order(g["trial_id"].tolist(), seed, f"{tag}:{k}")[:take]

    keep = set(keep)
    # reindexed to the source manifest's own row order, so the subset reads as a
    # filtered view of it rather than in hash order
    return df[df["trial_id"].isin(keep)].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", action="append", required=True,
                    help="name of a key under `subsets:` in the generator config; repeatable")
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--manifest-dir", default="data/manifests")
    args = ap.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    config_md5 = hashlib.md5(config_path.read_bytes()).hexdigest()
    seed = int(config["seed"])
    subsets = config.get("subsets") or {}
    man_dir = Path(args.manifest_dir)

    for name in args.subset:
        if name not in subsets:
            sys.exit(f"unknown subset '{name}'. Known: {sorted(subsets)}")
        spec = subsets[name]
        n = int(spec["n_trials"])
        source = spec["source"]
        stratify_by = spec.get("stratify_by")

        src_csv = man_dir / f"{source}.csv"
        if not src_csv.exists():
            sys.exit(f"source manifest {src_csv} does not exist -- build it first "
                     f"with scripts/build_manifest.py --split {source}")
        src = pd.read_csv(src_csv)
        if n > len(src):
            sys.exit(f"{name}: asked for {n} trials but {source} has only {len(src)}")

        sub = stratified_subset(src, n, seed, name, stratify_by)
        assert len(sub) == n, f"{name}: got {len(sub)} rows, wanted {n}"

        out_csv = man_dir / f"{name}.csv"
        sub.to_csv(out_csv, index=False)

        src_meta_path = src_csv.with_suffix(".meta.yaml")
        src_meta = yaml.safe_load(src_meta_path.read_text()) if src_meta_path.exists() else {}
        mix = (sub[stratify_by].value_counts().to_dict() if stratify_by else {})
        meta = {
            "generated": date.today().isoformat(),
            "generator": "scripts/make_subset_manifest.py",
            "subset": name,
            "seed": seed,
            "config": str(config_path),
            "config_md5": config_md5,
            "git_commit": git_commit(),
            "n_trials": len(sub),
            "n_requested": n,
            # the audio is NOT re-rendered: this subset reads the source split's
            # rendered trials, so the source's own provenance is what applies
            "source_manifest": str(src_csv),
            "source_n_trials": len(src),
            "source_audio_dir": f"data/rendered/{source}",
            "source_manifest_md5": hashlib.md5(src_csv.read_bytes()).hexdigest(),
            "source_built_at_commit": src_meta.get("git_commit", "UNKNOWN"),
            "source_config_md5": src_meta.get("config_md5", "UNKNOWN"),
            "noise_split": src_meta.get("noise_split", "UNKNOWN"),
            "stratify_by": stratify_by,
            "condition_mix": {str(k): int(v) for k, v in sorted(mix.items())},
            "n_speakers": int(sub["target_speaker"].nunique()),
        }
        out_meta = man_dir / f"{name}.meta.yaml"
        out_meta.write_text(yaml.safe_dump(meta, sort_keys=False))

        print(f"{name}: {len(sub)} trials from {source} ({len(src)}), "
              f"{meta['n_speakers']} target speakers -> {out_csv}")
        if stratify_by:
            for k in sorted(mix):
                got = 100 * mix[k] / len(sub)
                want = 100 * (src[stratify_by] == k).sum() / len(src)
                print(f"    {k:<16} {mix[k]:>5}  {got:5.2f}%  (source {want:5.2f}%)")


if __name__ == "__main__":
    main()
