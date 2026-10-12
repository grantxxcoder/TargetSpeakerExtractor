#!/usr/bin/env python3
"""Write the two M8 training manifests from ami_train (decisions-m8.md 2026-10-12).

    ../tse_venv/bin/python scripts/make_ami_manifests.py

  amift_train.csv   every ami_train row x `--repeat`. AMI only (arm A).
  amimix_train.csv  the same rows, plus an EQUAL number of sir0_train rows
                    (arm B, 50/50 replay against forgetting).

REPEAT, NOT LONGER EPOCHS. AMI windows are 10-20 s and training crops are 4 s, so
one pass sees about a quarter of the audio. Listing each row 4 times makes one
epoch cover each window about once; the repeats get DIFFERENT crops because the
loader seeds the crop on (seed, epoch, row index). No loader change needed.

STRATIFIED sir0 SAMPLE. The synthetic half keeps sir0_train's condition mix
(same idea as make_subset_manifest.py): loss.w was calibrated against a measured
target-absent rate, and a drifted mix would change what silence is worth.

Both manifests carry `audio_dir` on every row, so one manifest reads trials from
data/rendered/ami_train and data/rendered/sir0_train (dataset_loader._audio_dir).
"""

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.estimates.runner import git_commit  # noqa: E402


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def stratified(df, n, seed):
    """n rows of df, condition mix preserved (largest-remainder rounding), seeded."""
    share = df["condition"].value_counts() * n / len(df)
    counts = share.astype(int)
    for cond in (share - counts).sort_values(ascending=False).index[:n - counts.sum()]:
        counts[cond] += 1
    return pd.concat(df[df.condition == c].sample(int(k), random_state=seed)
                     for c, k in counts.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--ami", default="ami_train")
    ap.add_argument("--synthetic", default="sir0_train")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mdir = Path(args.manifest_dir)
    ami_path, syn_path = mdir / f"{args.ami}.csv", mdir / f"{args.synthetic}.csv"
    ami = pd.read_csv(ami_path).assign(audio_dir=args.ami)
    syn = pd.read_csv(syn_path).assign(audio_dir=args.synthetic, remixable=1)
    ami_rep = pd.concat([ami] * args.repeat, ignore_index=True)
    syn_half = stratified(syn, len(ami_rep), args.seed)
    mix = pd.concat([ami_rep, syn_half], ignore_index=True, sort=False)
    mix = mix.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    common = {"generated": date.today().isoformat(), "generator": "scripts/make_ami_manifests.py",
              "seed": args.seed, "repeat": args.repeat, "git_commit": git_commit(),
              "decision": "decisions-m8.md 2026-10-12"}
    for name, df, sources, extra in (
            ("amift_train", ami_rep, [ami_path], {}),
            ("amimix_train", mix, [ami_path, syn_path],
             {"ami_rows": len(ami_rep), "synthetic_rows": len(syn_half),
              "synthetic_condition_mix": syn_half["condition"].value_counts().to_dict(),
              "synthetic_source_condition_mix": syn["condition"].value_counts().to_dict()})):
        out = mdir / f"{name}.csv"
        df.to_csv(out, index=False)
        meta = {**common, "manifest": str(out), "n_rows": len(df),
                "sources": {str(p): md5(p) for p in sources}, **extra}
        out.with_suffix(".meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"{name}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
