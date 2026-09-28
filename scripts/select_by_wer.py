#!/usr/bin/env python3
"""Pick the checkpoint by WORD ERROR RATE, not by the signal-domain score.

    ../tse_venv/bin/python scripts/select_by_wer.py \
        --config experiments/configs/bsrnn_cue_parts.yaml \
        --checkpoints 'models/model_sir0_cueparts-e*.pt' \
        --split sir0_val --limit 103 --dry-run

WHY THIS EXISTS, AND WHY REWEIGHTING THE OLD RULE WAS NOT ENOUGH.

`train.py:selection_score` ranks epochs on `L_pres + w_m*L_MR + w_g*L_gain`.
Measured 2026-09-23 on item 1a, that rule kept epoch 9 -- which is WORSE THAN
DOING NOTHING on content (ASR LCF-WER 65.63 against a 65.22 floor) -- and
rejected epoch 15, which beats the floor by 12.5 points and is the best content
result the project has produced.

Every repair inside the signal domain was tried against the logged history and
none of them reach epoch 15 (`scripts/reselect_epochs.py`):

  * dropping the volume term `L_gain`      -> still keeps 9
  * `L_pres` alone                          -> still keeps 9
  * `L_MR` alone                            -> keeps 15, but muting IMPROVES
                                               L_MR (2026-08-28), so a rule that
                                               selects on it selects silence
  * any top-k shortlist                     -> epoch 15 ranks 10th of 14

So the criterion is not mis-weighted, it is measuring the wrong thing, and the
replacement is the quantity the project actually cares about. The
family-separation rule was withdrawn 2026-09-15, so scoring checkpoints with a
listener during development is now permitted.

TWO RULES THAT TRAVEL WITH THIS SCRIPT.

1. SELECT ON ONE SPLIT, REPORT ON ANOTHER. Choosing the argmin over N epochs on
   the same 103 trials you then quote is the winner's curse: with a ~8-point
   paired resolution you will pick the luckiest epoch and the headline will be
   optimistic. Select on `sir0_val`; report on `sir0_privval` or `eval_private`.
   This script refuses to select on a holdout split for that reason.

2. THE SILENCE BAR STILL APPLIES. WER is computed on target-present trials, so
   it says nothing about a model that talks through silence. Eligibility is read
   from each checkpoint's own stored `row` (`L_abs`) exactly as `train.py`
   applies it, and ineligible checkpoints are scored but never chosen.

COST, MEASURED (docs/run_times.md, CPU, whole-clip): ~28 min to render 103
estimates and ~6-8 min for the ASR pass, so ~35 min per checkpoint. Use
`--limit` and a `keep_stride` set rather than scoring every epoch.
"""

import argparse
import glob
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Never select on these: they are the reported holdout. CLAUDE.md, and
# decisions-m4.md 2026-09-15 (the holdout moved from the model to the data).
HOLDOUT_SPLITS = {"sir0_privval", "eval_private"}


def epoch_of(checkpoint_path):
    stem = Path(checkpoint_path).stem
    tail = stem.rsplit("-e", 1)[-1] if "-e" in stem else stem.rsplit("_e", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


def eligibility(checkpoint_path, bar):
    """(L_abs, eligible). Read from the checkpoint's own stored val row."""
    if bar is None:
        return None, True
    try:
        import torch
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:                      # noqa: BLE001
        print(f"    could not read eligibility ({exc}); treating as eligible")
        return None, True
    row = blob.get("row") or {}
    l_abs = row.get("L_abs")
    if l_abs is None:
        return None, True
    return float(l_abs), float(l_abs) <= float(bar)


def run(command, dry_run):
    print("    $ " + " ".join(command))
    if dry_run:
        return True
    return subprocess.run(command, cwd=ROOT).returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", required=True,
                    help="glob, e.g. 'models/model_sir0_cueparts-e*.pt'")
    ap.add_argument("--split", default="sir0_val",
                    help="MANIFEST split, what scripts/evaluate.py wants "
                         "(sir0_val, eval_public, ...). Also what the holdout "
                         "guard below checks.")
    # The two scripts disagree about what `--split` means and it is a real trap:
    # make_estimates.py takes the RUN split (`sir0`) and renders that pair's VAL
    # manifest, while evaluate.py takes the manifest name (`sir0_val`). Passing
    # one to the other fails with "invalid choice", which is how this was found.
    ap.add_argument("--run-split", default="sir0",
                    help="RUN split, what scripts/make_estimates.py wants "
                         "(sir0, sir0ext, mid, full, smoke). Its VAL manifest "
                         "must be the --split above.")
    ap.add_argument("--condition", default="both")
    ap.add_argument("--limit", type=int, default=None,
                    help="trials per checkpoint; the cost knob")
    ap.add_argument("--listener", default="asr", choices=["asr", "judge"],
                    help="asr is the default: cheap, cached, and r=0.825 with "
                         "the judge (2026-09-15-judge-predictability)")
    ap.add_argument("--select-abs-max", type=float, default=-10.0,
                    help="silence bar, as in train.py; pass nan to disable")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and the plan, run nothing")
    args = ap.parse_args()

    if args.split in HOLDOUT_SPLITS:
        ap.error(f"--split {args.split} is a reported holdout. Select on "
                 f"sir0_val and report on the holdout, never the reverse.")

    # nan disables the bar; nan != nan is the test.
    bar = args.select_abs_max if args.select_abs_max == args.select_abs_max else None
    checkpoints = sorted(glob.glob(str(ROOT / args.checkpoints)), key=epoch_of)
    if not checkpoints:
        ap.error(f"no checkpoints matched {args.checkpoints!r}")

    out_dir = Path(args.out or ROOT / "experiments" / "results" /
                   f"{date.today().isoformat()}-select-by-wer")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"selecting on {args.split} (rendered via --split {args.run_split})"
          f"/{args.condition} by {args.listener} "
          f"LCF-WER, {len(checkpoints)} checkpoints, "
          f"limit={args.limit}, silence bar L_abs <= {bar}")
    print("REPORT THE WINNER ON A HOLDOUT SPLIT, NOT ON THIS ONE.\n")

    rows = []
    for checkpoint in checkpoints:
        epoch = epoch_of(checkpoint)
        l_abs, eligible = eligibility(checkpoint, bar)
        print(f"  epoch {epoch}: {Path(checkpoint).name}"
              f"  L_abs={l_abs}  eligible={eligible}")
        est_dir = out_dir / f"est-e{epoch:03d}"
        eval_dir = out_dir / f"eval-e{epoch:03d}"
        ok = run([args.python, "scripts/make_estimates.py",
                  "--split", args.run_split, "--config", args.config,
                  "--checkpoint", checkpoint, "--out", str(est_dir)]
                 + (["--condition", args.condition] if args.condition else [])
                 + (["--limit", str(args.limit)] if args.limit else []), args.dry_run)
        if ok:
            ok = run([args.python, "scripts/evaluate.py",
                      "--split", args.split, "--condition", args.condition,
                      "--est", str(est_dir), "--metrics", "content",
                      "--listener", args.listener, "--out", str(eval_dir)]
                     + (["--limit", str(args.limit)] if args.limit else []), args.dry_run)
        wer = None
        results_json = eval_dir / "results.json"
        if ok and results_json.exists():
            blob = json.loads(results_json.read_text())
            wer = blob.get("systems", {}).get("estimate", {}).get("lcf_wer")
        rows.append({"epoch": epoch, "checkpoint": Path(checkpoint).name,
                     "L_abs": l_abs, "eligible": eligible, "lcf_wer": wer})

    scored = [r for r in rows if r["lcf_wer"] is not None and r["eligible"]]
    scored.sort(key=lambda r: r["lcf_wer"])
    print("\n  rank by LCF-WER (lower is better, eligible only):")
    for rank, row in enumerate(scored, 1):
        print(f"    {rank:2d}. epoch {row['epoch']:3d}  {row['lcf_wer']:.3f}")
    winner = scored[0]["epoch"] if scored else None
    print(f"\n  WINNER: epoch {winner}" if winner is not None
          else "\n  no eligible checkpoint was scored")

    (out_dir / "summary.json").write_text(json.dumps(
        {"split": args.split, "condition": args.condition,
         "listener": args.listener, "limit": args.limit,
         "select_abs_max": bar, "config": args.config,
         "winner_epoch": winner, "rows": rows}, indent=2))
    print(f"  wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
