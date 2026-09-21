"""Render the mix-back family: s_alpha = alpha * estimate + (1 - alpha) * mixture.

    python3 scripts/make_mixback.py --est experiments/results/2026-09-01-est-sir0-5000 \
        --alphas 0,0.25,0.5,0.75,1 --out-root experiments/results/2026-09-21-mixback
    python3 scripts/make_mixback.py --est ... --dry-run

WHY THIS EXISTS AS A SCRIPT. The 2026-09-01 sweep (decisions-m3.md) produced
`sweep_alpha_rows.json` and then vanished -- no runner survived in the repo, so
the result cannot be reproduced or pointed at a different listener. The panel
(decisions-m4.md 2026-09-21) needs exactly this family scored through four
listeners, so the runner has to exist as code.

WHAT IT WRITES, AND WHY THAT SHAPE. One ESTIMATE DIRECTORY per alpha, in the
same layout scripts/make_estimates.py produces: <out>/alpha<a>/<trial_id>/
estimate.wav plus a meta.yaml. That means every alpha is scored by the EXISTING
pipeline with no new scoring code at all:

    python3 scripts/evaluate.py --est <out>/alpha0.25 --listener judge \\
        --judge-model gemini-3.5-transcribe --judge-structured auto

Caching is safe by construction: the judge keys estimate.wav by CONTENT HASH, so
two alphas are two different keys and no answer is ever served for the wrong
blend. decisions-m4.md, judge.py `audio_fingerprint`.

THE BLEND IS PROVABLY LINEAR and that is load-bearing, not decorative. The iSTFT
is linear, so blending waveforms and interpolating masks are the same operation.
Nothing in the sweep is confounded by the blending itself, which is what let the
2026-09-01 run validate itself: SDR, SIR and SAR came out perfectly monotonic in
alpha, exactly as a linear blend predicts (decisions-m3.md 2026-09-01).

alpha = 1 is the model untouched. alpha = 0 is the raw mixture, which IS the
floor anchor -- so the floor comes free inside the family and needs no separate
scoring pass.

decisions-m4.md 2026-09-21, decisions-pending.md D11.
"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True)
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:                                               # noqa: BLE001
        return "unknown"


def parse_alphas(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {value}")
        values.append(value)
    if not values:
        raise ValueError("no alphas given")
    return values


def alpha_tag(value):
    """Directory-safe, and STABLE: 0.5 must never render as both 0.5 and 0.50,
    or two runs of the same sweep would write two different directories and the
    judge would be asked to pay for the same audio twice."""
    return f"{value:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--est", required=True,
                        help="a scripts/make_estimates.py output directory")
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--condition", default="both")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="recorded for provenance; the blend is deterministic")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import soundfile
    import yaml
    from src.live_model_metric.evaluate import load_trials

    alphas = parse_alphas(args.alphas)
    est_dir = Path(args.est)
    if not est_dir.exists():
        raise SystemExit(f"no estimate directory at {est_dir}")
    out_root = Path(args.out_root or
                    REPO_ROOT / "experiments/results" /
                    f"{date.today().isoformat()}-mixback-{est_dir.name}")

    trials = load_trials(args.split, condition=args.condition, limit=args.limit,
                         data_root=args.data_root, manifest_dir=args.manifest_dir,
                         estimate_directory=str(est_dir))
    usable = [t for t in trials
              if t.estimate and t.estimate.exists() and t.mixture.exists()]

    print(f"{len(usable)} trials of {len(trials)} have both estimate and mixture")
    print(f"alphas: {', '.join(alpha_tag(a) for a in alphas)}")
    print(f"out:    {out_root}")
    print(f"writes: {len(usable) * len(alphas)} wav files "
          f"({len(alphas)} directories)")
    if args.dry_run:
        print("\ndry run -- nothing written.")
        return

    commit = git_commit()
    for alpha in alphas:
        alpha_dir = out_root / f"alpha{alpha_tag(alpha)}"
        alpha_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for trial in usable:
            estimate, rate = soundfile.read(trial.estimate, dtype="float32")
            mixture, mix_rate = soundfile.read(trial.mixture, dtype="float32")
            if rate != mix_rate:
                raise SystemExit(f"{trial.trial_id}: estimate {rate} Hz but "
                                 f"mixture {mix_rate} Hz")
            # Lengths can differ by a frame or two after STFT round-tripping.
            # TRUNCATE, never pad: padding invents samples and a pad at the end
            # of a clip is audible to a listener scoring the last word.
            n = min(len(estimate), len(mixture))
            blended = alpha * estimate[:n] + (1.0 - alpha) * mixture[:n]
            # float32, unnormalised, whole clip -- matching make_estimates.py.
            # NOT normalised on purpose: a per-alpha gain change would be a
            # second variable moving alongside alpha.
            out_path = alpha_dir / trial.trial_id / "estimate.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(out_path, blended.astype(np.float32), rate,
                            subtype="FLOAT")
            written += 1
        (alpha_dir / "meta.yaml").write_text(yaml.safe_dump({
            "date": date.today().isoformat(),
            "script": "scripts/make_mixback.py",
            "git_commit": commit,
            "seed": args.seed,
            "alpha": float(alpha),
            "formula": "s_alpha = alpha * estimate + (1 - alpha) * mixture",
            "source_estimates": str(est_dir),
            "source_estimates_meta": str(est_dir / "meta.yaml"),
            "split": args.split,
            "condition": args.condition,
            "n_trials": written,
            "audio": "estimate.wav, float32, unnormalised, whole clip",
            "note": ("alpha=1 is the source model untouched; alpha=0 is the raw "
                     "mixture, i.e. the floor anchor."),
        }, sort_keys=False))
        print(f"  alpha={alpha_tag(alpha)}: {written} files -> {alpha_dir}")

    print(f"\nScore one with:\n"
          f"  python3 scripts/evaluate.py --split {args.split} "
          f"--condition {args.condition} \\\n"
          f"    --est {out_root}/alpha1 --systems estimate --metrics content \\\n"
          f"    --listener judge --judge-model <id> --judge-structured auto")


if __name__ == "__main__":
    main()
