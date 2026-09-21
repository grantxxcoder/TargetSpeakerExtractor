"""D19 control: does iSTFT -> wav -> re-STFT INVENT frequency structure?

    ../tse_venv/bin/python scripts/check_roundtrip_inflation.py --limit 103

WHY. scripts/measure_effective_mask_flatness.py measures every SYSTEM from its
rendered wav (so the numbers pass through iSTFT and a fresh analysis STFT) but
measures the IDEAL mask directly in the frequency domain (no round trip). If the
round trip inflates apparent frequency variation, that asymmetry alone could
produce the 2026-09-15 table -- in which our baseline scored a LOWER volume-knob
share (0.380) than the ideal mask itself (0.505), which is not possible as a
statement about structure.

THE CONTROL. Build the oracle system: apply the ideal magnitude mask to the
mixture, keep the mixture's phase, iSTFT it, re-analyse it exactly as a rendered
estimate is. Its mask is the ideal mask BY CONSTRUCTION, so any difference from
the direct ideal is the round trip and nothing else.

  direct     |S| / |X|                          -- no round trip
  roundtrip  |STFT(iSTFT(M_ideal * X))| / |X|   -- identical mask, one round trip

A modified spectrogram is generally INCONSISTENT: no waveform has exactly that
STFT. iSTFT picks the least-squares waveform and re-analysis projects onto the
consistent set, which perturbs every cell. This measures that perturbation on our
own audio rather than assuming it is small.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.estimates.runner import read_trials, git_commit                  # noqa: E402
from measure_mask_flatness import flatness, read_mono                     # noqa: E402
from measure_effective_mask_flatness import N_FFT, HOP, CLIP_MAX, LOUD_DB  # noqa: E402
from train import SPLIT_MANIFESTS                                          # noqa: E402


def stft(x):
    return torch.stft(torch.from_numpy(np.asarray(x, dtype=np.float32)),
                      n_fft=N_FFT, hop_length=HOP,
                      window=torch.hann_window(N_FFT),
                      center=True, return_complex=True)


def istft(spec, length):
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP,
                       window=torch.hann_window(N_FFT),
                       center=True, length=length).numpy()


def stats(mask, target_mag):
    db = 20 * np.log10(target_mag + 1e-6)
    keep = (db > (db.max() - LOUD_DB)).any(axis=1)
    m = mask[keep] if keep.sum() >= 2 else mask
    share, dt, df = flatness(m)
    return {"share": share, "d_time": dt, "d_freq": df,
            "freq_over_time": df / dt if dt else np.nan}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="sir0", choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--condition", default="both")
    ap.add_argument("--limit", type=int, default=103)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    trials = read_trials(manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
                         audio_root=Path(args.data_root) / "rendered" / val_audio,
                         limit=args.limit, condition=args.condition)

    direct, roundtrip = [], []
    for i, t in enumerate(trials, 1):
        mixture = read_mono(t.directory / "mixture.wav")
        target = read_mono(t.directory / "target.wav")
        n = min(len(mixture), len(target))
        mixture, target = mixture[:n], target[:n]

        X, S = stft(mixture), stft(target)
        f = min(X.shape[0], S.shape[0])
        T = min(X.shape[1], S.shape[1])
        X, S = X[:f, :T], S[:f, :T]
        Xm, Sm = X.abs().numpy().astype(np.float64), S.abs().numpy().astype(np.float64)

        ideal = np.clip(Sm / np.maximum(Xm, 1e-8), 0.0, CLIP_MAX)

        # The oracle SYSTEM: ideal magnitude mask, mixture phase, through iSTFT.
        est_wav = istft(torch.from_numpy(ideal).to(torch.complex64) * X, length=n)
        Em = stft(est_wav).abs().numpy().astype(np.float64)[:f, :T]
        effective = np.clip(Em / np.maximum(Xm, 1e-8), 0.0, CLIP_MAX)

        direct.append(stats(ideal, Sm))
        roundtrip.append(stats(effective, Sm))
        if i % 25 == 0 or i == len(trials):
            print(f"  {i}/{len(trials)}", flush=True)

    def mean(rows, k):
        return float(np.nanmean([r[k] for r in rows]))

    keys = ["share", "freq_over_time", "d_freq", "d_time"]
    summary = {"n_trials": len(trials), "split": args.split,
               "condition": args.condition, "n_fft": N_FFT, "hop": HOP,
               "git_commit": git_commit(), "date": date.today().isoformat(),
               "direct": {k: mean(direct, k) for k in keys},
               "roundtrip": {k: mean(roundtrip, k) for k in keys}}

    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-roundtrip-inflation")
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1)

    lines = ["", f"  THE SAME IDEAL MASK, measured two ways   n={len(trials)}", "",
             f"  {'':26s} {'share':>8s} {'f/t':>8s} {'d_freq':>8s} {'d_time':>8s}",
             "  " + "-" * 62]
    for name, blob in (("direct (no round trip)", summary["direct"]),
                       ("through iSTFT + re-STFT", summary["roundtrip"])):
        lines.append(f"  {name:26s} {blob['share']:8.4f} {blob['freq_over_time']:8.4f} "
                     f"{blob['d_freq']:8.4f} {blob['d_time']:8.4f}")
    d, r = summary["direct"], summary["roundtrip"]
    lines += ["", f"  round trip moves share by {r['share'] - d['share']:+.4f} "
                  f"and f/t by {r['freq_over_time'] - d['freq_over_time']:+.4f}",
              "  Identical mask, so ANY difference here is the round trip alone.", ""]
    print("\n".join(lines))
    (out_root / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
