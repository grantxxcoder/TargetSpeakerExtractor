"""Look at the mask as a picture: what the model removes, and what it should.

    ../tse_venv/bin/python scripts/plot_mask_grid.py --trial sir0_val-42-000004

WHY THIS EXISTS. Every mask number in this project is an average over a
spectrogram, and an average cannot tell you WHERE the model is wrong. The
2026-09-12 measurement that our mask is three times SMOOTHER than the ideal one,
while zeroing a third of all bins, is the kind of finding that needs a picture
before anyone can reason about what to do differently.

Five panels, all on the same time and frequency axes:

  mixture         what the model hears
  target          what it is supposed to produce
  ideal mask      |target| / |mixture|, the exactly correct answer, which we can
                  compute because we own both stems
  our mask        what the model actually applied
  over-removal    where the target HAD energy and the model removed it anyway.
                  This is the deletion map -- the panel to look at first.

The ideal mask is clipped to [0, 2]. Bins where the mixture is near silent make
the ratio explode and carry no information, so plotting them unclipped would
flood the panel with meaningless white.

Nothing here trains, writes audio, or touches a checkpoint. Read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                      # noqa: E402
import soundfile as sf                  # noqa: E402
import torch                            # noqa: E402
import torch.nn.functional as F         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train import build_model            # noqa: E402


def capture_mask(model):
    """Patch the Estimator to record the complex mask it applies, per bin."""
    store = {}
    estimator = model.estimator
    original = estimator.forward

    def patched(feats, mix_bands):
        mags = []
        for i, bw in enumerate(estimator.band_widths):
            h = estimator.trunks[i](feats[:, i])
            B, _, T = h.shape
            m = F.glu(estimator.mask_heads[i](h), dim=1).reshape(B, 2, bw, T)
            if estimator.mask_floor > 0.0:
                magnitude = (m[:, 0].pow(2) + m[:, 1].pow(2) + 1e-12).sqrt()
                topped = m[:, 0] + (estimator.mask_floor - magnitude).clamp_min(0.0)
                m = torch.stack([topped, m[:, 1]], dim=1)
            mags.append((m[:, 0].pow(2) + m[:, 1].pow(2)).sqrt())
        store["mask"] = torch.cat(mags, dim=1)
        return original(feats, mix_bands)

    estimator.forward = patched
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/model_sir0_10000-e6.pt")
    ap.add_argument("--trial", default=None, help="default: first target-present trial")
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--data-root", default="data/rendered")
    ap.add_argument("--mask-floor", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=6.0,
                    help="how much of the clip to draw; whole clips are unreadable")
    ap.add_argument("--max-hz", type=float, default=8000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.estimator.mask_floor = args.mask_floor
    store = capture_mask(model)

    root = Path(args.data_root) / args.split
    if args.trial:
        directory = root / args.trial
    else:
        for directory in sorted(root.glob(f"{args.split}-*")):
            audio, _ = sf.read(directory / "target.wav", dtype="float32")
            if np.abs(audio).max() > 1e-4:
                break

    read = lambda name: torch.from_numpy(                       # noqa: E731
        sf.read(directory / name, dtype="float32")[0].astype(np.float32))
    mixture, target, enrolment = read("mixture.wav"), read("target.wav"), read("enrollment.wav")

    with torch.no_grad():
        model(mixture[None], enrolment[None])
        mask = store["mask"][0].numpy()
        X = model.stft(mixture[None])[0].abs().numpy()
        S = model.stft(target[None])[0].abs().numpy()

    sample_rate = checkpoint["config"]["data"]["sample_rate"]
    hop = checkpoint["config"]["model"]["stft"]["hop"]
    frames = min(mask.shape[1], int(args.seconds * sample_rate / hop))
    bins = int(args.max_hz / (sample_rate / 2) * X.shape[0])
    crop = lambda A: A[:bins, :frames]                          # noqa: E731
    X, S, mask = crop(X), crop(S), crop(mask)

    ideal = np.clip(S / np.maximum(X, 1e-8), 0.0, 2.0)
    db = lambda A: 20 * np.log10(A + 1e-6)                      # noqa: E731
    # Where the target HAD energy and the mask removed it. Restricted to bins
    # carrying real target energy, or the panel is dominated by silence.
    loud = db(S) > (db(S).max() - 50)
    over_removal = np.where(loud & (mask < 0.1), 1.0, 0.0)

    extent = [0, frames * hop / sample_rate, 0, args.max_hz / 1000]
    panels = [
        ("mixture (what it hears)", db(X), dict(cmap="magma", vmin=db(X).max()-70, vmax=db(X).max())),
        ("target (what it should produce)", db(S), dict(cmap="magma", vmin=db(X).max()-70, vmax=db(X).max())),
        ("IDEAL mask", ideal, dict(cmap="viridis", vmin=0, vmax=1.5)),
        (f"OUR mask (floor {args.mask_floor:g})", mask, dict(cmap="viridis", vmin=0, vmax=1.5)),
        ("OVER-REMOVAL: target was here, mask killed it", over_removal,
         dict(cmap="Reds", vmin=0, vmax=1)),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.1 * len(panels)), sharex=True)
    for ax, (title, image, style) in zip(axes, panels):
        ax.imshow(image, origin="lower", aspect="auto", extent=extent, **style)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("kHz", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("seconds", fontsize=8)
    fig.suptitle(f"{directory.name}   —   {Path(args.checkpoint).name}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out = Path(args.out or f"experiments/results/mask_grid_{directory.name}"
                           f"_floor{args.mask_floor:g}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"\nbins the mask kills (<0.1): {100*(mask<0.1).mean():.1f} %")
    print(f"of those, ones where the TARGET had real energy: "
          f"{100*over_removal.sum()/max((mask<0.1).sum(),1):.1f} %")
    print(f"mask roughness  ours {np.abs(np.diff(mask,axis=1)).mean():.4f}   "
          f"ideal {np.abs(np.diff(ideal,axis=1)).mean():.4f}")


if __name__ == "__main__":
    main()
