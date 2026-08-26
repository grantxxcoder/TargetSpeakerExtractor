#!/usr/bin/env python3
"""Is the speaker cue worth anything the model does not already have for free?

WHY THIS EXISTS
---------------
The model sees the mixture, so it gets LOUDNESS for nothing. Measured
2026-08-25 on `mid`: 90 % of two-speaker trials have the target LOUDER than the
interferer (median +6 dB), because `regimes.base` narrows `sir_db` to [0, 12].
So "keep the loud voice" is right ~90 % of the time and needs no voice sample at
all -- which is very likely why the model never learned to use one.

`docs/data/difficulty-dial.md` (2026-08-13) ranked `sir_db` #1 of 14 dials and
said it outright: "At -5 dB the interferer is louder than the target, so nothing
but the enrollment can identify which voice to keep." That narrowing was taken
as difficulty relief; this script measures what it cost in RELEVANCE.

THE TEST
--------
Among moments where someone is clearly talking, how often does a signal
correctly say the TARGET is the dominant voice? Scored as a hit rate: pick one
target-dominant moment and one interferer-dominant moment, and see which the
signal ranks higher. 50 % is a coin flip.

Two signals are compared, SPLIT BY WHO IS LOUDER:

    the speaker cue     needs the voice sample
    plain loudness      free, the model already has it

On target-louder trials loudness is expected to win, because loudness and target
dominance are the same thing there -- that is the shortcut. The question this
script answers is what happens on the trials where the interferer is LOUDER,
because that is the only population where the voice sample is the sole thing
that can work. If the cue cannot beat loudness even there, widening `sir_db`
will not help and the problem is elsewhere.

USAGE
-----
    python scripts/diagnose_cue_vs_loudness.py --split mid
    python scripts/diagnose_cue_vs_loudness.py --split mid --use train --limit 200
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as Fn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import SPLIT_MANIFESTS, get_data_loaders  # noqa: E402
from src.data.dataset_loader import TrialDataset  # noqa: E402
from src.models.stft import STFT  # noqa: E402


def hit_rate(score, label):
    """P(score ranks a True above a False). Rank-based, so it needs no threshold
    and is unaffected by the signals being on different scales -- loudness is in
    dB and the cue is a cosine, and they must still be comparable."""
    pos, neg = score[label], score[~label]
    if len(pos) < 3 or len(neg) < 3:
        return None
    r = torch.cat([pos, neg]).argsort().argsort().float()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="smoke | mid | full")
    ap.add_argument("--use", default="val", choices=["train", "val"],
                    help="which half of the split to measure (train has more "
                         "interferer-louder trials, which is the population of interest)")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--limit", type=int, default=None, help="stop after N trials")
    ap.add_argument("--scale", type=float, default=None,
                    help="TF-Map logit scale to test (default: the config's)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    scale = args.scale if args.scale is not None else float(cfg["model"]["tfmap_scale"])
    torch.manual_seed(int(cfg["seed"]))
    stft = STFT(cfg["model"]["stft"]["n_fft"], cfg["model"]["stft"]["hop"],
                cfg["data"]["sample_rate"])

    # the split's own loader, so cropping matches training exactly
    manifest, audio_dir = SPLIT_MANIFESTS[args.split][0 if args.use == "train" else 1]
    ds = TrialDataset(manifest_csv=Path(args.manifest_dir) / f"{manifest}.csv",
                      data_root=Path(args.data_root), split=audio_dir,
                      chunk_s=cfg["data"]["chunk_s"],
                      sample_rate=cfg["data"]["sample_rate"],
                      seed=cfg["seed"], random_crop=False)
    print(f"{manifest}.csv, audio from rendered/{audio_dir}, {len(ds)} trials, "
          f"tfmap_scale={scale}")

    # bins by who is the louder voice. -inf..0 is the population that matters.
    BINS = [(-99.0, 0.0, "interferer LOUDER (sir < 0)"),
            (0.0, 6.0, "target louder by 0-6 dB"),
            (6.0, 99.0, "target louder by 6+ dB")]
    acc = {b[2]: {"cue": [], "loud": []} for b in BINS}
    n_used = 0

    with torch.no_grad():
        for i in range(len(ds)):
            if args.limit is not None and n_used >= args.limit:
                break
            it = ds[i]
            if it["meta"]["condition"] != "both" or it["crop_absent"]:
                continue
            x, s_t, e = it["mixture"][None], it["target"][None], it["enrollment"][None]
            if float(s_t.abs().max()) == 0:
                continue
            sir = it["meta"]["sir_db"]
            key = next(b[2] for b in BINS if b[0] <= sir < b[1])

            Xm, Em = stft(x).abs(), stft(e).abs()
            bx, be = Fn.normalize(Xm, p=2, dim=1), Fn.normalize(Em, p=2, dim=1)
            sim = torch.matmul(bx.transpose(1, 2), be)
            h = torch.softmax(sim * scale, dim=-1)
            cue = (h * sim).sum(-1)[0]

            E_t = stft(s_t).abs().pow(2).sum(1)[0]        # target only
            E_o = stft(x - s_t).abs().pow(2).sum(1)[0]    # other talker + noise
            loud = Xm.pow(2).sum(1)[0]

            busy = loud > loud.median()                   # someone is talking
            if int(busy.sum()) < 8:
                continue
            label = (E_t > E_o)[busy]                     # is the TARGET dominant here?
            if int(label.sum()) < 3 or int((~label).sum()) < 3:
                continue

            a_cue, a_loud = hit_rate(cue[busy], label), hit_rate(loud[busy], label)
            if a_cue is None or a_loud is None:
                continue
            acc[key]["cue"].append(a_cue); acc[key]["loud"].append(a_loud)
            n_used += 1
            if n_used % 25 == 0:
                print(f"  {n_used} trials", end="\r", flush=True)

    print(f"\nHow often is the TARGET correctly identified as the dominant voice?")
    print(f"(coin flip = 50.0 %)\n")
    print(f"{'who is louder':<28} {'n':>4} {'speaker cue':>12} {'loudness':>10} {'cue - loud':>11}")
    for _, _, key in BINS:
        a = acc[key]
        if not a["cue"]:
            print(f"{key:<28} {0:>4}   (no usable trials)")
            continue
        c, l = 100 * st.mean(a["cue"]), 100 * st.mean(a["loud"])
        print(f"{key:<28} {len(a['cue']):>4} {c:11.1f}% {l:9.1f}% {c - l:+10.1f}")

    print("\nRead the FIRST row. That is the only population where loudness cannot")
    print("answer the question, so it is the only place the voice sample has to earn")
    print("its keep. If the cue does not clearly beat loudness there, widening")
    print("sir_db will not make the model start using the enrolment.")


if __name__ == "__main__":
    main()
