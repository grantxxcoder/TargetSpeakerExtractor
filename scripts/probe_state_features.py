"""Can a probe read speaker state off the separator's own features? D14 head A.

    ../tse_venv/bin/python scripts/probe_state_features.py \
        --checkpoint models/model_sir0_10000-e6.pt --limit 100

WHY THIS EXISTS, AND WHY IT COMES BEFORE THE ARM. Head A adds a classifier on
the separator's features and trains it with cross-entropy, hoping to push
speaker-state information INTO those features. That is a ~10 h training arm.
But the prior question is free: do the features of the ALREADY-TRAINED baseline
encode state? If a 516-parameter linear probe reads it off, the information is
there and head A's auxiliary loss may be unnecessary -- go straight to the gate.
If nothing reads it, we learn what head A has to fix, and how much capacity it
needs, before spending a session.

WHAT IT ANSWERS, all on ONE feature extraction:
  1. how much state is linearly decodable from the mean-pooled features
     (exactly D14's specified head: 516 params)
  2. how much is lost by mean-pooling over the 32 bands (a band-resolved probe)
  3. whether nonlinearity buys anything (an MLP of the same order)
  4. D14's control: ablate the enrolment. If accuracy holds without the cue,
     the probe is a voice-activity detector and the "who" half is unearned.
  5. D14's control: same- vs cross-gender. The extractor already leans on
     gender (56.1 % vs 44.4 %, 2026-08-30) and a frame classifier is an easier
     place to hide it.

THE FOUR STATES come from src/data/state_labels.py: target is bit 0, interferer
bit 1, so 0 none / 1 target / 2 interferer / 3 both.

Probes are fit on a TRIAL-DISJOINT split. Frames within a trial are massively
correlated, so a frame-level split would report memorisation as accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.state_labels import states_at_frames  # noqa: E402
from train import build_model  # noqa: E402

N_STATES = 4
STATE_NAMES = ["none", "target only", "interferer only", "both"]


def parse_spans(text):
    """'0.34:8.55|8.60:12.46' -> [(0.34, 8.55), (8.60, 12.46)]"""
    if not text:
        return []
    out = []
    for chunk in text.split("|"):
        a, b = chunk.split(":")
        out.append((float(a), float(b)))
    return out


def load_labels(path):
    with open(path, newline="") as handle:
        return {r["trial_id"]: (parse_spans(r["target_spans"]),
                                parse_spans(r["interferer_spans"]))
                for r in csv.DictReader(handle)}


def load_manifest(path):
    with open(path, newline="") as handle:
        return {r["trial_id"]: r for r in csv.DictReader(handle)}


def read_wav(path, sample_rate):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    assert sr == sample_rate, f"{path} is {sr} Hz, expected {sample_rate}"
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


@torch.no_grad()
def features_for(model, mixture, enrolment, capture):
    """One forward; the hook leaves z in `capture`. -> (T, K, N) float32."""
    capture.clear()
    model(mixture[None], enrolment[None])
    z = capture["z"][0]                       # (K, N, T)
    return z.permute(2, 0, 1).contiguous()    # (T, K, N)


def collect(args, model, capture, labels, manifest, trial_dirs, hop_s):
    """Walk the trials once, returning per-trial features and labels."""
    rows = []
    for count, directory in enumerate(trial_dirs, 1):
        trial_id = directory.name
        if trial_id not in labels:
            continue
        target_spans, interferer_spans = labels[trial_id]
        mixture = read_wav(directory / "mixture.wav", args.sample_rate)
        if args.max_seconds:
            mixture = mixture[:int(args.max_seconds * args.sample_rate)]
        same_gender = manifest.get(trial_id, {}).get("same_gender", "")

        for which in ("target", "interferer"):
            cue = ("enrollment.wav" if which == "target"
                   else "interferer_enrollment.wav")
            if not (directory / cue).exists():
                continue
            enrolment = read_wav(directory / cue, args.sample_rate)
            # Roles swap with the direction: the SPANS swap with them, which is
            # the whole reason the state encoding was made order-independent.
            mine, theirs = ((target_spans, interferer_spans) if which == "target"
                            else (interferer_spans, target_spans))

            z = features_for(model, mixture, enrolment, capture)
            n_frames = z.shape[0]
            # The STFT pads by (n_fft - hop) and uses center=False, so frame f is
            # centred one hop EARLIER than f*hop. offset_s = -hop_s realigns it.
            states = states_at_frames(mine, theirs, n_frames, hop_s,
                                      offset_s=-hop_s)
            row = {"trial_id": trial_id, "direction": which,
                   "same_gender": same_gender,
                   "mean": z.mean(dim=1).numpy().astype(np.float32),
                   "states": states.astype(np.int64)}
            if args.band_stride > 0:
                row["bands"] = (z[::args.band_stride]
                                .reshape(-1, z.shape[1] * z.shape[2])
                                .numpy().astype(np.float16))
                row["states_bands"] = states[::args.band_stride].astype(np.int64)

            if args.ablate_enrolment:
                # D14 control 1: a DIFFERENT speaker's cue, same mixture. If the
                # probe still works, it never needed the cue.
                other = read_wav(directory / ("interferer_enrollment.wav"
                                 if which == "target" else "enrollment.wav"),
                                 args.sample_rate)
                za = features_for(model, mixture, other, capture)
                row["mean_ablated"] = za.mean(dim=1).numpy().astype(np.float32)
            rows.append(row)

        if count % 10 == 0:
            print(f"  {count}/{len(trial_dirs)} trials", flush=True)
    return rows


def fit_probe(x_fit, y_fit, x_test, y_test, make_module, steps=400, seed=42):
    """Class-weighted probe. Returns (balanced accuracy, per-class recall)."""
    torch.manual_seed(seed)
    module = make_module()
    counts = np.bincount(y_fit, minlength=N_STATES).astype(np.float64)
    weight = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0),
                          dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.Adam(module.parameters(), lr=1e-2)

    xf = torch.from_numpy(x_fit).float()
    yf = torch.from_numpy(y_fit)
    for step in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(module(xf), yf)
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        pred = module(torch.from_numpy(x_test).float()).argmax(dim=1).numpy()
    recalls = []
    for state in range(N_STATES):
        mask = y_test == state
        recalls.append(float((pred[mask] == state).mean()) if mask.any() else float("nan"))
    present = [r for r in recalls if r == r]
    return (float(np.mean(present)) if present else float("nan")), recalls, pred


def stack(rows, key_x, key_y):
    return (np.concatenate([r[key_x] for r in rows]),
            np.concatenate([r[key_y] for r in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/model_sir0_10000-e6.pt")
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--band-stride", type=int, default=8,
                    help="keep every Nth frame for the band-resolved probe; "
                         "the full tensor is 32x wider and will not fit otherwise")
    ap.add_argument("--ablate-enrolment", action="store_true", default=True)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--max-seconds", type=float, default=8.0,
                    help="truncate each mixture. CPU forward is ~0.39x realtime, "
                         "so whole 20 s clips cost 51 min against 18 for 8 s, and "
                         "8 s already yields ~1000 frames per example.")
    ap.add_argument("--fit-fraction", type=float, default=0.7)
    ap.add_argument("--out", default="experiments/results/2026-09-11-state-probe")
    args = ap.parse_args()

    labels_path = args.labels or f"data/index/state_{args.split}.csv"
    labels = load_labels(labels_path)
    manifest = load_manifest(f"data/manifests/{args.split}.csv")

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    hop_s = float(ckpt["config"]["model"]["stft"]["hop"]) / args.sample_rate
    print(f"checkpoint {args.checkpoint}  (epoch {ckpt.get('epoch')})   hop {hop_s*1000:.0f} ms")

    capture = {}
    model.separator.register_forward_hook(
        lambda module, inputs, output: capture.__setitem__("z", output.detach()))

    root = Path(args.data_root) / "rendered" / args.split
    trial_dirs = sorted(d for d in root.iterdir() if (d / "mixture.wav").exists())
    if args.limit:
        trial_dirs = trial_dirs[:args.limit]
    print(f"{len(trial_dirs)} trials, both directions\n")

    rows = collect(args, model, capture, labels, manifest, trial_dirs, hop_s)
    print(f"\ncollected {len(rows)} examples")

    ids = sorted({r["trial_id"] for r in rows})
    cut = int(len(ids) * args.fit_fraction)
    fit_ids = set(ids[:cut])
    fit = [r for r in rows if r["trial_id"] in fit_ids]
    test = [r for r in rows if r["trial_id"] not in fit_ids]
    print(f"trial-disjoint split: {len(fit_ids)} fit trials, {len(ids)-len(fit_ids)} test\n")

    xf, yf = stack(fit, "mean", "states")
    xt, yt = stack(test, "mean", "states")
    hist = np.bincount(yt, minlength=N_STATES)
    print("test-set frame counts: " + "  ".join(
        f"{STATE_NAMES[i]} {hist[i]}" for i in range(N_STATES)))
    print(f"chance (balanced) = {100/N_STATES:.1f} %\n")

    results = {}
    variants = [
        ("linear, mean-pooled  (D14's head, 516 params)",
         xf, yf, xt, yt, lambda: nn.Linear(128, N_STATES)),
        ("MLP,    mean-pooled  (hidden 128, 17k params)",
         xf, yf, xt, yt, lambda: nn.Sequential(nn.Linear(128, 128), nn.GELU(),
                                               nn.Linear(128, N_STATES))),
    ]
    if args.band_stride > 0:
        bf, byf = stack(fit, "bands", "states_bands")
        bt, byt = stack(test, "bands", "states_bands")
        variants.append((f"linear, band-resolved ({bf.shape[1]*N_STATES+N_STATES} params)",
                         bf, byf, bt, byt,
                         lambda d=bf.shape[1]: nn.Linear(d, N_STATES)))
    if args.ablate_enrolment:
        af, _ = stack(fit, "mean_ablated", "states")
        at, _ = stack(test, "mean_ablated", "states")
        variants.append(("linear, mean-pooled, WRONG enrolment (control)",
                         af, yf, at, yt, lambda: nn.Linear(128, N_STATES)))

    print(f"{'probe':<50}{'bal acc':>9}   per-class recall")
    print("-" * 100)
    for name, a, b, c, d, make in variants:
        acc, recalls, pred = fit_probe(a, b, c, d, make)
        results[name] = {"balanced_accuracy": acc, "per_class_recall": recalls}
        rs = "  ".join(f"{STATE_NAMES[i][:4]} {recalls[i]*100:5.1f}" for i in range(N_STATES))
        print(f"{name:<50}{acc*100:8.1f} %   {rs}")
        if name.startswith("linear, mean-pooled  ("):
            base_pred = pred

    # D14 control 2: same- vs cross-gender, on the specified head.
    print()
    gender = np.concatenate([np.full(len(r["states"]),
                                     1 if str(r["same_gender"]) in ("1", "1.0") else 0)
                             for r in test])
    for flag, label in [(1, "same gender"), (0, "different gender")]:
        mask = gender == flag
        if mask.sum() == 0:
            continue
        rec = []
        for state in range(N_STATES):
            sub = mask & (yt == state)
            if sub.any():
                rec.append(float((base_pred[sub] == state).mean()))
        print(f"  {label:<20} balanced acc {np.mean(rec)*100:5.1f} %   "
              f"({int(mask.sum())} frames)")
        results[f"gender:{label}"] = float(np.mean(rec))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
