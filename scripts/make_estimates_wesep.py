"""Write estimate.wav for every trial in a split, using the PRETRAINED WeSep baseline.

MUST BE RUN UNDER THE WESEP VIRTUALENV, not ours:

    ../wesep_venv/bin/python scripts/make_estimates_wesep.py \
        --split sir0 --condition both \
        --pretrain ../wesep_pretrained/tfmap_context_causal_100 \
        --out experiments/results/2026-09-03-est-wesep-tfmap-causal

Why a separate script instead of a flag on scripts/make_estimates.py: the two
systems cannot share an interpreter. Ours needs torch 2.13 / numpy 2.5.2, which
requirements.txt pins because the rendered trials and every VAD figure depend on
those exact versions; WeSep needs torch 2.7.1 and downgrades numpy to 1.26.4.
They meet in src/estimates/runner.py, which is stdlib + numpy + soundfile + yaml
only and is therefore importable from either venv. That shared runner is what
makes this a comparison of two models rather than a comparison of two pipelines.

BORROWED, NOT OURS. This is the REAL-TSE Challenge baseline toolkit
(Wang et al., Interspeech 2024, "WeSep: A Scalable and Flexible Toolkit Towards
Generalizable Target Speaker Extraction"; challenge fork REAL-TSE/wesep-real-tse).
We run their published checkpoint as a system under test on OUR data with OUR
metric. We do not use their evaluation toolkit, and no number produced here is
comparable to any published REAL-TSE result -- different data, different metric,
different protocol.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT. The checkpoint is
OUT OF DOMAIN on our trials, by its own training config, which this script
copies into the provenance of every run so the caveat cannot be separated from
the result:

  * trained on Libri2Mix train-100 CLEAN -- anechoic, whereas our mixtures are
    reverberant (pyroomacoustics shoebox RIRs, T60 0.25-0.6 s)
  * noise_prob 0, reverb_enroll_prob 0, noise_enroll_prob 0 -- no augmentation
    of any kind, whereas our trials carry real recorded noise
  * loss SISDR against Libri2Mix's DRY source, whereas our reference is the full
    reverberant target (A1, decisions-m0.md 2026-08-13). It was trained to
    produce a different signal from the one we score against, so the signal
    metrics penalise it for behaving as trained -- and the content metrics
    largely do not, because dereverberated speech still transcribes.
  * 33.5 M parameters against our 7.19 M, ~14.6 M of which is a VoxCeleb
    ECAPA-TDNN speaker encoder trained jointly, which our model has no
    equivalent of.

So a worse score here is NOT evidence that our model is better. What this row
buys is a system we did not build and did not tune, which is the only way the
divergence question stops resting on one checkpoint (milestones.md M6).
"""

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.estimates.runner import read_trials, write_estimates  # noqa: E402
from src.run_log import timed  # noqa: E402

# The val half of SPLIT_MANIFESTS in scripts/train.py, as (manifest, audio_dir).
# DUPLICATED, and the duplication is deliberate: train.py imports torch and our
# model code, none of which exists in the WeSep venv. tests/test_estimates.py
# asserts this stays identical to train.py's copy, so a divergence fails a test
# rather than silently rendering the wrong audio against the right manifest.
VAL_SPLITS = {
    "smoke": ("smoke_val", "smoke_val"),
    "mid":   ("mid_val",   "val"),
    "sir0":  ("sir0_val",  "sir0_val"),
    "full":  ("val",       "val"),
}


def sha256(path, chunk=1 << 20):
    """Hash the weights. A Google Drive link can be re-uploaded silently, so the
    hash is the only thing tying a result to these exact parameters -- the same
    reason the DNSMOS ONNX models carry theirs."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe_checkpoint(pretrain_dir):
    """Facts about the borrowed model, pulled from its own config and weights.

    Recorded in provenance so the out-of-domain caveat travels with the result
    instead of living only in this docstring.
    """
    pretrain_dir = Path(pretrain_dir)
    config_path, weights_path = pretrain_dir / "config.yaml", pretrain_dir / "avg_model.pt"
    for p in (config_path, weights_path):
        if not p.exists():
            raise SystemExit(f"missing {p}. --pretrain must be a directory holding "
                             f"avg_model.pt and config.yaml, e.g. "
                             f"../wesep_pretrained/tfmap_context_causal_100")

    cfg = yaml.safe_load(open(config_path))
    separator = cfg.get("model_args", {}).get("tse_model", {}).get("separator", {})
    dataset = cfg.get("dataset_args", {})
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = state["models"][0] if isinstance(state.get("models"), list) else state
    n_params = sum(v.numel() for v in weights.values() if hasattr(v, "numel"))

    return {
        "name": pretrain_dir.name,
        "path": str(pretrain_dir),
        "avg_model_sha256": sha256(weights_path),
        "parameters_millions": round(n_params / 1e6, 2),
        "causal": separator.get("causal"),
        "native_resample_rate": dataset.get("resample_rate"),
        # The out-of-domain evidence, quoted from the checkpoint's own config.
        "trained_on": cfg.get("train_data"),
        "training_loss": cfg.get("loss"),
        "noise_prob": dataset.get("noise_prob"),
        "reverb_enroll_prob": dataset.get("reverb_enroll_prob"),
        "num_avg": cfg.get("num_avg"),
        "seed": cfg.get("seed"),
    }


def build_extractor(pretrain_dir, sample_rate, device, output_norm):
    """Load WeSep's Extractor and wrap it in the runner's contract.

    Two settings here are part of the MEASURING INSTRUMENT, not conveniences:

    set_vad(False) -- WeSep can gate on its own Silero VAD (5.1.2) and return
    None for a silent enrollment. Our gate is Silero 6.2.1 (B2), applied
    identically to the judge and the offline ASR by
    src/live_model_metric/speech_gate.py. Enabling WeSep's would bury a second,
    differently-versioned, unlogged gate INSIDE a system under test, where it
    would mute clips before our gate ever saw them and contaminate NRR and the
    absent-trial rows.

    set_output_norm(False) by default -- WeSep otherwise rescales its output to
    0.9 peak. Our estimates are written unnormalised on purpose, so leaving it
    on would hand WeSep a level correction our model does not get. That is a
    difference between pipelines, not between extractors. Overridable, because
    the choice has to be deliberate and logged either way.

    We call extract_speech_from_pcm rather than extract_speech, which also
    sidesteps WeSep's wavform_norm: that flag only affects how IT loads a file,
    and here the runner has already read the audio the same way for both systems.
    """
    import wesep                                            # noqa: PLC0415

    model = wesep.load_model_local(str(pretrain_dir))
    model.set_resample_rate(sample_rate)
    model.set_vad(False)
    model.set_device(device)
    model.set_output_norm(output_norm)

    def extract(mixture, enrollment, sr):
        estimate = model.extract_speech_from_pcm(
            torch.from_numpy(mixture).unsqueeze(0), sr,
            torch.from_numpy(enrollment).unsqueeze(0), sr)
        if estimate is None:
            return None                       # the runner turns this into an error
        return estimate.reshape(-1).cpu().numpy()

    return extract


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(VAL_SPLITS))
    ap.add_argument("--pretrain", required=True,
                    help="a WeSep checkpoint directory holding avg_model.pt and "
                         "config.yaml, e.g. ../wesep_pretrained/tfmap_context_causal_100")
    ap.add_argument("--out", default=None,
                    help="default experiments/results/<today>-est-wesep-<checkpoint>")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--condition", default=None,
                    help="render only this condition, e.g. 'both'. Must MATCH what "
                         "scripts/make_estimates.py was run with -- two systems "
                         "rendered on different subsets are not comparable.")
    ap.add_argument("--limit", type=int, default=None, help="first N trials only")
    ap.add_argument("--sample-rate", type=int, default=16000,
                    help="our data is 16 kHz throughout; the run refuses if the "
                         "checkpoint was trained at a different rate, because then "
                         "a resampler silently joins the instrument")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--output-norm", action="store_true",
                    help="let WeSep rescale its output to 0.9 peak. OFF by default; "
                         "see build_extractor for why, and log it if you turn it on")
    args = ap.parse_args()

    checkpoint = describe_checkpoint(args.pretrain)
    native = checkpoint["native_resample_rate"]
    if native is not None and int(native) != args.sample_rate:
        raise SystemExit(
            f"{checkpoint['name']} was trained at {native} Hz but our audio is "
            f"{args.sample_rate} Hz. Resampling would become part of the measuring "
            f"instrument and has to be a recorded decision, not a default -- pass "
            f"--sample-rate {native} deliberately if that is what you mean.")

    print(f"  checkpoint: {checkpoint['name']}  "
          f"{checkpoint['parameters_millions']} M params  "
          f"causal={checkpoint['causal']}")
    print(f"  trained on: {checkpoint['trained_on']}  loss={checkpoint['training_loss']}")
    print("  OUT OF DOMAIN on our trials -- see this script's docstring before "
          "quoting anything from this run.")

    extract = build_extractor(args.pretrain, args.sample_rate, args.device,
                              args.output_norm)

    manifest, audio_dir = VAL_SPLITS[args.split]
    trials = read_trials(
        manifest_csv=Path(args.manifest_dir) / f"{manifest}.csv",
        audio_root=Path(args.data_root) / "rendered" / audio_dir,
        limit=args.limit,
        condition=args.condition,
    )
    out_root = Path(args.out or f"experiments/results/{date.today().isoformat()}"
                                f"-est-wesep-{checkpoint['name']}")
    written = 0

    with timed("scripts/make_estimates_wesep.py",
               scope=lambda: f"{written} trials, {args.split}, {checkpoint['name']}",
               rate=lambda: f"{args.device}, whole-clip"):
        meta = write_estimates(
            extract=extract,
            trials=trials,
            out_root=out_root,
            sample_rate=args.sample_rate,
            provenance={
                "script": "scripts/make_estimates_wesep.py",
                "system": "wesep-real-tse-pretrained",
                "borrowed_from": "REAL-TSE/wesep-real-tse; Wang et al., "
                                 "Interspeech 2024. Their checkpoint, our data, "
                                 "our metric. NOT comparable to published "
                                 "REAL-TSE numbers.",
                "out_of_domain": "trained on clean anechoic Libri2Mix with no "
                                 "augmentation, against the dry source; our "
                                 "trials are reverberant and noisy and are "
                                 "scored against the reverberant target (A1). "
                                 "A worse score here is not evidence our model "
                                 "is better.",
                "split": args.split,
                "manifest": manifest,
                "condition": args.condition,
                "checkpoint": checkpoint,
                "device": args.device,
                "wesep_vad": False,
                "wesep_output_norm": args.output_norm,
                # str() because torch.__version__ is a TorchVersion, and
                # yaml.safe_dump refuses to represent a str subclass.
                "torch": str(torch.__version__),
            },
        )
        written = meta["n_trials"]


if __name__ == "__main__":
    main()
