#!/usr/bin/env python3
"""Stage and upload the frozen WavLM Base+ checkpoint as a Kaggle dataset.

WavLM is BORROWED, not ours: Chen et al., 2022, "WavLM: Large-Scale Self-
Supervised Pre-Training for Full Stack Speech Processing", IEEE JSTSP 16(6).
The weights are Microsoft's (unilm, MIT), fetched through torchaudio's
WAVLM_BASE_PLUS pipeline and snapshotted to ../wavlm_pretrained/ by the same
convention as ../wesep_pretrained/ and the DNSMOS ONNX models.

    ../tse_venv/bin/python scripts/upload_kaggle_wavlm.py --dry-run   # verify + stage
    ../tse_venv/bin/python scripts/upload_kaggle_wavlm.py             # then upload

WHY A DATASET AND NOT A RUNTIME DOWNLOAD. A Kaggle session with internet off
cannot reach download.pytorch.org, and internet-on is the setting that gets
throttled mid-run. A dataset mounts read-only at a fixed path and costs nothing
per session -- the same reason the audio and the code travel as datasets
(scripts/make_kaggle_bundle.py).

WHY VERIFY BEFORE UPLOADING. 377 MB is a PROJECTED ~16 min, scaling the
0.40 MB/s that make_kaggle_bundle.py records for the audio upload -- no upload
of this size has been timed yet, and `record()` below writes the real row into
docs/run_times.md the first time it runs. Either way a checkpoint that turns out
not to load is far cheaper to catch here than after the upload plus one Kaggle
session, so this builds the architecture, loads the weights STRICTLY and runs
one forward pass before anything is sent.

WHAT SHIPS ALONGSIDE THE WEIGHTS. load_wavlm.py, and it is not a convenience.
torch renamed weight-norm's stored buffers -- `weight_g`/`weight_v` became
`parametrizations.weight.original0/1` -- so a state dict serialised here under
torch 2.13 fails a strict load against a model built by an older torchaudio,
on exactly two keys out of 199. Kaggle's torch version is not ours to choose.
The loader renames whichever direction is needed; this script verifies THROUGH
that same file, so the thing tested is the thing that ships.

WavLM is a training-side proxy only. It is a different model family from the
live judge, which CLAUDE.md requires, and it must never appear in the judge
path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.run_log import record  # noqa: E402  (needs REPO on the path first)

# What WAVLM_BASE_PLUS actually is. Asserted rather than trusted: a truncated or
# swapped file still loads as *a* state dict, and that failure would surface as
# poor detector features rather than as an error here.
EXPECT_TENSORS = 199
EXPECT_PARAMS_M = 94.38          # 94,381,168 exactly
EXPECT_DIM = 768
EXPECT_LAYERS = 12
# 16,000 samples -> 49 frames. The conv feature extractor's 20 ms stride gives
# 50 Hz minus the receptive-field edge, so pin the count, not the rate.
EXPECT_FRAMES = 49

UPSTREAM_URL = "https://download.pytorch.org/torchaudio/models/wavlm_base_plus.pth"
HUB_CACHE = Path.home() / ".cache/torch/hub/checkpoints/wavlm_base_plus.pth"

# Ships INTO the dataset, so the Kaggle side has one tested way to load these
# weights with no network and no version guessing.
LOADER_PY = '''"""Load the frozen WavLM Base+ weights in this dataset. No network.

WavLM: Chen et al., 2022, IEEE JSTSP 16(6). Weights are Microsoft's (unilm, MIT),
via torchaudio's WAVLM_BASE_PLUS pipeline. Borrowed, not ours.

    import sys; sys.path.insert(0, "/kaggle/input/DATASET")
    from load_wavlm import load
    wavlm = load("/kaggle/input/DATASET/wavlm_base_plus.pt", device="cuda")
    feats, _ = wavlm.extract_features(wav_16k)   # 12 layers, (B, T, 768), 50 Hz

TWO THINGS THIS FILE EXISTS TO GET RIGHT.

1. `torchaudio.models.wavlm_base()`, never `WAVLM_BASE_PLUS.get_model()`. The
   pipeline object downloads its weights, which is the one thing a session with
   internet off cannot do. The architecture is identical -- proved by a strict
   load of all 199 tensors.

2. Weight-norm key names. torch's parametrization API renamed the two buffers
   of the positional conv from `weight_g`/`weight_v` to
   `parametrizations.weight.original0/1`. Which pair a checkpoint carries
   depends on the torch that SAVED it; which pair the model wants depends on the
   torch that BUILT it. Mismatch = a strict-load failure on 2 of 199 keys, and
   loading non-strict instead would leave the positional conv at random init and
   quietly degrade every feature. So: rename, then load strictly.

NOTE the raw waveform goes in unnormalised. Base+ sets
`_normalize_waveform=False`; only WavLM Large layer-norms its input. Normalising
anyway shifts every feature with no error to show for it.
"""

import torch
import torchaudio

LEGACY = ("encoder.transformer.pos_conv_embed.conv.weight_g",
          "encoder.transformer.pos_conv_embed.conv.weight_v")
MODERN = ("encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original0",
          "encoder.transformer.pos_conv_embed.conv.parametrizations.weight.original1")


def align_weight_norm(state_dict, wanted_keys):
    """Return (state_dict, renamed) with the weight-norm buffers named the way
    `wanted_keys` names them. Renames nothing when the two already agree."""
    have, want = set(state_dict), set(wanted_keys)
    if set(LEGACY) <= have and set(MODERN) <= want:
        ren = dict(zip(LEGACY, MODERN))
    elif set(MODERN) <= have and set(LEGACY) <= want:
        ren = dict(zip(MODERN, LEGACY))
    else:
        return state_dict, {}
    return {ren.get(k, k): v for k, v in state_dict.items()}, ren


def load(path, device="cpu", quiet=False):
    """The frozen encoder: eval mode, gradients off, weights loaded strictly."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = torchaudio.models.wavlm_base()
    sd, renamed = align_weight_norm(sd, model.state_dict().keys())
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval().requires_grad_(False)
    if renamed and not quiet:
        print(f"load_wavlm: renamed {len(renamed)} weight-norm buffers to match "
              f"torch {torch.__version__}")
    return model
'''


def sha256(path: Path) -> str:
    """Chunked, because a 377 MB read_bytes() is 377 MB of resident memory for
    no reason."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_staged_loader(out: Path):
    """Import load_wavlm.py FROM THE STAGING DIR, so verification exercises the
    file that ships rather than a copy of its logic living here.

    Bytecode writing is off for the duration: the import would otherwise drop a
    __pycache__ into the staging dir, and every file in that dir gets uploaded.
    A stale .pyc mounted read-only on Kaggle is a confusing thing to ship.
    """
    was = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("load_wavlm",
                                                      out / "load_wavlm.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = was


def verify(ckpt: Path, out: Path) -> dict:
    """Build the model through the staged loader, then run one forward pass.

    torch is imported inside the function so --help and a bad path fail
    instantly rather than after a multi-second CUDA import.
    """
    import torch

    loader = import_staged_loader(out)
    model = loader.load(ckpt)            # strict load; raises on any drift

    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    n_tensors, n_params = len(sd), sum(v.numel() for v in sd.values())

    with torch.no_grad():
        feats, _ = model.extract_features(torch.zeros(1, 16000))

    facts = {
        "tensors": n_tensors,
        "params_m": round(n_params / 1e6, 3),
        "layers": len(feats),
        "frames_per_second_of_audio": feats[-1].shape[1],
        "hidden_dim": feats[-1].shape[2],
        "torch": torch.__version__,
        "torchaudio": __import__("torchaudio").__version__,
        "size_bytes": ckpt.stat().st_size,
        "sha256": sha256(ckpt),
    }

    bad = []
    if n_tensors != EXPECT_TENSORS:
        bad.append(f"{n_tensors} tensors, expected {EXPECT_TENSORS}")
    if abs(facts["params_m"] - EXPECT_PARAMS_M) > 0.01:
        bad.append(f"{facts['params_m']} M params, expected {EXPECT_PARAMS_M} M")
    if facts["layers"] != EXPECT_LAYERS:
        bad.append(f"{facts['layers']} encoder layers, expected {EXPECT_LAYERS}")
    if facts["hidden_dim"] != EXPECT_DIM:
        bad.append(f"dim {facts['hidden_dim']}, expected {EXPECT_DIM}")
    if facts["frames_per_second_of_audio"] != EXPECT_FRAMES:
        bad.append(f"{facts['frames_per_second_of_audio']} frames per second, "
                   f"expected {EXPECT_FRAMES}")
    if bad:
        sys.exit("this is not WavLM Base+ as we know it:\n  " + "\n  ".join(bad)
                 + "\n  Re-snapshot it before uploading.")

    print(f"  verified through load_wavlm.py: {n_tensors} tensors, "
          f"{facts['params_m']} M params, strict load OK")
    print(f"  forward:  1 s of audio -> {facts['layers']} layers x "
          f"{facts['frames_per_second_of_audio']} frames x {facts['hidden_dim']}")
    print(f"  sha256:   {facts['sha256']}")
    return facts


def check_against_hub(ckpt: Path, out: Path, facts: dict) -> None:
    """Tensor-compare the snapshot against torchaudio's own download, when that
    download is still in the hub cache.

    Provenance, not paranoia: the snapshot is a re-serialisation, so its bytes
    and its hash differ from upstream's by construction. Comparing tensors is
    the only way to show the two hold the same weights, and that is the evidence
    that lets PROVENANCE.md name the upstream URL as the source.

    Key names are aligned first, through the staged loader. The two files really
    do disagree there -- upstream is `weight_g`/`weight_v`, ours is the
    parametrization naming -- and a raw key comparison would report weights that
    are in fact identical as a provenance failure.
    """
    if not HUB_CACHE.exists():
        print(f"  (no hub copy at {HUB_CACHE}; skipping the upstream comparison)")
        return
    import torch
    loader = import_staged_loader(out)
    a = torch.load(ckpt, map_location="cpu", weights_only=True)
    b = torch.load(HUB_CACHE, map_location="cpu", weights_only=True)
    b, renamed = loader.align_weight_norm(b, a.keys())
    if a.keys() != b.keys():
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        sys.exit(f"snapshot and {HUB_CACHE} hold different keys -- the snapshot "
                 f"is not this pipeline's weights.\n  only in snapshot: {only_a[:4]}"
                 f"\n  only in hub:      {only_b[:4]}")
    diff = [k for k in a if a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]
    if diff:
        sys.exit(f"{len(diff)} tensors differ from {HUB_CACHE}, first: {diff[0]}")
    facts["upstream_sha256"] = sha256(HUB_CACHE)
    facts["upstream_weight_norm_renamed"] = len(renamed)
    print(f"  upstream: tensor-identical to the torchaudio download"
          + (f" after renaming {len(renamed)} weight-norm buffers" if renamed else "")
          + f" (sha256 {facts['upstream_sha256'][:16]}...)")


def provenance(facts: dict, name: str, slug: str) -> str:
    """The README that travels with the weights. Short on purpose; the load
    snippet is the part that gets used."""
    up = facts.get("upstream_sha256", "not compared (hub cache absent)")
    ds = slug.split("/")[-1]
    return f"""# WavLM Base+ — frozen, for the speaker-state detector

Borrowed weights. Not produced by this project.

| | |
|---|---|
| file | `{name}` |
| source | torchaudio `WAVLM_BASE_PLUS` -> {UPSTREAM_URL} |
| upstream sha256 | `{up}` |
| this file sha256 | `{facts['sha256']}` |
| size | {facts['size_bytes'] / 2**20:.1f} MB |
| params | {facts['params_m']} M in {facts['tensors']} tensors |
| encoder | {facts['layers']} layers x {facts['hidden_dim']} dim |
| sample rate | 16000 Hz |
| frame rate | 50 Hz (20 ms stride); our STFT hop is 8 ms / 125 Hz |
| waveform norm | **off** — `WAVLM_BASE_PLUS._normalize_waveform is False`. Base+ takes the raw waveform; only WavLM Large layer-norms its input. Normalising anyway shifts every feature silently. |
| licence | MIT (Microsoft unilm). Kaggle dataset kept **private**: these are not ours to redistribute. |
| serialised with | torch {facts['torch']}, torchaudio {facts['torchaudio']} |
| staged | {date.today().isoformat()} |

Cite: Chen et al., 2022, *WavLM: Large-Scale Self-Supervised Pre-Training for
Full Stack Speech Processing*, IEEE JSTSP 16(6).

Training-side proxy only. A different model family from the live judge, and
never in the judge path.

## Load it on Kaggle, offline

```python
import sys
sys.path.insert(0, "/kaggle/input/{ds}")
from load_wavlm import load

wavlm = load("/kaggle/input/{ds}/{name}", device="cuda")
feats, _ = wavlm.extract_features(wav_16k)   # {facts['layers']} layers, each (B, T, {facts['hidden_dim']})
```

`load_wavlm.py` builds the architecture with `torchaudio.models.wavlm_base()`
rather than `WAVLM_BASE_PLUS.get_model()`, which downloads, and it renames the
two weight-norm buffers if this box's torch names them differently from the one
that saved the file. Then it loads strictly, so a silent partial load cannot
happen.
"""


def copy_if_changed(src: Path, dst: Path) -> bool:
    """Size-only compare, as in make_kaggle_bundle.py: the checkpoint is
    write-once, and hashing 377 MB on every re-stage buys nothing."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def stage_loader(out: Path) -> None:
    """Written BEFORE verification, because verification imports it."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "load_wavlm.py").write_text(LOADER_PY)


def stage_rest(ckpt: Path, out: Path, slug: str, title: str, licence: str,
               facts: dict) -> None:
    copied = copy_if_changed(ckpt, out / ckpt.name)
    print(f"  {ckpt.name}: {'copied' if copied else 'already current'} "
          f"({ckpt.stat().st_size / 2**20:.0f} MB)")

    # Kaggle reads this and this only. `licenses` is not free text: the API
    # rejects a name outside its own list, hence --license with `other` as the
    # default -- MIT is not one of the names it accepts.
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": slug, "licenses": [{"name": licence}]},
        indent=2) + "\n")

    (out / "PROVENANCE.md").write_text(provenance(facts, ckpt.name, slug))
    # Machine-readable twin, so a later run can assert the hash rather than a
    # human re-reading the markdown.
    (out / "wavlm_base_plus.json").write_text(json.dumps(
        {"source": UPSTREAM_URL, "pipeline": "torchaudio WAVLM_BASE_PLUS",
         "citation": "Chen et al., 2022, WavLM, IEEE JSTSP 16(6)",
         "licence": "MIT (Microsoft unilm)", "staged": date.today().isoformat(),
         **facts}, indent=2, sort_keys=True) + "\n")
    # Belt and braces against the __pycache__ that an earlier version of this
    # script left here: `kaggle datasets create -p` uploads whatever is in the
    # dir, so anything not meant for the dataset has to go.
    for junk in out.glob("__pycache__"):
        shutil.rmtree(junk)

    print("  wrote load_wavlm.py, dataset-metadata.json, PROVENANCE.md, "
          "wavlm_base_plus.json")
    extra = sorted(f.name for f in out.iterdir()
                   if f.name not in {"load_wavlm.py", "dataset-metadata.json",
                                     "PROVENANCE.md", "wavlm_base_plus.json",
                                     ckpt.name})
    if extra:
        sys.exit(f"unexpected files in {out}, they would be uploaded: {extra}")


def kaggle_cli() -> str:
    """The `kaggle` next to the interpreter running this, not whatever is on
    PATH. Run with ../tse_venv/bin/python and you get that venv's CLI, which is
    the one holding kaggle 2.2.4."""
    sibling = Path(sys.executable).parent / "kaggle"
    if sibling.exists():
        return str(sibling)
    found = shutil.which("kaggle")
    if not found:
        sys.exit("no `kaggle` CLI. Run this with ../tse_venv/bin/python, which "
                 "has kaggle installed.")
    return found


def dataset_exists(cli: str, slug: str) -> bool:
    """`create` on an existing slug fails and `version` on a missing one fails,
    so ask first rather than parsing whichever error comes back."""
    r = subprocess.run([cli, "datasets", "status", slug],
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).lower()
    return r.returncode == 0 and "404" not in out and "not found" not in out


def upload(out: Path, slug: str, message: str) -> None:
    cli = kaggle_cli()
    if dataset_exists(cli, slug):
        cmd = [cli, "datasets", "version", "-p", str(out), "-m", message]
        print(f"  {slug} exists -> new version")
    else:
        # PRIVATE by default: --public is deliberately not offered. These are
        # Microsoft's weights, so a public mirror under our account is a
        # redistribution with nothing to gain from it.
        cmd = [cli, "datasets", "create", "-p", str(out)]
        print(f"  {slug} is new -> create (private)")
    print("  " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd)
    wall = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"kaggle exited {r.returncode} after {wall:.0f} s -- nothing "
                 f"else was done. The staged dir is intact; fix and re-run.")
    mb = sum(f.stat().st_size for f in out.iterdir() if f.is_file()) / 2**20
    record("upload_kaggle_wavlm.py", f"{mb:.0f} MB -> {slug}", wall,
           rate=f"{mb / max(wall, 1):.2f} MB/s measured")
    print(f"\nuploaded in {wall / 60:.1f} min ({mb / max(wall, 1):.2f} MB/s). "
          f"Kaggle processes it for a minute or two before it will mount.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=REPO.parent / "wavlm_pretrained/wavlm_base_plus.pt",
                    help="the snapshot to upload (default: ../wavlm_pretrained/)")
    ap.add_argument("--stage", type=Path,
                    default=Path.home() / "kaggle_upload/wavlm",
                    help="staging dir, following ~/kaggle_upload/<name>")
    ap.add_argument("--slug", default="grantbooysen/tse-wavlm-base-plus",
                    help="Kaggle dataset id, <user>/<slug>")
    ap.add_argument("--title", default="tse-wavlm-base-plus")
    ap.add_argument("--license", default="other", dest="licence",
                    help="a name Kaggle accepts. MIT is not one of them; the "
                         "real licence is recorded in PROVENANCE.md")
    ap.add_argument("-m", "--message", default=None,
                    help="version message; used only if the dataset exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify and stage, upload nothing")
    ap.add_argument("--skip-verify", action="store_true",
                    help="escape hatch for a box without torch. Not for a first "
                         "upload -- verification is the point of this script.")
    args = ap.parse_args()

    if not args.ckpt.exists():
        sys.exit(f"missing {args.ckpt}\n"
                 f"  Snapshot it first: the weights come from torchaudio's "
                 f"WAVLM_BASE_PLUS pipeline ({UPSTREAM_URL}).")

    print(f"checkpoint {args.ckpt}")
    print(f"staging -> {args.stage}")
    stage_loader(args.stage)

    if args.skip_verify:
        facts = {"tensors": EXPECT_TENSORS, "params_m": EXPECT_PARAMS_M,
                 "layers": EXPECT_LAYERS, "hidden_dim": EXPECT_DIM,
                 "frames_per_second_of_audio": EXPECT_FRAMES,
                 "torch": "not checked", "torchaudio": "not checked",
                 "size_bytes": args.ckpt.stat().st_size,
                 "sha256": sha256(args.ckpt)}
        print("  --skip-verify: NOT loaded, NOT run. The numbers recorded are "
              "the expected ones, not measured ones.")
    else:
        facts = verify(args.ckpt, args.stage)
        check_against_hub(args.ckpt, args.stage, facts)

    stage_rest(args.ckpt, args.stage, args.slug, args.title, args.licence, facts)

    ds = args.slug.split("/")[-1]
    if args.dry_run:
        print("\n--dry-run: nothing uploaded. When it looks right:")
        print("  ../tse_venv/bin/python scripts/upload_kaggle_wavlm.py")
    else:
        print(f"uploading {args.slug}")
        upload(args.stage, args.slug,
               args.message or f"WavLM Base+ {facts['sha256'][:12]}")

    print(f"\nOn Kaggle: Add Input -> {args.slug}. It mounts at "
          f"/kaggle/input/{ds}/, weights and loader together, so:")
    print(f'  import sys; sys.path.insert(0, "/kaggle/input/{ds}")')
    print(f'  from load_wavlm import load')
    print(f'  wavlm = load("/kaggle/input/{ds}/{args.ckpt.name}", device="cuda")')
    print("\nPin the hash wherever the detector's config lands:")
    print(f"  wavlm:\n    path: /kaggle/input/{ds}/{args.ckpt.name}"
          f"\n    sha256: {facts['sha256']}\n    frame_rate_hz: 50"
          f"\n    normalize_waveform: false")


if __name__ == "__main__":
    main()
