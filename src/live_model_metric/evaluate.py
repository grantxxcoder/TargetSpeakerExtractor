"""Score every metric this project has, on one split, in one call.

Designed for both a terminal and a notebook. The terminal wrapper is
scripts/evaluate.py; from a notebook:

    from src.live_model_metric.evaluate import evaluate
    results = evaluate(split="sir0_val", estimate_directory="experiments/results/...")
    print(results.table())
    results.frame()                      # pandas, if you want to plot it

Three metric families, each optional so a quick run stays quick:

    content      LCF-WER + its error breakdown, ICR@k, NRR      seconds (cached)
    signal       SDR / SIR / SAR                                ~1 min
    perceptual   DNSMOS P.808 and P.835                         ~5 min per system

WHAT THIS DOES NOT DO. It does not render estimates -- that is
scripts/make_estimates.py and takes ~25 minutes, which would make a "quick" call
a surprise. If the estimate directory is missing it says exactly what to run.
"""

import csv
import json
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

CONTENT, SIGNAL, PERCEPTUAL = "content", "signal", "perceptual"
ALL_METRICS = (CONTENT, SIGNAL, PERCEPTUAL)
ALL_SYSTEMS = ("floor", "estimate", "ceiling")

TRANSCRIPT_CACHE = Path("experiments/results/transcripts.csv")
ASR_MODEL_SIZE = "small.en"

# This file lives at <repo>/src/live_model_metric/evaluate.py, so the repo root is
# two directories up. Deriving it from the module rather than the working
# directory means a notebook opened anywhere resolves the same paths as the CLI.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path, root=None):
    """Relative paths hang off the repo root, absolute paths are left alone."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else Path(root or REPO_ROOT) / path


def _cache_key(audio_path, model_size=ASR_MODEL_SIZE):
    stat = Path(audio_path).stat()
    return (f"{model_size}|{Path(audio_path).parent.name}|{Path(audio_path).name}"
            f"|{int(stat.st_mtime)}|{stat.st_size}")


def _load_cache(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    with open(cache_path, newline="") as handle:
        return {row["key"]: row["text"] for row in csv.DictReader(handle)}


def _git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


@dataclass
class Trial:
    trial_id: str
    condition: str
    target_text: str
    interferer_text: str
    mixture: Path
    clean: Path
    interferer: Path
    estimate: Path = None
    signal_to_interference_db: float = None


def load_trials(split, condition="both", limit=None,
                data_root="data", manifest_dir="data/manifests",
                estimate_directory=None, repo_root=None):
    manifest = _resolve(manifest_dir, repo_root) / f"{split}.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"no manifest at {manifest}. Repo root resolved to "
            f"{_resolve('.', repo_root)}; pass repo_root= if that is wrong.")
    audio_root = _resolve(data_root, repo_root) / "rendered" / split
    estimate_root = _resolve(estimate_directory, repo_root)

    trials = []
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(handle):
            if condition and row["condition"] != condition:
                continue
            trial_directory = audio_root / row["trial_id"]
            if not (trial_directory / "meta.json").exists():
                continue
            meta = json.loads((trial_directory / "meta.json").read_text())
            trials.append(Trial(
                trial_id=row["trial_id"],
                condition=row["condition"],
                target_text=meta.get("target_text", ""),
                interferer_text=meta.get("interferer_text", ""),
                mixture=trial_directory / "mixture.wav",
                clean=trial_directory / "target.wav",
                interferer=trial_directory / "interferer.wav",
                estimate=(estimate_root / row["trial_id"] / "estimate.wav"
                          if estimate_root else None),
                signal_to_interference_db=float(row["sir_db"]) if row.get("sir_db") else None,
            ))
            if limit is not None and len(trials) >= limit:
                break
    return trials


def transcribe(audio_paths, cache_path=TRANSCRIPT_CACHE, model_size=ASR_MODEL_SIZE,
               allow_new=True, verbose=True, repo_root=None):
    """Transcribe, serving what the cache already has.

    `allow_new=False` refuses to run the ASR and raises instead, which is what a
    quick call wants: a silent 10-minute transcription pass is worse than an error.
    """
    cache_path = _resolve(cache_path, repo_root)
    cache = _load_cache(cache_path)
    texts, missing = [], []
    for path in audio_paths:
        if path is None or not Path(path).exists():
            texts.append(None)
            continue
        key = _cache_key(path, model_size)
        if key in cache:
            texts.append(cache[key])
        else:
            texts.append(None)
            missing.append((len(texts) - 1, Path(path), key))

    if not missing:
        return texts
    if not allow_new:
        raise RuntimeError(
            f"{len(missing)} clips are not transcribed and allow_new=False. "
            f"First: {missing[0][1]}. Re-run with allow_new=True to transcribe "
            f"them (~3 s per clip on CPU).")

    from faster_whisper import WhisperModel
    if verbose:
        print(f"  transcribing {len(missing)} clips not in cache "
              f"(~{len(missing) * 3 / 60:.0f} min)", flush=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    new_rows = []
    for count, (index, path, key) in enumerate(missing, 1):
        segments, _ = model.transcribe(str(path), language="en", beam_size=1,
                                       temperature=0.0, condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        texts[index] = text
        new_rows.append({"key": key, "model": model_size,
                         "trial_id": path.parent.name, "file": path.name, "text": text})
        if verbose and count % 25 == 0:
            print(f"    {count}/{len(missing)}", flush=True)

    exists = cache_path.exists()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "model", "trial_id", "file", "text"])
        if not exists:
            writer.writeheader()
        writer.writerows(new_rows)
    return texts


@dataclass
class Results:
    split: str
    condition: str
    n_trials: int
    systems: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def table(self):
        rows, columns = [], []
        for system, scores in self.systems.items():
            for key in scores:
                if key not in columns:
                    columns.append(key)
        header = f"{'system':<11}" + "".join(f"{c:>15}" for c in columns)
        rows.append(header)
        rows.append("-" * len(header))
        for system, scores in self.systems.items():
            line = f"{system:<11}"
            for column in columns:
                value = scores.get(column)
                line += f"{'—':>15}" if value is None else f"{value:>15.2f}"
            rows.append(line)
        return "\n".join(rows)

    def frame(self):
        import pandas as pd
        return pd.DataFrame(self.systems).T

    def save(self, out_directory, repo_root=None):
        out_directory = _resolve(out_directory, repo_root)
        out_directory.mkdir(parents=True, exist_ok=True)
        (out_directory / "results.json").write_text(json.dumps(
            {"split": self.split, "condition": self.condition,
             "n_trials": self.n_trials, "systems": self.systems,
             "provenance": self.provenance}, indent=2))
        (out_directory / "results.txt").write_text(self.table() + "\n")
        return out_directory


def evaluate(split="sir0_val", condition="both", estimate_directory=None,
             systems=ALL_SYSTEMS, metrics=ALL_METRICS, limit=None,
             data_root="data", manifest_dir="data/manifests",
             cache_path=TRANSCRIPT_CACHE, allow_new_transcripts=True,
             verbose=True, repo_root=None):
    """Score `systems` on `metrics` for one split. Returns `Results`.

    Relative paths resolve against the repo root, which is derived from this
    file's own location -- so the working directory does not matter and a
    notebook needs no `..` prefixes. Pass `repo_root` to override.
    """
    metrics = tuple(metrics)
    systems = tuple(systems)
    trials = load_trials(split, condition, limit, data_root, manifest_dir,
                         estimate_directory, repo_root)
    if not trials:
        raise RuntimeError(f"no trials loaded for split={split} condition={condition}")

    if "estimate" in systems:
        if estimate_directory is None:
            raise RuntimeError(
                "systems includes 'estimate' but estimate_directory is None. "
                "Render estimates first:\n"
                "  python scripts/make_estimates.py --split sir0 "
                "--checkpoint models/<ckpt>.pt --out experiments/results/<name>")
        missing = [t.trial_id for t in trials if not t.estimate.exists()]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(trials)} trials have no estimate.wav in "
                f"{estimate_directory} (first: {missing[0]}). Render them with "
                f"scripts/make_estimates.py")

    source_of = {"floor": lambda t: t.mixture,
                 "estimate": lambda t: t.estimate,
                 "ceiling": lambda t: t.clean}

    results = Results(split=split, condition=condition, n_trials=len(trials))
    results.provenance = {
        "date": date.today().isoformat(),
        "git_commit": _git_commit(),
        "split": split, "condition": condition, "n_trials": len(trials),
        "systems": list(systems), "metrics": list(metrics),
        "estimate_directory": str(estimate_directory) if estimate_directory else None,
        "listener": f"faster-whisper {ASR_MODEL_SIZE} int8 cpu greedy",
        "listener_role": "STAND-IN for the judge, NOT a live-model result",
    }

    for system in systems:
        if verbose:
            print(f"\n=== {system} ===", flush=True)
        scores = {}
        paths = [source_of[system](t) for t in trials]

        if CONTENT in metrics:
            from .lcf_wer import compute_lcf_wer
            from .icr import compute_icr
            from .nrr import compute_nrr
            responses = transcribe(paths, cache_path,
                                   allow_new=allow_new_transcripts,
                                   verbose=verbose, repo_root=repo_root)
            targets = [t.target_text for t in trials]
            interferers = [t.interferer_text for t in trials]
            word = compute_lcf_wer(targets, responses)
            leak = compute_icr(responses, targets, interferers)
            quiet = compute_nrr(responses, targets)
            scores.update(lcf_wer=word.word_error_rate,
                          substitutions=word.substitution_rate,
                          deletions=word.deletion_rate,
                          insertions=word.insertion_rate,
                          icr_at_2=leak.headline,
                          mean_leak=leak.mean_leaked_percent,
                          nrr=quiet.nrr)

        if SIGNAL in metrics:
            import soundfile
            from .separation import decompose
            sdr, sir, sar = [], [], []
            for trial, path in zip(trials, paths):
                target, _ = soundfile.read(trial.clean)
                interferer, _ = soundfile.read(trial.interferer)
                mixture, _ = soundfile.read(trial.mixture)
                estimate, _ = soundfile.read(path)
                n = min(len(target), len(interferer), len(mixture))
                noise = mixture[:n] - target[:n] - interferer[:n]
                s = decompose(estimate, target, interferer, noise)
                sdr.append(s.signal_to_distortion_db)
                sir.append(s.signal_to_interference_db)
                sar.append(s.signal_to_artefact_db)
            scores.update(sdr=sum(sdr)/len(sdr), sir=sum(sir)/len(sir),
                          sar=sum(sar)/len(sar))

        if PERCEPTUAL in metrics:
            from .dnsmos import score_audio_file, mean_scores
            if verbose:
                print(f"  DNSMOS on {len(paths)} clips (~{len(paths)*2.8/60:.0f} min)",
                      flush=True)
            mean = mean_scores([score_audio_file(p) for p in paths])
            scores.update(p808=mean.p808_mos, sig=mean.sig,
                          bak=mean.bak, ovrl=mean.ovrl)

        results.systems[system] = scores
        if verbose:
            print("  " + "  ".join(
                f"{k}={v:.2f}" for k, v in scores.items() if v is not None), flush=True)

    return results
