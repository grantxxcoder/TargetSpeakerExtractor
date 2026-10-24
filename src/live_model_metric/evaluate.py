"""Score the reported metrics on one split in one call. scripts/evaluate.py is the command-line wrapper.

Three metric families, each optional so a quick run stays quick:

    content      LCF-WER and its error breakdown, ICR@2, FR@2      seconds when cached
    signal       SDR, SIR and SAR                                 about 1 minute
    perceptual   DNSMOS P.808 and P.835                           about 5 minutes per system

It does not render estimates, which is scripts/make_estimates.py. Trials and results are plain dictionaries.
"""

import csv
import json
from datetime import date
from importlib.metadata import version
from pathlib import Path

import soundfile
from faster_whisper import WhisperModel

from .dnsmos import mean_scores, score_audio_file
from .fabrication import compute_fr
from .icr import compute_icr
from .judge import BACKEND, DEFAULT_CACHE, MODEL_ID, PROMPT_FILE, MissingCredentials, QuotaExhausted, judge_transcribe, make_judge, prompt_sha
from .lcf_wer import compute_lcf_wer
from .separation import decompose
from .speech_gate import condition_lookup, decide, log_decision, vad_seconds_fn

CONTENT, SIGNAL, PERCEPTUAL = "content", "signal", "perceptual"
ALL_METRICS = (CONTENT, SIGNAL, PERCEPTUAL)
ALL_SYSTEMS = ("floor", "estimate", "ceiling")
ASR, JUDGE = "asr", "judge"
ALL_LISTENERS = (ASR, JUDGE)

# The default listener, and the in-loop probe's. The report's tables use large-v3-turbo through experiments/configs/eval_offline_asr_turbo.yaml (decisions-m3.md 2026-09-28).
ASR_MODEL_SIZE = "small.en"
# Every reported judge run was paced at 10 requests a minute (docs/run_times.md).
JUDGE_REQUESTS_PER_MINUTE = 10

# Paths are resolved from the repo root, worked out from this file's own location, so the working directory does not matter.
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data/manifests"
RENDERED_DIR = REPO_ROOT / "data/rendered"
TRANSCRIPT_CACHE = REPO_ROOT / "experiments/results/transcripts.csv"


# this makes a relative path hang off the repo root and leaves an absolute path alone
def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


# this gives the transcript cache key for a clip. It includes the file's timestamp and size, so a rewritten clip is transcribed again
def transcript_cache_key(audio_path, model_size):
    stat = Path(audio_path).stat()
    return f"{model_size}|{Path(audio_path).parent.name}|{Path(audio_path).name}|{int(stat.st_mtime)}|{stat.st_size}"


# this reads the transcript cache as key to text. A missing file is an empty cache
def load_transcript_cache():
    if not TRANSCRIPT_CACHE.exists():
        return {}
    cache = {}
    with open(TRANSCRIPT_CACHE, newline="") as handle:
        for row in csv.DictReader(handle):
            cache[row["key"]] = row["text"]
    return cache


# this reads one dictionary per trial from the split manifest: its id, condition, both scripts, the paths of its clips, and its SIR
def load_trials(split, condition, estimate_directory):
    manifest = MANIFEST_DIR / f"{split}.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"No manifest at {manifest}.")
    audio_root = RENDERED_DIR / split
    estimate_root = resolve_path(estimate_directory)

    trials = []
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(handle):
            if condition and row["condition"] != condition:
                continue
            trial_directory = audio_root / row["trial_id"]
            if not (trial_directory / "meta.json").exists():
                continue
            meta = json.loads((trial_directory / "meta.json").read_text())
            estimate = None
            if estimate_root:
                estimate = estimate_root / row["trial_id"] / "estimate.wav"
            signal_to_interference_db = None
            if row.get("sir_db"):
                signal_to_interference_db = float(row["sir_db"])
            trials.append({"trial_id": row["trial_id"], "condition": row["condition"], "target_text": meta.get("target_text", ""), "interferer_text": meta.get("interferer_text", ""), "mixture": trial_directory / "mixture.wav", "clean": trial_directory / "target.wav", "interferer": trial_directory / "interferer.wav", "estimate": estimate, "signal_to_interference_db": signal_to_interference_db})
    return trials


# this transcribes clips with the offline ASR, serving what the cache already holds and appending what it had to transcribe
def transcribe(audio_paths, model_size):
    cache = load_transcript_cache()
    texts = []
    missing = []
    for path in audio_paths:
        if path is None or not Path(path).exists():
            texts.append(None)
            continue
        key = transcript_cache_key(path, model_size)
        if key in cache:
            texts.append(cache[key])
        else:
            texts.append(None)
            missing.append((len(texts) - 1, Path(path), key))

    if not missing:
        return texts

    print(f"  transcribing {len(missing)} clips not in cache with {model_size}", flush=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    new_rows = []
    for count, (index, path, key) in enumerate(missing, 1):
        segments, info = model.transcribe(str(path), language="en", beam_size=1, temperature=0.0, condition_on_previous_text=False)
        segment_texts = []
        for segment in segments:
            segment_texts.append(segment.text.strip())
        text = " ".join(segment_texts).strip()
        texts[index] = text
        new_rows.append({"key": key, "model": model_size, "trial_id": path.parent.name, "file": path.name, "text": text})
        if count % 25 == 0:
            print(f"    {count}/{len(missing)}", flush=True)

    exists = TRANSCRIPT_CACHE.exists()
    TRANSCRIPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_CACHE, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "model", "trial_id", "file", "text"])
        if not exists:
            writer.writeheader()
        writer.writerows(new_rows)
    return texts


# Generated by Claude Code
def results_table(results):
    rows, columns = [], []
    for system, scores in results["systems"].items():
        for key in scores:
            if key not in columns:
                columns.append(key)
    header = f"{'system':<11}" + "".join(f"{c:>15}" for c in columns)
    rows.append(header)
    rows.append("-" * len(header))
    for system, scores in results["systems"].items():
        line = f"{system:<11}"
        for column in columns:
            value = scores.get(column)
            line += f"{'—':>15}" if value is None else f"{value:>15.2f}"
        rows.append(line)
    return "\n".join(rows)


# this writes results.json and results.txt into out_directory
def save_results(results, out_directory):
    out_directory = resolve_path(out_directory)
    out_directory.mkdir(parents=True, exist_ok=True)
    (out_directory / "results.json").write_text(json.dumps({"split": results["split"], "condition": results["condition"], "n_trials": results["n_trials"], "systems": results["systems"], "provenance": results["provenance"]}, indent=2))
    (out_directory / "results.txt").write_text(results_table(results) + "\n")
    return out_directory


# this turns clips into response texts through the chosen listener, with the speech gate applied first
def listen(paths, listener, split, judge_repeat, asr_model):
    """The gate is applied here, once, for whichever listener is chosen, so the judge and the ASR are gated by the same rule (metric-definitions.md 3.1).

    A blocked clip returns an empty response, which is scored as all deletions. Returns the responses, the gate decisions, and the judge dictionary (None for the ASR).
    """
    lookup = condition_lookup(split)
    # The VAD model only loads when an estimate is in the batch. The rendered clips are decided from the trial's condition.
    needs_vad = False
    for path in paths:
        if path and Path(path).stem.lower() == "estimate":
            needs_vad = True
    vad = None
    if needs_vad:
        vad = vad_seconds_fn()
    decisions = []
    for path in paths:
        decision = decide(path, lookup(path), vad)
        log_decision(decision, path, listener, split, lookup(path))
        decisions.append(decision)

    passing = []
    for index, decision in enumerate(decisions):
        if not decision["fired"]:
            passing.append(index)
    if len(passing) < len(paths):
        print(f"  speech gate blocked {len(paths) - len(passing)}/{len(paths)} clips (answered locally, no listener call)", flush=True)

    responses = [""] * len(paths)
    if listener == JUDGE:
        judge = make_judge(judge_repeat, DEFAULT_CACHE, JUDGE_REQUESTS_PER_MINUTE, True)
        for count, index in enumerate(passing, 1):
            try:
                responses[index] = judge_transcribe(judge, paths[index])
            except (QuotaExhausted, MissingCredentials):
                # The quota or the key, not one bad clip. Everything paid for is on disk, so stop and let a re-run resume.
                raise
            except Exception as error:
                # One clip must never kill the run (a safety filter took down a whole pass on 2026-09-02). Record it and score it as a non-response.
                judge["failures"].append((str(paths[index]), f"{type(error).__name__}: {error}"))
                responses[index] = ""
                print(f"    FAILED {Path(paths[index]).parent.name}/{Path(paths[index]).name}: {type(error).__name__} -- scored as a non-response, continuing", flush=True)
            if count % 25 == 0:
                print(f"    judged {count}/{len(passing)}  (calls {judge['calls_made']}, cache {judge['cache_hits']})", flush=True)
        if judge["failures"]:
            print(f"  {len(judge['failures'])} clip(s) failed and were scored as non-responses -- see judge_failures in the results", flush=True)
        return responses, decisions, judge

    passing_paths = []
    for index in passing:
        passing_paths.append(paths[index])
    texts = transcribe(passing_paths, asr_model)
    for index, text in zip(passing, texts):
        responses[index] = text
    return responses, decisions, None


# this gives the clip a system is scored on: the mixture for the floor, the model's estimate, or the clean target for the ceiling
def system_path(trial, system):
    if system == "floor":
        return trial["mixture"]
    if system == "estimate":
        return trial["estimate"]
    return trial["clean"]


# this scores the chosen systems on the chosen metric families for one split, and returns split, condition, n_trials, systems and provenance
def evaluate(split, condition, estimate_directory, systems, metrics, listener, judge_repeat, asr_model):
    metrics = tuple(metrics)
    systems = tuple(systems)
    trials = load_trials(split, condition, estimate_directory)
    if not trials:
        raise RuntimeError(f"No trials loaded for split={split} condition={condition}.")

    if "estimate" in systems:
        if estimate_directory is None:
            raise RuntimeError("The systems include the estimate but no estimate directory was given. Render estimates first with scripts/make_estimates.py.")
        missing = []
        for trial in trials:
            if not trial["estimate"].exists():
                missing.append(trial["trial_id"])
        if missing:
            raise RuntimeError(f"{len(missing)} of {len(trials)} trials have no estimate.wav in {estimate_directory} (the first is {missing[0]}). Render them with scripts/make_estimates.py.")

    results = {"split": split, "condition": condition, "n_trials": len(trials), "systems": {}, "provenance": {}}
    results["provenance"] = {"date": date.today().isoformat(), "split": split, "condition": condition, "n_trials": len(trials), "systems": list(systems), "metrics": list(metrics), "estimate_directory": str(estimate_directory) if estimate_directory else None, "speech_gate": "on"}
    if listener == ASR:
        results["provenance"]["listener"] = f"faster-whisper=={version('faster-whisper')} {asr_model} int8 cpu greedy"
        results["provenance"]["listener_role"] = "STAND-IN for the judge, NOT a live-model result"
    else:
        # CLAUDE.md: every live-model result records the exact model ID, the exact prompt, the input modality and the run date.
        results["provenance"]["listener"] = MODEL_ID
        results["provenance"]["listener_role"] = "THE JUDGE -- a live-model result"
        results["provenance"]["judge_backend"] = BACKEND
        results["provenance"]["judge_modality"] = "audio-in / text-out"
        results["provenance"]["judge_prompt_file"] = str(PROMPT_FILE)
        results["provenance"]["judge_prompt_sha256_12"] = prompt_sha()
        # Which repeat of the same estimates this is. The rendered clips are judged once and always serve that answer.
        results["provenance"]["judge_repeat"] = judge_repeat

    targets = []
    interferers = []
    for trial in trials:
        targets.append(trial["target_text"])
        interferers.append(trial["interferer_text"])

    for system in systems:
        print(f"\n=== {system} ===", flush=True)
        scores = {}
        paths = []
        for trial in trials:
            paths.append(system_path(trial, system))

        if CONTENT in metrics:
            responses, gate_decisions, used_judge = listen(paths, listener, split, judge_repeat, asr_model)
            gate_blocked = 0
            for decision in gate_decisions:
                if decision["fired"]:
                    gate_blocked += 1
            scores["gate_blocked"] = gate_blocked
            if used_judge is not None:
                scores["judge_calls"] = used_judge["calls_made"]
                scores["judge_cache_hits"] = used_judge["cache_hits"]
                scores["judge_failed"] = len(used_judge["failures"])
                # A safety filter that fires on some systems' outputs and not others is a bias in the benchmark, so both counts are reported for every system.
                scores["filter_blocked"] = used_judge["filter_blocks"]
                scores["filter_transient"] = used_judge["filter_transients"]
                if used_judge["failures"]:
                    failure_records = []
                    for clip, error in used_judge["failures"]:
                        failure_records.append({"clip": clip, "error": error[:200]})
                    results["provenance"].setdefault("judge_failures", {})[system] = failure_records
            word = compute_lcf_wer(targets, responses)
            leak = compute_icr(responses, targets, interferers)
            # Content invented, in neither speaker's script. Compare systems on invented_per_trial, and read it against the ceiling row, which is not 0.
            made_up = compute_fr(responses, targets, interferers)
            # The share of empty responses, which catches a muting system that would otherwise score a clean ICR (decisions-m4.md 2026-09-03).
            no_response = None
            if made_up["trials_total"]:
                no_response = 100.0 * made_up["trials_empty"] / made_up["trials_total"]
            scores["lcf_wer"] = word["word_error_rate"]
            scores["substitutions"] = word["substitution_rate"]
            scores["deletions"] = word["deletion_rate"]
            scores["insertions"] = word["insertion_rate"]
            scores["icr_at_2"] = leak["icr_at_2"]
            scores["mean_leak"] = leak["mean_leaked_percent"]
            scores["fr_at_2"] = made_up["fr_at_2"]
            scores["invented_per_trial"] = made_up["invented_per_trial"]
            scores["mean_invented"] = made_up["mean_invented_percent"]
            scores["no_response"] = no_response

        if SIGNAL in metrics:
            sdr = []
            sir = []
            sar = []
            for trial, path in zip(trials, paths):
                target, sample_rate = soundfile.read(trial["clean"])
                interferer, sample_rate = soundfile.read(trial["interferer"])
                mixture, sample_rate = soundfile.read(trial["mixture"])
                estimate, sample_rate = soundfile.read(path)
                length = min(len(target), len(interferer), len(mixture))
                noise = mixture[:length] - target[:length] - interferer[:length]
                separation = decompose(estimate, target, interferer, noise)
                sdr.append(separation["signal_to_distortion_db"])
                sir.append(separation["signal_to_interference_db"])
                sar.append(separation["signal_to_artefact_db"])
            scores["sdr"] = sum(sdr) / len(sdr)
            scores["sir"] = sum(sir) / len(sir)
            scores["sar"] = sum(sar) / len(sar)

        if PERCEPTUAL in metrics:
            print(f"  DNSMOS on {len(paths)} clips (~{len(paths)*2.8/60:.0f} min)", flush=True)
            clip_scores = []
            for path in paths:
                clip_scores.append(score_audio_file(path))
            mean = mean_scores(clip_scores)
            scores["p808"] = mean["p808_mos"]
            scores["sig"] = mean["sig"]
            scores["bak"] = mean["bak"]
            scores["ovrl"] = mean["ovrl"]

        results["systems"][system] = scores
        # Generated by Claude Code
        print("  " + "  ".join(f"{k}={v:.2f}" for k, v in scores.items() if v is not None), flush=True)

    return results
