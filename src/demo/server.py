"""The local web server behind the live extraction demo (scripts/demo_live.py).

It uses only the standard library, so the research environment gains no dependency, and the live step streams one JSON object per line.
There is one presenter and one session, so all state lives in one dictionary and each upload replaces the last.
The live step is paced like a microphone: a chunk reaches the extractor only once its last sample would have been spoken, so the compute time and lateness on screen are real, not a replay.
"""

import base64
import hashlib
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from src.demo.mixer import build_mixture, draw_utterances, level_enrollment, pick_example, speaker_sexes, trim_silence
from src.live_model_metric.judge import BACKEND, MODEL_ID, PROMPT_FILE, judge_clip, make_judge, prompt_sha, prompt_text
from src.live_model_metric.lcf_wer import count_errors, normalise_text
from src.models.bsrnn import extract
from src.models.stft import stft_forward, stft_latency_ms
from src.models.streaming import causal_stft_push, causal_stft_start, stream_flush, stream_push, stream_start
from train import build_model, build_speaker_encoder, speaker_embedding

PAGE = Path(__file__).with_name("index.html")
ROLES = ("target", "enrollment", "interferer")
# The column labels are plain words, not mixture, floor and ceiling.
GEMINI_INPUTS = {"mixture": "No processing", "extracted": "After our model", "target": "Target alone (perfect)"}


# this hashes a file, so the record says exactly which checkpoint and config ran
def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# this packs an array as base64 for the page
def base64_array(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


# this encodes audio as 16-bit wav bytes for the browser to play
def wav_bytes(audio, sample_rate):
    buffer = BytesIO()
    sf.write(buffer, np.clip(audio, -1, 1), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# this lines up the normalised reference and transcript word by word for display. The error count itself comes from lcf_wer.count_errors, so it matches the metric
def alignment(reference, hypothesis):
    reference_text = normalise_text(reference)
    hypothesis_text = normalise_text(hypothesis)
    tokens = []
    if not reference_text.split():
        for word in hypothesis_text.split():
            tokens.append({"w": word, "op": "ok"})
        return tokens
    output = jiwer.process_words([reference_text], [hypothesis_text])
    reference_words = output.references[0]
    hypothesis_words = output.hypotheses[0]
    for chunk in output.alignments[0]:
        reference_part = reference_words[chunk.ref_start_idx:chunk.ref_end_idx]
        hypothesis_part = hypothesis_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type == "equal":
            for word in hypothesis_part:
                tokens.append({"w": word, "op": "ok"})
        elif chunk.type == "substitute":
            for heard, said in zip(hypothesis_part, reference_part):
                tokens.append({"w": heard, "op": "sub", "ref": said})
        elif chunk.type == "insert":
            for word in hypothesis_part:
                tokens.append({"w": word, "op": "ins"})
        else:
            for word in reference_part:
                tokens.append({"w": word, "op": "del"})
    return tokens


# this loads the config, the example voices and the model, and warms the model up
def start_demo(config_path):
    config = yaml.safe_load(Path(config_path).read_text())
    demo = {
        "config_path": Path(config_path),
        "config": config,
        "mixture_config": config["mixture"],
        "sample_rate": config["mixture"]["sample_rate"],
        "live_lock": threading.Lock(),
        # The whole-clip check holds this. A live run waits for it, because the check uses every core and sharing them made one chunk take 243 ms (measured).
        "verify_lock": threading.Lock(),
        "sources": {},
        "source_info": {},
        "reference_text": None,
        "example_meta": None,
        "trial": None,
        "run_dir": None,
        "record": None,
        "example_count": 0,
        "mix_count": 0,
        "version": 0,
        "voice_draw_count": 0,
        # The presenter's pick of interferer voice, as an index into voices, and what was drawn from it. Both are None when the interferer came from anywhere else.
        "interferer_voice": None,
        "interferer_draw": None,
    }
    demo["voices"] = load_voices(demo)
    torch.set_num_threads(int(config["model"]["threads"]))
    load_model(demo)
    warm_up(demo)
    return demo


# this loads the checkpoint and its speaker encoder, both from the checkpoint's own config, as make_estimates.py does
def load_model(demo):
    device = demo["config"]["model"]["device"]
    path = REPO / demo["config"]["model"]["checkpoint"]
    start = time.perf_counter()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    demo["device"] = device
    demo["model"] = model
    demo["encoder"] = build_speaker_encoder(checkpoint["config"], device)
    system = "extension" if hasattr(model, "context") else "baseline"
    demo["checkpoint_meta"] = {"path": str(path.relative_to(REPO)), "md5": md5_file(path), "class": system, "epoch": checkpoint.get("epoch"), "train_seed": checkpoint.get("seed")}
    print(f"loaded {demo['checkpoint_meta']['path']} ({system}, epoch {checkpoint.get('epoch')}) in {time.perf_counter() - start:.1f} s", flush=True)


# this embeds an enrolment for the extension, or gives None for the baseline
def embed(demo, enrollment):
    enrollment_tensor = torch.from_numpy(enrollment)[None].to(demo["device"])
    with torch.inference_mode():
        return speaker_embedding(demo["encoder"], enrollment_tensor)


# this pays for the first few calls, which are several times slower, before the audience is watching
def warm_up(demo):
    generator = np.random.default_rng(0)
    sample_rate = demo["sample_rate"]
    enrollment = (generator.standard_normal(sample_rate * 2) * 0.05).astype(np.float32)
    stream = stream_start(demo["model"], enrollment, embed(demo, enrollment))
    chunk = sample_rate * demo["config"]["stream"]["chunk_ms"] // 1000
    with torch.inference_mode():
        for warmup_index in range(15):
            stream_push(stream, (generator.standard_normal(chunk) * 0.05).astype(np.float32))


# this is the longest a source may be
def max_samples(demo):
    return int(demo["config"]["stream"]["max_seconds"] * demo["sample_rate"])


# this lists the speakers of the example split, so no example voice can come from eval_private
def split_speakers(demo):
    splits = yaml.safe_load((REPO / "experiments/configs/splits.yaml").read_text())
    speakers = []
    for speaker in splits[demo["config"]["examples"]["split"]]:
        speakers.append(str(speaker))
    return speakers


# this checks the interferer voices the presenter can pick. Each must be a speaker of the example split
def load_voices(demo):
    examples = demo["config"]["examples"]
    sexes = speaker_sexes(REPO / examples["librispeech_dir"], split_speakers(demo))
    voices = []
    for voice in examples["interferer_voices"]:
        speaker = str(voice["speaker"])
        if speaker not in sexes:
            raise ValueError(f"interferer voice {speaker} is not a {examples['split']} speaker")
        voices.append({"label": voice["label"], "speaker": speaker, "sex": sexes[speaker]})
    return voices


# this stores an uploaded or recorded source, with its silence trimmed
def set_source(demo, role, pcm):
    audio = trim_silence(np.frombuffer(pcm, dtype="<f4").astype(np.float32), demo["sample_rate"])
    audio = audio[:max_samples(demo)]
    if len(audio) < demo["sample_rate"] // 2:
        raise ValueError(f"{role}: under 0.5 s of sound after trimming silence")
    demo["sources"][role] = audio
    demo["source_info"][role] = "uploaded"
    if role == "interferer":
        demo["interferer_voice"] = None
        demo["interferer_draw"] = None
    if role == "target":
        # An example's transcript no longer applies.
        demo["reference_text"] = None
    demo["trial"] = None
    return source_view(demo, role)


# this removes a source
def clear_source(demo, role):
    demo["sources"].pop(role, None)
    demo["source_info"].pop(role, None)
    if role == "interferer":
        demo["interferer_voice"] = None
        demo["interferer_draw"] = None
    demo["trial"] = None
    return {"ok": True}


# this describes a source for the page
def source_view(demo, role):
    demo["version"] += 1
    voice = None
    if role == "interferer":
        voice = demo["interferer_voice"]
    return {"role": role, "seconds": len(demo["sources"][role]) / demo["sample_rate"], "label": demo["source_info"][role], "voice": voice, "url": f"/api/audio/src/{role}.wav?v={demo['version']}"}


# this draws an example target, enrolment and interferer from the example split
def draw_example(demo):
    seed = int(demo["config"]["seed"]) + demo["example_count"]
    demo["example_count"] += 1
    examples = demo["config"]["examples"]
    interferer_speaker = None
    if demo["interferer_voice"] is not None:
        interferer_speaker = demo["voices"][demo["interferer_voice"]]["speaker"]
    excluded_speakers = set()
    for voice in demo["voices"]:
        excluded_speakers.add(voice["speaker"])
    pick = pick_example(np.random.default_rng(seed), REPO / examples["librispeech_dir"], split_speakers(demo), examples["target_seconds"], demo["mixture_config"]["enrollment_max_s"], interferer_speaker=interferer_speaker, exclude=excluded_speakers)
    pick["seed"] = seed
    return pick


# this loads an example into all three sources
def example(demo):
    pick = draw_example(demo)
    for role in ROLES:
        demo["sources"][role] = pick[role][:max_samples(demo)]
    ids = pick["ids"]
    voice = None
    if demo["interferer_voice"] is not None:
        voice = demo["voices"][demo["interferer_voice"]]
    if voice is None:
        interferer_label = f"LibriSpeech · speaker {ids['interferer_speaker']}"
    else:
        interferer_label = f"LibriSpeech · {voice['label']}, speaker {ids['interferer_speaker']}"
    demo["source_info"] = {
        "target": f"LibriSpeech {ids['target']} · speaker {ids['target_speaker']}",
        "enrollment": f"LibriSpeech {ids['enrollment']} · same speaker, different sentence",
        "interferer": interferer_label,
    }
    if voice is not None:
        demo["interferer_draw"] = {"voice": voice["label"], "speaker": voice["speaker"], "seed": pick["seed"], "utterances": ids["interferer"]}
    example_meta = {"seed": pick["seed"]}
    example_meta.update(ids)
    demo["example_meta"] = example_meta
    demo["reference_text"] = pick["reference_text"]
    demo["trial"] = None
    views = []
    for role in ROLES:
        views.append(source_view(demo, role))
    return {"sources": views, "reference_text": demo["reference_text"]}


# this draws a fresh interferer sample from one of the fixed voices, replacing the interferer
def pick_voice(demo, index):
    if not 0 <= index < len(demo["voices"]):
        raise ValueError(f"no interferer voice {index}")
    voice = demo["voices"][index]
    seed = int(demo["config"]["seed"]) + 2000 + demo["voice_draw_count"]
    demo["voice_draw_count"] += 1
    # Enough speech to cover the target. build_mixture loops it if it falls short.
    if "target" in demo["sources"]:
        seconds = len(demo["sources"]["target"]) / demo["sample_rate"]
    else:
        seconds = demo["config"]["examples"]["target_seconds"][1]
    audio, utterances = draw_utterances(np.random.default_rng(seed), REPO / demo["config"]["examples"]["librispeech_dir"], voice["speaker"], seconds)
    demo["sources"]["interferer"] = audio[:max_samples(demo)]
    demo["source_info"]["interferer"] = f"LibriSpeech · {voice['label']}, speaker {voice['speaker']} ({voice['sex']})"
    demo["interferer_voice"] = index
    demo["interferer_draw"] = {"voice": voice["label"], "speaker": voice["speaker"], "seed": seed, "utterances": utterances}
    demo["trial"] = None
    return source_view(demo, "interferer")


# this builds the mixture the way the trials were built, saves every stem, and starts the run's record
def mix(demo, params):
    for role in ("target", "enrollment"):
        if role not in demo["sources"]:
            raise ValueError(f"add a {role} first")
    sample_rate = demo["sample_rate"]
    seed = int(demo["config"]["seed"]) + 1000 + demo["mix_count"]
    demo["mix_count"] += 1
    generator = np.random.default_rng(seed)
    interferer_meta = {"source": demo["source_info"].get("interferer")}
    if demo["interferer_voice"] is not None and "interferer" in demo["sources"]:
        interferer_meta = {"source": "voice"}
        interferer_meta.update(demo["interferer_draw"])
    if "interferer" not in demo["sources"]:
        pick = draw_example(demo)
        demo["sources"]["interferer"] = pick["interferer"]
        demo["source_info"]["interferer"] = f"auto: LibriSpeech speaker {pick['ids']['interferer_speaker']}"
        interferer_meta = {"source": "auto", "seed": pick["seed"], "utterances": pick["ids"]["interferer"]}
    snr_db = params.get("snr_db")
    if snr_db is not None:
        snr_db = float(snr_db)
    stems = build_mixture(demo["sources"]["target"], demo["sources"]["interferer"], demo["mixture_config"], generator, float(params.get("sir_db", 0.0)), snr_db, bool(params.get("reverb")), REPO / demo["mixture_config"]["noise_dir"])
    enrollment = level_enrollment(demo["sources"]["enrollment"], demo["mixture_config"])

    stamp = datetime.now()
    run_dir = REPO / demo["config"]["output_dir"] / stamp.strftime("%Y-%m-%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    demo["run_dir"] = run_dir
    # Written as 16-bit wav, like render_trials.py, then read back, so the extractor and Gemini receive the identical file, sample for sample.
    for name, audio in (("mixture", stems["mixture"]), ("target", stems["target"]), ("interferer", stems["interferer"]), ("enrollment", enrollment)):
        sf.write(run_dir / f"{name}.wav", audio, sample_rate, subtype="PCM_16")
    mixture = sf.read(run_dir / "mixture.wav", dtype="float32")[0]
    enrollment = sf.read(run_dir / "enrollment.wav", dtype="float32")[0]

    magnitude = stft_forward(demo["model"].stft, torch.from_numpy(mixture)[None])[0].abs()
    level_db = 20 * torch.log10(magnitude + 1e-10)
    top_db = float(torch.quantile(level_db.flatten()[::7], 0.999))
    demo["trial"] = {"mixture": mixture, "enrollment": enrollment, "emb": embed(demo, enrollment), "vmax_db": top_db, "vmin_db": top_db - 80.0}

    sources = {}
    for role in ROLES:
        sources[role] = demo["source_info"].get(role)
    mixture_meta = dict(stems["meta"])
    mixture_meta["seconds"] = len(mixture) / sample_rate
    demo["record"] = {
        "what": "live presentation demo run -- NOT an experiment result",
        "date": stamp.date().isoformat(),
        "time": stamp.strftime("%H:%M:%S"),
        "config": str(demo["config_path"].relative_to(REPO)),
        "config_md5": md5_file(demo["config_path"]),
        "checkpoint": demo["checkpoint_meta"],
        "seed": seed,
        "sources": sources,
        "example": demo["example_meta"] if demo["reference_text"] else None,
        "interferer": interferer_meta,
        "mixture": mixture_meta,
        "enrollment_seconds": len(enrollment) / sample_rate,
    }
    save_record(demo)
    return {"seconds": len(mixture) / sample_rate, "meta": stems["meta"], "url": f"/api/audio/run/mixture.wav?v={demo['mix_count']}", "run_dir": str(run_dir.relative_to(REPO)), "interferer": source_view(demo, "interferer"), "reference_text": demo["reference_text"]}


# this writes the run's record next to its audio
def save_record(demo):
    (demo["run_dir"] / "record.json").write_text(json.dumps(demo["record"], indent=2, default=str))


# this runs the live extraction, one run at a time, after any previous whole-clip check has finished
def live(demo, send):
    if demo["trial"] is None:
        raise ValueError("build the mixture first")
    if not demo["live_lock"].acquire(blocking=False):
        raise ValueError("a live run is already in progress")
    try:
        with demo["verify_lock"]:
            pass
        stream_live(demo, send)
    finally:
        demo["live_lock"].release()


# this streams the mixture through the extractor 80 ms at a time, paced like a microphone, sending both spectrograms as it goes
def stream_live(demo, send):
    trial = demo["trial"]
    sample_rate = demo["sample_rate"]
    mixture = trial["mixture"]
    chunk_ms = demo["config"]["stream"]["chunk_ms"]
    chunk = sample_rate * chunk_ms // 1000
    stft = demo["model"].stft
    stream = stream_start(demo["model"], trial["enrollment"], trial["emb"])
    # The spectrogram of the audio actually emitted.
    heard = causal_stft_start(stft)
    total_samples = len(mixture)
    # The frame count the whole-clip STFT gives, rounded up.
    total_frames = (total_samples + stft.padding + stft.hop_length - 1) // stft.hop_length
    bottom_db = trial["vmin_db"]
    top_db = trial["vmax_db"]
    lookahead_ms = stft_latency_ms(stft)
    send({"type": "start", "n_samples": total_samples, "sr": sample_rate, "chunk": chunk, "hop": stft.hop_length, "n_frames": total_frames, "bins": stft.n_fft // 2 + 1, "playout_delay_ms": demo["config"]["stream"]["playout_delay_ms"], "lookahead_ms": lookahead_ms, "lead_ms": demo["config"]["stream"]["start_lead_ms"]})

    # this turns a (frequency, frames) magnitude into (frames, frequency) bytes of dB for the page to draw
    def to_bytes(magnitude):
        level_db = 20 * np.log10(magnitude + 1e-10)
        return (np.clip((level_db - bottom_db) / (top_db - bottom_db), 0, 1) * 255).astype(np.uint8).T

    outputs = []
    compute_ms = []
    samples_sent = 0
    frame = 0
    estimate_frame = 0
    start = time.perf_counter() + demo["config"]["stream"]["start_lead_ms"] / 1000
    for chunk_start in range(0, total_samples, chunk):
        # When a microphone would have delivered this chunk.
        due = start + min(chunk_start + chunk, total_samples) / sample_rate
        wait = due - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        compute_start = time.perf_counter()
        last_chunk = chunk_start + chunk >= total_samples
        with torch.inference_mode():
            result = stream_push(stream, mixture[chunk_start:chunk_start + chunk])
            if last_chunk:
                tail = stream_flush(stream)
                joined = {}
                for key in result:
                    joined[key] = np.concatenate([result[key], tail[key]], axis=-1)
                result = joined
        elapsed_ms = (time.perf_counter() - compute_start) * 1000
        compute_ms.append(elapsed_ms)
        outputs.append(result["audio"])
        mixture_frames = result["mix_mag"].shape[1]
        with torch.inference_mode():
            heard_frames = causal_stft_push(heard, result["audio"])
            if last_chunk:
                # Zero-padded to the whole-clip frame count, as stft_forward does.
                padding = np.zeros(total_frames * stft.hop_length - total_samples, dtype=np.float32)
                heard_frames = torch.cat([heard_frames, causal_stft_push(heard, padding)], dim=-1)
        heard_magnitude = heard_frames[0].abs().numpy()
        send({"type": "chunk", "frame": frame, "frames": mixture_frames, "est_frame": estimate_frame, "est_frames": heard_magnitude.shape[1], "mix": base64_array(to_bytes(result["mix_mag"])), "est": base64_array(to_bytes(heard_magnitude)), "pcm_start": samples_sent, "pcm": base64_array(result["audio"].astype("<f4")), "compute_ms": elapsed_ms, "lag_ms": (time.perf_counter() - due) * 1000})
        samples_sent += len(result["audio"])
        frame += mixture_frames
        estimate_frame += heard_magnitude.shape[1]

    extracted = np.concatenate(outputs).astype(np.float32)
    # Written as float, like the estimate.wav files every judge score was computed on.
    sf.write(demo["run_dir"] / "extracted.wav", extracted, sample_rate, subtype="FLOAT")
    compute = np.array(compute_ms)
    # The latency follows scripts/measure_rtf.py: chunk plus lookahead plus compute.
    stats = {"chunk_ms": chunk_ms, "n_chunks": len(compute), "threads": torch.get_num_threads(), "compute_mean_ms": float(compute.mean()), "compute_p95_ms": float(np.percentile(compute, 95)), "compute_max_ms": float(compute.max()), "rtf": float(compute.mean() / chunk_ms), "lookahead_ms": lookahead_ms, "latency_mean_ms": float(chunk_ms + lookahead_ms + compute.mean())}
    trial["extracted"] = extracted
    demo["record"]["stream"] = stats
    demo["record"].pop("whole_clip_check", None)
    save_record(demo)
    message = {"type": "done"}
    message.update(stats)
    message["url"] = f"/api/audio/run/extracted.wav?v={time.time():.0f}"
    send(message)
    threading.Thread(target=verify, args=(demo, mixture, extracted), daemon=True).start()


# this runs the whole clip through the model in one pass and records the largest difference from the streamed output
def verify(demo, mixture, streamed):
    trial = demo["trial"]
    with demo["verify_lock"], torch.inference_mode():
        whole = extract(demo["model"], torch.from_numpy(mixture)[None], torch.from_numpy(trial["enrollment"])[None], trial["emb"])[0].numpy()
    demo["record"]["whole_clip_check"] = {"max_abs_diff": float(np.abs(whole - streamed).max()), "peak": float(np.abs(whole).max())}
    save_record(demo)


# this sends the mixture, the extracted audio and the clean target to Gemini with the benchmark's judge prompt, and scores each transcript
def gemini(demo, params):
    if demo["trial"] is None or "extracted" not in demo["trial"]:
        raise ValueError("run the live extraction first")
    run_dir = demo["run_dir"]

    # this asks the judge about one clip, with its own cache file in the run folder
    def call(name):
        judge = make_judge(0, run_dir / f"gemini_{name}.csv", 0, False)
        status, text = judge_clip(judge, run_dir / f"{name}.wav")
        return name, status, text

    answers = {}
    with ThreadPoolExecutor(len(GEMINI_INPUTS)) as pool:
        for name, status, text in pool.map(call, GEMINI_INPUTS):
            answers[name] = (status, text)

    reference = (params.get("reference_text") or "").strip() or demo["reference_text"]
    reference_source = "given" if reference else "gemini:target"
    if not reference:
        reference = answers["target"][1]
    results = {}
    recorded = {}
    for name, (status, text) in answers.items():
        errors = count_errors(reference, text)
        word_error_rate = None
        if errors["reference_word_count"]:
            word_error_rate = errors["total_errors"] / errors["reference_word_count"]
        recorded[name] = {"label": GEMINI_INPUTS[name], "status": status, "transcript": text, "errors": errors["total_errors"], "reference_words": errors["reference_word_count"], "wer": word_error_rate}
        results[name] = dict(recorded[name])
        results[name]["tokens"] = alignment(reference, text)
        results[name]["url"] = f"/api/audio/run/{name}.wav?v={time.time():.0f}"
    judge_meta = {"model_id": MODEL_ID, "backend": BACKEND, "prompt_file": str(PROMPT_FILE.relative_to(REPO)), "prompt": prompt_text(), "prompt_sha256_12": prompt_sha(), "modality": "audio-in / text-out", "run_date": datetime.now().date().isoformat()}
    demo["record"]["judge"] = judge_meta
    demo["record"]["reference"] = {"text": reference, "source": reference_source}
    demo["record"]["gemini"] = recorded
    save_record(demo)
    return {"results": results, "judge": judge_meta, "reference": {"text": normalise_text(reference), "source": reference_source}, "whole_clip_check": demo["record"].get("whole_clip_check")}


# this reports the whole-clip check once it has finished
def status(demo):
    record = demo["record"] or {}
    return {"whole_clip_check": record.get("whole_clip_check")}


# this is what the page needs at start-up
def config_view(demo):
    stream_config = demo["config"]["stream"]
    defaults = {}
    for key in ("sir_db", "snr_db", "reverb"):
        defaults[key] = demo["mixture_config"][key]
    return {"chunk_ms": stream_config["chunk_ms"], "playout_delay_ms": stream_config["playout_delay_ms"], "max_seconds": stream_config["max_seconds"], "defaults": defaults, "checkpoint": demo["checkpoint_meta"], "threads": torch.get_num_threads(), "interferer_voices": demo["voices"], "judge": {"model_id": MODEL_ID, "prompt": prompt_text()}, "benchmark_note": demo["config"]["display"]["benchmark_note"]}


# this returns a source or a saved run file as wav bytes, or None if there is no such file
def audio_file(demo, kind, name):
    if kind == "src" and name in demo["sources"]:
        return wav_bytes(demo["sources"][name], demo["sample_rate"])
    if kind == "run" and demo["run_dir"] is not None:
        path = demo["run_dir"] / f"{name}.wav"
        if path.exists() and path.parent == demo["run_dir"]:
            return path.read_bytes()
    return None


# this builds the request handler. http.server needs a handler class, which is why this one stays a class
def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        # this keeps the terminal readable by leaving out the audio requests
        def log_message(self, format, *args):
            if not self.path.startswith("/api/audio"):
                sys.stderr.write(f"  {self.command} {self.path.split('?')[0]}\n")

        # this sends one JSON reply
        def json_reply(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # this reads the request body
        def read_body(self):
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        # this runs an action and replies with its result, or with the error. A ValueError is the presenter's mistake, anything else is a bug
        def guarded(self, action):
            try:
                self.json_reply(action())
            except ValueError as error:
                self.json_reply({"error": str(error)}, 400)
            except Exception as error:
                traceback.print_exc()
                self.json_reply({"error": f"{type(error).__name__}: {error}"}, 500)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                body = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/config":
                self.guarded(lambda: config_view(demo))
            elif path == "/api/status":
                self.guarded(lambda: status(demo))
            elif path.startswith("/api/audio/"):
                # The parts are '', api, audio, kind and name.wav.
                parts = path.split("/")
                data = None
                if len(parts) == 5:
                    data = audio_file(demo, parts[3], parts[4].removesuffix(".wav"))
                if data is None:
                    return self.json_reply({"error": "no such audio"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.json_reply({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            if path.startswith("/api/source/"):
                role = path.rsplit("/", 1)[-1]
                if role not in ROLES:
                    return self.json_reply({"error": f"unknown role {role}"}, 400)
                body = self.read_body()
                self.guarded(lambda: set_source(demo, role, body))
            elif path == "/api/clear/interferer":
                self.guarded(lambda: clear_source(demo, "interferer"))
            elif path == "/api/example":
                self.guarded(lambda: example(demo))
            elif path == "/api/voice":
                params = json.loads(self.read_body() or b"{}")
                self.guarded(lambda: pick_voice(demo, int(params.get("voice", -1))))
            elif path == "/api/mix":
                params = json.loads(self.read_body() or b"{}")
                self.guarded(lambda: mix(demo, params))
            elif path == "/api/gemini":
                params = json.loads(self.read_body() or b"{}")
                self.guarded(lambda: gemini(demo, params))
            elif path == "/api/live":
                self.live_reply()
            else:
                self.json_reply({"error": "not found"}, 404)

        # this streams the live run back, one JSON object per line, flushed as each chunk is processed
        def live_reply(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            # this sends one line
            def send(payload):
                self.wfile.write((json.dumps(payload) + "\n").encode())
                self.wfile.flush()

            try:
                live(demo, send)
            except (BrokenPipeError, ConnectionResetError):
                print("  live: browser disconnected, run stopped", flush=True)
            except Exception as error:
                traceback.print_exc()
                try:
                    send({"type": "error", "error": str(error)})
                except OSError:
                    pass

    return Handler


# this starts the demo and serves it until interrupted
def serve(config_path):
    demo = start_demo(config_path)
    host = demo["config"]["server"]["host"]
    port = demo["config"]["server"]["port"]
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    server.daemon_threads = True
    print(f"\n  demo ready: http://{host}:{port}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
