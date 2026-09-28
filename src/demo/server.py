"""Local web server for the live extraction demo (scripts/demo_live.py).

Standard library only (http.server, newline-delimited JSON over a streamed
response), so the pinned research venv gains no dependency. One presenter, one
session: state is held on the Demo object and each upload replaces the last.

The live step is paced like a microphone: chunk k is handed to the extractor
only once its last sample would have been spoken, so compute time and lateness
shown on screen are real, not a replay of an offline render.
"""

import base64
import hashlib
import json
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from src.demo.mixer import build_mixture, level_enrollment, pick_example, trim_silence
from src.live_model_metric.judge import Judge, prompt_sha, prompt_text
from src.live_model_metric.lcf_wer import count_errors, normalise_text
from src.models.streaming import CausalSTFT, StreamingExtractor

REPO = Path(__file__).resolve().parents[2]
PAGE = Path(__file__).with_name("index.html")
ROLES = ("target", "enrollment", "interferer")
# Column labels: plain words, not mixture / floor / ceiling.
GEMINI_INPUTS = {"mixture": "No processing", "extracted": "After our model",
                 "target": "Target alone (perfect)"}


def _git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:                                      # noqa: BLE001
        return "unknown"


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def _wav_bytes(x, sr):
    buf = BytesIO()
    sf.write(buf, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _alignment(reference, hypothesis):
    """Word-level diff of the NORMALISED texts, for display only; the error
    count itself comes from lcf_wer.count_errors so it matches the metric."""
    import jiwer
    ref, hyp = normalise_text(reference), normalise_text(hypothesis)
    if not ref.split():
        return [{"w": w, "op": "ok"} for w in hyp.split()]
    out = jiwer.process_words([ref], [hyp])
    rw, hw, tokens = out.references[0], out.hypotheses[0], []
    for ch in out.alignments[0]:
        r, h = rw[ch.ref_start_idx:ch.ref_end_idx], hw[ch.hyp_start_idx:ch.hyp_end_idx]
        if ch.type == "equal":
            tokens += [{"w": w, "op": "ok"} for w in h]
        elif ch.type == "substitute":
            tokens += [{"w": a, "op": "sub", "ref": b} for a, b in zip(h, r)]
        elif ch.type == "insert":
            tokens += [{"w": w, "op": "ins"} for w in h]
        else:
            tokens += [{"w": w, "op": "del"} for w in r]
    return tokens


class Demo:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.mcfg = self.cfg["mixture"]
        self.sr = self.mcfg["sample_rate"]
        self.live_lock = threading.Lock()
        # Held by the whole-clip check. A live run waits for it: the check uses
        # every core, and sharing them made one chunk take 243 ms (measured).
        self.verify_lock = threading.Lock()
        self.sources, self.source_info = {}, {}
        self.reference_text = None
        self.trial = None            # set by mix()
        self.run_dir = None
        self.record = None
        self.n_examples = self.n_mixes = self.version = 0
        torch.set_num_threads(int(self.cfg["model"]["threads"]))
        self._load_model()
        self._warm_up()

    # -- model -------------------------------------------------------------
    def _load_model(self):
        sys.path.insert(0, str(REPO / "scripts"))
        from train import build_context_encoder, build_model, context_kwargs
        self._context_kwargs = context_kwargs
        dev = self.device = self.cfg["model"]["device"]
        path = REPO / self.cfg["model"]["checkpoint"]
        t = time.perf_counter()
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        self.model = build_model(ckpt["config"])
        self.model.load_state_dict(ckpt["model"])
        self.model.to(dev).eval()
        # Built from the CHECKPOINT's config, as make_estimates.py does.
        self.encoder = build_context_encoder(ckpt["config"], dev)
        self.ckpt_meta = {"path": str(path.relative_to(REPO)), "md5": _md5(path),
                          "class": type(self.model).__name__,
                          "epoch": ckpt.get("epoch"), "train_seed": ckpt.get("seed")}
        print(f"loaded {self.ckpt_meta['path']} ({self.ckpt_meta['class']}, "
              f"epoch {ckpt.get('epoch')}) in {time.perf_counter() - t:.1f} s", flush=True)

    def _embed(self, enrollment):
        e = torch.from_numpy(enrollment)[None].to(self.device)
        with torch.inference_mode():
            return self._context_kwargs(self.encoder, e).get("enrol_embedding")

    def _warm_up(self):
        """The first few calls are several times slower; pay that before the
        audience is watching."""
        rng = np.random.default_rng(0)
        enr = (rng.standard_normal(self.sr * 2) * 0.05).astype(np.float32)
        s = StreamingExtractor(self.model, enr, self._embed(enr))
        chunk = self.sr * self.cfg["stream"]["chunk_ms"] // 1000
        with torch.inference_mode():
            for _ in range(15):
                s.push((rng.standard_normal(chunk) * 0.05).astype(np.float32))

    # -- sources -----------------------------------------------------------
    def set_source(self, role, pcm):
        x = trim_silence(np.frombuffer(pcm, dtype="<f4").astype(np.float32), self.sr)
        x = x[:int(self.cfg["stream"]["max_seconds"] * self.sr)]
        if len(x) < self.sr // 2:
            raise ValueError(f"{role}: under 0.5 s of sound after trimming silence")
        self.sources[role] = x
        self.source_info[role] = "uploaded"
        if role == "target":
            self.reference_text = None       # an example's transcript no longer applies
        self.trial = None
        return self._source_view(role)

    def clear_source(self, role):
        self.sources.pop(role, None)
        self.source_info.pop(role, None)
        self.trial = None
        return {"ok": True}

    def _source_view(self, role):
        self.version += 1
        return {"role": role, "seconds": len(self.sources[role]) / self.sr,
                "label": self.source_info[role],
                "url": f"/api/audio/src/{role}.wav?v={self.version}"}

    def _draw_example(self):
        seed = int(self.cfg["seed"]) + self.n_examples
        self.n_examples += 1
        ex = self.cfg["examples"]
        splits = yaml.safe_load((REPO / "experiments/configs/splits.yaml").read_text())
        pick = pick_example(np.random.default_rng(seed), REPO / ex["librispeech_dir"],
                            [str(s) for s in splits[ex["split"]]], ex["target_seconds"],
                            self.mcfg["enrollment_max_s"])
        pick["seed"] = seed
        return pick

    def example(self):
        pick = self._draw_example()
        for role in ROLES:
            self.sources[role] = pick[role][:int(self.cfg["stream"]["max_seconds"] * self.sr)]
        ids = pick["ids"]
        self.source_info = {
            "target": f"LibriSpeech {ids['target']} · speaker {ids['target_speaker']}",
            "enrollment": f"LibriSpeech {ids['enrollment']} · same speaker, different sentence",
            "interferer": f"LibriSpeech · speaker {ids['interferer_speaker']}"}
        self.example_meta = {"seed": pick["seed"], **ids}
        self.reference_text = pick["reference_text"]
        self.trial = None
        return {"sources": [self._source_view(r) for r in ROLES],
                "reference_text": self.reference_text}

    # -- mixture -----------------------------------------------------------
    def mix(self, params):
        for role in ("target", "enrollment"):
            if role not in self.sources:
                raise ValueError(f"add a {role} first")
        seed = int(self.cfg["seed"]) + 1000 + self.n_mixes
        self.n_mixes += 1
        rng = np.random.default_rng(seed)
        interferer_meta = {"source": self.source_info.get("interferer")}
        if "interferer" not in self.sources:
            pick = self._draw_example()
            self.sources["interferer"] = pick["interferer"]
            self.source_info["interferer"] = f"auto: LibriSpeech speaker {pick['ids']['interferer_speaker']}"
            interferer_meta = {"source": "auto", "seed": pick["seed"],
                               "utterances": pick["ids"]["interferer"]}
        snr = params.get("snr_db")
        stems = build_mixture(self.sources["target"], self.sources["interferer"], self.mcfg,
                              rng, float(params.get("sir_db", 0.0)),
                              None if snr is None else float(snr), bool(params.get("reverb")),
                              REPO / self.mcfg["noise_dir"])
        enrollment = level_enrollment(self.sources["enrollment"], self.mcfg)

        stamp = datetime.now()
        self.run_dir = REPO / self.cfg["output_dir"] / stamp.strftime("%Y-%m-%d-%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # PCM_16, like render_trials.py, then READ BACK: the extractor and
        # Gemini must receive the identical file, sample for sample.
        for name, x in (("mixture", stems["mixture"]), ("target", stems["target"]),
                        ("interferer", stems["interferer"]), ("enrollment", enrollment)):
            sf.write(self.run_dir / f"{name}.wav", x, self.sr, subtype="PCM_16")
        read = lambda n: sf.read(self.run_dir / f"{n}.wav", dtype="float32")[0]  # noqa: E731
        mixture, enrollment = read("mixture"), read("enrollment")

        X = self.model.stft(torch.from_numpy(mixture)[None])[0].abs()
        db = 20 * torch.log10(X + 1e-10)
        vmax = float(torch.quantile(db.flatten()[::7], 0.999))
        self.trial = {"mixture": mixture, "enrollment": enrollment,
                      "emb": self._embed(enrollment), "vmax_db": vmax, "vmin_db": vmax - 80.0}

        self.record = {
            "what": "live presentation demo run -- NOT an experiment result",
            "date": stamp.date().isoformat(), "time": stamp.strftime("%H:%M:%S"),
            "config": str(self.config_path.relative_to(REPO)),
            "config_md5": _md5(self.config_path), "git_commit": _git_commit(),
            "checkpoint": self.ckpt_meta, "seed": seed,
            "sources": {r: self.source_info.get(r) for r in ROLES},
            "example": getattr(self, "example_meta", None) if self.reference_text else None,
            "interferer": interferer_meta,
            "mixture": {**stems["meta"], "seconds": len(mixture) / self.sr},
            "enrollment_seconds": len(enrollment) / self.sr,
        }
        self._save_record()
        return {"seconds": len(mixture) / self.sr, "meta": stems["meta"],
                "url": f"/api/audio/run/mixture.wav?v={self.n_mixes}",
                "run_dir": str(self.run_dir.relative_to(REPO)),
                "interferer": self._source_view("interferer"),
                "reference_text": self.reference_text}

    def _save_record(self):
        (self.run_dir / "record.json").write_text(json.dumps(self.record, indent=2, default=str))

    # -- live --------------------------------------------------------------
    def live(self, send):
        if self.trial is None:
            raise ValueError("build the mixture first")
        if not self.live_lock.acquire(blocking=False):
            raise ValueError("a live run is already in progress")
        try:
            with self.verify_lock:          # wait out a previous run's check
                pass
            self._live(send)
        finally:
            self.live_lock.release()

    def _live(self, send):
        tr, sr = self.trial, self.sr
        mix = tr["mixture"]
        chunk_ms = self.cfg["stream"]["chunk_ms"]
        chunk = sr * chunk_ms // 1000
        stft = self.model.stft
        stream = StreamingExtractor(self.model, tr["enrollment"], tr["emb"])
        heard = CausalSTFT(stft)       # spectrogram of the audio actually emitted
        n = len(mix)
        vmin, vmax = tr["vmin_db"], tr["vmax_db"]
        lookahead_ms = stft.latency_ms(self.model.lookahead_frames)
        send({"type": "start", "n_samples": n, "sr": sr, "chunk": chunk, "hop": stft.hop_length,
              "n_frames": -(-(n + stft.padding) // stft.hop_length),
              "bins": stft.n_fft // 2 + 1, "playout_delay_ms": self.cfg["stream"]["playout_delay_ms"],
              "lookahead_ms": lookahead_ms, "lead_ms": self.cfg["stream"]["start_lead_ms"]})

        def q(mag):                    # (F, m) magnitude -> (m, F) uint8 of dB
            d = 20 * np.log10(mag + 1e-10)
            return (np.clip((d - vmin) / (vmax - vmin), 0, 1) * 255).astype(np.uint8).T

        outs, compute, frame, est_frame = [], [], 0, 0
        t0 = time.perf_counter() + self.cfg["stream"]["start_lead_ms"] / 1000
        for i in range(0, n, chunk):
            due = t0 + min(i + chunk, n) / sr     # when a mic would have delivered it
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            tc = time.perf_counter()
            with torch.inference_mode():
                r = stream.push(mix[i:i + chunk])
                if i + chunk >= n:
                    tail = stream.flush()
                    r = {k: np.concatenate([r[k], tail[k]], axis=-1) for k in r}
            dt = (time.perf_counter() - tc) * 1000
            compute.append(dt)
            start = sum(len(o) for o in outs)
            outs.append(r["audio"])
            m = r["mix_mag"].shape[1]
            with torch.inference_mode():
                E = heard.push(r["audio"])
                if i + chunk >= n:
                    # zero-pad to the whole-clip frame count, as STFT.forward does
                    n_frames = -(-(n + stft.padding) // stft.hop_length)
                    tail = np.zeros(n_frames * stft.hop_length - n, dtype=np.float32)
                    E = torch.cat([E, heard.push(tail)], dim=-1)
            E = E[0].abs().numpy()
            send({"type": "chunk", "frame": frame, "frames": m,
                  "est_frame": est_frame, "est_frames": E.shape[1],
                  "mix": _b64(q(r["mix_mag"])), "est": _b64(q(E)),
                  "pcm_start": start, "pcm": _b64(r["audio"].astype("<f4")),
                  "compute_ms": dt, "lag_ms": (time.perf_counter() - due) * 1000})
            frame += m
            est_frame += E.shape[1]

        extracted = np.concatenate(outs).astype(np.float32)
        # FLOAT, like the estimate.wav files every judge score was computed on.
        sf.write(self.run_dir / "extracted.wav", extracted, sr, subtype="FLOAT")
        c = np.array(compute)
        stats = {"chunk_ms": chunk_ms, "n_chunks": len(c), "threads": torch.get_num_threads(),
                 "compute_mean_ms": float(c.mean()), "compute_p95_ms": float(np.percentile(c, 95)),
                 "compute_max_ms": float(c.max()), "rtf": float(c.mean() / chunk_ms),
                 "lookahead_ms": lookahead_ms,
                 # scripts/measure_rtf.py's convention: chunk + lookahead + compute
                 "latency_mean_ms": float(chunk_ms + lookahead_ms + c.mean())}
        self.trial["extracted"] = extracted
        self.record["stream"] = stats
        self.record.pop("whole_clip_check", None)
        self._save_record()
        send({"type": "done", **stats, "url": f"/api/audio/run/extracted.wav?v={time.time():.0f}"})
        if self.cfg["stream"].get("verify_whole_clip", True):
            threading.Thread(target=self._verify, args=(mix, extracted), daemon=True).start()

    def _verify(self, mix, streamed):
        """Whole-clip forward on the same input; log the largest difference."""
        tr = self.trial
        kw = {"enrol_embedding": tr["emb"]} if tr["emb"] is not None else {}
        with self.verify_lock, torch.inference_mode():
            whole = self.model(torch.from_numpy(mix)[None],
                               torch.from_numpy(tr["enrollment"])[None], **kw)[0].numpy()
        self.record["whole_clip_check"] = {
            "max_abs_diff": float(np.abs(whole - streamed).max()),
            "peak": float(np.abs(whole).max())}
        self._save_record()

    # -- Gemini ------------------------------------------------------------
    def gemini(self, params):
        if self.trial is None or "extracted" not in self.trial:
            raise ValueError("run the live extraction first")
        j = self.cfg["judge"]
        prompt_file = REPO / j["prompt_file"]

        def call(name):
            judge = Judge(model_id=j["model_id"], prompt_file=prompt_file,
                          backend=j["backend"], requests_per_minute=0, verbose=False,
                          cache_path=self.run_dir / f"gemini_{name}.csv")
            return name, *judge.judge(self.run_dir / f"{name}.wav")

        with ThreadPoolExecutor(len(GEMINI_INPUTS)) as pool:
            answers = {name: (status, text) for name, status, text in pool.map(call, GEMINI_INPUTS)}

        reference = (params.get("reference_text") or "").strip() or self.reference_text
        ref_source = "given" if reference else "gemini:target"
        if not reference:
            reference = answers["target"][1]
        results = {}
        for name, (status, text) in answers.items():
            e = count_errors(reference, text)
            results[name] = {
                "label": GEMINI_INPUTS[name], "status": status, "transcript": text,
                "errors": e.total_errors, "reference_words": e.reference_word_count,
                "wer": e.total_errors / e.reference_word_count if e.reference_word_count else None,
                "tokens": _alignment(reference, text),
                "url": f"/api/audio/run/{name}.wav?v={time.time():.0f}"}
        judge_meta = {"model_id": j["model_id"], "backend": j["backend"],
                      "prompt_file": j["prompt_file"], "prompt": prompt_text(prompt_file),
                      "prompt_sha256_12": prompt_sha(prompt_file),
                      "modality": "audio-in / text-out",
                      "run_date": datetime.now().date().isoformat()}
        self.record["judge"] = judge_meta
        self.record["reference"] = {"text": reference, "source": ref_source}
        self.record["gemini"] = {k: {kk: v[kk] for kk in ("label", "status", "transcript",
                                                         "errors", "reference_words", "wer")}
                                 for k, v in results.items()}
        self._save_record()
        return {"results": results, "judge": judge_meta,
                "reference": {"text": normalise_text(reference), "source": ref_source},
                "whole_clip_check": self.record.get("whole_clip_check")}

    def status(self):
        return {"whole_clip_check": (self.record or {}).get("whole_clip_check")}

    def config_view(self):
        s, j = self.cfg["stream"], self.cfg["judge"]
        return {"chunk_ms": s["chunk_ms"], "playout_delay_ms": s["playout_delay_ms"],
                "max_seconds": s["max_seconds"],
                "defaults": {k: self.mcfg[k] for k in ("sir_db", "snr_db", "reverb")},
                "checkpoint": self.ckpt_meta, "threads": torch.get_num_threads(),
                "judge": {"model_id": j["model_id"], "prompt": prompt_text(REPO / j["prompt_file"])},
                "benchmark_note": self.cfg.get("display", {}).get("benchmark_note", "")}

    def audio_file(self, kind, name):
        if kind == "src" and name in self.sources:
            return _wav_bytes(self.sources[name], self.sr)
        if kind == "run" and self.run_dir is not None:
            path = self.run_dir / f"{name}.wav"
            if path.exists() and path.parent == self.run_dir:
                return path.read_bytes()
        return None


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):         # keep the terminal readable
            if not self.path.startswith("/api/audio"):
                sys.stderr.write(f"  {self.command} {self.path.split('?')[0]}\n")

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _guard(self, fn):
            try:
                self._json(fn())
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:                   # noqa: BLE001
                traceback.print_exc()
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

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
                self._guard(demo.config_view)
            elif path == "/api/status":
                self._guard(demo.status)
            elif path.startswith("/api/audio/"):
                parts = path.split("/")                # '', api, audio, kind, name.wav
                data = (demo.audio_file(parts[3], parts[4].removesuffix(".wav"))
                        if len(parts) == 5 else None)
                if data is None:
                    return self._json({"error": "no such audio"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            if path.startswith("/api/source/"):
                role = path.rsplit("/", 1)[-1]
                if role not in ROLES:
                    return self._json({"error": f"unknown role {role}"}, 400)
                body = self._body()
                self._guard(lambda: demo.set_source(role, body))
            elif path == "/api/clear/interferer":
                self._guard(lambda: demo.clear_source("interferer"))
            elif path == "/api/example":
                self._guard(demo.example)
            elif path == "/api/mix":
                params = json.loads(self._body() or b"{}")
                self._guard(lambda: demo.mix(params))
            elif path == "/api/gemini":
                params = json.loads(self._body() or b"{}")
                self._guard(lambda: demo.gemini(params))
            elif path == "/api/live":
                self._live()
            else:
                self._json({"error": "not found"}, 404)

        def _live(self):
            """One JSON object per line, flushed as each chunk is processed."""
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def send(obj):
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()
            try:
                demo.live(send)
            except (BrokenPipeError, ConnectionResetError):
                print("  live: browser disconnected, run stopped", flush=True)
            except Exception as exc:                   # noqa: BLE001
                traceback.print_exc()
                try:
                    send({"type": "error", "error": str(exc)})
                except OSError:
                    pass
    return Handler


def serve(config_path, port=None):
    demo = Demo(config_path)
    host = demo.cfg["server"]["host"]
    port = port or demo.cfg["server"]["port"]
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    server.daemon_threads = True
    print(f"\n  demo ready: http://{host}:{port}\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
