"""Real-time factor and end-to-end latency of the BORROWED WeSep checkpoint.

    ../wesep_venv/bin/python scripts/measure_rtf_wesep.py \
        --pretrain ../wesep_pretrained/tfmap_context_causal_100

Runs in the WeSep venv, not ours: `wesep` is only installed there.

WHY A SECOND SCRIPT AND NOT A FLAG ON measure_rtf.py. That script calls
`build_model(config)` and `load_state_dict(checkpoint["model"])` from train.py,
which imports our model code -- none of which exists in the WeSep venv. The
MEASUREMENT PROTOCOL below is copied from it deliberately, term for term, so the
two rows in project-state.md's latency table are the same quantity. If you change
one, change both.

Borrowed model: Wang et al., Interspeech 2024, "WeSep: A Scalable and Flexible
Toolkit Towards Generalizable Target Speaker Extraction"; challenge fork
REAL-TSE/wesep-real-tse. Run as a system under test on OUR data with OUR
protocol. No number from this script is comparable to any published REAL-TSE
latency figure -- different hardware, different chunking, different convention.

TWO NUMBERS, ONE MEASUREMENT, same as measure_rtf.py:

    RTF      = processing time / audio duration. Must be < 1 or the input
               backlog grows without bound and delay climbs forever.
    latency  = chunk + algorithmic lookahead + processing time. Must fit the
               ~200-300 ms budget (decisions-m0.md 2026-08-07).

The RTF deadline is the tighter of the two. If RTF < 1 the latency budget is
satisfied automatically.

LOOKAHEAD IS READ FROM WESEP'S OWN CONFIG AND CONVERTED WITH OUR CONVENTION --
window fill + emit hop -- the same one src/models/stft.py:latency_ms() uses. At
win 512 / stride 128 @ 16 kHz that is 40.0 ms, which happens to equal our own
model's, so the latency columns are directly comparable. State the convention
when quoting it: CARTSE counts window - hop instead, and would call this 24 ms.

WHY 80 ms CHUNKS AND NOT WHOLE CLIPS. Timing a whole clip flatters streaming by
up to 32x (measured 2026-09-01): a whole-sequence LSTM call is one batched matmul
over every frame, whereas streaming is thousands of tiny matrix-vector products
and becomes launch-latency bound. The whole-clip WeSep figure already on record
(RTF 1.128, 2026-09-03) is BATCH THROUGHPUT and is not this number.

THIS IS AN ESTIMATE, NOT A MEASUREMENT, AND THE OUTPUT IS DISCARDED. Chunks are
processed INDEPENDENTLY because neither model has a stateful streaming path. The
audio produced is wrong -- discontinuous at every boundary, no recurrent context
-- but the TIMING is close, because passing a carried hidden state costs about
what passing a fresh one costs. Call it 10-20 % error and never report the figure
without that caveat.

THE ENROLMENT IS RE-EMBEDDED ON EVERY CHUNK, IN BOTH SYSTEMS, AND THAT IS WHY
THE COMPARISON IS STILL FAIR. WeSep's `extract_speech_from_pcm` recomputes the
enrolment fbank and runs its speaker branch per call; our `model(chunk,
enrollment)` likewise takes the raw 5 s enrolment and rebuilds its TF-Map cue per
call. A real deployment would hoist the speaker branch out of the loop for both
and both would get faster. So:

    default        full extract_speech_from_pcm -- EXACTLY the configuration
                   that produced the 2026-09-03 scores, and the protocol match
                   for our baseline's 0.528. This is the number for the table.
    --cache-fbank  enrolment fbank computed once, then model.model(chunk, feats)
                   directly. A LOWER BOUND: it removes the per-chunk fbank and
                   resample but the speaker encoder still runs inside the model,
                   so it is not a deployment figure either.

Neither mode hoists the speaker encoder. Do not present either as "WeSep's
streaming latency" -- present them as this protocol's numbers, with the caveat.

STREAMING STATUS IS SEPARATE AND STILL OPEN. A good RTF here does NOT establish
that WeSep can stream: causality is what decides that, and the scale-matched
probe (scripts/probe_wesep_causality.py) has not been run. A model with a
whole-utterance dependency can still process 80 ms chunks quickly while
producing the wrong audio. Report the two independently.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# git_commit lives in the estimates runner, not train.py: train.py imports our
# model code, which the WeSep venv does not have. Same reason as
# scripts/make_estimates_wesep.py.
from src.estimates.runner import git_commit  # noqa: E402
from src.run_log import timed  # noqa: E402

# Copied from measure_rtf.py. Keep in step.
WARMUP_CHUNKS = 20


def hardware_description(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def describe_checkpoint(pretrain_dir):
    """The framing facts, quoted from the checkpoint's own config, so the
    lookahead and the causality claim travel with the timing."""
    cfg = yaml.safe_load(open(Path(pretrain_dir) / "config.yaml"))
    separator = cfg.get("model_args", {}).get("tse_model", {}).get("separator", {})
    return {
        "win": int(separator.get("win", 0)),
        "stride": int(separator.get("stride", 0)),
        "causal_declared": bool(separator.get("causal", False)),
        "config_resample_rate": int(
            cfg.get("dataset_args", {}).get("resample_rate", 16000)),
    }


def build_model(pretrain_dir, sample_rate, device, output_norm):
    """Load WeSep exactly as scripts/make_estimates_wesep.py does, so the timing
    describes the configuration that produced the scores.

    VAD off: our speech gate is Silero 6.2.1 applied identically to every system
    by src/live_model_metric/speech_gate.py. Output norm off: switching it on
    would hand WeSep a level correction our model does not get.
    """
    import wesep  # noqa: PLC0415

    model = wesep.load_model_local(str(pretrain_dir))
    model.set_resample_rate(sample_rate)
    model.set_vad(False)
    model.set_device(device)
    model.set_output_norm(output_norm)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pretrain", required=True,
                        help="a WeSep checkpoint directory holding avg_model.pt "
                             "and config.yaml, e.g. "
                             "../wesep_pretrained/tfmap_context_causal_100")
    parser.add_argument("--chunk-ms", type=float, default=80.0)
    parser.add_argument("--audio-seconds", type=float, default=60.0,
                        help="how much audio to push through, per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4,
                        help="RTF is meaningless without this recorded")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enrollment-seconds", type=float, default=5.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--cache-fbank", action="store_true",
                        help="compute the enrolment fbank once instead of per "
                             "chunk. A LOWER BOUND, not the table number -- the "
                             "speaker encoder still runs per chunk. See module "
                             "docstring.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    facts = describe_checkpoint(args.pretrain)

    if facts["config_resample_rate"] != args.sample_rate:
        raise SystemExit(
            f"checkpoint was trained at {facts['config_resample_rate']} Hz but "
            f"--sample-rate is {args.sample_rate}. Resampling inside the timed "
            f"loop would be measured as model cost. Fix the flag.")

    model = build_model(args.pretrain, args.sample_rate, args.device,
                        output_norm=False)

    # Our convention, applied to their framing: window fill + emit hop.
    # src/models/stft.py:latency_ms().
    lookahead_ms = 1000.0 * (facts["win"] + facts["stride"]) / args.sample_rate

    chunk_samples = int(round(args.chunk_ms / 1000.0 * args.sample_rate))
    enroll_samples = int(args.enrollment_seconds * args.sample_rate)
    enrollment = torch.randn(1, enroll_samples)
    chunk = torch.randn(1, chunk_samples)
    # Two different bases, and they disagree, so both are labelled.
    # make_estimates_wesep.py sums the whole avg_model.pt state dict
    # (33.46 M, the figure in project-state.md); this counts the TSE model
    # as instantiated. The gap is checkpoint entries not live in the
    # forward pass. Quote the state-dict figure for "model size" and this
    # one for "what was timed".
    n_params = sum(p.numel() for p in model.model.parameters())

    print(f"  checkpoint        {args.pretrain}")
    print(f"  device            {args.device}  ({hardware_description(args.device)})")
    print(f"  threads           {args.threads}")
    print(f"  chunk             {args.chunk_ms:.0f} ms = {chunk_samples} samples")
    print(f"  model lookahead   {lookahead_ms:.1f} ms  "
          f"(win {facts['win']} + stride {facts['stride']} @ {args.sample_rate} Hz, "
          f"our convention)")
    print(f"  declared causal   {facts['causal_declared']}")
    print(f"  parameters timed  {n_params:,}  (state-dict total is 33.46 M)")
    print(f"  enrolment         {'fbank cached once (LOWER BOUND)' if args.cache_fbank else 're-embedded per chunk (table number)'}")

    if args.cache_fbank:
        # Bypass extract_speech_from_pcm's per-call fbank and resample. The
        # speaker encoder still runs inside model.model on every call.
        if getattr(model, "speaker_feat", False):
            feats = model.compute_fbank(enrollment, sample_rate=args.sample_rate,
                                        cmn=True).unsqueeze(0)
        else:
            feats = enrollment
        feats = feats.to(model.device)
        chunk_on_device = chunk.to(model.device)

        def one_chunk():
            with torch.no_grad():
                model.model(chunk_on_device, feats)
    else:
        def one_chunk():
            model.extract_speech_from_pcm(chunk, args.sample_rate,
                                          enrollment, args.sample_rate)

    for _ in range(WARMUP_CHUNKS):
        one_chunk()
    if model.device.type == "cuda":
        torch.cuda.synchronize()

    chunks_per_repeat = int(args.audio_seconds * 1000 / args.chunk_ms)
    per_chunk_ms = []
    for _ in range(args.repeats):
        for _ in range(chunks_per_repeat):
            start = time.perf_counter()
            one_chunk()
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            per_chunk_ms.append((time.perf_counter() - start) * 1000.0)

    per_chunk_ms.sort()

    def percentile(p):
        return per_chunk_ms[min(int(p / 100 * len(per_chunk_ms)),
                                len(per_chunk_ms) - 1)]

    mean_ms = statistics.fmean(per_chunk_ms)
    result = {
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        "system": "wesep-borrowed",
        "checkpoint": str(args.pretrain),
        "parameters_timed": n_params,
        "parameters_basis": "model.model as instantiated; state-dict total differs",
        "declared_causal": facts["causal_declared"],
        "win": facts["win"],
        "stride": facts["stride"],
        "enrollment_mode": "fbank_cached" if args.cache_fbank else "reembed_per_chunk",
        "wesep_vad": False,
        "wesep_output_norm": False,
        "device": args.device,
        "hardware": hardware_description(args.device),
        "threads": args.threads,
        "chunk_ms": args.chunk_ms,
        "chunks_timed": len(per_chunk_ms),
        "lookahead_ms": lookahead_ms,
        "lookahead_convention": "window fill + emit hop (src/models/stft.py)",
        "per_chunk_mean_ms": mean_ms,
        "per_chunk_p50_ms": percentile(50),
        "per_chunk_p95_ms": percentile(95),
        "per_chunk_p99_ms": percentile(99),
        "per_chunk_max_ms": per_chunk_ms[-1],
        "rtf_mean": mean_ms / args.chunk_ms,
        "rtf_p99": percentile(99) / args.chunk_ms,
        "latency_mean_ms": args.chunk_ms + lookahead_ms + mean_ms,
        "latency_p99_ms": args.chunk_ms + lookahead_ms + percentile(99),
        "keeps_up": bool(percentile(99) < args.chunk_ms),
        "note": "ESTIMATE: chunks processed independently, no state carried. "
                "Output discarded. 10-20 % error. Not a streaming measurement. "
                "Enrolment re-embedded per chunk unless --cache-fbank; the "
                "speaker encoder runs per chunk either way, as it does in our "
                "baseline's 0.528. Says nothing about whether WeSep is causal "
                "-- see scripts/probe_wesep_causality.py.",
        "not_comparable_to": "any published REAL-TSE latency figure",
    }

    print(f"\n  per chunk   mean {mean_ms:7.2f} ms   p50 {percentile(50):7.2f}"
          f"   p95 {percentile(95):7.2f}   p99 {percentile(99):7.2f}"
          f"   max {per_chunk_ms[-1]:7.2f}")
    print(f"  chunks timed      {len(per_chunk_ms)}")
    print()
    print(f"  RTF mean          {result['rtf_mean']:.4f}   "
          f"({'keeps up' if result['rtf_mean'] < 1 else 'FALLS BEHIND'}, needs < 1)")
    print(f"  RTF p99           {result['rtf_p99']:.4f}   "
          f"({'no chunk misses the deadline' if result['keeps_up'] else 'SOME CHUNKS MISS THE DEADLINE'})")
    print(f"  latency mean      {result['latency_mean_ms']:.1f} ms   "
          f"= {args.chunk_ms:.0f} chunk + {lookahead_ms:.0f} lookahead + {mean_ms:.1f} compute")
    print(f"  latency p99       {result['latency_p99_ms']:.1f} ms   (budget 200-300 ms)")
    print(f"\n  baseline for comparison: RTF 0.528 mean / 0.706 p99, "
          f"latency 162.2 / 176.5 ms (same protocol, 2026-09-01)")

    suffix = "-cachefbank" if args.cache_fbank else ""
    out_dir = Path(args.out or f"experiments/results/"
                               f"{date.today().isoformat()}-rtf-wesep-{args.device}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rtf.json").write_text(json.dumps(result, indent=2))
    print(f"\n  written to {out_dir}/rtf.json")


if __name__ == "__main__":
    with timed("scripts/measure_rtf_wesep.py", lambda: " ".join(sys.argv[1:])):
        main()
