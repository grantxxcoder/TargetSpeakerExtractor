"""Measure the real-time factor and end-to-end latency of the extractor.

    python scripts/measure_rtf.py --checkpoint models/model_sir0_5000-e7.pt
    python scripts/measure_rtf.py --chunk-ms 80 --threads 4 --device cuda

TWO NUMBERS, ONE MEASUREMENT. Both fall out of the time taken to process one
chunk:

    RTF      = processing time / audio duration.  Must be < 1 or the input
               backlog grows without bound and delay climbs forever.
    latency  = chunk + algorithmic lookahead + processing time.  Must fit the
               ~200-300 ms budget (decisions-m0.md 2026-08-07).

The RTF deadline is the tighter of the two: processing must finish inside the
chunk's own duration, which at 80 ms is stricter than the latency budget would
allow. **If RTF < 1 the latency budget is satisfied automatically.**

WHY CHUNKS AND NOT WHOLE CLIPS. Timing `forward()` on a whole clip flatters
streaming by up to 32x, measured 2026-09-01: a whole-sequence LSTM call is one
batched matrix multiply over every frame, whereas streaming is thousands of tiny
matrix-vector products and becomes launch-latency bound. Whole-clip RTF is
therefore not evidence of streaming ability.

WHY 80 ms. The chunk-size sweep saturates early: 8 ms -> 80 ms buys 9.3x of the
available efficiency, and 80 ms -> whole clip buys only 3.5x more. At 80 ms the
end-to-end delay is 120 ms plus processing, comfortably inside the budget, where
160 ms chunks would sit exactly on the 200 ms limit.

THIS IS AN ESTIMATE, NOT A MEASUREMENT, AND THE OUTPUT IS DISCARDED. Chunks are
processed INDEPENDENTLY because the model has no stateful streaming path. The
audio produced is wrong -- discontinuous at every boundary, no recurrent context
-- but the TIMING is close, because passing a carried hidden state costs the same
as passing a fresh one. Two things are missed: the small cost of state management
itself, and a difference in frames-per-chunk because `forward()` pads each chunk
where a streaming STFT would not. Call it 10-20 % error, and never report the
figure without that caveat.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_log import timed  # noqa: E402
from train import build_model, git_commit  # noqa: E402

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="models/model_sir0_5000-e7.pt")
    parser.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    parser.add_argument("--chunk-ms", type=float, default=80.0)
    parser.add_argument("--audio-seconds", type=float, default=60.0,
                        help="how much audio to push through, per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4,
                        help="RTF is meaningless without this recorded")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--enrollment-seconds", type=float, default=5.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    config = yaml.safe_load(open(args.config))
    sample_rate = int(config["data"]["sample_rate"])

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)

    # The model's own lookahead, from the STFT convention it documents.
    lookahead_ms = model.stft.latency_ms() if hasattr(model, "stft") else float("nan")

    chunk_samples = int(round(args.chunk_ms / 1000.0 * sample_rate))
    enrollment = torch.randn(1, int(args.enrollment_seconds * sample_rate), device=device)
    chunk = torch.randn(1, chunk_samples, device=device)

    print(f"  device            {args.device}  ({hardware_description(args.device)})")
    print(f"  threads           {args.threads}")
    print(f"  chunk             {args.chunk_ms:.0f} ms = {chunk_samples} samples")
    print(f"  model lookahead   {lookahead_ms:.1f} ms")
    print(f"  parameters        {sum(p.numel() for p in model.parameters()):,}")

    with torch.inference_mode():
        for _ in range(WARMUP_CHUNKS):
            model(chunk, enrollment)
        if device.type == "cuda":
            torch.cuda.synchronize()

        chunks_per_repeat = int(args.audio_seconds * 1000 / args.chunk_ms)
        per_chunk_ms = []
        for _ in range(args.repeats):
            for _ in range(chunks_per_repeat):
                start = time.perf_counter()
                model(chunk, enrollment)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                per_chunk_ms.append((time.perf_counter() - start) * 1000.0)

    per_chunk_ms.sort()
    def percentile(p):
        return per_chunk_ms[min(int(p / 100 * len(per_chunk_ms)), len(per_chunk_ms) - 1)]

    mean_ms = statistics.fmean(per_chunk_ms)
    result = {
        "date": date.today().isoformat(),
        "git_commit": git_commit(),
        "checkpoint": args.checkpoint,
        "device": args.device,
        "hardware": hardware_description(args.device),
        "threads": args.threads,
        "chunk_ms": args.chunk_ms,
        "chunks_timed": len(per_chunk_ms),
        "lookahead_ms": lookahead_ms,
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
                "Output discarded. 10-20 % error. Not a streaming measurement.",
    }

    print(f"\n  per chunk   mean {mean_ms:7.2f} ms   p50 {percentile(50):7.2f}"
          f"   p95 {percentile(95):7.2f}   p99 {percentile(99):7.2f}"
          f"   max {per_chunk_ms[-1]:7.2f}")
    print(f"  chunks timed      {len(per_chunk_ms)}")
    print()
    print(f"  RTF mean          {result['rtf_mean']:.4f}   "
          f"({'keeps up' if result['rtf_mean'] < 1 else 'FALLS BEHIND'}, needs < 1)")
    print(f"  RTF p99           {result['rtf_p99']:.4f}   "
          f"({'no chunk misses the deadline' if result['keeps_up'] else 'SOME CHUNKS MISS THE 80 ms DEADLINE'})")
    print(f"  latency mean      {result['latency_mean_ms']:.1f} ms   "
          f"= {args.chunk_ms:.0f} chunk + {lookahead_ms:.0f} lookahead + {mean_ms:.1f} compute")
    print(f"  latency p99       {result['latency_p99_ms']:.1f} ms   (budget 200-300 ms)")

    out_dir = Path(args.out or f"experiments/results/{date.today().isoformat()}-rtf-{args.device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rtf.json").write_text(json.dumps(result, indent=2))
    print(f"\n  written to {out_dir}/rtf.json")


if __name__ == "__main__":
    with timed("scripts/measure_rtf.py", lambda: " ".join(sys.argv[1:])):
        main()
