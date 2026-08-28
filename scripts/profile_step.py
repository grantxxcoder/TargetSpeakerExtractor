"""What a training step is SPENT ON, by module. CPU or GPU.

Companion to scripts/measure_train_cost.py, which answers "which batch size
fits and what does an epoch cost". This answers the question that leaves open:
*where inside the step does the time go*, and is the loader starving the device.
No batch sweep -- that script owns it, in subprocesses, for the reason below.

MEMORY SAFETY -- READ BEFORE RAISING --batch.
On 2026-08-28 an earlier version of this script killed the terminal:

    systemd-oomd: Killed .../app-org.gnome.Terminal.slice/vte-spawn-*.scope
    due to memory pressure for /user.slice/user-1000.slice/user@1000.service
    being 57.19% > 50.00% for > 20s with reclaim activity
    -> systemd-oomd killed 13 process(es) in this unit

systemd-oomd kills the whole CGROUP SCOPE, not the offending process, so the
terminal and every job in it die together. It triggers on sustained memory
PRESSURE (PSI > 50 % for 20 s), not on absolute exhaustion -- swap thrash is
enough. This is the same failure measure_train_cost.py's docstring records
against VSCode on 2026-08-24; that script isolates each batch size in a
subprocess for exactly this reason.

Defences here, in order:
  * CPU defaults to batch 1, not the config's batch_size (~5 GB of activations).
  * torch.profiler is OFF by default -- it retains a record per op and is the
    single largest memory adder. Lightweight forward hooks give the same
    attribution for free. --deep opts in.
  * The loader probe runs in its OWN invocation (--loader-only). DataLoader
    workers fork the parent, duplicating whatever the model already holds.
  * A pre-flight refuses to run if the estimated activations exceed half of
    available RAM.

On a shared desktop, ALSO put it in its own cgroup so a mistake kills only it:

    systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 -- \\
        ../tse_venv/bin/python scripts/profile_step.py

Usage:
    python scripts/profile_step.py                    # attribution, batch 1
    python scripts/profile_step.py --batch 3          # match the real run
    python scripts/profile_step.py --loader-only      # data loading alone
"""
import argparse, collections, sys, time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.bands import band_plan                                   # noqa: E402
from src.models.stft import STFT                                         # noqa: E402
# From train.py so this profiles the model that actually trains -- what
# build_model's own docstring says it is separate from main() for.
from scripts.train import build_model, build_loss_fn, get_data_loaders   # noqa: E402


def peak_rss_mb():
    """VmHWM: the kernel's own high-water mark, not a sampled guess.
    Same source measure_train_cost.py uses, so the numbers are comparable."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def available_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("inf")


def estimate_activations_mb(cfg, batch, T, K):
    """Analytic LSTM activation footprint, the term that dominates and the one
    that triggered the oomd kill. ~4x hidden for gate buffers, 4 bytes fp32."""
    s = cfg["model"]["separator"]
    N, H, R = s["feature_dim"], s["lstm_hidden"], s["num_repeat"]
    ex = batch * (2 if cfg["data"].get("both_directions") else 1)
    time_rnn = ex * K * T * H
    band_rnn = ex * T * K * (2 * H)
    return 4 * (time_rnn + band_rnn) * R * 4 / 1024**2


def synth(batch, cfg, device):
    """Tensors shaped like collate_pairs output, so timing is isolated from disk."""
    d = cfg["data"]
    n = int(d["sample_rate"] * d["chunk_s"])
    ex = batch * (2 if d.get("both_directions") else 1)
    g = torch.Generator().manual_seed(0)
    mix = torch.randn(ex, n, generator=g).to(device)
    tgt = torch.randn(ex, n, generator=g).to(device)
    enr = torch.randn(ex, d["sample_rate"] * 5, generator=g).to(device)
    absent = torch.zeros(ex, dtype=torch.bool, device=device)
    absent[::4] = True                       # ~25 %, near the measured 0.297
    return mix, tgt, enr, absent


def one_step(model, loss_fn, opt, data, scaler=None):
    mix, tgt, enr, absent = data
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(mix, enr)
        # LOSS IN FP32, ALWAYS. L_pres/L_abs/L_gain carry 1e-12 epsilons inside
        # log10; fp16's smallest normal is ~6e-5, so they underflow to zero and
        # the loss goes NaN. This cast is what makes AMP safe here.
        total, _ = loss_fn(tgt.float(), out.float(), mix.float(), absent)
        scaler.scale(total).backward(); scaler.step(opt); scaler.update()
    else:
        out = model(mix, enr)
        total, _ = loss_fn(tgt, out, mix, absent)
        total.backward(); opt.step()
    return float(total.detach())


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def attach_hooks(model):
    """Forward-hook timing. Costs nothing, unlike torch.profiler, and gives the
    per-module attribution and CALL COUNTS that the 32-band loops show up in."""
    tot, cnt, t0 = collections.defaultdict(float), collections.Counter(), {}
    watch = {"stft": model.stft, "tfmap": model.tfmap,
             "subband_norm": model.subband_norm, "separator": model.separator,
             "estimator": model.estimator}
    for i, blk in enumerate(model.separator.blocks):
        watch[f"block{i}.time_rnn"] = blk.time_rnn
        watch[f"block{i}.band_rnn"] = blk.band_rnn
    # `on` gates accumulation: without it the AMP timing loop accumulates into the
    # same counters as the fp32 one and the report mixes two precisions (seen as
    # calls=17 where 8 were expected). `cuda` forces a device sync around every
    # boundary -- CUDA kernels are ASYNCHRONOUS, so an unsynchronised
    # perf_counter measures QUEUING time, not execution: every module reads ~0
    # and whichever one happens to block last absorbs the whole queue. That is
    # what made estimator read 97.4 % and separator 0.8 % on the T4 on
    # 2026-08-28, exactly inverting the CPU result.
    state = {"on": False, "cuda": False}

    def make_pre(n):
        def pre(mod, inp):
            if not state["on"]:
                return None
            if state["cuda"]:
                torch.cuda.synchronize()
            t0[n] = time.perf_counter()
            return None
        return pre

    def make_post(n):
        # MUST return None. A forward hook that returns a value REPLACES the
        # module's output -- a lambda whose body is a tuple of two calls silently
        # turns stft's complex tensor into a tuple, and the failure surfaces
        # three lines later in TFMap as "'tuple' object has no attribute 'abs'".
        def post(mod, inp, out):
            if not state["on"]:
                return None
            if state["cuda"]:
                torch.cuda.synchronize()
            tot[n] += time.perf_counter() - t0[n]
            cnt[n] += 1
            return None
        return post

    for name, mod in watch.items():
        mod.register_forward_pre_hook(make_pre(name))
        mod.register_forward_hook(make_post(name))
    return tot, cnt, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--split", default="sir0")
    ap.add_argument("--batch", type=int, default=None,
                    help="trials per step; default 1 on CPU, config batch_size on GPU")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--loader-only", action="store_true",
                    help="time the data loader alone, model never built")
    ap.add_argument("--chunk-s", type=float, default=None,
                    help="override data.chunk_s. Use to test frame-count "
                         "alignment: T must make ex*T a multiple of 8 for the "
                         "fp16 tensor-core LSTM kernel (see E3f).")
    ap.add_argument("--amp-only", action="store_true",
                    help="skip the fp32 pass. Needed to find the AMP batch "
                         "ceiling: fp32 OOMs first and kills the process before "
                         "AMP is ever reached.")
    ap.add_argument("--deep", action="store_true",
                    help="also run torch.profiler (MEMORY HEAVY -- GPU or a cgroup only)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d, m = cfg["data"], cfg["model"]
    torch.manual_seed(cfg["seed"])

    # ---- data loading alone: separate invocation, model never built ----------
    if args.loader_only:
        print(f"=== data loading alone (model never built), {args.split} ===")
        tl, _ = get_data_loaders(args.split, Path(args.manifest_dir),
                                 Path(args.data_root), cfg)
        it = iter(tl); next(it)                      # warm the worker pool
        n = args.steps or 5
        t0 = time.perf_counter()
        for _ in range(n):
            next(it)
        per = (time.perf_counter() - t0) / n
        ex = int(d["batch_size"]) * (2 if d.get("both_directions") else 1)
        print(f"  {per:.3f} s/batch for {ex} examples "
              f"(num_workers={d.get('num_workers', 0)}, pin_memory unset)")
        print(f"  peak RSS {peak_rss_mb():.0f} MB")
        print("  Compare against s/step from a normal run: with num_workers>0 these")
        print("  overlap, so this is the worst case. If it exceeds s/step, the")
        print("  device is starving and E5's audio-compression question becomes live.")
        return

    if args.amp_only and device != "cuda":
        sys.exit("--amp-only needs CUDA: torch.amp.autocast('cuda') and GradScaler "
                 "are GPU-only, so on CPU this mode would measure nothing.")

    batch = args.batch if args.batch else (int(d["batch_size"]) if device == "cuda" else 1)
    steps = args.steps if args.steps else (12 if device == "cuda" else 1)
    if args.chunk_s:
        d["chunk_s"] = args.chunk_s
    # MEASURED from the real STFT, never a formula. src.models.stft pads by
    # (n_fft - hop) on BOTH sides for overlap-add ramp room, so the naive
    # (n - n_fft)//hop + 1 under-counts by 6 frames -- it reported T=497 where
    # the model actually produces 503, and T is what decides kernel alignment.
    _stft = STFT(m["stft"]["n_fft"], m["stft"]["hop"], d["sample_rate"])
    with torch.no_grad():
        T = _stft(torch.zeros(1, int(d["sample_rate"] * d["chunk_s"]))).shape[-1]
    K = len(band_plan(d["sample_rate"], m["stft"]["n_fft"], m["bands"]["plan"]))

    # ---- pre-flight: refuse before oomd has an opinion ------------------------
    est, avail = estimate_activations_mb(cfg, batch, T, K), available_mb()
    print(f"pre-flight: ~{est:.0f} MB LSTM activations at batch {batch} "
          f"({'fp16' if args.amp_only else 'fp32'} pass), {avail:.0f} MB available")
    if args.amp_only:
        print("  (estimate is the fp32 figure; fp16 measured ~1.9x lower)")
    if device == "cpu" and est > 0.5 * avail:
        sys.exit(f"REFUSING: estimate is {100*est/avail:.0f}% of available RAM. "
                 f"systemd-oomd kills the whole terminal scope at sustained "
                 f"pressure (see module docstring). Lower --batch, or wrap in "
                 f"systemd-run --user --scope -p MemoryMax=4G.")

    model = build_model(cfg).to(device).train()
    loss_fn = build_loss_fn(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ex = batch * (2 if d.get("both_directions") else 1)
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU x{torch.get_num_threads()}"

    print(f"\n{dev_name} | torch {torch.__version__}")
    print(f"params {sum(p.numel() for p in model.parameters()):,} | K={K} bands | T={T} frames")
    print(f"batch {batch} trials -> {ex} examples (both_directions={d.get('both_directions')})")
    print(f"time_rnn: batch B*K={ex*K}, seq={T}   band_rnn: batch B*T={ex*T}, seq={K} bidir")
    # The single largest speed factor found on the T4 (E3f): fp16 tensor-core
    # LSTM kernels need the batch dimension to be a multiple of 8. band_rnn's
    # batch is ex*T, so an odd T makes this depend on the batch size, and
    # missing it costs 4.09x -- measured, 6 repeats, 0.3 % spread.
    aligned = (ex * T) % 8 == 0
    print(f"  band_rnn batch {ex*T} % 8 = {(ex*T) % 8} -> "
          f"{'ALIGNED (fast fp16 kernel)' if aligned else 'NOT ALIGNED (~4x slower in fp16)'}")
    if not aligned:
        need = [b for b in range(1, 13) if (2 * b * T) % 8 == 0]
        print(f"  at T={T}, aligned batch sizes are {need}; or change chunk_s so T % 4 == 0")
    print()

    data = synth(batch, cfg, device)
    tot, cnt, hook_state = attach_hooks(model)

    # The scaler must exist BEFORE the warmup. Under --amp-only the warmup has to
    # run in fp16 as well: an fp32 warmup allocates the very footprint this mode
    # exists to avoid, and OOMs at exactly the batch sizes we are trying to
    # reach. That bug made the 2026-08-28 --amp-only sweep report an "fp32
    # ceiling" of batch 3 a second time, having measured nothing new.
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    one_step(model, loss_fn, opt, data, scaler if args.amp_only else None)
    sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print("=== step time ===")
    per, m32 = float("nan"), float("nan")
    if args.amp_only:
        print("  fp32  skipped (--amp-only)")
    else:
        t0 = time.perf_counter()
        for _ in range(steps):
            one_step(model, loss_fn, opt, data)
        sync(device)
        per = (time.perf_counter() - t0) / steps
        line = f"  fp32  {per:.3f} s/step"
        if device == "cuda":
            m32 = torch.cuda.max_memory_allocated() / 1024**3
            line += f"   peak {m32:.2f} GB"
        print(line + f"   peak RSS {peak_rss_mb():.0f} MB")

    if device == "cuda":
        sc = scaler
        one_step(model, loss_fn, opt, data, sc)          # warm the scaler
        sync(device); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            one_step(model, loss_fn, opt, data, sc)
        sync(device)
        pa = (time.perf_counter() - t0) / steps
        # m32 captured BEFORE the reset above: comparing the post-reset peak to
        # itself reports 1.00x and hides whatever AMP actually saved.
        ma = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  amp   {pa:.3f} s/step   peak {ma:.2f} GB   peak RSS {peak_rss_mb():.0f} MB")
        if not args.amp_only:
            print(f"  -> AMP {per/pa:.2f}x faster, {m32/ma:.2f}x less memory")
        per = pa if args.amp_only else per      # attribution header needs a step time
    else:
        print("  (AMP is CUDA-only -- rerun on the GPU box for the fp16 comparison)")

    # ---- attribution: a SEPARATE pass, fp32, hooks on, synchronised ----------
    # Separate because the timing loops above must not be slowed by per-module
    # syncs, and because mixing AMP steps into these counters silently averages
    # two precisions together.
    if args.amp_only:
        print("\n(attribution skipped: it is an fp32-only pass)")
        print("\nSee decisions-pending.md group E for what to do with these numbers.")
        return
    ATTR_STEPS = 3
    hook_state["on"] = True
    hook_state["cuda"] = (device == "cuda")
    tot.clear(); cnt.clear()
    for _ in range(ATTR_STEPS):
        one_step(model, loss_fn, opt, data)
    sync(device)
    hook_state["on"] = False

    fwd = sum(tot[k] for k in ("stft", "tfmap", "subband_norm", "separator", "estimator"))
    print(f"\n=== forward attribution ({ATTR_STEPS} synchronised fp32 steps; "
          f"forward = {fwd/ATTR_STEPS:.3f} s/step of {per:.3f} unsynchronised) ===")
    for k in ("stft", "tfmap", "subband_norm", "separator", "estimator"):
        print(f"  {k:<14} {tot[k]:7.3f}s  {100*tot[k]/fwd:5.1f}%   calls={cnt[k]}")
    tr = sum(v for k, v in tot.items() if "time_rnn" in k)
    br = sum(v for k, v in tot.items() if "band_rnn" in k)
    print(f"\n  inside separator ({100*tot['separator']/fwd:.0f}% of forward):")
    print(f"    time_rnn  seq={T:<4} batch={ex*K:<6} {tr:7.3f}s  {100*tr/fwd:5.1f}% of forward")
    print(f"    band_rnn  seq={K:<4} batch={ex*T:<6} {br:7.3f}s  {100*br/fwd:5.1f}% of forward")
    print(f"\n  band Python loops: SubbandNorm {len(model.subband_norm.blocks)} iters, "
          f"Estimator {len(model.estimator.trunks)} iters, per forward")

    if args.deep:
        print("\n=== torch.profiler (memory heavy) ===")
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
        with profile(activities=acts) as prof:
            one_step(model, loss_fn, opt, data); sync(device)
        key = "self_cuda_time_total" if device == "cuda" else "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=12))

    print("\nSee decisions-pending.md group E for what to do with these numbers.")


if __name__ == "__main__":
    main()
