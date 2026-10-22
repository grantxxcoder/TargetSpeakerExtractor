"""Derive the presence gate's beta (KL) and lam (supervised term) by gradient share, as derive_w_state.py does for D14.

    ../tse_venv/bin/python scripts/derive_presence_weights.py --config experiments/configs/bsrnn_presence_gate.yaml \
        --checkpoint models/model_sir0_cuecontext-wer-e21.pt --device cuda      # Kaggle: notebooks/kaggle_presence_weights.ipynb

For each new term T (the KL, eq. 16, and the BCE), at weight 1:

    share(T) = |grad T| / |grad base|        L2 over the extractor's parameters (everything except the gate)

The share is exactly linear in the weight, so weight = target share / share(T). Measured at two anchors: random init
(where a from-scratch run starts) and e21's weights with an untrained gate (a trained extractor). The suggestion follows
D14's precedent: a 15 % share. Both anchors are reported and neither is chosen: a from-scratch config reads
"init", and at "e21" a default-initialised gate is saturated (|logit| ~ 10-56, 2026-10-14), which the output flags. The global gradient norm at the suggested weights is reported
against grad_clip, because the BCE's gradient lands mostly on the gate's own 129 weights and could trigger clipping.

Writes experiments/results/<date>-presence-weights/meta.yaml.
"""

import argparse
import copy
import hashlib
import resource
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import (build_context_encoder, build_loss_fn, build_model, context_kwargs, gate_kwargs,  # noqa: E402
                           get_data_loaders, git_commit, other_kwargs, unpack, w_at_step)
from src.models.presence_gate import markov_kl, supervised_bce  # noqa: E402
from src.run_log import timed  # noqa: E402


def norms(grads, params, gate_ids):
    """L2 norm of a gradient over all parameters, over the extractor only, and over the gate only."""
    sq = {"all": 0.0, "extractor": 0.0, "gate": 0.0}
    for g, p in zip(grads, params):
        if g is None:
            continue
        s = float(g.pow(2).sum())
        sq["all"] += s
        sq["gate" if id(p) in gate_ids else "extractor"] += s
    return {k: v ** 0.5 for k, v in sq.items()}


def measure(model, loader, loss_fn, encoder, w, tau, n_batches, device):
    """Per-batch gradient norms of the base loss, the KL and the BCE, each at weight 1.

    MEMORY. Three backward passes share one graph, so it is retained for the first two and freed by the third, and every
    tensor that still points into it is deleted before the next batch. Without both, the 2026-10-14 laptop run held up
    to two retained graphs at once and systemd-oomd killed the terminal.
    """
    model.train()
    model.gate_tau = tau
    loss_fn.w = w
    params = [p for p in model.parameters() if p.requires_grad]
    gate_ids = {id(p) for p in model.presence_gate.parameters()}
    rows = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        mixture, target, enrollment, crop_absent = unpack(batch, device)
        out = model(mixture, enrollment, return_gate=True, **context_kwargs(encoder, enrollment))
        y, logits, y_ungated = out if len(out) == 3 else (*out, None)   # 3 outputs in "postfilter" mode
        # In "postfilter" mode the base here is the GATED term alone (the one the gate feels); the ungated term never
        # reaches the gate, so it is left out of the share on purpose.
        base, parts = loss_fn(target, y, mixture, crop_absent, **other_kwargs(loss_fn, batch, device), **gate_kwargs(loss_fn, batch, device, logits))
        pi = torch.sigmoid(logits)
        valid = batch["v_valid"].to(device).bool()
        kl = markov_kl(pi, loss_fn.a11, loss_fn.a01, loss_fn.rho).mean()
        bce = supervised_bce(logits[valid], batch["v"].to(device)[valid]).mean()
        row = {"L_kl": float(kl.detach()), "L_bce": float(bce.detach()), "pi_mean": float(pi.detach().mean()), "n_present": parts["n_present"], "n_absent": parts["n_absent"],
               # Frames where pi is pinned at 0 or 1 in fp32: the gate is saturated there and passes no gradient.
               "pi_saturated": float(((pi.detach() < 1e-6) | (pi.detach() > 1 - 1e-6)).float().mean()),
               "logit_abs_mean": float(logits.detach().abs().mean())}
        terms = (("base", base), ("kl", kl), ("bce", bce))
        for k, (name, term) in enumerate(terms):
            # Retain the graph only while a later term still needs it; the last backward frees it.
            grads = torch.autograd.grad(term, params, retain_graph=k < len(terms) - 1, allow_unused=True)
            row[name] = norms(grads, params, gate_ids)
            del grads
        rows.append(row)
        del y, logits, base, parts, pi, kl, bce, mixture, target, enrollment, crop_absent
        print(f"  batch {i}: base {row['base']['extractor']:.4g}  kl {row['kl']['extractor']:.4g}  bce {row['bce']['extractor']:.4g}  (extractor grad norms)", flush=True)
    return rows


def summarise(rows, target, grad_clip, on="extractor"):
    """Mean share at weight 1, the weight that gives the target share, and the global norm that weight produces.

    A batch whose base gradient is exactly 0 has no defined share and is skipped: at step 0 of the w schedule (w = 0)
    a batch of target-absent crops contributes nothing to the base loss. Found by the 2026-10-14 smoke test.
    """
    used = [r for r in rows if r["base"][on] > 0]
    if not used:
        return {"n_batches_used": 0, "n_batches_skipped": len(rows), "note": "every batch had a zero base gradient"}
    # `on` = "extractor" (the original rule) or "gate": in "postfilter" mode no gate term has an extractor gradient,
    # so the only coherent anchor is the gate's own 129 weights (decisions-m2.md 2026-10-18).
    share = {t: float(np.mean([r[t][on] / r["base"][on] for r in used])) for t in ("kl", "bce")}
    weight = {t: (target / share[t] if share[t] > 0 else float("inf")) for t in share}
    # Global norm with both terms on at the suggested weights, assuming no cancellation (an upper bound).
    upper = float(np.mean([r["base"]["all"] + weight["kl"] * r["kl"]["all"] + weight["bce"] * r["bce"]["all"] for r in used]))
    return {"n_batches_used": len(used), "n_batches_skipped": len(rows) - len(used),
            "share_at_weight_1": share, "weight_for_target": weight, "global_norm_upper_bound": upper,
            "base_global_norm": float(np.mean([r["base"]["all"] for r in used])), "grad_clip": grad_clip,
            "pi_saturated_mean": float(np.mean([r["pi_saturated"] for r in rows])),
            "logit_abs_mean": float(np.mean([r["logit_abs_mean"] for r in rows]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="experiments/configs/bsrnn_presence_gate.yaml")
    ap.add_argument("--checkpoint", default="models/model_sir0_cuecontext-wer-e21.pt")
    ap.add_argument("--split", default="sir0")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None, help="override data.batch_size (default: the config's)")
    ap.add_argument("--target-share", type=float, default=0.15)
    ap.add_argument("--share-on", choices=("extractor", "gate"), default=None,
                    help="which parameters the share is measured over. Default: 'gate' when the config's model.presence_mode is 'postfilter', else 'extractor'")
    ap.add_argument("--device", default="auto", help="auto = cuda if available, else cpu")
    ap.add_argument("--out-root", default="experiments/results")
    args = ap.parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)

    config = yaml.safe_load(Path(args.config).read_text())
    share_on = args.share_on or ("gate" if config["model"].get("presence_mode", "joint") == "postfilter" else "extractor")
    print(f"gradient share measured over the {share_on}'s parameters", flush=True)
    # beta = lam = 0, so the loss returns the base objective alone; the two terms are measured separately.
    measure_cfg = copy.deepcopy(config)
    measure_cfg["loss"]["presence"].update(beta=0.0, lam=0.0)
    measure_cfg["data"]["num_workers"] = 0
    if args.batch_size is not None:
        measure_cfg["data"]["batch_size"] = args.batch_size
    batch_size = int(measure_cfg["data"]["batch_size"])
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, _ = get_data_loaders(args.split, Path(args.manifest_dir), Path(args.data_root), measure_cfg)
    loss_fn = build_loss_fn(measure_cfg)
    encoder = build_context_encoder(config, device)
    grad_clip = float(config["training"]["grad_clip"])
    tau = float(config["loss"]["presence"]["tau_start"])

    anchors = {}
    peak_gpu_gb = {}
    with timed(f"scripts/derive_presence_weights.py --batches {args.batches} --batch-size {batch_size}",
               scope=lambda: f"sir0_train, {args.batches} batches of {batch_size} x 2 anchors",
               rate=lambda: f"{device.type}, three backward passes per batch"):
        # Random init, at step 0 of the w schedule: where a from-scratch run starts.
        torch.manual_seed(seed)
        model = build_model(config).to(device)
        print("anchor: random init", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        anchors["init"] = measure(model, train_loader, loss_fn, encoder, w_at_step(config, 0, len(train_loader)), tau, args.batches, device)
        if device.type == "cuda":
            peak_gpu_gb["init"] = torch.cuda.max_memory_allocated(device) / 2**30
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # e21's weights with an untrained gate: a trained extractor, at the final w.
        torch.manual_seed(seed)
        model = build_model(config)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        assert not unexpected and all(k.startswith("presence_gate.") for k in missing), (missing, unexpected)
        model = model.to(device)
        print("anchor: e21 + untrained gate", flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        anchors["e21"] = measure(model, train_loader, loss_fn, encoder, float(config["loss"]["w"]), tau, args.batches, device)
        if device.type == "cuda":
            peak_gpu_gb["e21"] = torch.cuda.max_memory_allocated(device) / 2**30
    # Peak resident memory of this process, for sizing the next run (Linux reports KiB).
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"peak memory: RSS {peak_rss_gb:.2f} GiB" + "".join(f", GPU {k} {v:.2f} GiB" for k, v in peak_gpu_gb.items()), flush=True)

    summary = {name: summarise(rows, args.target_share, grad_clip, share_on) for name, rows in anchors.items()}
    # Both anchors are reported; neither is chosen here. A from-scratch config (no training.init_weights) starts at
    # "init". At "e21" the untrained gate can be saturated: e21's features are ~300x larger than at init
    # (2026-10-14 check), so a default-initialised gate gives |logit| ~ 10-56 and pi pinned at 1, where its shares
    # describe the initialisation, not the terms.
    suggested = {name: s.get("weight_for_target") for name, s in summary.items()}
    for name, s in summary.items():
        if not s.get("n_batches_used"):
            print(f"\n{name}: no usable batch ({s.get('note')})")
            continue
        print(f"\n{name}: {s['n_batches_used']} batches used, {s['n_batches_skipped']} skipped (zero base gradient)")
        print(f"  gate: mean |logit| {s['logit_abs_mean']:.2f}, pi saturated on {100 * s['pi_saturated_mean']:.0f} % of frames"
              + ("   <-- SATURATED: these shares reflect the gate's init, not the terms" if s["pi_saturated_mean"] > 0.1 else ""))
        print(f"  share at weight 1  KL {100 * s['share_at_weight_1']['kl']:.2f} %  BCE {100 * s['share_at_weight_1']['bce']:.2f} %")
        print(f"  for a {100 * args.target_share:.0f} % share: beta {s['weight_for_target']['kl']:.4g}  lam {s['weight_for_target']['bce']:.4g}")
        print(f"  global grad norm: base {s['base_global_norm']:.3g}, with both terms <= {s['global_norm_upper_bound']:.3g} (grad_clip {grad_clip})")
    from_scratch = not (config.get("training", {}) or {}).get("init_weights")
    print(f"\nconfig starts {'from scratch -> read the init anchor' if from_scratch else 'from ' + str(config['training']['init_weights']) + ' -> read the e21 anchor'}")

    out = Path(args.out_root) / f"{date.today().isoformat()}-presence-weights"
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "date": date.today().isoformat(),
        "script": "scripts/derive_presence_weights.py",
        "git_commit": git_commit(),
        "seed": seed,
        "config": args.config,
        "config_md5": hashlib.md5(Path(args.config).read_bytes()).hexdigest(),
        "checkpoint": args.checkpoint,
        "checkpoint_md5": hashlib.md5(Path(args.checkpoint).read_bytes()).hexdigest(),
        "split": args.split,
        "batches": args.batches,
        "batch_size": batch_size,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_rss_gib": round(peak_rss_gb, 3),
        "peak_gpu_gib": {k: round(v, 3) for k, v in peak_gpu_gb.items()},
        "target_share": args.target_share,
        "rule": f"weight = target share / (|grad term| / |grad base|), L2 over {share_on} parameters, mean over batches",
        "share_on": share_on,
        "summary": summary,
        "suggested": {name: ({"beta": w["kl"], "lam": w["bce"]} if w else None) for name, w in suggested.items()},
        "config_starts_from": (config.get("training", {}) or {}).get("init_weights") or "scratch",
        "per_batch": anchors,
    }, sort_keys=False))
    print(f"Wrote {out}/meta.yaml")


if __name__ == "__main__":
    main()
