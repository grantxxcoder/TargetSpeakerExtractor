"""Measure L_state at known anchors and derive w_state. decisions-pending.md D14.

    ../tse_venv/bin/python scripts/derive_w_state.py --split sir0 \
        --teacher models/state_detector_notebook.pt \
        --checkpoint models/model_sir0_10000-e6.pt

Writes experiments/results/<date>-wstate-anchor-<split>/{per_crop.csv,meta.yaml}.

WHY THIS SCRIPT EXISTS
----------------------
MEASURED 2026-09-11: at w_state = 1.0 the state term contributes **0.68 % of the
gradient** reaching the waveform. L_state is a cross-entropy in nats (~0.10);
L_pres and friends are in dB (tens). The term is two orders of magnitude too
quiet to do anything, so a run at w_state = 1.0 would be indistinguishable from
the baseline and would look like the IDEA failing when it was the arithmetic.

Same problem `w_m` and `w_g` had, same treatment: derive it, do not guess it.
There is no meaningful dB conversion -- BCE is a log-probability, dB is a power
ratio -- and `L_MR` is the standing precedent that a term need not be in dB
provided its weight reconciles the units.

THE NUMBER THAT MATTERS MOST IS NOT THE WEIGHT
----------------------------------------------
It is the HEADROOM: L_state on the mixture minus L_state on the clean target.
That is the whole range the term can express. If it is small, no weight rescues
the term -- it would be amplifying noise -- and that is a result worth having
before a 6 h arm rather than after.

TWO CANDIDATE RULES, BOTH REPORTED
----------------------------------
1. LOSS-SCALE MATCH, which is w_m's documented rule ("30 % of |L_pres| at the
   do-nothing anchor", decisions-m2.md 2026-08-20):

       w_state = share * |L_pres(anchor)| / L_state(anchor)

   Ill-conditioned on sir0 at the passthrough anchor, where L_pres is ~-0.18 and
   the division blows up -- exactly why derive_w_g.py had to abandon this rule.
   Reported at the MODEL anchor too, where L_pres is a real number.

2. GRADIENT SHARE, which is what the 0.68 % measurement actually exposed:

       w_state such that |grad(with term) - grad(without)| / |grad(without)|
       equals a stated share

   Measured by two backward passes rather than derived in closed form, because
   the gradient norm is not a simple function of the loss values. Costs a
   handful of batches.

Neither is "correct". They price different things -- rule 1 makes the term
VISIBLE in the reported total, rule 2 makes it INFLUENTIAL in the update. Rule 2
is the one that matches the failure being fixed, so it is what `suggested`
follows, with rule 1 printed beside it as a cross-check.

THE ORACLE ANCHOR IS A WIRING CHECK, NOT A DATA POINT
-----------------------------------------------------
L_state on the clean target must be near zero: there is no second voice in it.
If it is not, the teacher is mis-wired -- wrong column, inverted BCE target, or
an enrolment that does not match the trial -- and every other number here is
meaningless. The script says so and stops short of a recommendation.

Terms are accumulated by train.py's own add_parts/epoch_report, so the
arithmetic matches history.csv rather than resembling it.
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.models.losses_state import LossBSRNNState  # noqa: E402
from src.models.state_teacher import StateTeacher  # noqa: E402
from src.run_log import timed  # noqa: E402
from train import (SPLIT_MANIFESTS, add_parts, build_loss_fn, build_model,  # noqa: E402
                   epoch_report, git_commit)

ORACLE, PASSTHROUGH, MODEL = "oracle", "passthrough", "model"
# Synthetic partial suppressions, in dB of interferer attenuation. These show
# the term's DYNAMIC RANGE -- whether it distinguishes -6 dB of leakage from
# -20 dB -- which no single anchor can.
PARTIAL_DB = (-6.0, -12.0, -20.0)
SHARES = (0.05, 0.10, 0.15, 0.20, 0.30)


def partial_name(db):
    return f"partial_{abs(int(db))}db"


def read_interferer(batch, data_root, split, sample_rate, n_samples):
    """The interferer stem for each example, cropped to match the batch.

    Read here rather than taken from the loader, which does not return it --
    it opens interferer.wav internally to build the second direction and then
    discards it. A derivation script reading its own reference signals is
    normal, and it keeps dataset_loader.py untouched for a measurement that is
    not part of training.

    random_crop=False, so the crop is deterministic and starts at 0 -- the same
    window the loader handed us. Verified additive 2026-09-10:
    mixture == target + interferer + noise exactly, because each stem is stored
    at the gain it contributes.
    """
    import soundfile as sf
    stems = []
    for trial_id, direction in zip(batch["trial_id"], batch["direction"]):
        # `direction` says which speaker was the TARGET for this example, so the
        # interferer is the other stem. Getting this backwards would make every
        # partial anchor measure the wrong speaker.
        name = "interferer.wav" if direction == "target" else "target.wav"
        audio, rate = sf.read(Path(data_root) / split / trial_id / name,
                              dtype="float32")
        assert rate == sample_rate, f"{trial_id}/{name} is {rate} Hz"
        audio = audio[:n_samples]
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        stems.append(audio)
    return torch.from_numpy(np.stack(stems))


def systems_for(batch, model, device, interferer=None):
    """The output to score under each anchor.

    oracle       the clean target. No second voice, so L_state must be ~0.
    passthrough  the mixture. Full interferer. The term's upper anchor.
    partial_*    target + beta*interferer at a known attenuation, synthesised
                 from the stems. Verified additive 2026-09-10:
                 mixture == target + interferer + noise exactly, because each
                 stem is stored at the gain it contributes. So these are real
                 suppression levels, not approximations -- but note they carry
                 NO masking artefact, which a real extractor output does.
    model        the checkpoint under test, if one was given.
    """
    mixture, target = batch["mixture"], batch["target"]

    systems = {ORACLE: target, PASSTHROUGH: mixture}

    if interferer is not None:
        noise = mixture - target - interferer
        for db in PARTIAL_DB:
            beta = 10.0 ** (db / 20.0)
            systems[partial_name(db)] = target + beta * interferer + noise

    if model is not None:
        with torch.no_grad():
            systems[MODEL] = model(mixture.to(device),
                                   batch["enrollment"].to(device)).cpu()
    return systems


def gradient_share(loss_fn, batch, w_state, device):
    """|grad(w_state) - grad(0)| / |grad(0)| on the waveform, for one batch.

    Two backward passes. Measured rather than derived: the gradient norm is not
    a simple function of the loss values, and the whole reason this script
    exists is that the loss VALUES looked reasonable while the gradient share
    was 0.68 %.
    """
    target = batch["target"].to(device)
    mixture = batch["mixture"].to(device)
    crop_absent = batch["crop_absent"].to(device)

    gradients = {}
    for weight in (0.0, w_state):
        estimate = (mixture.clone() * 0.5).requires_grad_(True)
        loss_fn.w_state = weight
        total, _ = loss_fn(target, estimate, mixture, crop_absent)
        total.backward()
        gradients[weight] = estimate.grad.detach().clone()

    base = gradients[0.0].abs().sum()
    delta = (gradients[w_state] - gradients[0.0]).abs().sum()
    return float(delta / base.clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    parser.add_argument("--split", default="sir0")
    parser.add_argument("--teacher", required=True,
                        help="the frozen state-detector checkpoint")
    parser.add_argument("--ecapa-dir", default="../ecapa_pretrained")
    parser.add_argument("--checkpoint", default=None,
                        help="extractor checkpoint, for the `model` anchor")
    # Paths are CLI arguments, not config keys -- same as train.py, so the two
    # cannot disagree about where the data is.
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--grad-batches", type=int, default=4,
                        help="batches used for the gradient-share measurement, "
                             "which needs two backward passes each")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    config = yaml.safe_load(Path(arguments.config).read_text())
    device = torch.device(arguments.device)
    w = float(config["loss"]["w"])
    w_m = float(config["loss"]["w_m"])
    w_g = float(config["loss"].get("w_g", 0.0))

    teacher = StateTeacher(arguments.teacher, arguments.ecapa_dir,
                           device=arguments.device)
    print(f"teacher: {teacher.describe()}")

    model = None
    if arguments.checkpoint:
        model = build_model(config).to(device).eval()
        state = torch.load(arguments.checkpoint, map_location=device,
                           weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"extractor: {arguments.checkpoint}")

    # The loss is the STATE subclass even at w_state = 0: the anchors need
    # L_state logged, and the parent class does not compute it.
    base_loss = build_loss_fn(config)
    loss_fn = LossBSRNNState(
        wm=w_m, w=w, wg=w_g, teacher=teacher, w_state=0.0,
        p=base_loss.p, tau_pres=base_loss.tau_pres, tau_abs=base_loss.tau_abs,
        windows=base_loss.windows, sample_rate=base_loss.sample_rate,
        gain_delta_db=base_loss.gain_delta_db)

    # Same construction train.py uses for its validation set, so the crops are
    # the ones the objective is actually evaluated on.
    (_, _), (manifest_val, audio_val) = SPLIT_MANIFESTS[arguments.split]
    data_root = Path(arguments.data_root)
    dataset = TrialDataset(
        manifest_csv=Path(arguments.manifest_dir) / f"{manifest_val}.csv",
        data_root=data_root,
        split=audio_val,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=config["seed"],
        random_crop=False,
        both_directions=bool(config["data"].get("both_directions", False)),
        # remix_gains MUST be off here, and it is the config default that has to
        # be overridden rather than inherited. MEASURED 2026-09-11: with it on,
        # the loader rebuilds the mixture at a fresh SIR each epoch, so the
        # mixture it returns is NOT target + interferer + noise from disk -- the
        # residual came back at 0.2x to 1.7x the mixture's own level. Every
        # partial_* anchor synthesised from those stems was garbage, and read as
        # WORSE than the raw mixture on L_pres, L_gain and L_abs.
        remix_gains=False,
    )
    loader = DataLoader(dataset, batch_size=int(config["data"]["batch_size"]),
                        shuffle=False, collate_fn=collate_pairs,
                        num_workers=int(config["data"].get("num_workers", 0)))
    n_samples = int(round(float(config["data"]["chunk_s"])
                          * int(config["data"]["sample_rate"])))

    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    rows, shares = [], defaultdict(list)

    with timed("scripts/derive_w_state.py",
               scope=lambda: f"{arguments.split}, {arguments.n_batches} batches"):
        for index, batch in enumerate(loader):
            if index >= arguments.n_batches:
                break
            loss_fn.enrolment_embedding = teacher.embed_enrolment(
                batch["enrollment"].to(device))

            interferer = read_interferer(batch, data_root / "rendered",
                                         audio_val,
                                         int(config["data"]["sample_rate"]),
                                         n_samples)
            for name, estimate in systems_for(batch, model, device,
                                              interferer).items():
                with torch.no_grad():
                    _, parts = loss_fn(batch["target"].to(device),
                                       estimate.to(device).float(),
                                       batch["mixture"].to(device),
                                       batch["crop_absent"].to(device))
                add_parts(sums[name], counts[name], parts)
                # L_state applies to EVERY crop, so it is accumulated against
                # the batch size rather than against the present/absent halves
                # add_parts knows about.
                sums[name]["L_state"] += parts["L_state"] * len(batch["target"])
                counts[name]["all"] += len(batch["target"])
                rows.append(dict(batch=index, system=name,
                                 **{k: parts.get(k) for k in
                                    ("L_pres", "L_MR", "L_gain", "L_abs",
                                     "L_state")}))

            if index < arguments.grad_batches:
                for candidate in (1.0,):
                    shares[candidate].append(
                        gradient_share(loss_fn, batch, candidate, device))
                loss_fn.w_state = 0.0

    stats = {}
    for name in sums:
        report = epoch_report(sums[name], counts[name], w, w_m, w_g)
        report["L_state"] = (sums[name]["L_state"] / counts[name]["all"]
                             if counts[name]["all"] else float("nan"))
        stats[name] = report

    order = [ORACLE] + [partial_name(db) for db in PARTIAL_DB] + [PASSTHROUGH]
    if MODEL in stats:
        order.append(MODEL)
    order = [n for n in order if n in stats]

    print(f"\n{'anchor':16s} {'L_state':>9s} {'L_pres':>9s} {'L_MR':>8s} "
          f"{'L_gain':>8s} {'L_abs':>8s}")
    for name in order:
        s = stats[name]
        print(f"  {name:14s} {s['L_state']:>9.4f} {s['L_pres']:>9.3f} "
              f"{s['L_MR']:>8.4f} {s['L_gain']:>8.3f} {s['L_abs']:>8.3f}")

    headroom = stats[PASSTHROUGH]["L_state"] - stats[ORACLE]["L_state"]
    print(f"\nHEADROOM, mixture minus clean target: {headroom:.4f} nats")
    print("  the whole range the term can express. Small means no weight helps.")

    oracle_ok = stats[ORACLE]["L_state"] < 0.5 * stats[PASSTHROUGH]["L_state"]
    if not oracle_ok:
        print("\n*** WIRING CHECK FAILED ***")
        print(f"L_state on the CLEAN TARGET is {stats[ORACLE]['L_state']:.4f}, "
              f"not far below the mixture's {stats[PASSTHROUGH]['L_state']:.4f}.")
        print("The clean target contains no second voice, so the teacher is")
        print("mis-wired: wrong output column, inverted BCE target, or an")
        print("enrolment that does not match the trial. Every number above is")
        print("meaningless until that is fixed. No weight is recommended.")

    measured_share = float(np.mean(shares[1.0])) if shares[1.0] else float("nan")
    print(f"\nGRADIENT SHARE at w_state = 1.0: {100 * measured_share:.3f} % "
          f"(n={len(shares[1.0])} batches)")
    print("  2026-09-11 reference on synthetic audio: 0.68 %")

    print(f"\n{'target share':>13s} {'w_state (gradient)':>20s} "
          f"{'w_state (loss-scale, model anchor)':>36s}")
    suggested = None
    for share in SHARES:
        # The share is very nearly linear in w_state over this range -- the term
        # is a small perturbation on a much larger gradient -- so scaling the
        # measurement is sound. Verify with --grad-batches on the chosen value
        # rather than trusting the extrapolation.
        by_gradient = share / measured_share if measured_share > 0 else float("nan")
        anchor = stats.get(MODEL, stats[PASSTHROUGH])
        by_loss_scale = (share * abs(anchor["L_pres"]) / anchor["L_state"]
                         if anchor["L_state"] > 0 else float("nan"))
        # %.4g, not %.1f: the first run's answer was 0.0027 and printed as
        # "0.0", which reads as "no weight is recommended".
        print(f"{share:>13.2f} {by_gradient:>20.4g} {by_loss_scale:>36.4g}")
        if share == 0.15:
            suggested = by_gradient

    print()
    if oracle_ok and suggested is not None and np.isfinite(suggested):
        print(f"SUGGESTED w_state = {suggested:.4g}   "
              f"(15 % of the gradient, rule 2)")
        # VERIFY rather than extrapolate. The linear scaling above assumes the
        # term is a small perturbation on the base gradient; at w_state = 1.0 it
        # measured 5571 % on real audio, which is nowhere near that regime, and
        # a value extrapolated from it would be wrong by whatever the curvature
        # is. Two more batches is cheap insurance against a six-hour arm run at
        # the wrong weight.
        verified = []
        for index, batch in enumerate(loader):
            if index >= arguments.grad_batches:
                break
            loss_fn.enrolment_embedding = teacher.embed_enrolment(
                batch["enrollment"].to(device))
            verified.append(gradient_share(loss_fn, batch, suggested, device))
        loss_fn.w_state = 0.0
        achieved = float(np.mean(verified)) if verified else float("nan")
        print(f"VERIFIED at that weight: {100 * achieved:.2f} % of the "
              f"gradient (target 15 %)")
        if np.isfinite(achieved) and abs(achieved - 0.15) > 0.05:
            print("  The extrapolation missed. Scale by the ratio and verify "
                  f"again: try w_state = {suggested * 0.15 / achieved:.4g}")
    else:
        print("No recommendation. Fix the wiring check, or the term has no "
              "headroom to weight.")
    print("\nRule 1 (loss-scale) is printed as a CROSS-CHECK, not an "
          "alternative. It is ill-conditioned wherever |L_pres| is near zero "
          "-- which is the passthrough anchor on sir0, and the reason "
          "derive_w_g.py abandoned it.")

    out_dir = Path(arguments.out or
                   f"experiments/results/{date.today().isoformat()}"
                   f"-wstate-anchor-{arguments.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_crop.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(dict(
        date=date.today().isoformat(),
        git_commit=git_commit(),
        config=arguments.config,
        split=arguments.split,
        seed=int(config.get("seed", 42)),
        teacher=arguments.teacher,
        teacher_describe=teacher.describe(),
        extractor=arguments.checkpoint,
        n_batches=arguments.n_batches,
        w=w, w_m=w_m, w_g=w_g,
        anchors={k: {m: float(v) for m, v in stats[k].items()
                     if isinstance(v, (int, float))} for k in stats},
        headroom_nats=float(headroom),
        oracle_wiring_ok=bool(oracle_ok),
        gradient_share_at_1=measured_share,
        suggested_w_state=(float(suggested) if suggested is not None
                           and np.isfinite(suggested) else None),
        verified_share_at_suggested=(float(achieved)
                                     if "achieved" in dir() else None),
    ), sort_keys=False))
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
