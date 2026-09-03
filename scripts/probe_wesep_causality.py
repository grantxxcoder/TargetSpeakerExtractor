"""TEMPORARY -- delete after use. Not part of the project; do not commit.

Does the WeSep checkpoint depend on FUTURE audio? Run under the WeSep venv:

    ../wesep_venv/bin/python tmp_causality_probe.py
    ../wesep_venv/bin/python tmp_causality_probe.py --pretrain ../wesep_pretrained/tfmap_context_100

Protocol is ours from 2026-08-24, the same one WeSep's own check_causal() uses:
replace everything after time T, then measure how much the output BEFORE T moved.
A causal model cannot move at all. Ours measured 1.68e-08.

Three passes, because the obvious version of this test is confounded:

  0. DETERMINISM -- the same input twice. Establishes the noise floor. Every
     other number is meaningless if this is not ~0, and it is not guaranteed:
     a model left in train mode, or with dropout live, moves on its own.

  A. SCALE-MATCHED perturbation -- the future is replaced by noise at the SAME
     RMS as what it replaced. THIS IS THE ONE THAT ANSWERS THE QUESTION. Global
     statistics barely move, so anything left is real lookahead.

  B. LOUD perturbation (5x) -- the naive test. If B is large while A is at the
     floor, the model is causal but normalises over the whole input, which still
     breaks streaming, but for a shallower reason: it is fixable with a running
     normaliser, whereas true lookahead is architectural.

The A-versus-B split is the whole point. A first probe on 2026-09-03 ran only
the B case and looked like proof of non-causality; it was not.
"""

import argparse

import numpy as np
import torch
import wesep

SR = 16000
CLIP_S, ENROL_S = 8, 2
CUTS_S = (2, 4, 6)
OURS_REFERENCE = 1.68e-08          # our BSRNN, measured 2026-08-24


def build(pretrain, device):
    model = wesep.load_model_local(pretrain)
    model.set_resample_rate(SR)
    model.set_vad(False)            # our gate is Silero 6.2.1 in speech_gate.py
    model.set_device(device)
    model.set_output_norm(False)    # a 0.9-peak rescale would mask the effect
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain", default="../wesep_pretrained/tfmap_context_causal_100")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = build(args.pretrain, args.device)
    rng = np.random.default_rng(args.seed)
    mixture = (rng.standard_normal(SR * CLIP_S) * 0.1).astype(np.float32)
    enrolment = (rng.standard_normal(SR * ENROL_S) * 0.1).astype(np.float32)

    def run(x):
        out = model.extract_speech_from_pcm(
            torch.from_numpy(x).unsqueeze(0), SR,
            torch.from_numpy(enrolment).unsqueeze(0), SR)
        return out.reshape(-1).cpu().numpy()

    print(f"checkpoint : {args.pretrain}")
    print(f"probe      : {CLIP_S}s mixture, {ENROL_S}s enrolment, seed {args.seed}\n")

    base = run(mixture)

    # ---- 0. determinism floor ------------------------------------------------
    floor = float(np.abs(run(mixture) - base).max())
    print(f"0. DETERMINISM  same input twice -> max change {floor:.3e}")
    if floor > 1e-6:
        print("   WARNING: the model is not deterministic. Every number below is\n"
              "   noise until this is explained -- check for train mode / dropout.\n")
    else:
        print("   deterministic, so anything above this floor is a real effect.\n")

    # ---- A / B. future perturbation -----------------------------------------
    def probe(cut_s, mode):
        x = mixture.copy()
        tail = x[cut_s * SR:]
        new = rng.standard_normal(len(tail)).astype(np.float32)
        if mode == "matched":
            # same RMS as the segment being replaced -> global stats ~unchanged
            new *= np.sqrt((tail ** 2).mean()) / max(np.sqrt((new ** 2).mean()), 1e-12)
        else:
            new *= 0.5                                   # ~5x louder
        x[cut_s * SR:] = new
        out = run(x)
        n = min(len(out), len(base))
        before = np.abs(out[:n] - base[:n])[:cut_s * SR]
        ref = max(np.abs(base[:cut_s * SR]).max(), 1e-12)
        return float(before.max()), float(before.max() / ref)

    results = {}
    for mode, title in (("matched", "A. SCALE-MATCHED future  <-- THE ANSWER"),
                        ("loud",    "B. 5x LOUDER future      (global-stats check)")):
        print(title)
        print(f"   {'perturb from':>13} | {'max change before cut':>21} | {'vs signal':>10}")
        worst = 0.0
        for cut in CUTS_S:
            absolute, relative = probe(cut, mode)
            worst = max(worst, absolute)
            print(f"   {cut:>11} s | {absolute:21.3e} | {relative:9.2%}")
        results[mode] = worst
        print()

    # ---- verdict -------------------------------------------------------------
    a, b = results["matched"], results["loud"]
    causal = a <= max(floor * 10, 1e-7)
    print("=" * 66)
    print(f"scale-matched worst : {a:.3e}   (ours 2026-08-24: {OURS_REFERENCE:.2e})")
    print(f"loud worst          : {b:.3e}")
    print("-" * 66)
    if causal and b <= max(floor * 10, 1e-7):
        print("CAUSAL. Future audio does not reach the past under either probe.")
        print("Streaming is architecturally possible. Throughput is the open\n"
              "question, not causality -- batch RTF was 1.128 on this CPU.")
    elif causal:
        print("CAUSAL, BUT NORMALISES GLOBALLY. Matched perturbation does not")
        print("reach the past; a louder future does. No lookahead, but it scales")
        print("by whole-input statistics, so a naive stream would drift. Fixable")
        print("with a running/fixed normaliser -- an implementation job.")
    else:
        print("NOT CAUSAL AS RUN. Future audio changes earlier output even at")
        print("matched level, so there is a genuine whole-utterance dependency.")
        print("It cannot feed a live transcriber in this form, whatever the")
        print("config flag says. Report the config claim and this measurement.")
    print("=" * 66)
    print("\nCaveat: this probes the model AS CALLED through")
    print("extract_speech_from_pcm(). A non-causal wrapper around a causal core")
    print("fails here too -- which is the right answer for 'can it stream today'.")


if __name__ == "__main__":
    main()
