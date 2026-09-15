import torch
import torch.nn as nn
import numpy as np
from src.models.stft import STFT

class LossBSRNN:
    """Loss terms for the M2 objective. decisions-m2.md 2026-08-20.

    Arg order is (reference, s_output) throughout -- the reverse of the usual
    (pred, target) convention, so keep it consistent. The reference is the
    target for the present term and the mixture for the absent one.
    """
    def __init__(self, wm, w, p=0.3, tau_pres=0.001, tau_abs=0.01, windows=(8, 16, 32, 64),
                 sample_rate=16000, wg=0.0, gain_delta_db=3.0,
                 w_struct=0.0, struct_floor_db=-40.0):
        self.tau_pres = tau_pres
        self.tau_abs = tau_abs
        self.wm = wm            # weight on L_MR, inside the present branch
        self.w = w              # weight on the ABSENT half. Do not confuse with wm.
        self.wg = wg            # weight on L_gain, also inside the present branch.
                                # DEFAULTS TO 0.0 = term disabled, so an existing
                                # config reproduces its old numbers byte for byte.
        self.gain_delta_db = gain_delta_db   # L_gain deadzone half-width, in dB
        # D17, the mask-structure term. DEFAULTS TO 0.0 = disabled, so every
        # config written before 2026-09-13 reproduces its old numbers exactly.
        self.w_struct = w_struct
        # A mixture bin this far below the clip's peak is treated as silent and
        # excluded: the ideal mask is |target| / |mixture| and that ratio is
        # meaningless where the denominator is noise. Without this the term
        # trains the model to copy the quietest, least reliable cells.
        self.struct_floor_db = struct_floor_db
        self.p = p
        self.sample_rate = sample_rate
        # windows are MILLISECONDS. A tuple, not a list: a mutable default is
        # shared across every instance ever constructed.
        self.windows = tuple(windows)

    def energy(self, x):
        # sum of squared samples, per example. (B, T) -> (B,)
        return x.pow(2).sum(dim=-1)

    def _loss_target_present(self, s_target, s_output, tau_pres=0.001):
        """L_pres, floored SI-SDR. CARTSE eq (1). (B, T) -> (B,)

        Lower is better. Range [-30, inf): -30 when s_output == s_target.
        MUST be masked to crops where the target speaks -- when s_target is all
        zero, alpha is 0/0 and the NaN destroys every weight in the model.
        """
        # alpha: the one volume knob that best explains the output as "the
        # target, turned up or down". keepdim so it broadcasts back over T.
        alpha = (s_output * s_target).sum(dim=-1, keepdim=True) / self.energy(s_target).unsqueeze(-1)
        s_projected = alpha * s_target          # a multiple of the TARGET, not of s_output

        # DEVIATION 1 from CARTSE eq (1), which floors on tau*||s||^2: not
        # scale-invariant, so amplifying paid without bound (g=100 -> -70 dB).
        # Flooring on ||s_proj||^2 makes numerator and floor scale together.
        numerator = self.energy(s_projected)
        denominator = self.energy(s_output - s_projected) + tau_pres * numerator

        return -10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12))

    def _loss_target_absent(self, x_input, s_output, tau_abs=0.01):
        """L_abs, push-to-silence. CARTSE eq (2), normalised. (B, T) -> (B,)

        No target argument: the right answer IS silence, so there is nothing to
        compare against and the mixture takes the target's place as the
        yardstick. No leading minus -- this is already lower-is-better.

            0   emitted the mixture unchanged, i.e. did nothing
          -10   suppressed 10 dB
          -30   floor: 30 dB down or better, i.e. silent
           >0   AMPLIFYING. A bug, not a bad score -- flag it in the run log.
        """
        # DEVIATION from CARTSE eq (2): dividing by ||x||^2 makes it
        # scale-invariant, so 0 dB means "did nothing" on every trial. eta is
        # absent -- it and w appear only as w*eta, so it lives in w.
        numerator = self.energy(s_output) + tau_abs * self.energy(x_input)
        denominator = self.energy(x_input)

        return 10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12))

    def _loss_gain_match(self, s_target, s_output, delta_db=3.0):
        """L_gain, deadzone output-level match. (B, T) -> (B,)

        PRESENT CROPS ONLY. Lower is better, 0 inside +-delta_db of the target's
        level, |error_db| - delta_db outside it.

        Nothing else in the objective opposes a mute: L_pres is scale-invariant
        (Deviation 1), L_abs rewards silence, and L_MR was measured to REWARD
        muting, not penalise it (decisions-m2.md 2026-08-28). Deviation 7, ours.

        Symmetric and minimised AT correct level, so unlike the bug Deviation 1
        fixed there is no gain direction that pays without bound. The deadzone
        also puts .abs()'s kink at 0 inside the zeroed region. dB, not percent:
        10 % amplitude is 0.83 dB. eps inside the sqrt, not a clamp on the
        result -- a clamp strands a fully-muted model with no gradient back up.
        RMS, not the renderer's BS.1770: not differentiable, and both signals are
        measured identically so the comparison stays symmetric.
        """
        eps = 1e-12
        rms_output = (s_output.pow(2).mean(dim=-1) + eps).sqrt()
        rms_target = (s_target.pow(2).mean(dim=-1) + eps).sqrt()
        error_db = 20 * torch.log10(rms_output / rms_target)
        return (error_db.abs() - delta_db).clamp_min(0.0)

    def _loss_multi_res_stft(self, s_target, s_output, windows, p=0.3):
        """L_MR, multi-resolution compressed magnitude + complex L1.
        Yu et al., Interspeech 2023 eq (3). (B, T) -> (B,)

        Lower is better. Range [0, inf), exactly 0 at s_output == s_target.

        PRESENT CROPS ONLY. With an all-zero target both L1 terms collapse into
        "minimise output energy", duplicating _loss_target_absent in
        unnormalised non-dB units and making the silence weight unknowable.

        `windows` is in MILLISECONDS. Scale-VARIANT but it does NOT pin the
        output gain, despite what decisions-m2.md 2026-08-20 claimed: muting the
        mixture ~21 dB IMPROVES it, 0.2735 -> 0.2438 (200 sir0_val crops,
        2026-08-28). Levels are pinned by _loss_gain_match; this prices detail.
        """
        summation = s_output.new_zeros(s_output.shape[0])

        for window_ms in windows:
            n_fft = int(round(window_ms * self.sample_rate / 1000))

            # torch.stft, NOT src.models.stft.STFT: that one is the streaming
            # front end, and reusing it would let a latency change alter the loss.
            # Causality constrains the model, not the objective.
            window = torch.hann_window(n_fft, device=s_output.device, dtype=s_output.dtype)
            stft_kwargs = dict(n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                               window=window, center=True, return_complex=True)
            S_target = torch.stft(s_target, **stft_kwargs)      # (B, F, N) complex
            S_output = torch.stft(s_output, **stft_kwargs)

            # NOT torch.abs(): infinite gradient at the origin, where silent
            # T-F bins sit. Same eps on both, so silence still differences to ~0.
            magnitude_target = (S_target.real.pow(2) + S_target.imag.pow(2) + 1e-8).sqrt()
            magnitude_output = (S_output.real.pow(2) + S_output.imag.pow(2) + 1e-8).sqrt()

            # L1 not L2 (L2's gradient vanishes on the quiet-band errors this
            # term exists to catch). MEAN not sum: sum inflates by ~1e5 and makes
            # L_pres invisible. Reduce over (F, N) only -- the masks need
            # per-example values.
            compressed_magnitude = (magnitude_target.pow(p)
                                    - magnitude_output.pow(p)).abs().mean(dim=(-2, -1))

            # Complex term, uncompressed as in eq (3). L1(real)+L1(imag), not
            # the modulus, which reintroduces the sqrt singularity.
            complex_term = ((S_target.real - S_output.real).abs().mean(dim=(-2, -1))
                            + (S_target.imag - S_output.imag).abs().mean(dim=(-2, -1)))

            summation = summation + compressed_magnitude + complex_term

        return summation / len(windows)         # the 1/I in eq (3)

    def _loss_mask_shape(self, mask, oracle, mixture_mag):
        """L_struct: match the mask's FREQUENCY SHAPE to the ideal mask's.

        mask, oracle, mixture_mag: (B, F, T). Returns (B,).

        WHY THIS TERM EXISTS. MEASURED 2026-09-12/13: 84.2 % of our mask's
        variance is explained by a single number per frame, and it varies 6.5x
        less across frequency than the ideal mask. The model applies a broadband
        gain -- loud when the target speaks, quiet when it does not -- which is
        voice activity detection, not extraction. Two voices overlapping in time
        occupy the same frequencies, so only a per-cell decision can separate
        them and a broadband gain cannot do it even in principle.

        Nothing in the M2 objective opposes this. L_pres is scale-invariant SI-SDR
        over the waveform; L_gain and L_abs are broadband; L_MR has frequency
        resolution but sums over bins, so matching the loud low-frequency bins
        captures most of it. None of the four ever asks which BIN the gain went
        into. The flat mask is not a failure to reach the objective -- it IS the
        objective's cheapest optimum.

        AND IT IS NOT A DATA PROBLEM. Measured 2026-09-13 across checkpoints at
        ~1,989 / ~4,976 / ~9,955 training trials: doubling the data at a matched
        epoch left the volume-knob share unchanged (-0.003, inside noise) and
        made the frequency/time ratio significantly WORSE (-0.162
        [-0.205, -0.118]). More epochs do the same. Both axes converge the model
        ONTO the knob. Scaling the training set is refuted as a fix.

        THE PER-FRAME MEAN IS REMOVED FROM BOTH SIDES, and that is the whole
        design. The model's per-frame gain is roughly right already, and three
        existing terms (L_pres, L_gain, L_abs) argue about level. Supervising the
        raw mask would spend most of the gradient re-teaching what is not broken
        and add a fourth voice to an argument about loudness, making any result
        unattributable. What is left after the subtraction is pure shape across
        frequency -- which bins get more than their frame's average and which get
        less. This mirrors postprocess_mask.py, which holds each frame's level
        fixed for exactly the same reason.

        BORROWED WITH A DIFFERENCE. Supervising a mask against the ideal ratio
        mask is standard (Wang, Narayanan & Wang, IEEE/ACM TASLP 2014, "On
        training targets for supervised speech separation"). There the ideal
        ratio mask is the WHOLE training target and the acceptance test is signal
        quality. Here it is an auxiliary shape-only constraint at a small derived
        weight, the primary objective stays signal-domain, and the acceptance
        test is downstream content fidelity (LCF-WER per SIR band). The
        difference matters: mask approximation weights an error in an inaudible
        bin the same as one in a loud bin, which is why it is NOT allowed to
        become the objective.

        ENERGY WEIGHTED, and bins below struct_floor_db excluded. The ideal mask
        is |target| / |mixture| and that explodes where the mixture is near
        silent. Unweighted, this term's largest gradients would come from the
        least trustworthy cells in the spectrogram.
        """
        # Bins where the mixture carries real energy. Per clip, not per batch:
        # a loud clip would otherwise set the threshold for a quiet one.
        peak = mixture_mag.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        floor = peak * (10.0 ** (self.struct_floor_db / 20.0))
        weight = (mixture_mag >= floor).to(mask.dtype)

        # Per-frame mean over the SELECTED bins only -- including excluded bins
        # in the mean would let silence drag the shape of a loud frame.
        counts = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        mask_mean = (mask * weight).sum(dim=1, keepdim=True) / counts
        oracle_mean = (oracle * weight).sum(dim=1, keepdim=True) / counts

        shape_ours = (mask - mask_mean) * weight
        shape_ideal = (oracle - oracle_mean) * weight

        # Energy weighting on top of selection: a loud bin's shape matters more
        # than a barely-audible one's. Normalised per clip so the term does not
        # scale with input level.
        energy = mixture_mag * weight
        energy = energy / energy.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        return ((shape_ours - shape_ideal).abs() * energy).sum(dim=(1, 2))

    def __call__(self, s_target, s_output, x_input, crop_absent,
                 mask=None, oracle_mask=None, mixture_mag=None):
        """The full M2 objective. decisions-m2.md 2026-08-20.

            L = (1 - w) * mean_present[ L_pres + wm * L_MR + wg * L_gain ] + w * mean_absent[ L_abs ]

        L_gain is OFF at wg = 0.0, which reproduces the 2026-08-20 objective.
        Returns (scalar total, dict of per-term values for logging).

        crop_absent MUST come from the loader, not the manifest label: 5.8 % of
        `both`/`target_only` crops land entirely in target silence
        (decisions-m1.md 2026-08-18), and an all-zero target down the L_pres
        path is a NaN.
        """
        crop_absent = crop_absent.bool()
        present = ~crop_absent

        # .item() costs one device sync per step, paid deliberately: NaN * 0 =
        # NaN in backward, so rows must be SELECTED first, not masked after.
        n_present = int(present.sum().item())
        n_absent = int(crop_absent.sum().item())

        nan = float("nan")
        total = s_output.new_zeros(())
        # Constant key set: a missing half is a gap in the curve, not a
        # missing column.
        parts = {"L_pres": nan, "L_MR": nan, "L_gain": nan, "L_abs": nan,
                 "L_struct": nan,
                 "n_present": n_present, "n_absent": n_absent}

        if n_present:
            target_p, output_p = s_target[present], s_output[present]

            # Catches the likeliest caller error (branching on the manifest
            # label), whose only other symptom is an unexplained NaN.
            assert bool((target_p.abs().amax(dim=-1) > 0).all()), (
                "a crop flagged present has an all-zero target stem -- "
                "crop_absent disagrees with s_target. Use the loader's "
                "crop_absent, not the manifest condition label.")
            loss_present = self._loss_target_present(target_p, output_p, self.tau_pres).mean()
            loss_mr = self._loss_multi_res_stft(target_p, output_p, self.windows, self.p).mean()
            # Computed even at wg = 0 so it is logged before being weighted --
            # scripts/derive_w_g.py reads that to derive wg.
            loss_gain = self._loss_gain_match(target_p, output_p, self.gain_delta_db).mean()
            total = total + (1 - self.w) * (loss_present + self.wm * loss_mr + self.wg * loss_gain)
            parts["L_pres"] = float(loss_present.detach())
            parts["L_MR"] = float(loss_mr.detach())
            parts["L_gain"] = float(loss_gain.detach())

            # D17. PRESENT BRANCH ONLY: on a target-absent crop the ideal mask is
            # |silence| / |mixture|, which is noise divided by signal, and
            # supervising against it would teach the model to reproduce a random
            # pattern. The absent case is already handled by L_abs.
            #
            # Computed whenever the tensors are supplied, even at w_struct = 0,
            # so it is LOGGED before it is weighted -- that is what a derivation
            # script reads to set the weight, exactly as L_gain is computed at
            # wg = 0 for derive_w_g.py.
            if mask is not None and oracle_mask is not None:
                loss_struct = self._loss_mask_shape(
                    mask[present], oracle_mask[present], mixture_mag[present]).mean()
                parts["L_struct"] = float(loss_struct.detach())
                if self.w_struct:
                    total = total + (1 - self.w) * self.w_struct * loss_struct

        if n_absent:
            loss_absent = self._loss_target_absent(x_input[crop_absent],
                                                   s_output[crop_absent], self.tau_abs).mean()
            total = total + self.w * loss_absent    
            parts["L_abs"] = float(loss_absent.detach())

        # Both guards are required: at batch 12 and a 0.297 absent rate ~1.5 %
        # of batches have no absent crop, and mean over an empty selection is NaN.
        parts["total"] = float(total.detach())
        return total, parts
