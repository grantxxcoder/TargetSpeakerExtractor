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
    # Whether __call__ takes `s_other`. The state-loss subclasses have their own
    # __call__ signatures and set this False, so train.py asks rather than guesses.
    ACCEPTS_OTHER = True

    def __init__(self, wm, w, p=0.3, tau_pres=0.001, tau_abs=0.01, windows=(8, 16, 32, 64),
                 sample_rate=16000, wg=0.0, gain_delta_db=3.0,
                 w_struct=0.0, struct_floor_db=-40.0, w_interf=1.0):
        # D10, the interference weight. How much MORE an error that sounds like
        # the other speaker costs than any other error in L_pres. DEFAULTS TO
        # 1.0 = L_pres exactly as before, so every existing config reproduces its
        # old numbers byte for byte. decisions-m2.md 2026-10-02.
        self.w_interf = w_interf
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

    def _loss_target_present(self, s_target, s_output, tau_pres=0.001,
                             s_other=None, w_interf=1.0):
        """L_pres, floored SI-SDR. CARTSE eq (1). (B, T) -> (B,)

        Lower is better. Range [-30, inf): -30 when s_output == s_target.
        MUST be masked to crops where the target speaks -- when s_target is all
        zero, alpha is 0/0 and the NaN destroys every weight in the model.

        With `s_other` and `w_interf != 1`, the part of the error that sounds
        like the OTHER speaker is counted w_interf times instead of once (D10):

            denominator = ||error||^2 + (w_interf - 1) * ||e_interf||^2

        At w_interf = 1 (or no s_other) the arithmetic is unchanged.
        """
        # alpha: the one volume knob that best explains the output as "the
        # target, turned up or down". keepdim so it broadcasts back over T.
        alpha = (s_output * s_target).sum(dim=-1, keepdim=True) / self.energy(s_target).unsqueeze(-1)
        s_projected = alpha * s_target          # a multiple of the TARGET, not of s_output

        # DEVIATION 1 from CARTSE eq (1), which floors on tau*||s||^2: not
        # scale-invariant, so amplifying paid without bound (g=100 -> -70 dB).
        # Flooring on ||s_proj||^2 makes numerator and floor scale together.
        numerator = self.energy(s_projected)
        error_energy = self.energy(s_output - s_projected)
        if s_other is not None and w_interf != 1.0:
            # Never below zero: ||e_interf||^2 is an exact share of error_energy
            # (see _interference_part), so the sum is >= w_interf * ||e_interf||^2.
            error_energy = error_energy + (w_interf - 1.0) * self.energy(
                self._interference_part(s_target, s_output, s_other))
        denominator = error_energy + tau_pres * numerator

        return -10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12))

    def _interference_part(self, s_target, s_output, s_other, eps=1e-12):
        """e_interf: the part of the output that the OTHER speaker explains and
        the target does not. (B, T) -> (B, T). decisions-m2.md 2026-10-02, D10.

        The interference term of the BSS_EVAL decomposition (Vincent, Gribonval
        & Fevotte, IEEE TASLP 14(4), 2006), which is what AB-SDR re-weights
        (Ochiai et al., IEEE/ACM TASLP 32, 2024). BORROWED WITH A DIFFERENCE:
        AB-SDR boosts the ARTEFACT term; here the INTERFERENCE term is boosted,
        because for a live listener the other speaker's words are the worst
        error -- they are transcribed as if the target said them. The noise
        stem is not needed: only the interference part is weighted.

        Gram-Schmidt rather than a least-squares solve. Take the target's
        direction out of the other stem, then project the output onto what is
        left. That remainder is orthogonal to the target, so

            s_output - P_target(s_output) = e_interf + rest,  <e_interf, rest> = 0

        and ||e_interf||^2 is an exact share of L_pres's error energy.

        A silent other stem (target_only trials, a phantom interferer, a crop
        where the other speaker is quiet) gives e_interf = 0 exactly: every dot
        product is 0 and eps keeps the division finite.
        """
        target_energy = self.energy(s_target).unsqueeze(-1)
        other_perp = s_other - ((s_other * s_target).sum(dim=-1, keepdim=True)
                                / (target_energy + eps)) * s_target
        coef = ((s_output * other_perp).sum(dim=-1, keepdim=True)
                / (self.energy(other_perp).unsqueeze(-1) + eps))
        return coef * other_perp

    def _interference_share(self, s_target, s_output, s_other, eps=1e-12):
        """Fraction of L_pres's error energy that is the other speaker. (B,).

        In [0, 1]. LOGGED, never optimised: it says whether a weight on the
        interference term is moving the thing it targets, and it is computed at
        w_interf = 1 too, so the control run shows the same column.
        """
        alpha = (s_output * s_target).sum(dim=-1, keepdim=True) / self.energy(s_target).unsqueeze(-1)
        error_energy = self.energy(s_output - alpha * s_target)
        interf_energy = self.energy(self._interference_part(s_target, s_output, s_other))
        return interf_energy / (error_energy + eps)

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
                 mask=None, oracle_mask=None, mixture_mag=None, s_other=None):
        """The full M2 objective. decisions-m2.md 2026-08-20.

            L = (1 - w) * mean_present[ L_pres + wm * L_MR + wg * L_gain ] + w * mean_absent[ L_abs ]

        L_gain is OFF at wg = 0.0, which reproduces the 2026-08-20 objective.
        Returns (scalar total, dict of per-term values for logging).

        crop_absent MUST come from the loader, not the manifest label: 5.8 % of
        `both`/`target_only` crops land entirely in target silence
        (decisions-m1.md 2026-08-18), and an all-zero target down the L_pres
        path is a NaN.

        `s_other` is the OTHER speaker's stem for each example (D10). With
        w_interf != 1 the present branch optimises the interference-weighted
        L_pres instead of the plain one. `parts["L_pres"]` stays the PLAIN value
        either way, so every curve stays comparable with the control and with
        every earlier run; the optimised value is `parts["L_pres_w"]`.
        """
        # An arm configured for the term but handed no stem would train the
        # plain objective and still look like the arm. Refuse rather than
        # degrade -- the failure decisions-pending.md E8 guards against.
        if self.w_interf != 1.0 and s_other is None:
            raise ValueError(
                f"w_interf = {self.w_interf} but no s_other was passed, so the "
                f"interference weighting cannot be applied. Pass batch['other'].")
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
                 "L_struct": nan, "L_pres_w": nan, "interf_share": nan,
                 "n_present": n_present, "n_absent": n_absent}

        if n_present:
            target_p, output_p = s_target[present], s_output[present]

            # Catches the likeliest caller error (branching on the manifest
            # label), whose only other symptom is an unexplained NaN.
            assert bool((target_p.abs().amax(dim=-1) > 0).all()), (
                "a crop flagged present has an all-zero target stem -- "
                "crop_absent disagrees with s_target. Use the loader's "
                "crop_absent, not the manifest condition label.")
            other_p = None if s_other is None else s_other[present]
            if other_p is not None and self.w_interf != 1.0:
                # D10. The optimised term is the weighted one; the plain value is
                # computed without a graph, for the log only.
                loss_present_opt = self._loss_target_present(
                    target_p, output_p, self.tau_pres, other_p, self.w_interf).mean()
                with torch.no_grad():
                    loss_present = self._loss_target_present(target_p, output_p, self.tau_pres).mean()
            else:
                loss_present = self._loss_target_present(target_p, output_p, self.tau_pres).mean()
                loss_present_opt = loss_present
            loss_mr = self._loss_multi_res_stft(target_p, output_p, self.windows, self.p).mean()
            # Computed even at wg = 0 so it is logged before being weighted --
            # scripts/derive_w_g.py reads that to derive wg.
            loss_gain = self._loss_gain_match(target_p, output_p, self.gain_delta_db).mean()
            total = total + (1 - self.w) * (loss_present_opt + self.wm * loss_mr + self.wg * loss_gain)
            parts["L_pres"] = float(loss_present.detach())
            if other_p is not None:
                # Logged at w_interf = 1 too, so the control run has the column.
                parts["L_pres_w"] = float(loss_present_opt.detach())
                with torch.no_grad():
                    parts["interf_share"] = float(
                        self._interference_share(target_p, output_p, other_p).mean())
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
