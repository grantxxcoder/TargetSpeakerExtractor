import torch
import torch.nn as nn
import numpy as np
from src.models.stft import STFT

class LossBSRNN:
    """Loss terms for the M2 objective. decisions-m1.md 2026-08-20.

    Arg order is (reference, s_output) throughout -- the reverse of the usual
    (pred, target) convention, so keep it consistent. The reference is the
    target for the present term and the mixture for the absent one.
    """
    def __init__(self, wm, w, p=0.3, tau_pres=0.001, tau_abs=0.01, windows=(8, 16, 32, 64), sample_rate=16000):
        self.tau_pres = tau_pres
        self.tau_abs = tau_abs
        self.wm = wm            # weight on L_MR, inside the present branch
        self.w = w              # weight on the ABSENT half. Do not confuse with wm.
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

        # every term is an ENERGY (squared). tau caps the reward at 1/tau = 30 dB.
        #
        # DEVIATION from CARTSE eq (1), which floors on tau*||s||^2. That is NOT
        # scale-invariant: the numerator scales with the output gain g while the
        # floor does not, so a perfect-shape output scaled by g scores
        # -20log10(g) - 30 -- unbounded reward for amplifying (measured: g=5 ->
        # -43.98 dB, g=100 -> -70 dB). Flooring on ||s_proj||^2 makes numerator
        # and floor scale together, so they cancel: flat -30 dB at every gain.
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
        # DEVIATION from CARTSE eq (2) = eta * 10log10(||s_hat||^2 + tau*||x||^2),
        # not scale-invariant: scaling x and s_hat by g shifts it by 20log10(g), so
        # two loudness-matched silent outputs get different gradients. Dividing by
        # ||x||^2 fixes that and makes 0 dB mean "did nothing" on every trial.
        #
        # eta is deliberately absent. Under masked means it and w appear only as
        # the product w*eta -- two dials, one degree of freedom. It lives in w.
        numerator = self.energy(s_output) + tau_abs * self.energy(x_input)
        denominator = self.energy(x_input)

        return 10 * torch.log10((numerator + 1e-12) / (denominator + 1e-12))

    def _loss_multi_res_stft(self, s_target, s_output, windows, p=0.3):
        """L_MR, multi-resolution compressed magnitude + complex L1.
        Yu et al., Interspeech 2023 eq (3). (B, T) -> (B,)

        Lower is better. Range [0, inf), exactly 0 at s_output == s_target.

        PRESENT CROPS ONLY. With an all-zero target both L1 terms collapse into
        "minimise output energy", duplicating _loss_target_absent in
        unnormalised non-dB units and making the silence weight unknowable.

        `windows` is in MILLISECONDS and converted here. NOT scale-invariant,
        which is wanted: it compares magnitudes directly, so this is the term
        that pins the output gain -- _loss_target_present does not.
        """
        summation = s_output.new_zeros(s_output.shape[0])

        for window_ms in windows:
            n_fft = int(round(window_ms * self.sample_rate / 1000))

            # torch.stft with center=True, NOT src.models.stft.STFT. That one is
            # the streaming front end: center=False, cold-start left pad, framing
            # chosen for latency. The loss runs offline over a whole chunk, so
            # causality constrains the MODEL, not the objective -- and reusing it
            # would make a latency change silently alter the loss.
            window = torch.hann_window(n_fft, device=s_output.device, dtype=s_output.dtype)
            stft_kwargs = dict(n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                               window=window, center=True, return_complex=True)
            S_target = torch.stft(s_target, **stft_kwargs)      # (B, F, N) complex
            S_output = torch.stft(s_output, **stft_kwargs)

            # NOT torch.abs(): |z| = sqrt(re^2 + im^2) has infinite gradient at
            # the origin, x**0.3 has infinite gradient there too, and silent T-F
            # bins sit exactly at the origin. Both signals get the same eps
            # floor, so the difference entering the loss is still ~0 in silence.
            magnitude_target = (S_target.real.pow(2) + S_target.imag.pow(2) + 1e-8).sqrt()
            magnitude_output = (S_output.real.pow(2) + S_output.imag.pow(2) + 1e-8).sqrt()

            # L1, not torch.norm's L2: L2's gradient vanishes as the error
            # shrinks, so it stops caring about the quiet-band errors this term
            # exists to catch. MEAN over the ~F*N coefficients, not the sum that
            # ||.||_1 literally denotes -- sum inflates this by ~1e5 and makes
            # _loss_target_present numerically invisible with no error message.
            # Reduce over (F, N) only, never the batch: per-example values are
            # what the masked means need.
            compressed_magnitude = (magnitude_target.pow(p)
                                    - magnitude_output.pow(p)).abs().mean(dim=(-2, -1))

            # Complex term, UNCOMPRESSED as written in eq (3). L1(real)+L1(imag),
            # i.e. real and imag as two real channels -- not the modulus, which
            # reintroduces the sqrt singularity. Magnitude alone discards phase,
            # and the Estimator predicts a COMPLEX mask plus a COMPLEX residual.
            complex_term = ((S_target.real - S_output.real).abs().mean(dim=(-2, -1))
                            + (S_target.imag - S_output.imag).abs().mean(dim=(-2, -1)))

            summation = summation + compressed_magnitude + complex_term

        return summation / len(windows)         # the 1/I in eq (3)

    def __call__(self, s_target, s_output, x_input, crop_absent):
        """The full M2 objective. decisions-m1.md 2026-08-20.

            L = (1 - w) * mean_present[ L_pres + wm * L_MR ] + w * mean_absent[ L_abs ]

        s_target / s_output / x_input : (B, T)
        crop_absent                   : (B,) bool, straight from the loader.

        Returns (scalar total, dict of per-term values for logging).

        crop_absent MUST come from the loader, which computes it from the
        CROPPED target stem -- never from the manifest's condition label. 5.8 %
        of `both`/`target_only` crops land entirely in target silence
        (decisions-m1.md 2026-08-18), so branching on the label sends ~1 crop in
        17 down the L_pres path with an all-zero target, which is a NaN.
        """
        crop_absent = crop_absent.bool()
        present = ~crop_absent

        # .item() forces one device sync per step. Paid deliberately: the
        # alternative is computing every term on the full batch and masking
        # after, and NaN * 0 = NaN in the backward pass -- masking a NaN does
        # not remove it. Selecting the rows FIRST is the only safe form.
        n_present = int(present.sum().item())
        n_absent = int(crop_absent.sum().item())

        nan = float("nan")
        total = s_output.new_zeros(())
        # Constant key set so the logger's schema never changes: a missing half
        # shows up as a gap in the curve rather than a missing column.
        parts = {"L_pres": nan, "L_MR": nan, "L_abs": nan,
                 "n_present": n_present, "n_absent": n_absent}

        if n_present:
            target_p, output_p = s_target[present], s_output[present]

            # Catches the single most likely caller error: branching on the
            # manifest condition label instead of the loader's crop_absent. That
            # sends all-zero targets down this path, where alpha is 0/0. Without
            # this the symptom is a NaN with no indication of the cause; one amax
            # over the selected rows is cheap beside four STFTs.
            assert bool((target_p.abs().amax(dim=-1) > 0).all()), (
                "a crop flagged present has an all-zero target stem -- "
                "crop_absent disagrees with s_target. Use the loader's "
                "crop_absent, not the manifest condition label.")
            loss_present = self._loss_target_present(target_p, output_p, self.tau_pres).mean()
            loss_mr = self._loss_multi_res_stft(target_p, output_p, self.windows, self.p).mean()
            total = total + (1 - self.w) * (loss_present + self.wm * loss_mr)
            parts["L_pres"] = float(loss_present.detach())
            parts["L_MR"] = float(loss_mr.detach())

        if n_absent:
            loss_absent = self._loss_target_absent(x_input[crop_absent],
                                                   s_output[crop_absent], self.tau_abs).mean()
            total = total + self.w * loss_absent    
            parts["L_abs"] = float(loss_absent.detach())

        # Both guards are required, not defensive. At batch 12 and the measured
        # 0.297 absent rate, ~1.5 % of batches contain no absent crop (0.703^12)
        # and ~5e-7 contain no present crop. An unguarded mean over an empty
        # selection is NaN.
        parts["total"] = float(total.detach())
        return total, parts
